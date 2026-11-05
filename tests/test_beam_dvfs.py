#!/usr/bin/env python3
"""Generate BEAM-like LLM serving workloads and sample Ascend AI-core frequency.

Only Python's standard library is needed. Run on the NPU host (not inside a
container that lacks npu-smi) while a vLLM OpenAI-compatible server is up.

Examples:
  python3 beam_dvfs_test.py --mode burst --url http://127.0.0.1:8000
  python3 beam_dvfs_test.py --mode phases --phase-seconds 25
  python3 beam_dvfs_test.py --mode burst --high-rps 20 --max-inflight 32
"""

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request


def get_json(url, *, payload=None, timeout=120):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read(1000).decode("utf-8", "replace")}


def smi_sample(device):
    cmd = ["npu-smi", "info", "-t", "common", "-i", str(device)]
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{cmd!r}: {exc}"
    if out.returncode:
        return None, f"{cmd!r} failed: {(out.stderr or out.stdout).strip()}"

    def field(name):
        m = re.search(r"^\s*" + re.escape(name) + r"\s*:\s*([\d.]+)",
                      out.stdout, re.MULTILINE | re.IGNORECASE)
        return float(m.group(1)) if m else None

    values = {
        "freq_mhz": field("Aicore curFreq(MHZ)"),
        "rated_mhz": field("Aicore Freq(MHZ)"),
        "power_w": field("NPU Real-time Power(W)"),
        "aicore_pct": field("Aicore Usage Rate(%)"),
    }
    if values["freq_mhz"] is None:
        return None, f"Current AI-core frequency missing in npu-smi output:\n{out.stdout[:1200]}"
    return values, None


def prompt_text(approx_tokens, request_id):
    # Exact prompt length is tokenizer-dependent; record actual usage from server.
    unit = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
    repeats = max(1, round(approx_tokens / 12))
    # Different first block prevents prefix caching from masking prefill work.
    return (f"Unique task {request_id:010d}. /no_think\n"
            + unit * repeats
            + "\nContinue a numbered list of practical computing concepts. "
              "Make the list long and do not stop early.")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def run(args):
    base = args.url.rstrip("/")
    if args.model:
        model = args.model
    else:
        status, res = await asyncio.to_thread(get_json, base + "/v1/models", timeout=10)
        if status != 200 or not res.get("data"):
            raise RuntimeError(f"GET /v1/models failed: HTTP {status}: {res}")
        model = res["data"][0]["id"]

    initial, error = await asyncio.to_thread(smi_sample, args.device)
    if error:
        raise RuntimeError(error)

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    samples_file = prefix.with_name(prefix.name + "_samples.csv")
    requests_file = prefix.with_name(prefix.name + "_requests.csv")
    t0 = time.monotonic()
    phase = "starting"
    sent = completed = errors = inflight = dropped = next_id = 0
    stop_polling = False
    tasks = set()

    if args.mode == "burst":
        phases = [
            ("idle-before", 0, 0, 0, args.idle_seconds),
            ("low-before", args.low_rps, 256, 64, args.phase_seconds),
            ("HIGH-burst", args.high_rps, 256, 64, args.phase_seconds),
            ("low-after", args.low_rps, 256, 64, args.phase_seconds),
        ]
    elif args.mode == "sweep":
        # Five prefill lengths exercise the synthetic profile's clock ladder.
        phases = [("idle-before", 0, 0, 0, args.idle_seconds)]
        for tokens in (64, 256, 512, 1024, 2048):
            phases.extend([
                (f"PREFILL-{tokens}", args.prefill_rps, tokens, 8, args.phase_seconds),
                (f"drain-{tokens}", 0, 0, 0, args.idle_seconds),
            ])
        phases.extend([
            ("DECODE-heavy", args.decode_rps, 64, 256, args.phase_seconds),
            ("drain-decode", 0, 0, 0, args.idle_seconds),
        ])
    else:
        phases = [
            ("idle-before", 0, 0, 0, args.idle_seconds),
            ("PREFILL-heavy", args.prefill_rps, 2048, 8, args.phase_seconds),
            ("drain-prefill", 0, 0, 0, args.idle_seconds),
            ("DECODE-heavy", args.decode_rps, 64, 256, args.phase_seconds),
            ("drain-decode", 0, 0, 0, args.idle_seconds),
        ]

    with samples_file.open("w", newline="") as sfile, \
         requests_file.open("w", newline="") as rfile:
        s = csv.writer(sfile)
        r = csv.writer(rfile)
        s.writerow(["elapsed_s", "utc", "phase", "sent", "completed", "errors",
                    "inflight", "dropped", "freq_mhz", "rated_mhz", "power_w", "aicore_pct"])
        r.writerow(["id", "phase", "start_s", "end_s", "status", "latency_s",
                    "approx_input_tokens", "requested_output_tokens", "actual_prompt_tokens",
                    "actual_completion_tokens", "error"])

        async def poll():
            last_report = -2.0
            while not stop_polling:
                start = time.monotonic()
                metrics, issue = await asyncio.to_thread(smi_sample, args.device)
                if issue:
                    print(f"WARNING: {issue}", flush=True)
                    metrics = {}
                now = time.monotonic() - t0
                s.writerow([f"{now:.3f}", utc_now(), phase, sent, completed, errors,
                            inflight, dropped, *(metrics.get(k) for k in
                            ("freq_mhz", "rated_mhz", "power_w", "aicore_pct"))])
                sfile.flush()
                if now - last_report >= 2.0:
                    last_report = now
                    print(f"{now:6.1f}s  {phase:15s}  "
                          f"freq={metrics.get('freq_mhz', 'NA')} MHz  "
                          f"power={metrics.get('power_w', 'NA')} W  "
                          f"active={inflight} done={completed} err={errors}", flush=True)
                await asyncio.sleep(max(0.05, args.sample_interval - (time.monotonic() - start)))

        async def send_request(req_id, req_phase, in_tokens, out_tokens):
            nonlocal inflight, completed, errors
            start = time.monotonic() - t0
            inflight += 1
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt_text(in_tokens, req_id)}],
                "temperature": 0.0,
                "max_tokens": out_tokens,
                "ignore_eos": True,  # vLLM extension; ensures decode-heavy requests stay long
                "stream": False,
            }
            status = 0
            result = {}
            issue = ""
            try:
                status, result = await asyncio.to_thread(
                    get_json, base + "/v1/chat/completions", payload=payload,
                    timeout=args.timeout)
                if status != 200:
                    issue = str(result.get("error", result))[:300]
            except Exception as exc:
                issue = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                end = time.monotonic() - t0
                inflight -= 1
                completed += 1
                errors += bool(issue)
                usage = result.get("usage") or {}
                r.writerow([req_id, req_phase, f"{start:.3f}", f"{end:.3f}", status,
                            f"{end - start:.3f}", in_tokens, out_tokens,
                            usage.get("prompt_tokens"), usage.get("completion_tokens"), issue])
                rfile.flush()
                if issue and errors <= 5:
                    print(f"  ERROR request {req_id}: HTTP {status} {issue}", flush=True)

        polling = asyncio.create_task(poll())
        try:
            print(f"Model: {model} | NPU: {args.device} | "
                  f"initial clock: {initial['freq_mhz']} MHz | "
                  f"out: {samples_file}, {requests_file}", flush=True)
            for name, rate, input_tokens, output_tokens, duration in phases:
                phase = name
                if name.startswith("drain-") and tasks:
                    print(f"\n--- {name}: waiting for {len(tasks)} outstanding requests ---", flush=True)
                    await asyncio.gather(*tasks)
                print(f"\n--- {name}: {rate:g} req/s, "
                      f"~{input_tokens} input / {output_tokens} output tokens, "
                      f"{duration:g}s ---", flush=True)
                end_time = time.monotonic() + duration
                next_arrival = time.monotonic()
                while time.monotonic() < end_time:
                    if not rate:
                        await asyncio.sleep(min(0.2, max(0.0, end_time - time.monotonic())))
                        continue
                    await asyncio.sleep(max(0.0, next_arrival - time.monotonic()))
                    next_arrival += 1.0 / rate
                    if time.monotonic() >= end_time:
                        break
                    if inflight >= args.max_inflight:
                        dropped += 1
                        continue
                    next_id += 1
                    sent += 1
                    task = asyncio.create_task(send_request(next_id, name, input_tokens, output_tokens))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            phase = "draining"
            print("\n--- Waiting for outstanding requests to finish ---", flush=True)
            if tasks:
                await asyncio.gather(*tasks)
            phase = "idle-after"
            await asyncio.sleep(args.idle_seconds)
        finally:
            stop_polling = True
            await polling

    unique_freqs = set()
    with samples_file.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["freq_mhz"]:
                unique_freqs.add(float(row["freq_mhz"]))
    print(f"\nDone. HTTP requests: sent={sent}, completed={completed}, "
          f"errors={errors}, dropped_at_concurrency_limit={dropped}.")
    print(f"Observed AI-core frequencies (MHz): {sorted(unique_freqs)}")
    print(f"CSV: {samples_file}\nRequests: {requests_file}")
    if len(unique_freqs) == 1:
        print("CLOCK DID NOT CHANGE. Check that BEAM S1/S2, the Ascend DVFS "
              "actuator, and the model-specific offline profile are enabled.")
    make_plot(samples_file, prefix.with_name(prefix.name + "_frequency.png"))


def make_plot(samples_file, plot_file):
    """Optional post-run plot. Skipped if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Optional: install matplotlib to also save a frequency timeline PNG.")
        return
    with samples_file.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    data = [(float(row["elapsed_s"]), float(row["freq_mhz"]), row["phase"])
            for row in rows if row["freq_mhz"]]
    if not data:
        return
    x = [row[0] for row in data]
    y = [row[1] for row in data]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(x, y, where="post", linewidth=1.8)
    ax.set(xlabel="Time since start (s)", ylabel="AI-core frequency (MHz)",
           title="Ascend NPU frequency under BEAM workloads")
    ax.grid(alpha=0.25)
    prev = None
    for t, _, ph in data:
        if ph != prev:
            ax.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.text(t, 1.02, ph, transform=ax.get_xaxis_transform(), rotation=35,
                    ha="left", va="bottom", fontsize=8)
            prev = ph
    fig.tight_layout()
    fig.savefig(plot_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {plot_file}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", help="Default: first ID returned by GET /v1/models")
    p.add_argument("--device", type=int, default=0, help="Host NPU ID for npu-smi")
    p.add_argument("--mode", choices=("burst", "phases", "sweep"), default="burst")
    p.add_argument("--phase-seconds", type=float, default=20.0)
    p.add_argument("--idle-seconds", type=float, default=8.0)
    p.add_argument("--low-rps", type=float, default=0.5)
    p.add_argument("--high-rps", type=float, default=4.0)
    p.add_argument("--prefill-rps", type=float, default=1.0)
    p.add_argument("--decode-rps", type=float, default=1.0)
    p.add_argument("--max-inflight", type=int, default=16)
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--out", default="beam_dvfs", help="Output path prefix")
    args = p.parse_args()
    for key in ("phase_seconds", "idle_seconds", "sample_interval", "timeout"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be > 0")
    if args.max_inflight < 1 or any(x < 0 for x in
                                     (args.low_rps, args.high_rps, args.prefill_rps, args.decode_rps)):
        p.error("Rates must be >= 0 and max-inflight must be >= 1")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
