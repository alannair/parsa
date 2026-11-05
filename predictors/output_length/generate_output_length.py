# Copyright (C) 2026 M.R. Siavash Katebzadeh <mr@katebzadeh.xyz>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictors.output_length.pools import (
    DEFAULT_ARXIV,
    DEFAULT_HUMANEVAL,
    DEFAULT_SHAREGPT,
    load_prompt_pool,
    normalize_active_pools,
)

CSV_COLUMNS = ("prompt", "type", "output_length")

DEFAULT_MODEL = "/mnt/nvme1/models/qwen3-4b"
DEFAULT_ORIGINAL_CSV = "dataset_output_length.csv"
PROGRESS_BAR_WIDTH = 24


def _stderr(*parts):
    print(*parts, file=sys.stderr)


def _parse_devices(devices):
    items = [item.strip() for item in str(devices or "").split(",") if item.strip()]
    if not items:
        raise ValueError("ascend_devices must list at least one device id")
    return items


def _chunked(items, size):
    size = max(1, int(size))
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _format_bar(done, total, width=PROGRESS_BAR_WIDTH):
    if total <= 0:
        return "." * width
    filled = int(width * done / total)
    filled = min(width, max(0, filled))
    return ("#" * filled) + ("." * (width - filled))


class InstanceProgress:
    def __init__(self, device_ids):
        self.device_ids = list(device_ids)
        self.state = {device_id: (0, 0) for device_id in self.device_ids}
        self.lock = threading.Lock()
        self._lines_printed = 0

    def set_total(self, device_id, total_batches):
        with self.lock:
            done, _total = self.state.get(device_id, (0, 0))
            self.state[device_id] = (done, int(total_batches))
            self._render_locked()

    def advance(self, device_id, done_batches):
        with self.lock:
            _done, total = self.state.get(device_id, (0, 0))
            self.state[device_id] = (int(done_batches), total)
            self._render_locked()

    def _render_locked(self):
        lines = []
        for device_id in self.device_ids:
            done, total = self.state.get(device_id, (0, 0))
            lines.append(
                "device %s [%s] %d/%d batches"
                % (device_id, _format_bar(done, total), done, total)
            )
        if self._lines_printed:
            sys.stderr.write("\033[%dA" % self._lines_printed)
        for line in lines:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()
        self._lines_printed = len(lines)

    def finish(self):
        with self.lock:
            if self._lines_printed:
                sys.stderr.write("\n")
                sys.stderr.flush()
                self._lines_printed = 0


def _url_ready(url, timeout_s=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return int(getattr(response, "status", 200)) < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _health_ready(base_url, timeout_s=2.0):
    root = base_url.rstrip("/")
    for path in ("/health", "/v1/models"):
        if _url_ready(root + path, timeout_s=timeout_s):
            return True
    return False


def _tail_log_file(log_path, max_lines=40):
    log_path = Path(log_path)
    if not log_path.is_file():
        return "(log file not found: %s)" % log_path
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return "(could not read log: %s)" % exc
    if not lines:
        return "(log is empty)"
    return "\n".join(lines[-max_lines:])


def wait_for_vllm(base_url, timeout_s, *, process=None, log_path=None, device_id=None):
    deadline = time.time() + max(0, int(timeout_s))
    label = base_url
    if device_id is not None:
        label = "%s (device %s)" % (base_url, device_id)
    while time.time() < deadline:
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                hint = ""
                if log_path:
                    hint = "\n--- vLLM log tail (%s) ---\n%s" % (log_path, _tail_log_file(log_path))
                raise RuntimeError(
                    "vLLM process exited with code %s before %s became ready%s"
                    % (exit_code, label, hint)
                )
        if _health_ready(base_url):
            return
        time.sleep(2)
    hint = ""
    if log_path:
        hint = "\n--- vLLM log tail (%s) ---\n%s" % (log_path, _tail_log_file(log_path))
    if process is not None and process.poll() is not None:
        raise RuntimeError(
            "vLLM process exited with code %s; %s not ready within %ss%s"
            % (process.poll(), label, timeout_s, hint)
        )
    raise TimeoutError(
        "vLLM server did not become ready at %s within %ss. "
        "Check the log, model path, NPU visibility, and port conflicts.%s"
        % (label, timeout_s, hint)
    )


def start_vllm_instance(model, device, max_num_seqs, port, vllm_root, log_path):
    max_batched_tokens = max(2048, int(max_num_seqs) * 256)
    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(max_batched_tokens),
        "--max-num-seqs",
        str(max_num_seqs),
        "--trust-remote-code",
    ]
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(vllm_root),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return process


def stop_process(process):
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait()


def stop_all(processes):
    for process in list(processes or []):
        stop_process(process)


def completion_tokens_for_prompt(base_url, model, prompt, timeout_s=720.0):
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        _stderr("completion failed status=%s detail=%s" % (exc.code, detail))
        return None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        _stderr("completion failed: %s" % exc)
        return None

    usage = body.get("usage") or {}
    value = usage.get("completion_tokens")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    choices = body.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, str) and text.strip():
            return len(text.split())
    return None


def average_output_length(base_url, model, prompt, num_tries):
    if num_tries < 1:
        raise ValueError("num_tries must be >= 1")
    lengths = []
    for _ in range(num_tries):
        value = completion_tokens_for_prompt(base_url, model, prompt)
        if value is not None:
            lengths.append(value)
    if not lengths:
        return None
    return int(sum(lengths) / len(lengths))


def _label_prompt(base_url, model, indexed_item, num_tries):
    index, prompt, prompt_type = indexed_item
    output_length = average_output_length(base_url, model, prompt, num_tries)
    if output_length is None:
        _stderr(
            "skipping prompt (all %d tries failed, first 80 chars): %s"
            % (num_tries, prompt[:80])
        )
        return None
    return index, prompt, prompt_type, output_length


def process_batch(base_url, model, batch, num_tries):
    if not batch:
        return []
    workers = min(len(batch), 8)
    results = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_label_prompt, base_url, model, item, num_tries) for item in batch]
        for future in as_completed(futures):
            row = future.result()
            if row is None:
                skipped += 1
            else:
                results.append(row)
    results.sort(key=lambda row: row[0])
    return results, skipped


def split_pool_across_devices(pool, device_ids):
    shards = [[] for _ in device_ids]
    for index, item in enumerate(pool):
        shard_index = index % len(device_ids)
        shards[shard_index].append((index, item[0], item[1]))
    return list(zip(device_ids, shards))


def run_device_shard(
    device_id,
    base_url,
    model,
    shard,
    num_tries,
    max_num_seqs,
    progress,
    csv_writer,
):
    batches = list(_chunked(shard, max_num_seqs))
    progress.set_total(device_id, len(batches))
    for batch_index, batch in enumerate(batches, start=1):
        batch_rows, batch_skipped = process_batch(base_url, model, batch, num_tries)
        csv_writer.add_skips(batch_skipped)
        for _index, prompt, prompt_type, output_length in batch_rows:
            csv_writer.append(prompt, prompt_type, output_length)
        progress.advance(device_id, batch_index)


def start_vllm_fleet(model, device_ids, max_num_seqs, base_port, vllm_root, out_dir):
    processes = []
    for offset, device_id in enumerate(device_ids):
        port = int(base_port) + offset
        log_path = out_dir / ("vllm_device_%s.log" % device_id)
        _stderr("vLLM device %s -> port %d log %s" % (device_id, port, log_path))
        process = start_vllm_instance(model, device_id, max_num_seqs, port, vllm_root, log_path)
        processes.append((device_id, port, process, log_path))
    return processes


def wait_for_vllm_fleet(processes, timeout_s):
    for device_id, port, process, log_path in processes:
        base_url = "http://127.0.0.1:%d" % int(port)
        wait_for_vllm(
            base_url,
            timeout_s,
            process=process,
            log_path=log_path,
            device_id=device_id,
        )


class IncrementalCsvWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.row_count = 0
        self.skipped_count = 0
        self._handle = open(self.path, "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._handle.flush()

    def append(self, prompt, prompt_type, output_length):
        with self.lock:
            self._writer.writerow(
                {
                    "prompt": prompt,
                    "type": int(prompt_type),
                    "output_length": int(output_length),
                }
            )
            self._handle.flush()
            self.row_count += 1

    def add_skips(self, count):
        if count:
            with self.lock:
                self.skipped_count += int(count)

    def close(self):
        with self.lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


def read_labeled_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("labeled CSV not found: %s" % path)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV %s missing columns: %s" % (path, sorted(missing)))
        for row in reader:
            prompt = str(row.get("prompt") or "").strip()
            if not prompt:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "type": int(row["type"]),
                    "output_length": int(float(row["output_length"])),
                }
            )
    return rows


def write_labeled_csv_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_labeled_csv(base_path, new_rows_path, output_path=None):
    base_rows = read_labeled_csv_rows(base_path)
    new_rows = read_labeled_csv_rows(new_rows_path)
    merged = base_rows + new_rows
    out = Path(output_path or base_path)
    write_labeled_csv_rows(out, merged)
    return out, len(base_rows), len(new_rows), len(merged)


def generate_output_length_csv(
    model=DEFAULT_MODEL,
    num_tries=None,
    csv_name="dataset_output_length.csv",
    max_num_seqs=25,
    ascend_devices=None,
    sharegpt_path=DEFAULT_SHAREGPT,
    arxiv_path=DEFAULT_ARXIV,
    humaneval_path=DEFAULT_HUMANEVAL,
    num_prompts=0,
    output_dir=None,
    vllm_root=None,
    port=8400,
    vllm_startup_secs=300,
    manage_vllm=True,
    active_pools=None,
    append_to=None,
    merged_csv=None,
):
    if num_tries is None:
        raise ValueError("num_tries is required")
    if num_tries < 1:
        raise ValueError("num_tries must be >= 1")
    device_ids = _parse_devices(ascend_devices or os.environ.get("ASCEND_DEVICES", "0,1"))

    out_dir = Path(output_dir or (REPO_ROOT / "predictors" / "output_length"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / csv_name

    pool = load_prompt_pool(
        sharegpt_path=sharegpt_path,
        arxiv_path=arxiv_path,
        humaneval_path=humaneval_path,
        num_prompts=num_prompts,
        active_pools=active_pools,
    )
    _stderr(
        "labeling %d prompts (pools=%s, num_tries=%d)"
        % (len(pool), ",".join(sorted(normalize_active_pools(active_pools))), int(num_tries))
    )

    csv_writer = IncrementalCsvWriter(csv_path)
    vllm_fleet = []
    progress = InstanceProgress(device_ids)

    try:
        if manage_vllm:
            vllm_root_path = Path(vllm_root or os.environ.get("VLLM_ROOT", "/vllm-workspace/vllm"))
            _stderr(
                "starting %d vLLM instances (tp=1 each) model=%s devices=%s base_port=%d batch_size=%d"
                % (len(device_ids), model, ",".join(device_ids), int(port), int(max_num_seqs))
            )
            vllm_fleet = start_vllm_fleet(model, device_ids, max_num_seqs, port, vllm_root_path, out_dir)
            wait_for_vllm_fleet(vllm_fleet, vllm_startup_secs)

        device_shards = split_pool_across_devices(pool, device_ids)
        port_by_device = {}
        if manage_vllm:
            for device_id, instance_port, _process, _log_path in vllm_fleet:
                port_by_device[device_id] = instance_port
        else:
            for offset, device_id in enumerate(device_ids):
                port_by_device[device_id] = int(port) + offset

        def _run_one(device_id, shard):
            instance_port = port_by_device[device_id]
            base_url = "http://127.0.0.1:%d" % int(instance_port)
            run_device_shard(
                device_id,
                base_url,
                model,
                shard,
                num_tries,
                max_num_seqs,
                progress,
                csv_writer,
            )

        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            futures = [
                executor.submit(_run_one, device_id, shard)
                for device_id, shard in device_shards
            ]
            for future in as_completed(futures):
                future.result()

        progress.finish()
        _stderr(
            "wrote %s (%d rows, %d skipped)"
            % (csv_path, csv_writer.row_count, csv_writer.skipped_count)
        )
        if append_to:
            append_path = Path(append_to)
            if not append_path.is_absolute():
                append_path = out_dir / append_path
            merged_path = Path(merged_csv) if merged_csv else append_path
            if not merged_path.is_absolute():
                merged_path = out_dir / merged_path
            _merged_path, base_n, new_n, total_n = append_labeled_csv(
                append_path, csv_path, merged_path
            )
            _stderr(
                "merged %s + %s -> %s (%d + %d = %d rows)"
                % (append_path.name, csv_path.name, merged_path.name, base_n, new_n, total_n)
            )
            return merged_path
        return csv_path
    finally:
        progress.finish()
        csv_writer.close()
        if manage_vllm:
            stop_all([process for _device, _port, process, _log in vllm_fleet])


def build_parser():
    parser = argparse.ArgumentParser(
        prog="predictors.output_length.generate_output_length",
        description="Generate output-length labels by running vLLM on prompt pools.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path for vllm serve")
    parser.add_argument(
        "--num-tries",
        type=int,
        default=1,
        help="Completions per prompt (1 = single run, no averaging)",
    )
    parser.add_argument(
        "--csv-name",
        default="dataset_output_length.csv",
        help="Output CSV for this run (summarization-only labels)",
    )
    parser.add_argument(
        "--pools",
        default="conversation,summarization,code",
        help="Comma-separated pools: conversation, code, summarization",
    )
    parser.add_argument(
        "--append-to",
        default=None,
        help="After labeling, append new rows onto this existing labeled CSV",
    )
    parser.add_argument(
        "--merged-csv",
        default=None,
        help="Write combined CSV here (default: same path as --append-to)",
    )
    parser.add_argument("--max-num-seqs", default=25, type=int, help="Batch size and vLLM --max-num-seqs per instance")
    parser.add_argument(
        "--ascend-devices",
        default=os.environ.get("ASCEND_DEVICES", "0,1"),
        help="Comma-separated Ascend device ids (one vLLM instance per device, tp=1)",
    )
    parser.add_argument("--sharegpt-path", default=DEFAULT_SHAREGPT)
    parser.add_argument("--arxiv-path", default=DEFAULT_ARXIV)
    parser.add_argument("--humaneval-path", default=DEFAULT_HUMANEVAL)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=0,
        help="Max prompts per pool (0 = use all prompts from each pool)",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "predictors" / "output_length"))
    parser.add_argument("--vllm-root", default=os.environ.get("VLLM_ROOT", "/vllm-workspace/vllm"))
    parser.add_argument("--port", default=8400, type=int, help="Base port; instance i uses port+i")
    parser.add_argument(
        "--vllm-startup-secs",
        default=900,
        type=int,
        help="Wait up to N seconds per vLLM instance for /health or /v1/models (Ascend often needs >300s)",
    )
    parser.add_argument(
        "--no-manage-vllm",
        action="store_true",
        help="Use already-running vLLM servers at base_port, base_port+1, ...",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    generate_output_length_csv(
        model=args.model,
        num_tries=args.num_tries,
        csv_name=args.csv_name,
        max_num_seqs=args.max_num_seqs,
        ascend_devices=args.ascend_devices,
        sharegpt_path=args.sharegpt_path,
        arxiv_path=args.arxiv_path,
        humaneval_path=args.humaneval_path,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir,
        vllm_root=args.vllm_root,
        port=args.port,
        vllm_startup_secs=args.vllm_startup_secs,
        manage_vllm=not args.no_manage_vllm,
        active_pools=args.pools,
        append_to=args.append_to,
        merged_csv=args.merged_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
