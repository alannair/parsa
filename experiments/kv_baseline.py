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
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "experiments" / "_runs" / "kv-baseline"
SUMMARY_CSV = RUNS_ROOT / "kv-baseline-summary.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_config import _client_extra_args, load_experiments
from run import plan_serving_profile
from src.serving_profiles import apply_serving_config, runtime_package_context


SUMMARY_FIELDS = [
    "experiment",
    "workload",
    "max_num_batched_tokens",
    "max_num_seqs",
    "frequency_mhz",
    "max_tokens",
    "total_num_reqs",
    "kv_mean_utilization",
    "kv_p95_utilization",
    "kv_peak_utilization",
    "kv_host_swap_samples",
    "energy_joules",
    "energy_joules_per_output_token",
    "tokens_per_second",
    "mean_ttft",
    "99.9th_ttft",
    "mean_tpot",
    "99.9th_tpot",
    "total_output_tokens",
    "num_requests",
    "result_json",
]


def _emit(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)


def _slugify(name):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return slug or "experiment"


def _write_presets(run_dir, lanes):
    from src.presets import load_presets_json

    presets_path = run_dir / "presets.json"
    with open(presets_path, "w", encoding="utf-8") as handle:
        json.dump(lanes, handle, ensure_ascii=False, indent=2)
    load_presets_json(str(presets_path))
    return presets_path


def _build_baseline_env(exp, presets_path, result_json, run_dir):
    env = os.environ.copy()
    env.pop("ROUTING_POLICY", None)
    env.pop("ROUTING_KWARGS", None)
    env.pop("CLIENT_EXTRA_ARGS", None)
    env["ONLINE_SERVING"] = str(REPO_ROOT)
    env["MODEL"] = exp["model"]
    env["TP_SIZE"] = str(exp["tp"])
    env["PRESETS_JSON"] = str(presets_path)
    env["CLIENT_MAX_ARRIVALS"] = str(exp["total_num_reqs"])
    env["RESULT_JSON"] = str(result_json)
    env["DEMO_STATE_DIR"] = str(run_dir / "state")
    env["SKIP_ROUTER"] = "1"
    env["POWER_MONITOR_ENABLED"] = "1"
    env["KV_MONITOR_ENABLED"] = "1"
    client = dict(exp.get("client") or {})
    tuning = dict(exp.get("tuning") or {})
    max_seqs = tuning.get("max_num_seqs")
    if max_seqs is not None:
        client["max_in_flight"] = int(max_seqs)
    client_args = _client_extra_args(client)
    if client_args:
        env["CLIENT_EXTRA_ARGS"] = shlex.join(client_args)
    return apply_serving_config(env, exp.get("serving"))


def _result_row(exp, result_json):
    row = {field: "" for field in SUMMARY_FIELDS}
    row["experiment"] = exp.get("name") or ""
    row["workload"] = exp.get("workload") or ""
    row["result_json"] = str(result_json)
    tuning = dict(exp.get("tuning") or {})
    client = dict(exp.get("client") or {})
    row["max_num_batched_tokens"] = tuning.get("max_num_batched_tokens", "")
    row["max_num_seqs"] = tuning.get("max_num_seqs", "")
    row["frequency_mhz"] = tuning.get("frequency_mhz", "")
    row["max_tokens"] = client.get("max_tokens", "")
    row["total_num_reqs"] = exp.get("total_num_reqs", "")
    if not result_json.is_file():
        return row
    with open(result_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mapping = {
        "kv_mean_utilization": "kv_mean_utilization",
        "kv_p95_utilization": "kv_p95_utilization",
        "kv_peak_utilization": "kv_peak_utilization",
        "kv_host_swap_samples": "kv_host_swap_samples",
        "energy_joules": "energy_joules",
        "energy_joules_per_output_token": "energy_joules_per_output_token",
        "tokens_per_second": "tokens_per_second",
        "mean_ttft": "mean_ttft",
        "99.9th_ttft": "99.9th_ttft",
        "mean_tpot": "mean_tpot",
        "99.9th_tpot": "99.9th_tpot",
        "total_output_tokens": "total_output_tokens",
        "num_requests": "num_requests",
    }
    for out_key, payload_key in mapping.items():
        value = payload.get(payload_key)
        if value is not None:
            row[out_key] = value
    return row


def write_summary_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="experiments.kv_baseline",
        description="Run non-Parsa KV cache / energy / SLO baseline experiments (direct vLLM, no router).",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(REPO_ROOT / "experiments" / "kv-baseline.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print expanded experiment names and exit")
    parser.add_argument("--run-script", default=str(REPO_ROOT / "run.py"))
    parser.add_argument(
        "--summary-csv",
        default=str(SUMMARY_CSV),
        help="Aggregate motivation table written after all runs (default: experiments/_runs/kv-baseline/kv-baseline-summary.csv)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    experiments = load_experiments(args.config)
    if args.list:
        for exp in experiments:
            tuning = exp.get("tuning") or {}
            client = exp.get("client") or {}
            _emit(
                "%s workload=%s mbt=%s ms=%s f=%s max_tokens=%s reqs=%s"
                % (
                    exp["name"],
                    exp.get("workload") or "",
                    tuning.get("max_num_batched_tokens", ""),
                    tuning.get("max_num_seqs", ""),
                    tuning.get("frequency_mhz", ""),
                    client.get("max_tokens", ""),
                    exp["total_num_reqs"],
                )
            )
        return 0
    anchor = runtime_package_context()
    plans = []
    for exp in experiments:
        slug = _slugify(exp["name"])
        run_dir = RUNS_ROOT / slug
        env = _build_baseline_env(exp, run_dir / "presets.json", run_dir / "result.json", run_dir)
        extra_args, _, metadata, condition = plan_serving_profile(
            exp["model"], exp["tp"], shlex.split(env.get("VLLM_SERVE_EXTRA_ARGS", "")), env,
            versions=anchor["versions"], origins=anchor["origins"],
        )
        run_dir = RUNS_ROOT / (slug + "-" + condition)
        env["PRESETS_JSON"] = str(run_dir / "presets.json")
        env["RESULT_JSON"] = str(run_dir / "result.json")
        env["DEMO_STATE_DIR"] = str(run_dir / "state")
        env["VLLM_EXPECTED_SERVING_CONDITION"] = condition
        plans.append((exp, run_dir, env, metadata, extra_args))
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    last_code = 0
    summary_rows = []
    for exp, run_dir, env, metadata, extra_args in plans:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_presets(run_dir, exp["lanes"])
        result_json = Path(env["RESULT_JSON"])
        command = [sys.executable, str(args.run_script)]
        if args.dry_run:
            _emit(exp["name"])
            _emit("  command:", " ".join(command))
            _emit("  serving profile: id=%s status=%s reason=%s" % (metadata["profile_id"], metadata["status"], metadata["reason"]))
            _emit("  serving args:", shlex.join(extra_args))
            for key in (
                "MODEL",
                "TP_SIZE",
                "PRESETS_JSON",
                "CLIENT_MAX_ARRIVALS",
                "RESULT_JSON",
                "DEMO_STATE_DIR",
                "SKIP_ROUTER",
                "POWER_MONITOR_ENABLED",
                "KV_MONITOR_ENABLED",
                "CLIENT_EXTRA_ARGS",
                "VLLM_SERVING_OVERRIDES",
                "VLLM_EXPECTED_SERVING_CONDITION",
            ):
                if key in env:
                    _emit("  %s=%s" % (key, env[key]))
            summary_rows.append(_result_row(exp, result_json))
            continue
        completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env)
        summary_rows.append(_result_row(exp, result_json))
        if completed.returncode != 0:
            last_code = completed.returncode
            if args.fail_fast:
                write_summary_csv(summary_rows, Path(args.summary_csv))
                return last_code
    write_summary_csv(summary_rows, Path(args.summary_csv))
    if not args.dry_run:
        _emit("summary:", args.summary_csv)
    return last_code


if __name__ == "__main__":
    sys.exit(main())
