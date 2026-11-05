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
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "experiments" / "_runs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_config import _client_extra_args, load_experiments
from run import SERVING_METADATA_KEYS, plan_serving_profile
from src.serving_profiles import apply_serving_config, runtime_package_context


def _emit(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        # Piped output (e.g. `| head`) may close stdout early; exit quietly.
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)


def _slugify(name):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return slug or "experiment"


def _progress_status(completed, total, name, returncode):
    if total <= 1:
        return None
    return "[%d/%d] finished %s (exit %d)" % (completed, total, name, returncode)


def _write_presets(run_dir, lanes):
    from src.presets import load_presets_json

    presets_path = run_dir / "presets.json"
    with open(presets_path, "w", encoding="utf-8") as handle:
        json.dump(lanes, handle, ensure_ascii=False, indent=2)
    load_presets_json(str(presets_path))
    return presets_path


def _build_env(exp, presets_path, result_json, run_dir):
    env = os.environ.copy()
    env.setdefault("POWER_INTERVAL_S", "0.05")
    env.pop("ROUTING_POLICY", None)
    env.pop("ROUTING_KWARGS", None)
    env.pop("CLIENT_EXTRA_ARGS", None)
    env["ONLINE_SERVING"] = str(REPO_ROOT)
    env["MODEL"] = exp["model"]
    env["TP_SIZE"] = str(exp["tp"])
    env["PRESETS_JSON"] = str(presets_path)
    env["CLIENT_MAX_ARRIVALS"] = str(exp["total_num_reqs"])
    env["RESULT_JSON"] = str(result_json)
    if exp.get("routing_policy"):
        env["ROUTING_POLICY"] = str(exp.get("routing_policy"))
    if exp.get("routing_kwargs") is not None:
        env["ROUTING_KWARGS"] = json.dumps(exp.get("routing_kwargs"), ensure_ascii=False)
    client = dict(exp.get("client") or {})
    client_args = _client_extra_args(client)
    if client_args:
        env["CLIENT_EXTRA_ARGS"] = shlex.join(client_args)
    metadata = dict(exp.get("metadata") or {})
    if SERVING_METADATA_KEYS.intersection(metadata):
        raise ValueError("experiment metadata uses reserved serving_profile/serving_condition keys")
    metadata.update({
        "frequency_mhz": exp["tuning"]["frequency_mhz"],
        "max_num_seqs": exp["tuning"]["max_num_seqs"],
        "max_num_batched_tokens": exp["tuning"]["max_num_batched_tokens"],
        "repetition": exp.get("repetition"),
        "tp": exp["tp"],
    })
    env["EXPERIMENT_METADATA"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    return apply_serving_config(env, exp.get("serving"))


def build_parser():
    parser = argparse.ArgumentParser(prog="experiments.benchmarks")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(REPO_ROOT / "experiments" / "benchmarks.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print expanded experiment names and exit")
    parser.add_argument("--run-script", default=str(REPO_ROOT / "run.py"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    experiments = load_experiments(args.config)
    if args.list:
        for exp in experiments:
            tuning = exp.get("tuning") or {}
            _emit(
                "%s role=%s workload=%s mbt=%s ms=%s f=%s reqs=%s"
                % (
                    exp["name"],
                    exp.get("role") or "",
                    exp.get("workload") or "",
                    tuning.get("max_num_batched_tokens", ""),
                    tuning.get("max_num_seqs", ""),
                    tuning.get("frequency_mhz", ""),
                    exp["total_num_reqs"],
                )
            )
        return 0
    anchor = runtime_package_context()
    plans = []
    for exp in experiments:
        slug = _slugify(exp["name"])
        run_dir = RUNS_ROOT / slug
        env = _build_env(exp, run_dir / "presets.json", run_dir / "result.json", run_dir)
        extra_args, _, metadata, condition = plan_serving_profile(
            exp["model"], exp["tp"], shlex.split(env.get("VLLM_SERVE_EXTRA_ARGS", "")), env,
            versions=anchor["versions"], origins=anchor["origins"],
        )
        run_dir = RUNS_ROOT / (slug + "-" + condition)
        env["PRESETS_JSON"] = str(run_dir / "presets.json")
        env["RESULT_JSON"] = str(run_dir / "result.json")
        env["VLLM_EXPECTED_SERVING_CONDITION"] = condition
        plans.append((exp, run_dir, env, metadata, extra_args))
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    last_code = 0
    for index, (exp, run_dir, env, metadata, extra_args) in enumerate(plans, 1):
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_presets(run_dir, exp["lanes"])
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
                "ROUTING_POLICY",
                "ROUTING_KWARGS",
                "CLIENT_EXTRA_ARGS",
                "VLLM_SERVING_OVERRIDES",
                "VLLM_EXPECTED_SERVING_CONDITION",
            ):
                if key in env:
                    _emit("  %s=%s" % (key, env[key]))
            continue
        completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env)
        status = _progress_status(index, len(experiments), exp["name"], completed.returncode)
        if status:
            _emit(status)
        if completed.returncode != 0:
            last_code = completed.returncode
            if args.fail_fast:
                return last_code
    return last_code


if __name__ == "__main__":
    sys.exit(main())
