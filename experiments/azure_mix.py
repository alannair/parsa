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

"""Run an existing Azure-mix schedule and write actual token lengths to CSV."""

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_CSV = REPO_ROOT / "datasets" / "azure_mix_results.csv"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run import output_identity, plan_serving_profile
from src.serving_profiles import apply_serving_config, runtime_package_context, verify_runtime_package_context


def build_parser():
    parser = argparse.ArgumentParser(
        prog="experiments.azure_mix",
        description="Serve a model locally, run an existing Azure-mix schedule, and write token lengths",
    )
    parser.add_argument("--schedule", required=True, help="Existing schedule CSV; it is never rebuilt")
    parser.add_argument("--result-csv", default=str(DEFAULT_RESULT_CSV))
    parser.add_argument("--model", default="/models/qwen3-8b")
    parser.add_argument("--port", default=8010, type=int)
    parser.add_argument("--max-arrivals", default=50, type=int, help="0 runs all usable schedule rows")
    parser.add_argument("--startup-timeout", default=600, type=int)
    parser.add_argument("--vllm-log", default=str(REPO_ROOT / "datasets" / "azure_mix_vllm.log"))
    parser.add_argument("--serving-overrides", help="JSON serving tuning intent: {args: option map, env: string map}")
    return parser


def _visible_devices():
    devices = [item.strip() for item in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if not devices:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES must list at least one device")
    return devices


def _vllm_command(model, port, devices):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(len(devices)),
    ]


def _wait_for_vllm(port, process, timeout):
    url = "http://127.0.0.1:%d/v1/models" % port
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before becoming ready")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError("vLLM did not become ready within %d seconds" % timeout)


def _stop_vllm(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def verified_serving_sidecar(result_csv):
    """Return content-bound provenance, or None for missing/unverified records."""
    result_csv = Path(result_csv)
    try:
        record = json.loads(result_csv.with_suffix(".serving.json").read_text(encoding="utf-8"))
        content = result_csv.read_bytes()
    except (OSError, ValueError):
        return None
    if (not isinstance(record, dict) or not isinstance(record.get("serving_profile"), dict)
            or record["serving_profile"].get("status") != "applied"
            or record["serving_profile"].get("runtime_verified") is not True
            or not isinstance(record.get("serving_condition"), str) or not record["serving_condition"]
            or type(record.get("client_returncode")) is not int
            or type(record.get("result_csv_bytes")) is not int or record["result_csv_bytes"] != len(content)
            or record.get("result_csv_sha256") != hashlib.sha256(content).hexdigest()):
        return None
    return record


def main(argv=None):
    args = build_parser().parse_args(argv)
    schedule = Path(args.schedule).expanduser().resolve()
    result_csv = Path(args.result_csv).expanduser().resolve()
    log_path = Path(args.vllm_log).expanduser().resolve()
    if not schedule.is_file():
        raise FileNotFoundError("schedule does not exist: %s" % schedule)
    if result_csv == schedule:
        raise ValueError("--result-csv must not overwrite --schedule")
    devices = _visible_devices()
    serving = json.loads(args.serving_overrides) if args.serving_overrides is not None else None
    env = apply_serving_config(os.environ, serving)
    anchor = runtime_package_context()
    extra_args, updates, metadata, condition = plan_serving_profile(
        args.model, len(devices), shlex.split(env.get("VLLM_SERVE_EXTRA_ARGS", "")), env,
        versions=anchor["versions"], origins=anchor["origins"],
    )
    env.update(updates)
    verified = verify_runtime_package_context(anchor, REPO_ROOT, env)
    metadata.update(runtime_verified=True, verified_package_origins=verified["origins"],
                    runtime_verification_scope="local-server-cwd-and-env")
    vllm_command = [*_vllm_command(args.model, args.port, devices), *extra_args]
    sidecar = result_csv.with_suffix(".serving.json")
    if schedule in (log_path, sidecar) or log_path in (result_csv, sidecar):
        raise ValueError("schedule, result CSV, serving sidecar and vLLM log must use distinct paths")
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.client",
        "--base-url",
        "http://127.0.0.1:%d/v1" % args.port,
        "--model",
        args.model,
        "--prompt-source",
        "azure-mix",
        "--active-pools",
        "all",
        "--azure-mix-schedule-csv",
        str(schedule),
        "--max-arrivals",
        str(args.max_arrivals),
        "--result-csv",
        str(result_csv),
    ]
    print(shlex.join(vllm_command), flush=True)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            vllm_command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_vllm(args.port, process, args.startup_timeout)
            print(shlex.join(command), flush=True)
            before = output_identity(result_csv)
            completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
            after = output_identity(result_csv)
            code = int(completed.returncode)
            if after is not None and after != before:
                content = result_csv.read_bytes()
                record = {
                    "serving_profile": metadata, "serving_condition": condition,
                    "argv": vllm_command, "cwd": str(REPO_ROOT),
                    "effective_env": {"ASCEND_RT_VISIBLE_DEVICES": env["ASCEND_RT_VISIBLE_DEVICES"],
                                      **metadata["relevant_env"]},
                    "result_csv_sha256": hashlib.sha256(content).hexdigest(),
                    "result_csv_bytes": len(content), "client_returncode": code,
                }
                handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=sidecar.parent,
                                                     prefix=sidecar.name + ".", suffix=".tmp", delete=False)
                temporary = Path(handle.name)
                try:
                    with handle:
                        json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        handle.write("\n")
                    os.replace(temporary, sidecar)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                print("Client produced no current result CSV; leaving previous CSV/sidecar untouched", file=sys.stderr)
                if code == 0:
                    raise RuntimeError("Client exited successfully without a current result CSV")
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            _stop_vllm(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
