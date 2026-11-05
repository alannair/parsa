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


import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import kill as kill_module
from src.serving_profiles import (
    resolve_serving_profile, runtime_package_context, serving_condition_key, verify_runtime_package_context,
)


SERVING_METADATA_KEYS = frozenset({"serving_profile", "serving_condition"})


def _env_bool(name, default = False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _split_extra_args(value):
    text = str(value or "").strip()
    if not text:
        return []
    return shlex.split(text)


def output_identity(path):
    """Identify an output across in-place writes and atomic replacements."""
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def plan_serving_profile(model, tp, explicit_args, environment, versions=None, origins=None):
    """Resolve raw argv/env intent into (effective argv, env updates, metadata, condition).

    Qualifiers record context, with TP owned by the runner. Returned defaults
    belong only in the final lane launch; pass raw intent to child planners.
    """
    raw_qualifiers = environment.get("VLLM_SERVING_PROFILE_QUALIFIERS", "")
    try:
        qualifiers = json.loads(raw_qualifiers) if raw_qualifiers else {}
    except json.JSONDecodeError as exc:
        raise ValueError("VLLM_SERVING_PROFILE_QUALIFIERS must be a JSON object") from exc
    if not isinstance(qualifiers, dict):
        raise ValueError("VLLM_SERVING_PROFILE_QUALIFIERS must be a JSON object")
    if "tp" in qualifiers:
        raise ValueError("VLLM_SERVING_PROFILE_QUALIFIERS cannot override TP; use TP_SIZE")
    qualifiers["tp"] = int(tp)
    args, updates, metadata = resolve_serving_profile(
        model, explicit_args, environment, versions=versions,
        qualifiers=qualifiers,
    )
    metadata.update(
        versions_source="launcher-interpreter-distribution-metadata",
        launcher_interpreter=sys.executable,
        scope="local-launch-plan",
    )
    if origins is not None:
        metadata.update(package_origins=origins, package_origins_source="parent-interpreter-find-spec-unverified")
    return args, updates, metadata, serving_condition_key(metadata)


@dataclass
class LanePlanRow:
    preset_id: str
    port: str
    devices: str
    max_tokens: int
    max_seqs: int
    frequency_mhz: str
    start: bool


@dataclass
class RunConfig:
    online_serving: Path
    vllm_root: Path
    model: str
    tp_size: int
    ascend_devices: str
    wait_secs: int
    wait_bar_width: int
    router_port: int
    lane_p_port: str
    lane_d_port: str
    slo_max_latency_s: float
    presets_json: Path
    profile_json: Path
    router_extra_args: list
    routing_policy: str
    routing_kwargs: str
    client_max_arrivals: int
    client_extra_args: list
    kill_after_client: bool
    vllm_serve_extra_args: list
    power_monitor_enabled: bool
    power_interval_s: float
    kv_monitor_enabled: bool
    kv_interval_s: float
    kv_timeout_s: float
    result_json: Path
    demo_state_dir: Path
    rescue_preset_id: str
    lane_devices_json: str
    skip_vllm: bool
    skip_vllm_lanes: set
    skip_vllm_p: bool
    skip_vllm_d: bool
    skip_router: bool
    skip_set_freq_from_presets: bool
    experiment_metadata: dict = field(default_factory=dict)
    serving_extra_args: list = field(default_factory=list)
    serving_env_updates: dict = field(default_factory=dict)
    serving_profile: dict = field(default_factory=dict)
    serving_condition: str = ""
    serving_launch_env: dict | None = None

    @property
    def router_pid_path(self):
        return self.demo_state_dir / "router.pid"

    @property
    def router_log_path(self):
        return self.demo_state_dir / "router.log"

    @property
    def lane_plan_path(self):
        return self.demo_state_dir / "lane_plan.tsv"

    @property
    def pid_files_list_path(self):
        return self.demo_state_dir / "pid_files.list"

    @property
    def power_log_path(self):
        return self.demo_state_dir / "power.jsonl"

    @property
    def power_summary_path(self):
        return self.demo_state_dir / "power_summary.json"

    @property
    def kv_log_path(self):
        return self.demo_state_dir / "kv.jsonl"

    @property
    def kv_summary_path(self):
        return self.demo_state_dir / "kv_summary.json"


def load_config():
    ascend_p = os.environ.get("ASCEND_P_DEVICES", "0,1")
    ascend_d = os.environ.get("ASCEND_D_DEVICES", "2,3")
    ascend_devices = os.environ.get("ASCEND_DEVICES", f"{ascend_p},{ascend_d}")
    demo_state_dir = Path(os.path.abspath(os.path.expanduser(os.environ.get("DEMO_STATE_DIR", "/tmp/energy_aware_demo"))))
    result_default = Path.home() / "Projects" / "workspace" / "result.json"
    raw_metadata = os.environ.get("EXPERIMENT_METADATA", "")
    try:
        experiment_metadata = json.loads(raw_metadata) if raw_metadata else {}
    except json.JSONDecodeError as exc:
        raise ValueError("EXPERIMENT_METADATA must be a JSON object") from exc
    if not isinstance(experiment_metadata, dict):
        raise ValueError("EXPERIMENT_METADATA must be a JSON object")
    return RunConfig(
        online_serving=Path(os.environ.get("ONLINE_SERVING", str(REPO_ROOT))).resolve(),
        vllm_root=Path(os.environ.get("VLLM_ROOT", "/vllm-workspace/vllm")).resolve(),
        model=os.environ.get("MODEL", "Qwen/Qwen3-4B"),
        tp_size=max(1, _env_int("TP_SIZE", 2)),
        ascend_devices=ascend_devices,
        wait_secs=max(0, _env_int("WAIT_SECS", 180)),
        wait_bar_width=max(1, _env_int("WAIT_BAR_WIDTH", 40)),
        router_port=_env_int("ROUTER_PORT", 8010),
        lane_p_port=os.environ.get("LANE_P_PORT", "8100"),
        lane_d_port=os.environ.get("LANE_D_PORT", "8200"),
        slo_max_latency_s=_env_float("SLO_MAX_LATENCY_S", 120.0),
        presets_json=Path(os.environ.get("PRESETS_JSON", "src/presets.example.json")),
        profile_json=Path(os.environ.get("PROFILE_JSON", "src/profile.example.json")),
        router_extra_args=_split_extra_args(os.environ.get("ROUTER_EXTRA_ARGS", "")),
        routing_policy=os.environ.get("ROUTING_POLICY", "").strip(),
        routing_kwargs=os.environ.get("ROUTING_KWARGS", "").strip(),
        client_max_arrivals=_env_int("CLIENT_MAX_ARRIVALS", 50),
        client_extra_args=_split_extra_args(os.environ.get("CLIENT_EXTRA_ARGS", "")),
        kill_after_client=_env_bool("KILL_AFTER_CLIENT", True),
        vllm_serve_extra_args=_split_extra_args(os.environ.get("VLLM_SERVE_EXTRA_ARGS", "")),
        power_monitor_enabled=_env_bool("POWER_MONITOR_ENABLED", True),
        power_interval_s=_env_float("POWER_INTERVAL_S", 1.0),
        kv_monitor_enabled=_env_bool("KV_MONITOR_ENABLED", True),
        kv_interval_s=_env_float("KV_MONITOR_INTERVAL_S", 1.0),
        kv_timeout_s=_env_float("KV_MONITOR_TIMEOUT_S", 1.0),
        result_json=Path(os.path.abspath(os.path.expanduser(os.environ.get("RESULT_JSON", str(result_default))))),
        demo_state_dir=demo_state_dir,
        rescue_preset_id=os.environ.get("RESCUE_PRESET_ID", "rescue"),
        lane_devices_json=os.environ.get("LANE_DEVICES_JSON", "").strip(),
        skip_vllm=_env_bool("SKIP_VLLM", False),
        skip_vllm_lanes={item.strip() for item in os.environ.get("SKIP_VLLM_LANES", "").split(",") if item.strip()},
        skip_vllm_p=_env_bool("SKIP_VLLM_P", False),
        skip_vllm_d=_env_bool("SKIP_VLLM_D", False),
        skip_router=_env_bool("SKIP_ROUTER", False),
        skip_set_freq_from_presets=_env_bool("SKIP_SET_FREQ_FROM_PRESETS", False),
        experiment_metadata=experiment_metadata,
    )


def lane_slug(preset_id):
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(preset_id)).strip("_") or "lane"


def port_from_url(url):
    return str(url).rsplit(":", 1)[-1].rstrip("/")


def parse_devices(value):
    if value is None:
        return None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def plan_lanes(cfg, persist=True):
    presets_path = cfg.presets_json if cfg.presets_json.is_absolute() else cfg.online_serving / cfg.presets_json
    with open(presets_path, "r", encoding="utf-8") as handle:
        presets = json.load(handle)
    if not isinstance(presets, list) or not presets:
        raise SystemExit("PRESETS_JSON must be a non-empty JSON array")

    overrides = {}
    if cfg.lane_devices_json:
        parsed = json.loads(cfg.lane_devices_json)
        if not isinstance(parsed, dict):
            raise SystemExit("LANE_DEVICES_JSON must decode to a JSON object")
        overrides = {str(key): parse_devices(value) or "" for key, value in parsed.items()}

    skip_ports = set()
    if cfg.skip_vllm_p:
        skip_ports.add(str(cfg.lane_p_port))
    if cfg.skip_vllm_d:
        skip_ports.add(str(cfg.lane_d_port))

    pool = [item.strip() for item in cfg.ascend_devices.split(",") if item.strip()]
    pool_idx = 0
    rows = []

    for item in presets:
        if not isinstance(item, dict):
            raise SystemExit("each preset must be an object")
        preset_id = str(item.get("preset_id") or "").strip()
        backend_url = str(item.get("backend_url") or "").strip()
        if not preset_id or not backend_url:
            raise SystemExit("each preset requires preset_id and backend_url")
        port = port_from_url(backend_url)
        max_tokens = int(item.get("max_num_batched_tokens") or 0)
        max_seqs_raw = item.get("max_num_seqs")
        if max_tokens <= 0:
            raise SystemExit("preset %s: max_num_batched_tokens must be > 0" % preset_id)
        max_seqs = 128 if max_seqs_raw is None else int(max_seqs_raw)
        if max_seqs <= 0:
            raise SystemExit("preset %s: max_num_seqs must be > 0" % preset_id)
        mhz_value = item.get("frequency_mhz")
        mhz = "" if mhz_value is None else str(int(mhz_value))

        devices = overrides.get(preset_id)
        if devices is None:
            devices = parse_devices(item.get("ascend_devices"))
        if devices is None:
            devices = parse_devices((item.get("metadata") or {}).get("ascend_devices"))
        if devices is None:
            chunk = pool[pool_idx : pool_idx + cfg.tp_size]
            if len(chunk) < cfg.tp_size:
                raise SystemExit(
                    "not enough ASCEND_DEVICES for preset %s (need %d, have %d left)"
                    % (preset_id, cfg.tp_size, len(pool) - pool_idx)
                )
            devices = ",".join(chunk)
            pool_idx += cfg.tp_size

        start = not (
            cfg.skip_vllm
            or preset_id in cfg.skip_vllm_lanes
            or port in skip_ports
        )
        rows.append(
            LanePlanRow(
                preset_id=preset_id,
                port=port,
                devices=devices,
                max_tokens=max_tokens,
                max_seqs=max_seqs,
                frequency_mhz=mhz,
                start=start,
            )
        )

    if persist:
        write_lane_plan(cfg, rows)
    return rows


def write_lane_plan(cfg, rows):
    cfg.lane_plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.lane_plan_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                "\t".join(
                    [
                        row.preset_id,
                        row.port,
                        row.devices,
                        str(row.max_tokens),
                        str(row.max_seqs),
                        row.frequency_mhz,
                        "1" if row.start else "0",
                    ]
                )
                + "\n"
            )


def load_lane_plan(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            preset_id, port, devices, max_tokens, max_seqs, mhz, start = line.split("\t")
            rows.append(
                LanePlanRow(
                    preset_id=preset_id,
                    port=port,
                    devices=devices,
                    max_tokens=int(max_tokens),
                    max_seqs=int(max_seqs),
                    frequency_mhz=mhz,
                    start=start == "1",
                )
            )
    return rows


def projected_process_env(cfg, environment):
    env = dict(environment)
    env["ONLINE_SERVING"] = str(cfg.online_serving)
    env["DEMO_STATE_DIR"] = str(cfg.demo_state_dir)
    pypath = str(cfg.online_serving)
    if env.get("PYTHONPATH"):
        pypath = f"{pypath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pypath
    env.setdefault("ENERGY_AWARE_SEQ_PRED_BERT_PKL", str(cfg.online_serving / "src" / "models"))
    return env


def setup_process_env(cfg):
    os.environ.update(projected_process_env(cfg, os.environ))


def require_owned_lanes(cfg, rows):
    if (cfg.skip_vllm or cfg.skip_vllm_lanes or cfg.skip_vllm_p or cfg.skip_vllm_d
            or any(not row.start for row in rows)):
        raise ValueError("Mandatory serving profiles require owned lanes; SKIP_VLLM/per-lane skips and external server attestation are unsupported")


def preflight_serving_runtime(cfg, rows, anchor):
    require_owned_lanes(cfg, rows)
    cfg.serving_profile["runtime_verified"] = False
    env = projected_process_env(cfg, os.environ)
    env.update(cfg.serving_env_updates)
    verified = verify_runtime_package_context(anchor, cfg.vllm_root, env)
    cfg.serving_launch_env = env
    cfg.serving_profile.update(runtime_verified=True, verified_package_origins=verified["origins"],
                               runtime_verification_scope="local-server-cwd-and-env")


def record_manifest(cfg):
    manifest = cfg.demo_state_dir / "manifest.txt"
    lines = [
        f"ONLINE_SERVING={cfg.online_serving}",
        f"MODEL={cfg.model}",
        f"TP_SIZE={cfg.tp_size}",
        f"ASCEND_DEVICES={cfg.ascend_devices}",
        f"PRESETS_JSON={cfg.presets_json}",
        f"PROFILE_JSON={cfg.profile_json}",
        f"RESULT_JSON={cfg.result_json}",
        f"ROUTER_PORT={cfg.router_port}",
        f"RESCUE_PRESET_ID={cfg.rescue_preset_id}",
        f"DEMO_STATE_DIR={cfg.demo_state_dir}",
        f"LANE_PLAN={cfg.lane_plan_path}",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wait_with_progress(total_secs, bar_width):
    if total_secs <= 0:
        return
    for second in range(total_secs):
        done = int(bar_width * (second + 1) / total_secs)
        bar = "#" * done + "." * (bar_width - done)
        sys.stderr.write(f"\rwaiting [{bar}] {second + 1}/{total_secs}s")
        sys.stderr.flush()
        time.sleep(1)
    sys.stderr.write("\n")


def _append_pid_file(cfg, pid_path, pid):
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    with open(cfg.pid_files_list_path, "a", encoding="utf-8") as handle:
        handle.write(f"{pid_path}\n")
    all_pids = cfg.demo_state_dir / "all.pids"
    with open(all_pids, "a", encoding="utf-8") as handle:
        handle.write(f"{pid}\n")


def start_background_process(cmd, log_path, cfg, pid_path, **popen_kwargs):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(popen_kwargs.pop("cwd", cfg.online_serving)),
        env=popen_kwargs.pop("env", os.environ.copy()),
        start_new_session=True,
        **popen_kwargs,
    )
    log_handle.close()
    _append_pid_file(cfg, pid_path, process.pid)
    return process


def specialization_gap_enabled(cfg):
    kwargs = json.loads(cfg.routing_kwargs) if cfg.routing_kwargs else {}
    return isinstance(kwargs, dict) and kwargs.get("specialization_gap") is not None


def op_conflict_enabled(cfg):
    kwargs = json.loads(cfg.routing_kwargs) if cfg.routing_kwargs else {}
    return isinstance(kwargs, dict) and (kwargs.get("op_conflict") is not None
        or isinstance(kwargs.get("specialization_gap"), dict)
        and kwargs["specialization_gap"].get("label_key") == "op_conflict")


def experiment_energy_enabled(cfg):
    return specialization_gap_enabled(cfg) or op_conflict_enabled(cfg)


def set_freq_from_presets(cfg, rows):
    if cfg.skip_set_freq_from_presets:
        return
    seen = set()
    energy_meter = cfg.online_serving / "src" / "energy_meter.py"
    for row in rows:
        if not row.start or not row.frequency_mhz:
            continue
        key = (row.frequency_mhz, row.devices)
        if key in seen:
            continue
        seen.add(key)
        subprocess.run(
            [
                sys.executable,
                str(energy_meter),
                "freq",
                "--freq",
                row.frequency_mhz,
                "--devices",
                row.devices,
            ],
            cwd=str(cfg.online_serving),
            check=experiment_energy_enabled(cfg),
        )


def start_vllm_lanes(cfg, rows):
    if cfg.serving_profile.get("status") != "applied":
        raise ValueError("An active serving profile plan is required before launch; use plan_serving_profile")
    require_owned_lanes(cfg, rows)
    if any(row.start for row in rows) and cfg.serving_launch_env is None:
        raise ValueError("Local serving runtime package preflight is required before launch")
    cfg.pid_files_list_path.write_text("", encoding="utf-8")
    (cfg.demo_state_dir / "all.pids").write_text("", encoding="utf-8")
    processes = []
    extra_args = cfg.serving_extra_args
    cfg.serving_profile["lanes"] = []
    for row in rows:
        slug = lane_slug(row.preset_id)
        log_path = cfg.demo_state_dir / f"vllm_{slug}.log"
        pid_path = cfg.demo_state_dir / f"vllm_{slug}.pid"
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            cfg.model,
            "--port",
            row.port,
            "--tensor-parallel-size",
            str(cfg.tp_size),
            "--enable-chunked-prefill",
            "--scheduling-policy",
            "priority",
            "--max-num-batched-tokens",
            str(row.max_tokens),
            "--max-num-seqs",
            str(row.max_seqs),
            "--trust-remote-code",
            *extra_args,
        ]
        env = cfg.serving_launch_env.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = row.devices
        processes.append(
            start_background_process(
                cmd,
                log_path,
                cfg,
                pid_path,
                cwd=cfg.vllm_root,
                env=env,
            )
        )
        cfg.serving_profile["lanes"].append({
            "preset_id": row.preset_id, "status": "launched",
            "runtime_verified": True,
            "profile_applied": cfg.serving_profile.get("status") == "applied",
            "command": cmd, "cwd": str(cfg.vllm_root),
            "env": {"ASCEND_RT_VISIBLE_DEVICES": row.devices,
                    **{key: env[key] for key in cfg.serving_profile.get("relevant_env", {}) if key in env}},
        })
    return processes


def start_router(cfg):
    if cfg.skip_router:
        return None
    presets = cfg.presets_json if cfg.presets_json.is_absolute() else cfg.online_serving / cfg.presets_json
    profile = cfg.profile_json if cfg.profile_json.is_absolute() else cfg.online_serving / cfg.profile_json
    cmd = [
        sys.executable,
        "-m",
        "src.cli",
        "--presets",
        str(presets),
        "--profile",
        str(profile),
        "--slo-max-latency-s",
        str(cfg.slo_max_latency_s),
        "--rescue-preset-id",
        cfg.rescue_preset_id,
        "--host",
        "0.0.0.0",
        "--port",
        str(cfg.router_port),
    ]
    if cfg.routing_policy:
        cmd.extend(["--routing-policy", cfg.routing_policy])
    if cfg.routing_kwargs:
        kwargs = json.loads(cfg.routing_kwargs)
        routing_kwargs = cfg.routing_kwargs
        if isinstance(kwargs, dict) and kwargs.get("op_conflict") is not None:
            if kwargs.get("specialization_gap") is not None or not isinstance(kwargs["op_conflict"], dict):
                raise ValueError("op_conflict must be an object and cannot also configure specialization_gap")
            # The CLI's existing experiment gate avoids loading production predictors.
            kwargs["specialization_gap"] = dict(kwargs.pop("op_conflict"), label_key="op_conflict")
            routing_kwargs = json.dumps(kwargs)
        cmd.extend(["--routing-kwargs", routing_kwargs])
    cmd.extend(cfg.router_extra_args)
    return start_background_process(cmd, cfg.router_log_path, cfg, cfg.router_pid_path)


def collect_power_devices(rows):
    seen = set()
    ordered = []
    for row in rows:
        for item in row.devices.split(","):
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
    return ",".join(ordered)


def collect_kv_targets(cfg, rows):
    presets_path = cfg.presets_json if cfg.presets_json.is_absolute() else cfg.online_serving / cfg.presets_json
    with open(presets_path, "r", encoding="utf-8") as handle:
        presets = json.load(handle)
    if not isinstance(presets, list):
        return ""
    preset_map = {}
    for item in presets:
        if not isinstance(item, dict):
            continue
        preset_id = str(item.get("preset_id") or "").strip()
        backend_url = str(item.get("backend_url") or "").strip().rstrip("/")
        if preset_id and backend_url:
            preset_map[preset_id] = item
    targets = []
    for row in rows:
        if not row.start:
            continue
        item = preset_map.get(row.preset_id)
        if item is None:
            continue
        capabilities = dict(item.get("capabilities") or {})
        if not capabilities.get("kv_projection", False):
            continue
        backend_url = str(item.get("backend_url") or "").strip().rstrip("/")
        if backend_url:
            targets.append("%s=%s" % (row.preset_id, backend_url))
    return ",".join(targets)


def client_base_url(cfg, rows):
    override = os.environ.get("CLIENT_BASE_URL", "").strip()
    if override:
        text = override.rstrip("/")
        if not text.endswith("/v1"):
            text = text + "/v1"
        return text
    if cfg.skip_router:
        for row in rows:
            if row.start:
                return "http://127.0.0.1:%s/v1" % row.port
    return "http://127.0.0.1:%s/v1" % cfg.router_port


def run_client(cfg, rows):
    """Return (exit status, fresh JSON); zero exit without output is a failure."""
    for token in cfg.client_extra_args:
        option = token.split("=", 1)[0].replace("_", "-")
        # src.client has no short output alias; argparse accepts long prefixes.
        if option.startswith("--") and len(option) > 2 and "--result-json".startswith(option):
            raise ValueError("Client output is runner-owned; use RESULT_JSON instead of CLIENT_EXTRA_ARGS --result-json aliases/abbreviations")
    cmd = [
        sys.executable,
        "-m",
        "src.client",
        "--base-url",
        client_base_url(cfg, rows),
        "--model",
        cfg.model,
        "--max-arrivals",
        str(cfg.client_max_arrivals),
        "--result-json",
        str(cfg.result_json),
    ]
    if cfg.client_extra_args:
        cmd.extend(cfg.client_extra_args)
    else:
        cmd.extend(
            [
                "--prompt-source",
                "azure-mix",
                "--azure-mix-schedule-csv",
                "/datasets/azure-mix/scheduled_prompts.csv",
            ]
        )
    started_s = time.time()
    before = output_identity(cfg.result_json)
    completed = subprocess.run(cmd, cwd=str(cfg.online_serving))
    after = output_identity(cfg.result_json)
    ended_s = time.time()
    code = int(completed.returncode)
    fresh = after is not None and after != before
    if not fresh:
        print("Client produced no current result JSON; leaving previous output untouched", file=sys.stderr)
        return code or 1, False
    result = _load_result_object(cfg.result_json)
    if experiment_energy_enabled(cfg):
        result["benchmark_start_s"] = started_s
        result["benchmark_end_s"] = ended_s
        with open(cfg.result_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return code, True


def _load_result_object(path):
    with open(path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or ("requests" in result and (
        not isinstance(result["requests"], list) or any(not isinstance(row, dict) for row in result["requests"])
    )):
        raise ValueError("Current client result must be a JSON object with an object-array requests field when present")
    return result


def stop_process(process):
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def merge_power_into_result(cfg, result_fresh=False):
    if not result_fresh or not cfg.power_summary_path.is_file() or not cfg.result_json.is_file():
        return
    result = _load_result_object(cfg.result_json)
    with open(cfg.power_summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    result["energy_joules"] = float(summary.get("total_joules", 0.0) or 0.0)
    result["energy_num_samples"] = int(summary.get("samples", 0) or 0)
    result["energy_per_npu_joules"] = dict(summary.get("per_npu") or {})
    result["power_samples_series"] = list(summary.get("power_samples_series") or [])
    output_tokens = float(result.get("total_output_tokens") or 0.0)
    if output_tokens > 0:
        result["energy_joules_per_output_token"] = float(result["energy_joules"]) / output_tokens
    with open(cfg.result_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def merge_experiment_metadata(cfg, result_fresh=False):
    if not result_fresh or not cfg.result_json.is_file():
        return
    result = _load_result_object(cfg.result_json)
    result.update(cfg.experiment_metadata)
    result["serving_profile"] = cfg.serving_profile
    result["serving_condition"] = cfg.serving_condition
    with open(cfg.result_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def merge_kv_into_result(cfg, result_fresh=False):
    if not result_fresh or not cfg.kv_summary_path.is_file() or not cfg.result_json.is_file():
        return
    result = _load_result_object(cfg.result_json)
    with open(cfg.kv_summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    result["kv_samples"] = int(summary.get("samples", 0) or 0)
    result["kv_duration_s"] = float(summary.get("duration_s", 0.0) or 0.0)
    result["kv_host_swap_samples"] = int(summary.get("host_swap_samples", 0) or 0)
    result["kv_mean_utilization"] = summary.get("mean_utilization")
    result["kv_p95_utilization"] = summary.get("p95_utilization")
    result["kv_peak_utilization"] = summary.get("peak_utilization")
    result["kv_mean_peak_util_p95"] = summary.get("mean_peak_util_p95")
    result["kv_by_lane"] = dict(summary.get("by_lane") or {})
    result["kv_samples_series"] = list(summary.get("kv_samples_series") or [])
    with open(cfg.result_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def validate_op_conflict_boundary(sample, expected):
    if not isinstance(sample, dict) or not expected:
        raise ValueError("op_conflict boundary requires a sample and configured devices")
    timestamp = sample.get("timestamp")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("op_conflict boundary timestamp must be finite and nonnegative")
    readings = sample.get("per_npu")
    if not isinstance(readings, dict) or set(readings) != set(expected):
        raise ValueError("op_conflict boundary must meter exactly the configured lane devices")
    for device, frequency in expected.items():
        reading = readings[device]
        power = reading.get("power_sample") if isinstance(reading, dict) else None
        if type(power) not in (int, float) or not math.isfinite(power) or power <= 0:
            raise ValueError("op_conflict boundary requires positive finite power on device %s" % device)
        observed = reading.get("frequency")
        if type(observed) not in (int, float) or observed != frequency:
            raise ValueError("op_conflict boundary frequency mismatch on device %s" % device)


def append_power_boundary(cfg, devices, initial=False):
    """Experiment-only boundary sample, using the same meter and integration as normal runs."""
    snapshot_path = cfg.demo_state_dir / ("power_initial.json" if initial else "power_final.json")
    subprocess.run(
        [sys.executable, str(cfg.online_serving / "src" / "energy_meter.py"), "snapshot",
         "--output", str(snapshot_path), "--devices", devices],
        cwd=str(cfg.online_serving), check=True,
    )
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        sample = json.load(handle)
    if op_conflict_enabled(cfg):
        expected = {device: int(row.frequency_mhz) for row in load_lane_plan(cfg.lane_plan_path)
                    for device in row.devices.split(",")}
        validate_op_conflict_boundary(sample, expected)
    with open(cfg.power_log_path, "w" if initial else "a", encoding="utf-8") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")


def summarize_power(cfg, power_process, result_fresh=False):
    stop_process(power_process)
    if not result_fresh or not cfg.power_log_path.is_file():
        return
    energy_meter = cfg.online_serving / "src" / "energy_meter.py"
    with open(cfg.power_summary_path, "w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, str(energy_meter), "summarize", "--log", str(cfg.power_log_path)],
            cwd=str(cfg.online_serving),
            stdout=handle,
            check=op_conflict_enabled(cfg),
        )
    merge_power_into_result(cfg, result_fresh=result_fresh)


def summarize_kv(cfg, kv_process, result_fresh=False):
    stop_process(kv_process)
    if not result_fresh or not cfg.kv_log_path.is_file():
        return
    kv_meter = cfg.online_serving / "src" / "kv_meter.py"
    with open(cfg.kv_summary_path, "w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, str(kv_meter), "summarize", "--log", str(cfg.kv_log_path)],
            cwd=str(cfg.online_serving),
            stdout=handle,
            check=False,
        )
    merge_kv_into_result(cfg, result_fresh=result_fresh)


def _kill_wait_timeout_s():
    raw = os.environ.get("KILL_WAIT_TIMEOUT_S", "60")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 60.0


def prepare_services(cfg):
    kill_module.kill_all(cfg.demo_state_dir, force=False, wait_timeout_s=_kill_wait_timeout_s())


def teardown_services(cfg):
    if cfg.kill_after_client:
        kill_module.kill_all(cfg.demo_state_dir, force=False, wait_timeout_s=_kill_wait_timeout_s())


def main():
    cfg = load_config()
    if SERVING_METADATA_KEYS.intersection(cfg.experiment_metadata):
        raise ValueError("EXPERIMENT_METADATA uses reserved serving_profile/serving_condition keys")
    anchor = runtime_package_context()
    cfg.serving_extra_args, cfg.serving_env_updates, cfg.serving_profile, cfg.serving_condition = plan_serving_profile(
        cfg.model, cfg.tp_size, cfg.vllm_serve_extra_args, os.environ,
        versions=anchor["versions"], origins=anchor["origins"],
    )
    expected = os.environ.get("VLLM_EXPECTED_SERVING_CONDITION")
    if expected is not None and expected != cfg.serving_condition:
        raise ValueError("VLLM_EXPECTED_SERVING_CONDITION mismatch: expected %s, selected %s" % (expected, cfg.serving_condition))
    print("serving profile id=%s status=%s reason=%s condition=%s" % (
        cfg.serving_profile["profile_id"], cfg.serving_profile["status"], cfg.serving_profile["reason"],
        cfg.serving_condition,
    ), file=sys.stderr)
    experiment = experiment_energy_enabled(cfg)
    rows = plan_lanes(cfg, persist=False)
    require_owned_lanes(cfg, rows)
    if op_conflict_enabled(cfg) and (not cfg.power_monitor_enabled or cfg.skip_set_freq_from_presets
            or cfg.skip_router):
        raise ValueError("op_conflict requires both serving lanes, router, frequency setting and power monitoring")
    if op_conflict_enabled(cfg) and (len(rows) != 2 or any(not row.frequency_mhz for row in rows)):
        raise ValueError("op_conflict requires exactly two frequency-configured serving lanes")
    preflight_serving_runtime(cfg, rows, anchor)
    cfg.demo_state_dir.mkdir(parents=True, exist_ok=True)
    setup_process_env(cfg)
    prepare_services(cfg)

    power_process = None
    kv_process = None
    client_code = 1

    try:
        record_manifest(cfg)
        write_lane_plan(cfg, rows)
        set_freq_from_presets(cfg, rows)
        start_vllm_lanes(cfg, rows)
        start_router(cfg)
        wait_with_progress(cfg.wait_secs, cfg.wait_bar_width)

        if cfg.power_monitor_enabled:
            devices = collect_power_devices(rows)
            if devices:
                if experiment:
                    append_power_boundary(cfg, devices, initial=True)
                energy_meter = cfg.online_serving / "src" / "energy_meter.py"
                power_process = start_background_process(
                    [
                        sys.executable,
                        str(energy_meter),
                        "monitor",
                        "--output",
                        str(cfg.power_log_path),
                        "--interval",
                        str(cfg.power_interval_s),
                        "--devices",
                        devices,
                        *(["--append"] if experiment else []),
                    ],
                    cfg.demo_state_dir / "power_monitor.log",
                    cfg,
                    cfg.demo_state_dir / "power_monitor.pid",
                )

        if cfg.kv_monitor_enabled:
            targets = collect_kv_targets(cfg, rows)
            if targets:
                kv_meter = cfg.online_serving / "src" / "kv_meter.py"
                kv_process = start_background_process(
                    [
                        sys.executable,
                        str(kv_meter),
                        "monitor",
                        "--output",
                        str(cfg.kv_log_path),
                        "--interval",
                        str(cfg.kv_interval_s),
                        "--targets",
                        targets,
                        "--timeout-s",
                        str(cfg.kv_timeout_s),
                    ],
                    cfg.demo_state_dir / "kv_monitor.log",
                    cfg,
                    cfg.demo_state_dir / "kv_monitor.pid",
                )

        client_code, result_fresh = run_client(cfg, rows)
        if not result_fresh:
            return client_code
        if experiment and power_process is not None:
            if power_process.poll() is not None:
                name = "op-conflict" if op_conflict_enabled(cfg) else "specialization-gap"
                raise RuntimeError("%s power monitor exited before measurement completed" % name)
            # The monitor must be stopped before the final append to avoid interleaved writes.
            stop_process(power_process)
            power_process = None
            append_power_boundary(cfg, collect_power_devices(rows))
        summarize_power(cfg, power_process, result_fresh=result_fresh)
        power_process = None
        summarize_kv(cfg, kv_process, result_fresh=result_fresh)
        kv_process = None
        merge_experiment_metadata(cfg, result_fresh=result_fresh)
        return client_code
    finally:
        stop_process(power_process)
        stop_process(kv_process)
        teardown_services(cfg)


if __name__ == "__main__":
    sys.exit(main())
