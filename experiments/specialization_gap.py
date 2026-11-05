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
import hashlib
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.azure_mix import SCHEDULE_FIELDS, load_schedule_events, normalize_pool_name
from run import plan_serving_profile
from src.serving_profiles import apply_serving_config, normalize_serving_config, runtime_package_context, serving_condition_key


def _json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _slug(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(value)).strip("-") or "value"


def load_config(path):
    config = _json(Path(path))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    required = ("model", "tp", "source_schedule", "candidate_ops", "fixed_max_num_batched_tokens", "lanes", "routing", "mixtures", "seed", "total_requests")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("missing config keys: %s" % ", ".join(missing))
    if len(config["lanes"]) != 2:
        raise ValueError("lanes must contain exactly two entries")
    for key in ("tp", "total_requests", "fixed_max_num_batched_tokens"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("%s must be a positive integer" % key)
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if config["total_requests"] < 2:
        raise ValueError("total_requests must be >= 2 so neutral controls can use both lanes")
    if "max_num_batched_tokens" in config and config["max_num_batched_tokens"] != config["fixed_max_num_batched_tokens"]:
        raise ValueError("max_num_batched_tokens conflicts with fixed_max_num_batched_tokens")
    if not config["candidate_ops"]:
        raise ValueError("candidate_ops must not be empty")
    for op in config["candidate_ops"]:
        if "max_num_batched_tokens" in op:
            raise ValueError("candidate OPs cannot set max_num_batched_tokens; use fixed_max_num_batched_tokens globally")
        if set(op) != {"frequency_mhz", "max_num_seqs"}:
            raise ValueError("each candidate_op must contain only frequency_mhz and max_num_seqs")
        if any(type(value) is not int or value <= 0 for value in op.values()):
            raise ValueError("OP frequency_mhz and max_num_seqs must be positive integers")
    if len({(op["frequency_mhz"], op["max_num_seqs"]) for op in config["candidate_ops"]}) != len(config["candidate_ops"]):
        raise ValueError("candidate_ops must not contain duplicate OPs")
    devices = []
    for lane in config["lanes"]:
        for key in ("id", "url", "device_map"):
            if key not in lane:
                raise ValueError("each lane requires %s" % key)
        if "max_num_batched_tokens" in lane:
            raise ValueError("lanes cannot override fixed_max_num_batched_tokens")
        if not isinstance(lane["device_map"], list) or len(lane["device_map"]) != config["tp"]:
            raise ValueError("each lane device_map must contain exactly tp devices")
        normalized = []
        for device in lane["device_map"]:
            text = str(device).strip()
            if type(device) not in (str, int) or not text.isascii() or not text.isdecimal():
                raise ValueError("device IDs must be nonnegative decimal integers")
            normalized.append(str(int(text)))
        lane["device_map"] = normalized
        devices.extend(normalized)
        url = urlsplit(lane["url"])
        if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or not url.port or url.path not in ("", "/"):
            raise ValueError("lane URLs must be local HTTP endpoints with explicit ports")
    if len(set(devices)) != len(devices):
        raise ValueError("lane device maps must be disjoint and contain no duplicates")
    if any(not isinstance(lane["id"], str) or not lane["id"].strip() for lane in config["lanes"]) or len({lane["id"] for lane in config["lanes"]}) != 2:
        raise ValueError("lane IDs must be nonempty and distinct")
    if len({urlsplit(lane["url"]).port for lane in config["lanes"]}) != 2:
        raise ValueError("lanes must use distinct ports")
    if config["routing"].get("routing_policy") not in ("oracle_class_affinity", "round_robin"):
        raise ValueError("experiment routing must be oracle_class_affinity or round_robin")
    kwargs = config["routing"].get("routing_kwargs", {})
    if kwargs not in ({}, {"feedback": {"enabled": False}}):
        raise ValueError("experiment routing_kwargs cannot add adaptive/predictor-dependent routing knobs")
    for mixture in config["mixtures"]:
        if set(mixture) != {"name", "summarization_pct", "conversation_pct"}:
            raise ValueError("mixtures require name, summarization_pct, conversation_pct")
        if not isinstance(mixture["name"], str) or not mixture["name"].strip():
            raise ValueError("mixture names must be nonempty strings")
        percentages = (mixture["summarization_pct"], mixture["conversation_pct"])
        if any(type(value) is not int or not 0 <= value <= 100 for value in percentages) or sum(percentages) != 100:
            raise ValueError("mixture percentages must sum to 100")
        if all(percentages) and not all(_class_counts(config, mixture).values()):
            raise ValueError("total_requests is too small to represent both classes in a mixed cohort")
    if not config["mixtures"] or len({_slug(item["name"]) for item in config["mixtures"]}) != len(config["mixtures"]):
        raise ValueError("mixtures must have unique nonempty output names")
    if any(config.get("slo", {}).get(key) is None for key in ("ttft_p99_9_s", "tpot_p99_9_s")):
        raise ValueError("both configured TTFT and TPOT p99.9 SLO limits are required")
    for key, value in config.get("slo", {}).items():
        if key not in ("ttft_p99_9_s", "tpot_p99_9_s", "max_latency_s"):
            raise ValueError("unknown experiment SLO field: %s" % key)
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError("SLO limits must be finite and positive")
    if "serving" in config:
        config["serving"] = normalize_serving_config(config["serving"])
    return config


def _mixture_filter(config, requested):
    mixtures = config["mixtures"]
    if not requested:
        return mixtures
    requested = set(requested.split(","))
    found = [item for item in mixtures if item["name"] in requested]
    if len(found) != len(requested):
        raise ValueError("unknown mixture filter")
    return found


def banks(config, requested=None, op_indices=None, mapping=None):
    indices = range(len(config["candidate_ops"]))
    if op_indices:
        indices = sorted(set(int(item) for item in op_indices.split(",")))
        if any(index < 0 or index >= len(config["candidate_ops"]) for index in indices):
            raise ValueError("op index out of range")
    result = []
    for left in indices:
        for right in indices:
            if left == right:
                name = "shared-%d-%d" % (left, right)
                result.append({"name": name, "bank_id": name, "type": "shared", "ops": [left, right], "mapping": "neutral"})
            elif left < right:
                name = "specialized-%d-%d" % (left, right)
                for variant in ("pd", "dp"):
                    result.append({"name": name + "-" + variant, "bank_id": name, "type": "specialized", "ops": [left, right], "mapping": variant})
    if requested:
        # Diagonal specialized banks are aliases of the exact shared measurement, not extra runs.
        parts = requested.split("-")
        if len(parts) == 3 and parts[0] == "specialized" and parts[1] == parts[2]:
            requested = "shared-" + "-".join(parts[1:])
        result = [item for item in result if requested in (item["name"], item["bank_id"])]
        if not result:
            raise ValueError("unknown bank ID: %s" % requested)
    if mapping:
        result = [item for item in result if item["mapping"] == mapping]
        if not result:
            raise ValueError("mapping does not match any selected bank")
    return result


def _class_counts(config, mixture):
    prefill = config["total_requests"] * mixture["summarization_pct"] // 100
    return {"summarization": prefill, "conversation": config["total_requests"] - prefill}


def _routing(config, mixture, bank):
    active = [pool for pool, count in _class_counts(config, mixture).items() if count]
    oracle = (config["routing"]["routing_policy"] == "oracle_class_affinity"
              and len(active) == 2 and bank["ops"][0] != bank["ops"][1])
    prefill_lane = 0 if bank["mapping"] == "pd" else 1
    mapping = {"summarization": config["lanes"][prefill_lane]["id"],
               "conversation": config["lanes"][1 - prefill_lane]["id"]} if oracle else {}
    reason = "class_affinity" if oracle else ("homogeneous_control" if len(active) == 1 else "shared_bank" if bank["type"] == "shared" else "secondary_round_robin")
    return {
        "routing_mode": "oracle_class_affinity" if oracle else "round_robin",
        "prefill_target_lane": prefill_lane if oracle else "both",
        "decode_target_lane": 1 - prefill_lane if oracle else "both",
        "effective_mapping": bank["mapping"] if oracle else "neutral",
        "neutral_load_balancing": not oracle, "routing_reason": reason,
        "routing_kwargs": {"specialization_gap": {
            "lane_ids": [lane["id"] for lane in config["lanes"]],
            "active_pools": active, "pool_to_lane": mapping,
        }},
    }


def _realizations(config, mixture, selected_banks):
    result = []
    seen = set()
    for bank in selected_banks:
        if bank["type"] == "specialized" and _routing(config, mixture, bank)["neutral_load_balancing"]:
            bank = dict(bank, name=bank["bank_id"] + "-neutral", mapping="neutral", mapping_aliases=["pd", "dp"])
        if bank["name"] not in seen:
            result.append(bank)
            seen.add(bank["name"])
    return result


def _cohort(config, mixture):
    events = load_schedule_events(config["source_schedule"], active_pools="all", max_arrivals=0)
    total = int(config["total_requests"])
    timing = events[:total]
    if len(timing) < total:
        raise ValueError("source schedule has fewer events than total_requests")
    by_pool = {"summarization": [], "conversation": []}
    for event in events:
        pool = normalize_pool_name(event.get("pool"))
        if pool in by_pool:
            by_pool[pool].append(event)
    summarization_count = total * int(mixture["summarization_pct"]) // 100
    conversation_count = total - summarization_count
    if len(by_pool["summarization"]) < summarization_count or len(by_pool["conversation"]) < conversation_count:
        raise ValueError("source schedule lacks enough events for mixture %s" % mixture["name"])
    rng = random.Random(int(config["seed"]) + sum(ord(ch) for ch in mixture["name"]))
    selected = rng.sample(by_pool["summarization"], summarization_count) + rng.sample(by_pool["conversation"], conversation_count)
    rng.shuffle(selected)
    rows = []
    for timing_event, event in zip(timing, selected):
        row = {field: event.get(field, "") for field in SCHEDULE_FIELDS}
        row["arrival_sec"] = timing_event["arrival_sec"]
        row["pool"] = normalize_pool_name(event["pool"])
        rows.append(row)
    # Match the client's stable schedule ordering, including equal-timestamp type ties.
    rows.sort(key=lambda row: (float(row["arrival_sec"]), int(row["type"])))
    return rows, _cohort_hash(rows)


def _cohort_hash(rows):
    encoded = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_cohort(path, rows):
    if path.exists():
        if _cohort_hash(load_schedule_events(str(path), active_pools="all", max_arrivals=0)) != _cohort_hash(rows):
            raise ValueError("existing cohort differs; use a fresh --output-root: %s" % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEDULE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mixture_dir(root, mixture):
    return root / _slug(mixture["name"])


def _presets(config, bank):
    presets = []
    for lane, op_index in zip(config["lanes"], bank["ops"]):
        op = config["candidate_ops"][op_index]
        presets.append({
            "preset_id": lane["id"],
            "backend_url": lane["url"],
            "ascend_devices": lane["device_map"],
            "max_num_batched_tokens": int(config["fixed_max_num_batched_tokens"]),
            "max_num_seqs": int(op["max_num_seqs"]),
            "frequency_mhz": int(op["frequency_mhz"]),
        })
    return presets


def _run_env(config, schedule, presets, result, run_dir, routing):
    client = ["--prompt-source", "azure-mix", "--azure-mix-schedule-csv", str(schedule), "--active-pools", "all", "--stream-metrics", "--specialization-gap", "--temperature", "0", "--max-in-flight", "200"]
    env = os.environ.copy()
    if env.get("ROUTER_EXTRA_ARGS", "").strip():
        raise ValueError("ROUTER_EXTRA_ARGS must be unset for this controlled experiment")
    fixed_flags = {"--model", "--port", "--device", "--tensor-parallel-size", "-tp", "--data-parallel-size", "-dp", "--pipeline-parallel-size", "-pp", "--decode-context-parallel-size", "--prefill-context-parallel-size", "--max-num-batched-tokens", "--max-num-seqs", "--scheduling-policy", "--enable-chunked-prefill", "--no-enable-chunked-prefill"}
    if any(arg.split("=", 1)[0] in fixed_flags or arg.startswith("--") and any(flag.startswith(arg.split("=", 1)[0]) for flag in fixed_flags)
           for arg in shlex.split(env.get("VLLM_SERVE_EXTRA_ARGS", ""))):
        raise ValueError("VLLM_SERVE_EXTRA_ARGS cannot override the experiment's fixed serving knobs")
    for key in ("LANE_DEVICES_JSON", "CLIENT_BASE_URL", "SKIP_VLLM_LANES", "ROUTER_EXTRA_ARGS"):
        env.pop(key, None)
    env.update({
        "ONLINE_SERVING": str(REPO_ROOT),
        "MODEL": str(config["model"]),
        "TP_SIZE": str(config["tp"]),
        "PRESETS_JSON": str(run_dir / "presets.json"),
        "RESULT_JSON": str(result),
        "DEMO_STATE_DIR": str(run_dir / "demo-state"),
        "CLIENT_MAX_ARRIVALS": str(config["total_requests"]),
        "CLIENT_EXTRA_ARGS": shlex.join(client),
        "ROUTING_POLICY": routing["routing_mode"],
        "ROUTING_KWARGS": json.dumps(routing["routing_kwargs"], ensure_ascii=False),
        "ASCEND_DEVICES": ",".join(str(device) for lane in config["lanes"] for device in lane["device_map"]),
        "SKIP_VLLM": "0", "SKIP_VLLM_P": "0", "SKIP_VLLM_D": "0", "SKIP_ROUTER": "0",
        "SKIP_SET_FREQ_FROM_PRESETS": "0", "POWER_MONITOR_ENABLED": "1", "KILL_AFTER_CLIENT": "1",
        "RESCUE_PRESET_ID": "",
        "SLO_MAX_LATENCY_S": str(config.get("slo", {}).get("max_latency_s") or 120.0),
    })
    return apply_serving_config(env, config.get("serving"))


def _plan_serving(config, environments):
    """Plan every candidate against one runtime anchor; children retain only raw intent."""
    anchor = runtime_package_context()
    profile, condition = None, None
    for env in environments:
        _, _, metadata, selected = plan_serving_profile(
            config["model"], config["tp"], shlex.split(env.get("VLLM_SERVE_EXTRA_ARGS", "")), env,
            versions=anchor["versions"], origins=anchor["origins"],
        )
        if condition is not None and selected != condition:
            raise ValueError("candidates cannot mix serving conditions; use a fresh output root")
        if profile is None:
            profile, condition = metadata, selected
        env["VLLM_EXPECTED_SERVING_CONDITION"] = selected
    if profile is None:
        raise ValueError("no serving candidates selected")
    return profile, condition


def _check_experiment(root, experiment):
    path = root / "experiment.json"
    if path.exists():
        if _json(path) != experiment:
            raise ValueError("frozen experiment settings/profile/environment differ; use a fresh output root")
    elif any(next(root.glob(pattern), None) is not None for pattern in (
        "*/manifest.json", "*/cohort.csv", "*/cohort.json", "*/*/candidate.json", "*/*/rep-*",
    )):
        raise ValueError("existing results lack frozen serving identity; use a fresh output root")


def _frozen_serving_condition(experiment):
    profile = experiment.get("serving_profile")
    condition = experiment.get("serving_condition")
    if not isinstance(profile, dict) or profile.get("status") != "applied" or condition != serving_condition_key(profile):
        raise ValueError("missing or invalid frozen serving identity; use a fresh output root")
    return condition


def _planned(config, mixtures, selected_banks, repetitions, root, repetition=None, mapping=None):
    for mixture in mixtures:
        realized = [bank for bank in _realizations(config, mixture, selected_banks)
                    if not mapping or bank["mapping"] == mapping or mapping in bank.get("mapping_aliases", [])]
        if not realized:
            raise ValueError("mapping has no realization for mixture %s" % mixture["name"])
        for bank in realized:
            for index in ([repetition] if repetition is not None else range(1, repetitions + 1)):
                yield mixture, bank, index, _mixture_dir(root, mixture) / bank["name"] / ("rep-%d" % index)


def _persist(path, value):
    if path.exists():
        if _json(path) != value:
            raise ValueError("incompatible existing manifest; use a fresh --output-root: %s" % path)
    else:
        _dump(path, value)


def _candidate_manifest(config, mixture, bank, repetitions, cohort_hash, rows, schedule, serving_condition):
    presets = _presets(config, bank)
    return {
        "model": config["model"], "tp": config["tp"], "mixture": mixture, "bank": bank,
        "serving_condition": serving_condition,
        "seed": config["seed"], "cohort_hash": cohort_hash, "offered": len(rows),
        "class_counts": _class_counts(config, mixture), "expected_repetitions": repetitions,
        "source_schedule": config["source_schedule"], "schedule": str(schedule),
        "max_num_batched_tokens": config["fixed_max_num_batched_tokens"],
        "fixed_max_num_batched_tokens": config["fixed_max_num_batched_tokens"],
        "generation_settings": {"temperature": 0.0, "stream": True, "max_in_flight": 200, "max_tokens": None},
        "energy_scope": "sampled_boundary_inclusive_client_subprocess",
        "lane0": {"op": dict(config["candidate_ops"][bank["ops"][0]]), "preset": presets[0]},
        "lane1": {"op": dict(config["candidate_ops"][bank["ops"][1]]), "preset": presets[1]},
        "presets": presets, "npu_mapping": {item["preset_id"]: item["ascend_devices"] for item in presets},
        "slo": config.get("slo", {}), **_routing(config, mixture, bank),
    }


def run_mode(config, root, mixtures, selected_banks, repetitions, dry_run, repetition=None, mapping=None):
    plan = list(_planned(config, mixtures, selected_banks, repetitions, root, repetition, mapping))
    environments = {
        run_dir: _run_env(config, _mixture_dir(root, mixture) / "cohort.csv", _presets(config, bank),
                          run_dir / "result.json", run_dir, _routing(config, mixture, bank))
        for mixture, bank, _, run_dir in plan
    }
    profile, condition = _plan_serving(config, environments.values())
    runtime_keys = ("PATH", "PYTHONPATH", "VLLM_ROOT", "VLLM_SERVE_EXTRA_ARGS", "VLLM_USE_V1", "PROFILE_JSON", "WAIT_SECS", "ROUTER_PORT", "POWER_INTERVAL_S", "KV_MONITOR_ENABLED", "KV_MONITOR_INTERVAL_S", "KV_MONITOR_TIMEOUT_S")
    experiment = {"config": config, "repetitions": repetitions,
                  "runtime_environment": {key: os.environ.get(key) for key in runtime_keys},
                  "serving_profile": profile, "serving_condition": condition}
    _check_experiment(root, experiment)
    for _, _, _, run_dir in plan:
        if not dry_run and run_dir.exists():
            raise ValueError("refusing to overwrite an existing run; select another repetition or output root: %s" % run_dir)
    cohorts = {}
    for mixture in mixtures:
        mixture_dir = _mixture_dir(root, mixture)
        cohort_path = mixture_dir / "cohort.csv"
        if cohort_path.exists():
            rows = load_schedule_events(str(cohort_path), active_pools="all", max_arrivals=0)
            cohort_hash = _cohort_hash(rows)
            recorded = _json(mixture_dir / "manifest.json")
            if recorded["cohort_hash"] != cohort_hash or recorded["seed"] != config["seed"] or recorded["mixture"] != mixture:
                raise ValueError("persisted cohort identity mismatch: %s" % cohort_path)
        else:
            rows, cohort_hash = _cohort(config, mixture)
        counts = {pool: sum(row["pool"] == pool for row in rows) for pool in ("summarization", "conversation")}
        if len(rows) != config["total_requests"] or counts != _class_counts(config, mixture):
            raise ValueError("persisted cohort offered count or class mixture differs")
        cohorts[mixture["name"]] = (rows, cohort_hash)
    prepared = []
    for mixture, bank, index, run_dir in plan:
        rows, cohort_hash = cohorts[mixture["name"]]
        schedule = _mixture_dir(root, mixture) / "cohort.csv"
        presets = _presets(config, bank)
        result = run_dir / "result.json"
        candidate = _candidate_manifest(config, mixture, bank, repetitions, cohort_hash, rows, schedule, condition)
        manifest = dict(candidate, repetition=index, result=str(result))
        env = environments[run_dir]
        command = [sys.executable, str(REPO_ROOT / "run.py")]
        print("%s/%s/rep-%d mapping=%s aliases=%s effective=%s routing=%s P->%s D->%s cohort=%s: %s" % (
            mixture["name"], bank["name"], index, bank["mapping"], bank.get("mapping_aliases", []), candidate["effective_mapping"],
            candidate["routing_mode"], candidate["prefill_target_lane"], candidate["decode_target_lane"],
            cohort_hash, shlex.join(command)))
        if dry_run:
            continue
        prepared.append((run_dir, presets, manifest, candidate, command, env))
    if dry_run:
        return
    _persist(root / "experiment.json", experiment)
    for mixture in mixtures:
        rows, cohort_hash = cohorts[mixture["name"]]
        mixture_dir = _mixture_dir(root, mixture)
        cohort_path = mixture_dir / "cohort.csv"
        _write_cohort(cohort_path, rows)
        _persist(mixture_dir / "manifest.json", {"mixture": mixture, "seed": config["seed"], "cohort_hash": cohort_hash, "offered": len(rows), "schedule": str(cohort_path), "class_counts": _class_counts(config, mixture)})
    # Register all selected candidates before the first subprocess, retaining interrupted/missing runs.
    for run_dir, _, _, candidate, _, _ in prepared:
        _persist(run_dir.parent / "candidate.json", candidate)
    for run_dir, presets, manifest, candidate, command, env in prepared:
        run_dir.mkdir(parents=True, exist_ok=False)
        _dump(run_dir / "presets.json", presets)
        _dump(run_dir / "manifest.json", manifest)
        process = subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=False)
        _dump(run_dir / "execution.json", {"returncode": process.returncode})
        result_path = Path(manifest["result"])
        if result_path.is_file():
            result = _json(result_path)
            result["specialization_gap"] = manifest
            _dump(result_path, result)


def _metric(result, names):
    for name in names:
        value = result.get(name)
        if value is not None and not isinstance(value, bool):
            try:
                number = float(value)
                if math.isfinite(number) and number >= 0:
                    return number
            except (TypeError, ValueError):
                pass
    return None


def _measurement_status(run_dir, result, manifest):
    status = {"measurement_valid": False, "frequency_verified": False, "energy_coverage_valid": False,
              "energy_duration_s": None, "energy_lead_in_s": None, "energy_tail_s": None,
              "measurement_error": "missing raw power log or benchmark bounds"}
    path = run_dir / "demo-state" / "power.jsonl"
    if not path.is_file():
        return status
    try:
        with open(path, encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
        if len(samples) < 2 or result.get("energy_num_samples") != len(samples):
            raise ValueError("at least two matching raw/summary energy samples are required")
        expected = {str(device): manifest["lane%d" % lane]["op"]["frequency_mhz"]
                    for lane in (0, 1) for device in manifest["lane%d" % lane]["preset"]["ascend_devices"]}
        timestamps = []
        for sample in samples:
            timestamp = _metric(sample, ("timestamp",))
            if timestamp is None:
                raise ValueError("invalid raw sample timestamp")
            timestamps.append(timestamp)
            readings = sample.get("per_npu") or {}
            if set(readings) != set(expected):
                raise ValueError("metered devices differ from the two lane maps")
            for device, frequency in expected.items():
                power = _metric(readings[device], ("power_sample",))
                observed_frequency = _metric(readings[device], ("frequency",))
                if power is None or power <= 0:
                    raise ValueError("missing/invalid raw NPU power sample for device %s" % device)
                if observed_frequency != frequency:
                    raise ValueError("missing/mismatched sampled frequency for device %s" % device)
        status["frequency_verified"] = True
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("energy timestamps must strictly increase")
        started = _metric(result, ("benchmark_start_s",))
        ended = _metric(result, ("benchmark_end_s",))
        if started is None or ended is None or not timestamps[0] <= started < ended <= timestamps[-1]:
            raise ValueError("raw power samples do not bracket the complete client subprocess")
        status.update({"energy_coverage_valid": True, "energy_duration_s": timestamps[-1] - timestamps[0],
                       "energy_lead_in_s": started - timestamps[0], "energy_tail_s": timestamps[-1] - ended})
        per_npu = result.get("energy_per_npu_joules") or {}
        if set(per_npu) != set(expected) or any((_metric(per_npu, (device,)) or 0) <= 0 for device in expected):
            raise ValueError("missing/invalid integrated energy for a lane device")
        energy = _metric(result, ("energy_joules",))
        if energy is None or not math.isclose(energy, sum(_metric(per_npu, (device,)) for device in expected), rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("total measured energy differs from the metered per-device totals")
        status.update({"measurement_valid": True, "measurement_error": ""})
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        status["measurement_error"] = str(exc)
    return status


def _aggregate_rows(root, config):
    rows = []
    experiment = _json(root / "experiment.json")
    condition = _frozen_serving_condition(experiment)
    if experiment["config"] != config:
        raise ValueError("aggregation config differs from the frozen experiment config")
    repetitions = experiment["repetitions"]
    for candidate_path in sorted(root.rglob("candidate.json")):
        manifest = _json(candidate_path)
        bank = manifest["bank"]
        mixture = next((item for item in config["mixtures"] if item == manifest["mixture"]), None)
        if mixture is None or bank not in _realizations(config, mixture, banks(config)):
            raise ValueError("unknown mixture or bank in %s" % candidate_path)
        if candidate_path.parent.name != bank["name"] or candidate_path.parent.parent.name != _slug(mixture["name"]):
            raise ValueError("candidate directory does not match its mixture/bank identity")
        cohort_path = candidate_path.parent.parent / "cohort.csv"
        cohort = load_schedule_events(str(cohort_path), active_pools="all", max_arrivals=0)
        cohort_hash = _cohort_hash(cohort)
        expected = _candidate_manifest(config, mixture, bank, repetitions, cohort_hash, cohort, Path(manifest["schedule"]), condition)
        if manifest != expected:
            raise ValueError("candidate metadata/resources/cohort differ from frozen experiment: %s" % candidate_path)
        if len(cohort) != config["total_requests"] or {pool: sum(item["pool"] == pool for item in cohort) for pool in ("summarization", "conversation")} != _class_counts(config, mixture):
            raise ValueError("cohort work differs from configured mixture")
        for index in range(1, repetitions + 1):
            run_dir = candidate_path.parent / ("rep-%d" % index)
            result_path = run_dir / "result.json"
            result = _json(result_path) if result_path.is_file() else {}
            if result and result.get("serving_condition") != condition:
                raise ValueError("result serving condition differs from frozen experiment: %s" % result_path)
            execution_path = run_dir / "execution.json"
            execution = _json(execution_path) if execution_path.is_file() else {}
            run_manifest_path = run_dir / "manifest.json"
            run_manifest = _json(run_manifest_path) if run_manifest_path.is_file() else {}
            metadata_valid = bool(run_manifest) and all(run_manifest.get(key) == value for key, value in manifest.items())
            metadata_valid = metadata_valid and run_manifest.get("repetition") == index and result.get("specialization_gap") == run_manifest
            requests = result.get("requests") or []
            offered = manifest["offered"]
            completed = sum(item.get("ok") is True and item.get("stream_complete") is True for item in requests)
            identity_valid = sorted(item.get("prompt_idx", -1) for item in requests) == list(range(1, offered + 1))
            routing_valid = identity_valid
            for request in requests:
                request_index = request.get("prompt_idx")
                if type(request_index) is not int or not 1 <= request_index <= offered:
                    routing_valid = False
                    continue
                pool = cohort[request_index - 1]["pool"]
                target = (manifest["routing_kwargs"]["specialization_gap"]["pool_to_lane"].get(pool)
                          if manifest["routing_mode"] == "oracle_class_affinity"
                          else config["lanes"][(request_index - 1) % 2]["id"])
                if request.get("pool") != pool or request.get("ok") and request.get("lane") != target:
                    routing_valid = False
            ttft = _metric(result, ("99.9th_ttft", "ttft_p99_9_s"))
            tpot = _metric(result, ("99.9th_tpot", "tpot_p99_9_s"))
            required = manifest["slo"]
            slo_feasible = True
            for name, value in (("ttft_p99_9_s", ttft), ("tpot_p99_9_s", tpot)):
                if required.get(name) is not None:
                    slo_feasible = slo_feasible and value is not None and value <= float(required[name])
            energy = _metric(result, ("energy_joules",))
            samples = _metric(result, ("energy_num_samples",))
            measurement = _measurement_status(run_dir, result, manifest)
            energy_valid = energy is not None and energy > 0 and samples is not None and samples >= 2 and measurement["measurement_valid"]
            full_work = identity_valid and completed == offered
            comparable = metadata_valid and routing_valid and execution.get("returncode") == 0
            eligible = comparable and full_work and energy_valid and slo_feasible
            lane0, lane1 = manifest["lane0"]["op"], manifest["lane1"]["op"]
            rows.append({
                "model": manifest["model"], "tp": manifest["tp"], "seed": manifest["seed"], "mixture": mixture["name"],
                "bank": bank["name"], "bank_id": bank["bank_id"], "bank_type": bank["type"], "mapping": bank["mapping"],
                "effective_mapping": manifest["effective_mapping"], "repetition": index, "expected_repetitions": repetitions,
                "cohort_hash": cohort_hash, "max_num_batched_tokens": manifest["max_num_batched_tokens"],
                "lane0_frequency_mhz": lane0["frequency_mhz"], "lane0_max_num_seqs": lane0["max_num_seqs"],
                "lane1_frequency_mhz": lane1["frequency_mhz"], "lane1_max_num_seqs": lane1["max_num_seqs"],
                "lane0_device_map": json.dumps(config["lanes"][0]["device_map"]), "lane1_device_map": json.dumps(config["lanes"][1]["device_map"]),
                "routing_mode": manifest["routing_mode"], "prefill_target_lane": manifest["prefill_target_lane"],
                "decode_target_lane": manifest["decode_target_lane"], "neutral_load_balancing": manifest["neutral_load_balancing"],
                "routing_reason": manifest["routing_reason"], "routing_kwargs": json.dumps(manifest["routing_kwargs"], sort_keys=True),
                "mapping_aliases": json.dumps(bank.get("mapping_aliases", [])),
                "generation_settings": json.dumps(manifest["generation_settings"], sort_keys=True),
                "energy_scope": manifest["energy_scope"],
                "energy_joules": energy, "energy_num_samples": samples,
                "benchmark_start_s": result.get("benchmark_start_s"), "benchmark_end_s": result.get("benchmark_end_s"),
                **measurement,
                "total_input_tokens": result.get("total_input_tokens"), "total_output_tokens": result.get("total_output_tokens"),
                "total_num_tokens": result.get("total_num_tokens"), "total_elapsed_time": result.get("total_elapsed_time"),
                "tokens_per_second": result.get("tokens_per_second"), "mean_ttft_s": _metric(result, ("mean_ttft",)),
                "ttft_p99_9_s": ttft, "mean_tpot_s": _metric(result, ("mean_tpot",)), "tpot_p99_9_s": tpot,
                "ttft_limit_s": required.get("ttft_p99_9_s"), "tpot_limit_s": required.get("tpot_p99_9_s"),
                "offered": offered, "completed": completed, "failed": max(0, offered - completed),
                "rejected": 0 if full_work else "", "deferred": 0,
                "unfinished": max(0, offered - len(requests)) + sum(item.get("stream_complete") is False for item in requests),
                "result_present": result_path.is_file(), "returncode": execution.get("returncode"),
                "full_work": full_work, "comparable": comparable, "routing_valid": routing_valid,
                "energy_valid": energy_valid, "slo_feasible": slo_feasible, "eligible": eligible,
            })
    return rows


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root, config):
    rows = _aggregate_rows(root, config)
    identity_fields = ["model", "tp", "seed", "mixture", "cohort_hash", "bank", "bank_id", "bank_type", "mapping", "mapping_aliases", "effective_mapping", "routing_mode", "prefill_target_lane", "decode_target_lane", "neutral_load_balancing", "routing_reason", "routing_kwargs", "generation_settings", "energy_scope", "lane0_frequency_mhz", "lane0_max_num_seqs", "lane1_frequency_mhz", "lane1_max_num_seqs", "lane0_device_map", "lane1_device_map", "max_num_batched_tokens", "ttft_limit_s", "tpot_limit_s"]
    counts = ["offered", "completed", "failed", "rejected", "deferred", "unfinished"]
    latency = ["mean_ttft_s", "ttft_p99_9_s", "mean_tpot_s", "tpot_p99_9_s"]
    measurement_fields = ["benchmark_start_s", "benchmark_end_s", "energy_duration_s", "energy_lead_in_s", "energy_tail_s", "measurement_valid", "frequency_verified", "energy_coverage_valid", "measurement_error"]
    fields = identity_fields + ["repetition", "expected_repetitions", "energy_joules", "energy_num_samples", "total_input_tokens", "total_output_tokens", "total_num_tokens", "total_elapsed_time", "tokens_per_second"] + measurement_fields + counts + latency + ["result_present", "returncode", "full_work", "comparable", "routing_valid", "energy_valid", "slo_feasible", "eligible"]
    _write_csv(root / "runs.csv", rows, fields)
    banks_summary = []
    for mixture in sorted({row["mixture"] for row in rows}):
        if len({row["cohort_hash"] for row in rows if row["mixture"] == mixture}) != 1:
            raise ValueError("cannot compare different cohorts for mixture %s" % mixture)
        for bank in sorted({row["bank"] for row in rows if row["mixture"] == mixture}):
            group = [row for row in rows if row["mixture"] == mixture and row["bank"] == bank]
            valid = [row["energy_joules"] for row in group if row["eligible"]]
            record = {key: group[0][key] for key in identity_fields}
            record.update({"repetitions": len(group), "valid_repetitions": len(valid),
                           "mean_energy_j": statistics.mean(valid) if valid else None,
                           "std_energy_j": statistics.stdev(valid) if len(valid) > 1 else None,
                           "slo_feasible": all(row["slo_feasible"] for row in group),
                           "feasible": len(valid) == len(group),
                           "measurements_complete": all(row["result_present"] and row["returncode"] == 0 and row["energy_valid"] and row["comparable"] and row["full_work"] for row in group)})
            # Count columns are per-repetition means; missing planned runs remain in the denominator.
            for key in counts + latency:
                values = [row[key] for row in group if row[key] is not None and row[key] != ""]
                record[key] = statistics.mean(values) if len(values) == len(group) else None
            banks_summary.append(record)
    for row in banks_summary:
        pair = [item for item in banks_summary if item["mixture"] == row["mixture"] and item["bank_id"] == row["bank_id"] and item["feasible"]]
        best = min(pair, key=lambda item: (item["mean_energy_j"], item["bank"])) if pair else None
        row["best_mapping_for_pair"] = best is row
    bank_fields = identity_fields + ["repetitions", "valid_repetitions", "mean_energy_j", "std_energy_j"] + counts + latency + ["slo_feasible", "feasible", "measurements_complete", "best_mapping_for_pair"]
    _write_csv(root / "banks.csv", banks_summary, bank_fields)
    summary = []
    for mixture in config["mixtures"]:
        group = [row for row in banks_summary if row["mixture"] == mixture["name"]]
        feasible = [row for row in group if row["feasible"]]
        shared_candidates = [row for row in feasible if row["bank_type"] == "shared"]
        shared = min(shared_candidates, key=lambda row: (row["mean_energy_j"], row["bank"])) if shared_candidates else None
        # Exact same diagonal records are eligible in both searches: no noisy duplicate baseline runs.
        specialized = min(feasible, key=lambda row: (row["mean_energy_j"], row["bank_type"] != "shared", row["bank"])) if feasible else None
        shared_energy = shared["mean_energy_j"] if shared else None
        specialized_energy = specialized["mean_energy_j"] if specialized else None
        gap = shared_energy - specialized_energy if shared and specialized else None
        if gap is not None and gap < 0:
            raise ValueError("specialized search must contain the identical shared records")
        record = {
            "mixture": mixture["name"], "cohort_hash": group[0]["cohort_hash"] if group else None,
            "seed": config["seed"], "max_num_batched_tokens": config["fixed_max_num_batched_tokens"],
            "search_complete": {row["bank"] for row in group} == {bank["name"] for bank in _realizations(config, mixture, banks(config))} and all(row["measurements_complete"] for row in group),
            "selection_scope": "measured_feasible_candidates",
            "best_shared_bank": shared["bank"] if shared else None, "best_shared_energy_j": shared_energy,
            "best_specialized_bank": specialized["bank"] if specialized else None,
            "best_specialized_mapping": specialized["mapping"] if specialized else None,
            "best_specialized_effective_mapping": specialized["effective_mapping"] if specialized else None,
            "best_specialized_energy_j": specialized_energy, "specialization_gap_j": gap,
            "specialization_gap_pct": gap / shared_energy * 100 if gap is not None and shared_energy > 0 else None,
        }
        for prefix, winner in (("shared", shared), ("specialized", specialized)):
            for key in latency + ["routing_mode", "offered", "completed", "slo_feasible"]:
                record[prefix + "_" + key] = winner[key] if winner else None
        summary.append(record)
    summary_fields = ["mixture", "cohort_hash", "seed", "max_num_batched_tokens", "search_complete", "selection_scope", "best_shared_bank", "best_shared_energy_j", "best_specialized_bank", "best_specialized_mapping", "best_specialized_effective_mapping", "best_specialized_energy_j", "specialization_gap_j", "specialization_gap_pct"] + [prefix + "_" + key for prefix in ("shared", "specialized") for key in latency + ["routing_mode", "offered", "completed", "slo_feasible"]]
    _write_csv(root / "summary.csv", summary, summary_fields)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="experiments.specialization_gap")
    parser.add_argument("config", nargs="?", default=str(Path(__file__).with_name("specialization-gap.json")))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--mixture")
    parser.add_argument("--bank", help="exact bank ID, e.g. specialized-0-1")
    parser.add_argument("--mapping", choices=("pd", "dp", "neutral"), help="class-to-lane variant; shared diagonals use neutral")
    parser.add_argument("--routing", choices=("oracle_class_affinity", "round_robin"), help="primary oracle experiment or optional secondary round-robin control")
    parser.add_argument("--op-indices", help="candidate OP indices to enumerate, e.g. 0,1")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--repetition", type=int, help="run only this 1-based repetition within --repetitions")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "experiments" / "_specialization-gap"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.routing:
        config["routing"]["routing_policy"] = args.routing
    root = Path(args.output_root).expanduser().resolve()
    if args.aggregate:
        return aggregate(root, config)
    mixtures = _mixture_filter(config, args.mixture)
    requested_bank = args.bank
    if requested_bank and requested_bank.endswith("-neutral"):
        if args.mapping not in (None, "neutral"):
            raise ValueError("neutral bank ID conflicts with --mapping")
        requested_bank = requested_bank.removesuffix("-neutral")
        args.mapping = "neutral"
    selected_banks = banks(config, requested_bank, args.op_indices)
    repetitions = int(args.repetitions if args.repetitions is not None else config.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if args.repetition is not None and not 1 <= args.repetition <= repetitions:
        raise ValueError("--repetition must be within 1..repetitions")
    if args.list:
        for mixture, bank, repetition, run_dir in _planned(config, mixtures, selected_banks, repetitions, root, args.repetition, args.mapping):
            routing = _routing(config, mixture, bank)
            print("%s mapping=%s aliases=%s effective=%s routing=%s P->%s D->%s lane0=%s lane1=%s max_num_batched_tokens=%s" % (
                run_dir, bank["mapping"], bank.get("mapping_aliases", []), routing["effective_mapping"], routing["routing_mode"],
                routing["prefill_target_lane"], routing["decode_target_lane"],
                config["candidate_ops"][bank["ops"][0]], config["candidate_ops"][bank["ops"][1]],
                config["fixed_max_num_batched_tokens"]))
        return 0
    run_mode(config, root, mixtures, selected_banks, repetitions, args.dry_run, args.repetition, args.mapping)
    return 0


if __name__ == "__main__":
    sys.exit(main())
