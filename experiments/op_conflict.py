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

"""Offline OP selection, followed by a frozen two-lane synthetic specialization sweep.

--select freezes selection.json from characterization result JSONs; --inspect is read-only.
Runs and --aggregate only consume that manifest, never reselect using experiment outcomes.
P/D optima minimize mean characterized joules/request; the shared compromise minimizes
the sum of each shape's energy normalized to its own optimum, over OPs feasible for both shapes.
The three selected roles may coincide: do not manufacture an OP conflict.
TPOT is inapplicable to one-output-token shapes; every applicable SLO must hold per run.
Energy remains measured, sampled-boundary-inclusive energy of both physical lanes.
"""

import argparse
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

from experiments import specialization_gap as gap
from src.client import _load_op_conflict_events, _percentile
from src.energy_meter import summarize_log
from src.serving_profiles import normalize_serving_config

SHAPES = ("prefill-heavy", "decode-heavy")
banks = gap.banks
RUNTIME_KEYS = ("PATH", "PYTHONPATH", "VLLM_ROOT", "VLLM_SERVE_EXTRA_ARGS", "VLLM_USE_V1",
                "PROFILE_JSON", "WAIT_SECS", "ROUTER_PORT", "POWER_INTERVAL_S")


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError("%s must be a positive integer" % name)


def load_config(path):
    config = gap._json(Path(path))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    required = ("model", "tp", "characterization_root", "characterization_glob", "characterization_repetitions",
                 "fixed_max_num_batched_tokens", "maximum_frequency_mhz", "lanes", "shapes", "slo", "mixtures", "synthetic_token_id",
                 "arrival_interval_s", "request_timeout_seconds", "seed", "total_requests", "repetitions")
    if any(key not in config for key in required):
        raise ValueError("missing OP-conflict config fields")
    if "candidate_ops" in config:
        raise ValueError("OPs must come from the frozen characterization selection")
    unknown = set(config) - set(required) - {"_license", "serving"}
    if unknown:
        raise ValueError("unknown OP-conflict config fields: %s" % ", ".join(sorted(unknown)))
    for key in ("tp", "characterization_repetitions", "fixed_max_num_batched_tokens", "maximum_frequency_mhz", "total_requests", "repetitions"):
        _positive_int(config[key], key)
    if config["total_requests"] < 2 or type(config["seed"]) is not int:
        raise ValueError("total_requests must be >= 2 and seed must be an integer")
    if type(config["synthetic_token_id"]) is not int or config["synthetic_token_id"] < 0:
        raise ValueError("synthetic_token_id must be a nonnegative integer")
    interval = config["arrival_interval_s"]
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("arrival_interval_s must be finite and positive")
    timeout = config["request_timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout_seconds must be finite and positive")
    if not isinstance(config["model"], str) or not config["model"].strip():
        raise ValueError("model must be a nonempty string")
    if not all(isinstance(config[key], str) and config[key].strip() for key in ("characterization_root", "characterization_glob")):
        raise ValueError("characterization_root and characterization_glob must be nonempty strings")
    if not isinstance(config["shapes"], dict) or set(config["shapes"]) != set(SHAPES):
        raise ValueError("exactly prefill-heavy and decode-heavy shapes are required")
    for shape in config["shapes"].values():
        if not isinstance(shape, dict) or set(shape) != {"prompt_tokens", "output_tokens"}:
            raise ValueError("each shape requires only prompt_tokens and output_tokens")
        for key, value in shape.items():
            _positive_int(value, key)
    if not isinstance(config["slo"], dict) or set(config["slo"]) != {"ttft_p99_9_s", "tpot_p99_9_s"}:
        raise ValueError("both TTFT and TPOT p99.9 SLO limits are required")
    for value in config["slo"].values():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("SLO limits must be finite and positive")
    if not isinstance(config["lanes"], list) or len(config["lanes"]) != 2:
        raise ValueError("exactly two physical lanes are required")
    devices, ports, ids = [], [], []
    for lane in config["lanes"]:
        if (not isinstance(lane, dict) or set(lane) != {"id", "url", "device_map"}
                or not isinstance(lane["id"], str) or not lane["id"].strip() or not isinstance(lane["url"], str)):
            raise ValueError("lanes require id, url, device_map and cannot override serving knobs")
        if not isinstance(lane["device_map"], list) or len(lane["device_map"]) != config["tp"]:
            raise ValueError("each lane requires exactly tp devices")
        canonical = []
        for device in lane["device_map"]:
            text = str(device).strip()
            if type(device) not in (str, int) or not text.isascii() or not text.isdecimal():
                raise ValueError("device IDs must be nonnegative decimal integers")
            canonical.append(str(int(text)))
        lane["device_map"] = canonical
        url = urlsplit(lane["url"])
        if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost")
                or not url.port or url.path not in ("", "/") or url.query or url.fragment or url.username):
            raise ValueError("lane URLs must be local HTTP endpoints with explicit ports")
        devices.extend(canonical)
        ports.append(url.port)
        ids.append(lane["id"])
    if len(set(devices)) != len(devices) or len(set(ports)) != 2 or len(set(ids)) != 2:
        raise ValueError("lane device maps, ports and IDs must be disjoint/distinct")
    if not isinstance(config["mixtures"], list) or len(config["mixtures"]) != 3:
        raise ValueError("exactly three mixtures are required")
    for mixture in config["mixtures"]:
        if not isinstance(mixture, dict) or set(mixture) != {"name", "prefill_pct", "decode_pct"}:
            raise ValueError("mixtures require name, prefill_pct, decode_pct")
        if not isinstance(mixture["name"], str) or not mixture["name"].strip():
            raise ValueError("mixture names must be nonempty")
        percentages = (mixture["prefill_pct"], mixture["decode_pct"])
        if any(type(value) is not int or not 0 <= value <= 100 for value in percentages) or sum(percentages) != 100:
            raise ValueError("mixture percentages must sum to 100")
        if any(config["total_requests"] * value % 100 for value in percentages):
            raise ValueError("total_requests must represent the mixture percentages exactly")
    if ({item["prefill_pct"] for item in config["mixtures"]} != {0, 50, 100}
            or len({gap._slug(item["name"]) for item in config["mixtures"]}) != 3):
        raise ValueError("three unique 100/0, 50/50 and 0/100 mixtures are required")
    if "serving" in config:
        config["serving"] = normalize_serving_config(config["serving"])
    return config


def _slo_status(config, shape, ttft, tpot):
    return (ttft is not None and ttft <= config["slo"]["ttft_p99_9_s"]
            and (config["shapes"][shape]["output_tokens"] == 1
                 or tpot is not None and tpot <= config["slo"]["tpot_p99_9_s"]))


def _characterization_valid(result, config, shape):
    lengths = config["shapes"][shape]
    requests = result.get("requests") or []
    count = result.get("num_requests")
    energy = gap._metric(result, ("energy_joules",))
    per_npu = result.get("energy_per_npu_joules") or {}
    return bool(
        type(count) is int and count > 0 and isinstance(requests, list) and len(requests) == count
        and all(isinstance(item, dict) and type(item.get("prompt_idx")) is int for item in requests)
        and result.get("shape_valid") is True
        and result.get("prompt_tokens") == lengths["prompt_tokens"]
        and result.get("output_tokens") == lengths["output_tokens"]
        and sorted(item.get("prompt_idx", -1) for item in requests) == list(range(1, count + 1))
        and all(item.get("ok") is True and item.get("shape_valid") is True
                and item.get("stream_complete") is not False
                and item.get("input_num_tokens") == lengths["prompt_tokens"]
                and item.get("output_num_tokens") == lengths["output_tokens"] for item in requests)
        and energy is not None and energy > 0
        and (gap._metric(result, ("energy_num_samples",)) or 0) >= 2
        and isinstance(per_npu, dict) and len(per_npu) == config["tp"]
        and all((gap._metric(per_npu, (device,)) or 0) > 0 for device in per_npu)
        and math.isclose(energy, sum(float(value) for value in per_npu.values()), rel_tol=1e-9, abs_tol=1e-6)
        and _slo_status(config, shape, gap._metric(result, ("99.9th_ttft",)), gap._metric(result, ("99.9th_tpot",)))
    )


def select(config, source_root, serving_condition=None):
    source_root = Path(source_root).expanduser().resolve()
    grouped, sources, excluded, offered_counts = {}, [], [], set()
    relevant, conditions = [], set()
    for path in sorted(source_root.glob(config["characterization_glob"])):
        result = gap._json(path)
        shape = result.get("request_shape") if isinstance(result, dict) else None
        if shape not in SHAPES:
            continue
        source = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        sources.append(source)
        if (result.get("model_name") != config["model"] or result.get("tp") != config["tp"]
                or result.get("max_num_batched_tokens") != config["fixed_max_num_batched_tokens"]):
            excluded.append(dict(source, reason="model/tp/fixed batching mismatch"))
            continue
        frequency, seqs, repetition = (result.get(key) for key in ("frequency_mhz", "max_num_seqs", "repetition"))
        for key, value in (("frequency_mhz", frequency), ("max_num_seqs", seqs), ("repetition", repetition)):
            _positive_int(value, key)
        if frequency > config["maximum_frequency_mhz"]:
            excluded.append(dict(source, reason="frequency exceeds configured maximum"))
            continue
        condition = result.get("serving_condition") or "legacy"
        if not isinstance(condition, str):
            raise ValueError("invalid characterization serving condition: %s" % path)
        conditions.add(condition)
        relevant.append((source, result, shape, frequency, seqs, repetition, condition))
    available = ", ".join(sorted(conditions)) or "(none)"
    if serving_condition is None:
        if len(conditions) > 1:
            raise ValueError("multiple serving conditions; choose --serving-condition from: " + available)
        selected_condition = next(iter(conditions), "legacy")
    else:
        if serving_condition not in conditions:
            raise ValueError("serving condition %r not found; available: %s" % (serving_condition, available))
        selected_condition = serving_condition
    for source, result, shape, frequency, seqs, repetition, condition in relevant:
        if condition != selected_condition:
            excluded.append(dict(source, reason="serving condition mismatch", serving_condition=condition))
            continue
        key = (shape, frequency, seqs)
        group = grouped.setdefault(key, {})
        if repetition in group:
            raise ValueError("duplicate characterization repetition for %s: %s" % (key, source["path"]))
        group[repetition] = result
        if type(result.get("num_requests")) is int and result["num_requests"] > 0:
            offered_counts.add(result["num_requests"])
    if len(offered_counts) > 1:
        raise ValueError("characterization runs have different offered request counts")
    expected = set(range(1, config["characterization_repetitions"] + 1))
    eligible, diagnostics = {}, []
    for (shape, frequency, seqs), group in sorted(grouped.items()):
        valid = set(group) == expected and all(_characterization_valid(row, config, shape) for row in group.values())
        record = {"request_shape": shape, "frequency_mhz": frequency, "max_num_seqs": seqs,
                  "repetitions": sorted(group), "eligible": valid,
                  "invalid_repetitions": [index for index, row in sorted(group.items())
                                          if not _characterization_valid(row, config, shape)],
                  "missing_repetitions": sorted(expected - set(group))}
        if valid:
            energies = [gap._metric(row, ("energy_joules",)) / row["num_requests"] for row in group.values()]
            record.update(mean_energy_per_request_j=statistics.mean(energies),
                          std_energy_per_request_j=statistics.stdev(energies) if len(energies) > 1 else None)
            eligible[(shape, frequency, seqs)] = record["mean_energy_per_request_j"]
        diagnostics.append(record)
    winners = {}
    for role, shape in zip(("prefill", "decode"), SHAPES):
        ops = [(frequency, seqs) for candidate_shape, frequency, seqs in eligible if candidate_shape == shape]
        if not ops:
            raise ValueError("no fully measured SLO-feasible characterization OP for %s" % shape)
        winners[role] = min(ops, key=lambda item: (eligible[(shape, *item)], item))
    common = {(frequency, seqs) for shape, frequency, seqs in eligible if shape == SHAPES[0]}
    common &= {(frequency, seqs) for shape, frequency, seqs in eligible if shape == SHAPES[1]}
    if not common:
        raise ValueError("no characterization OP feasible for both shapes; no shared compromise")
    normalized = {
        item: sum(eligible[(shape, *item)] / eligible[(shape, *winners[role])]
                  for shape, role in zip(SHAPES, ("prefill", "decode")))
        for item in common
    }
    winners["shared"] = min(common, key=lambda item: (normalized[item], item))
    selected_ops = sorted(set(winners.values()))
    # Retain enough measured near-optimal common alternatives for a meaningful two-lane sweep,
    # without replacing coincident roles with made-up conflicting OPs.
    for item in sorted(common, key=lambda value: (normalized[value], value)):
        if len(selected_ops) >= 3:
            break
        if item not in selected_ops:
            selected_ops.append(item)
    selected_ops.sort()
    return {
        "format": "parsa-op-conflict-selection-v1", "config": config,
        "serving_condition": selected_condition,
        "characterization_root": str(source_root), "sources": sources, "excluded_sources": excluded,
        "criterion": "per-shape minimum mean J/request; shared minimum sum of P/D energy normalized to their respective optima; measured alternatives added in normalized-rank order until three distinct OPs; ties by frequency then max_num_seqs",
        "candidate_ops": [{"frequency_mhz": frequency, "max_num_seqs": seqs} for frequency, seqs in selected_ops],
        "roles": {role: selected_ops.index(item) for role, item in winners.items()},
        "op_conflict_observed": winners["prefill"] != winners["decode"], "characterization": diagnostics,
    }


def load_selection(root, config):
    path = root / "selection.json"
    if not path.is_file():
        raise ValueError("missing frozen selection.json; use --select before running")
    selection = gap._json(path)
    if (not isinstance(selection, dict) or selection.get("format") != "parsa-op-conflict-selection-v1"
            or selection.get("config") != config):
        raise ValueError("selection does not match the experiment config; use a fresh output root")
    if "serving_condition" in selection and (
        not isinstance(selection["serving_condition"], str) or not selection["serving_condition"].strip()
    ):
        raise ValueError("invalid frozen characterization serving condition")
    # Missing condition means legacy. Keep the old object intact for frozen selection hashes.
    ops = selection.get("candidate_ops")
    if not isinstance(ops, list) or not ops or len(ops) > 3:
        raise ValueError("invalid selected OP set")
    for item in ops:
        if not isinstance(item, dict) or set(item) != {"frequency_mhz", "max_num_seqs"}:
            raise ValueError("selected OPs can only set frequency_mhz and max_num_seqs")
        for key, value in item.items():
            _positive_int(value, key)
        if item["frequency_mhz"] > config["maximum_frequency_mhz"]:
            raise ValueError("selected OP exceeds the configured maximum frequency")
    if len({(item["frequency_mhz"], item["max_num_seqs"]) for item in ops}) != len(ops):
        raise ValueError("duplicate selected OPs")
    roles = selection.get("roles")
    if (not isinstance(roles, dict) or set(roles) != {"prefill", "decode", "shared"}
            or any(type(index) is not int or not 0 <= index < len(ops) for index in roles.values())):
        raise ValueError("invalid selected role indices")
    return selection


def class_counts(config, mixture):
    prefill = config["total_requests"] * mixture["prefill_pct"] // 100
    return {SHAPES[0]: prefill, SHAPES[1]: config["total_requests"] - prefill}


def cohort(config, mixture):
    labels = [shape for shape, count in class_counts(config, mixture).items() for _ in range(count)]
    random.Random(config["seed"] + sum(ord(ch) for ch in mixture["name"])).shuffle(labels)
    return [{"request_index": index, "arrival_sec": (index - 1) * config["arrival_interval_s"],
             "request_shape": shape, "token_id": config["synthetic_token_id"], **config["shapes"][shape]}
            for index, shape in enumerate(labels, 1)]


def routing(config, mixture, bank):
    active = [shape for shape, count in class_counts(config, mixture).items() if count]
    oracle = len(active) == 2 and bank["type"] == "specialized"
    p_lane = 0 if bank["mapping"] == "pd" else 1
    return {"lane_ids": [lane["id"] for lane in config["lanes"]], "active_pools": active,
            "pool_to_lane": {SHAPES[0]: config["lanes"][p_lane]["id"],
                             SHAPES[1]: config["lanes"][1 - p_lane]["id"]} if oracle else {}}


def realizations(config, mixture, selected_banks):
    result = {}
    for bank in selected_banks:
        if bank["type"] == "specialized" and len([count for count in class_counts(config, mixture).values() if count]) == 1:
            bank = dict(bank, name=bank["bank_id"] + "-neutral", mapping="neutral", mapping_aliases=["pd", "dp"])
        result[bank["name"]] = bank
    return list(result.values())


def candidate_manifest(config, selection, mixture, bank, repetitions, events, schedule, serving_condition):
    resolved = dict(config, candidate_ops=selection["candidate_ops"])
    presets = gap._presets(resolved, bank)
    options = routing(config, mixture, bank)
    return {"model": config["model"], "tp": config["tp"], "mixture": mixture, "bank": bank,
            "serving_condition": serving_condition,
            "selection_hash": gap._cohort_hash([selection]), "cohort_hash": gap._cohort_hash(events),
            "seed": config["seed"], "offered": len(events), "class_counts": class_counts(config, mixture),
            "expected_repetitions": repetitions, "schedule": str(schedule), "shapes": config["shapes"],
            "fixed_max_num_batched_tokens": config["fixed_max_num_batched_tokens"],
        "generation_settings": {"temperature": 0, "ignore_eos": True, "stream": True,
                                     "min_tokens_equals_max_tokens": True, "max_in_flight": config["total_requests"],
                                     "request_timeout_seconds": config["request_timeout_seconds"]},
            "energy_scope": "sampled_boundary_inclusive_client_subprocess", "slo": config["slo"],
            "routing_mode": "oracle_class_affinity" if options["pool_to_lane"] else "round_robin",
            "routing_kwargs": {"op_conflict": options}, "presets": presets,
            **{"lane%d" % index: {"op": selection["candidate_ops"][bank["ops"][index]], "preset": presets[index]}
               for index in (0, 1)}}


def _run_env(config, schedule, manifest, run_dir):
    # Reuse the fixed-knob override guard and mandatory two-lane/measurement environment.
    env = gap._run_env(config, schedule, manifest["presets"], run_dir / "result.json", run_dir,
                       {"routing_mode": manifest["routing_mode"], "routing_kwargs": manifest["routing_kwargs"]})
    env.pop("EXPERIMENT_METADATA", None)
    env["KV_MONITOR_ENABLED"] = "0"
    env["CLIENT_EXTRA_ARGS"] = shlex.join([
        "--prompt-source", "synthetic", "--op-conflict", "--op-conflict-cohort", str(schedule),
        "--stream-metrics", "--temperature", "0", "--max-in-flight", str(config["total_requests"]),
        "--request-timeout-seconds", str(config["request_timeout_seconds"]),
    ])
    router_port = int(env.get("ROUTER_PORT", "8010"))
    if not 1 <= router_port <= 65535 or router_port in {urlsplit(lane["url"]).port for lane in config["lanes"]}:
        raise ValueError("router port must be valid and distinct from both serving lanes")
    interval = float(env.get("POWER_INTERVAL_S", "1.0"))
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("POWER_INTERVAL_S must be finite and positive")
    return env


def _experiment(config, selection, repetitions, serving_profile, serving_condition):
    return {"config": config, "selection_hash": gap._cohort_hash([selection]), "repetitions": repetitions,
            "serving_profile": serving_profile, "serving_condition": serving_condition,
            "runtime_environment": {key: os.environ.get(key) for key in RUNTIME_KEYS}}


def planned(config, mixtures, selected_banks, repetitions, root, repetition=None, mapping=None):
    for mixture in mixtures:
        realized = [bank for bank in realizations(config, mixture, selected_banks)
                    if not mapping or bank["mapping"] == mapping or mapping in bank.get("mapping_aliases", [])]
        if not realized:
            raise ValueError("mapping has no realization for mixture %s" % mixture["name"])
        for bank in realized:
            for index in ([repetition] if repetition is not None else range(1, repetitions + 1)):
                yield mixture, bank, index, root / gap._slug(mixture["name"]) / bank["name"] / ("rep-%d" % index)


def run_mode(config, selection, root, mixtures, selected_banks, repetitions, dry_run=False, repetition=None, mapping=None):
    prepared = []
    for mixture, bank, index, run_dir in planned(config, mixtures, selected_banks, repetitions, root, repetition, mapping):
        events = cohort(config, mixture)
        schedule = run_dir.parent.parent / "cohort.json"
        if schedule.exists() and _load_op_conflict_events(schedule) != events:
            raise ValueError("frozen cohort differs; use a fresh output root")
        candidate = candidate_manifest(config, selection, mixture, bank, repetitions, events, schedule, None)
        manifest = dict(candidate, repetition=index, result=str(run_dir / "result.json"))
        env = _run_env(config, schedule, candidate, run_dir)
        print("%s mapping=%s routing=%s ops=%s cohort=%s" % (
            run_dir, bank["mapping"], candidate["routing_mode"], bank["ops"], candidate["cohort_hash"]))
        prepared.append((run_dir, events, candidate, manifest, env))
    profile, condition = gap._plan_serving(config, (env for _, _, _, _, env in prepared))
    experiment = _experiment(config, selection, repetitions, profile, condition)
    gap._check_experiment(root, experiment)
    for run_dir, _, candidate, manifest, _ in prepared:
        if not dry_run and run_dir.exists():
            raise ValueError("refusing to overwrite existing run: %s" % run_dir)
        candidate["serving_condition"] = manifest["serving_condition"] = condition
    if dry_run:
        return 0
    gap._persist(root / "experiment.json", experiment)
    for run_dir, events, candidate, _, _ in prepared:
        gap._persist(run_dir.parent.parent / "cohort.json", events)
        gap._persist(run_dir.parent / "candidate.json", candidate)
    code = 0
    # Register every candidate first so interrupted/failed repetitions remain visible to aggregation.
    for run_dir, _, candidate, manifest, env in prepared:
        run_dir.mkdir(parents=True, exist_ok=False)
        gap._dump(run_dir / "presets.json", candidate["presets"])
        gap._dump(run_dir / "manifest.json", manifest)
        command = [sys.executable, str(REPO_ROOT / "run.py")]
        with open(run_dir / "stdout.log", "w", encoding="utf-8") as stdout, open(run_dir / "stderr.log", "w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=stdout, stderr=stderr, check=False)
        gap._dump(run_dir / "execution.json", {"command": command, "cwd": str(REPO_ROOT), "returncode": process.returncode})
        code = process.returncode or code
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = gap._json(result_path)
            result["op_conflict"] = manifest
            gap._dump(result_path, result)
    return code


def measurement_status(run_dir, result, manifest):
    status = gap._measurement_status(run_dir, result, manifest)
    if status["measurement_valid"]:
        # Validate the summary against raw integration, not merely internally consistent totals.
        summary = summarize_log(run_dir / "demo-state" / "power.jsonl")
        if any(not math.isclose(float(result["energy_per_npu_joules"][device]), energy, rel_tol=1e-9, abs_tol=1e-6)
               for device, energy in summary["per_npu"].items()):
            status.update(measurement_valid=False, measurement_error="integrated energy differs from raw power log")
    return status


def _run_record(path):
    if not path.is_file():
        return {}, "missing record"
    try:
        record = gap._json(path)
        if not isinstance(record, dict):
            raise ValueError("run record must be a JSON object")
        return record, ""
    except (OSError, ValueError) as exc:
        return {}, str(exc)


def _run_row(config, candidate, run_dir, index, events):
    result_path = run_dir / "result.json"
    result, result_error = _run_record(result_path)
    if result and result.get("serving_condition") != candidate["serving_condition"]:
        raise ValueError("result serving condition differs from frozen experiment: %s" % result_path)
    execution, execution_error = _run_record(run_dir / "execution.json")
    run_manifest, manifest_error = _run_record(run_dir / "manifest.json")
    expected_manifest = dict(candidate, repetition=index, result=str(result_path))
    metadata_valid = (not manifest_error and run_manifest == expected_manifest
                      and result.get("op_conflict") == expected_manifest)
    raw_requests = result.get("requests") or []
    requests = [item for item in raw_requests if isinstance(item, dict)] if isinstance(raw_requests, list) else []
    by_index = {item.get("prompt_idx"): item for item in requests if type(item.get("prompt_idx")) is int}
    identity = (len(requests) == len(events) and requests == raw_requests
                and set(by_index) == set(range(1, len(events) + 1)))
    options = candidate["routing_kwargs"]["op_conflict"]
    routing_valid, shape_valid = identity, identity
    completed = sum(item.get("ok") is True and item.get("stream_complete") is True for item in requests)
    latencies, slo_feasible = {}, identity
    for event in events:
        request = by_index.get(event["request_index"], {})
        target = (options["pool_to_lane"].get(event["request_shape"]) if options["pool_to_lane"]
                  else options["lane_ids"][(event["request_index"] - 1) % 2])
        routing_valid &= (request.get("request_shape") == event["request_shape"]
                          and (not request.get("ok") or request.get("lane") == target)
                          and not request.get("rescue", False))
        shape_valid &= (request.get("shape_valid") is True
                        and request.get("input_num_tokens") == event["prompt_tokens"]
                        and request.get("output_num_tokens") == event["output_tokens"]
                        and request.get("arrival_sec") == event["arrival_sec"])
    for shape in SHAPES:
        group = [request for request in requests if request.get("request_shape") == shape and request.get("ok")]
        ttfts = [gap._metric(request, ("ttft_s",)) for request in group]
        tpots = [gap._metric(request, ("tpot_mean_s",)) for request in group]
        ttft = _percentile(ttfts, 0.999) if group and all(value is not None for value in ttfts) else None
        tpot = _percentile(tpots, 0.999) if group and all(value is not None for value in tpots) else None
        latencies[shape + "_ttft_p99_9_s"] = ttft
        latencies[shape + "_tpot_p99_9_s"] = tpot
        if candidate["class_counts"][shape]:
            slo_feasible &= _slo_status(config, shape, ttft, tpot)
    measurement = measurement_status(run_dir, result, candidate)
    energy = gap._metric(result, ("energy_joules",))
    energy_valid = energy is not None and energy > 0 and measurement["measurement_valid"]
    full_work = identity and completed == len(events) and shape_valid
    comparable = metadata_valid and routing_valid and execution.get("returncode") == 0
    bank = candidate["bank"]
    return {"mixture": candidate["mixture"]["name"], "bank": bank["name"], "bank_id": bank["bank_id"],
            "bank_type": bank["type"], "mapping": bank["mapping"], "repetition": index,
            "expected_repetitions": candidate["expected_repetitions"], "model": candidate["model"], "tp": candidate["tp"],
            "selection_hash": candidate["selection_hash"], "cohort_hash": candidate["cohort_hash"], "seed": candidate["seed"],
            "fixed_max_num_batched_tokens": candidate["fixed_max_num_batched_tokens"],
            "routing_mode": candidate["routing_mode"], "routing_kwargs": json.dumps(candidate["routing_kwargs"], sort_keys=True),
            "presets": json.dumps(candidate["presets"], sort_keys=True), "energy_scope": candidate["energy_scope"],
            "energy_joules": energy, "energy_num_samples": result.get("energy_num_samples"),
            "benchmark_start_s": result.get("benchmark_start_s"), "benchmark_end_s": result.get("benchmark_end_s"),
            "total_input_tokens": result.get("total_input_tokens"), "total_output_tokens": result.get("total_output_tokens"),
            "total_elapsed_time": result.get("total_elapsed_time"), "tokens_per_second": result.get("tokens_per_second"),
            "offered": len(events), "completed": completed, "failed": max(0, len(events) - completed),
            "result_present": result_path.is_file(), "returncode": execution.get("returncode"),
            "result_error": result_error, "execution_error": execution_error, "manifest_error": manifest_error,
            "full_work": full_work, "shape_valid": shape_valid, "routing_valid": routing_valid,
            "metadata_valid": metadata_valid, "comparable": comparable, "energy_valid": energy_valid,
            "slo_feasible": slo_feasible, "eligible": comparable and full_work and energy_valid and slo_feasible,
            **latencies, **measurement}


def aggregate(root, config, selection):
    experiment = gap._json(root / "experiment.json")
    condition = gap._frozen_serving_condition(experiment)
    if experiment.get("config") != config or experiment.get("selection_hash") != gap._cohort_hash([selection]):
        raise ValueError("aggregation differs from frozen experiment/selection")
    repetitions = experiment["repetitions"]
    _positive_int(repetitions, "repetitions")
    resolved = dict(config, candidate_ops=selection["candidate_ops"])
    rows, summaries = [], []
    for candidate_path in sorted(root.glob("*/*/candidate.json")):
        candidate = gap._json(candidate_path)
        mixture = next((item for item in config["mixtures"] if item == candidate.get("mixture")), None)
        if mixture is None or candidate["bank"] not in realizations(config, mixture, banks(resolved)):
            raise ValueError("unknown candidate mixture/bank: %s" % candidate_path)
        bank = candidate["bank"]
        schedule = root / gap._slug(mixture["name"]) / "cohort.json"
        events = _load_op_conflict_events(schedule)
        expected = candidate_manifest(config, selection, mixture, bank, repetitions, events, schedule, condition)
        if (events != cohort(config, mixture) or candidate != expected
                or candidate_path.parent != schedule.parent / bank["name"]):
            raise ValueError("candidate resources/cohort differ from frozen experiment: %s" % candidate_path)
        group = [_run_row(config, candidate, candidate_path.parent / ("rep-%d" % index), index, events)
                 for index in range(1, repetitions + 1)]
        rows.extend(group)
        valid = [row["energy_joules"] for row in group if row["eligible"]]
        record = {key: group[0][key] for key in ("mixture", "bank", "bank_id", "bank_type", "mapping", "model", "tp",
            "selection_hash", "cohort_hash", "seed", "fixed_max_num_batched_tokens", "routing_mode", "routing_kwargs", "presets", "energy_scope")}
        record.update(repetitions=repetitions, valid_repetitions=len(valid), feasible=len(valid) == repetitions,
                      mean_energy_j=statistics.mean(valid) if valid else None,
                      std_energy_j=statistics.stdev(valid) if len(valid) > 1 else None,
                      measurements_complete=all(row["comparable"] and row["full_work"] and row["energy_valid"] for row in group),
                      slo_feasible=all(row["slo_feasible"] for row in group))
        for key in ("offered", "completed", "failed") + tuple(shape + suffix for shape in SHAPES for suffix in ("_ttft_p99_9_s", "_tpot_p99_9_s")):
            values = [row[key] for row in group]
            record[key] = statistics.mean(values) if all(value is not None for value in values) else None
        summaries.append(record)
    if not summaries:
        raise ValueError("no registered candidates to aggregate")
    for row in summaries:
        pair = [item for item in summaries if item["mixture"] == row["mixture"]
                and item["bank_id"] == row["bank_id"] and item["feasible"]]
        winner = min(pair, key=lambda item: (item["mean_energy_j"], item["bank"])) if pair else None
        row["best_mapping_for_pair"] = winner is row
    summary = []
    for mixture in config["mixtures"]:
        group = [row for row in summaries if row["mixture"] == mixture["name"]]
        feasible = [row for row in group if row["feasible"]]
        diagonals = [row for row in feasible if row["bank_type"] == "shared"]
        shared = min(diagonals, key=lambda row: (row["mean_energy_j"], row["bank"])) if diagonals else None
        specialized = min(feasible, key=lambda row: (row["mean_energy_j"], row["bank_type"] != "shared", row["bank"])) if feasible else None
        difference = shared["mean_energy_j"] - specialized["mean_energy_j"] if shared and specialized else None
        record = {"mixture": mixture["name"], "selection_hash": experiment["selection_hash"],
                  "cohort_hash": group[0]["cohort_hash"] if group else None,
                  "selection_scope": "measured_feasible_frozen_candidates",
                  "search_complete": {row["bank"] for row in group} == {bank["name"] for bank in realizations(config, mixture, banks(resolved))}
                     and all(row["measurements_complete"] for row in group),
                   "specialization_gap_j": difference,
                   "specialization_gap_pct": difference / shared["mean_energy_j"] * 100 if shared and difference is not None else None}
        for prefix, winner in (("shared", shared), ("specialized", specialized)):
            for key in ("bank", "mapping", "mean_energy_j", "std_energy_j", "valid_repetitions"):
                record[prefix + "_" + key] = winner[key] if winner else None
        summary.append(record)
    for name, data in (("runs", rows), ("banks", summaries), ("summary", summary)):
        gap._write_csv(root / (name + ".csv"), data, list(data[0]))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="experiments.op_conflict", description=__doc__)
    parser.add_argument("config", nargs="?", default=str(Path(__file__).with_name("op-conflict.json")))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--select", action="store_true", help="Freeze selection from offline characterization JSONs")
    modes.add_argument("--inspect", action="store_true", help="Print frozen selection, or prospective selection without writing")
    modes.add_argument("--aggregate", action="store_true")
    modes.add_argument("--list", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    parser.add_argument("--characterization-root", help="Source JSON root for --select / prospective --inspect only")
    parser.add_argument("--serving-condition", help="Characterization condition for offline --select / --inspect only (or legacy)")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "experiments" / "_op-conflict"))
    parser.add_argument("--mixture", help="Comma-separated mixture names")
    parser.add_argument("--bank", help="Bank/pair ID, e.g. specialized-0-1 or specialized-0-1-pd")
    parser.add_argument("--mapping", choices=("pd", "dp", "neutral"))
    parser.add_argument("--op-indices", help="Restrict frozen candidate OP indices for debugging")
    parser.add_argument("--repetitions", type=int, help="Default five; use a separate root for shorter debugging runs")
    parser.add_argument("--repetition", type=int, help="Run one 1-based repetition without changing planned repetitions")
    args = parser.parse_args(argv)
    if args.serving_condition is not None and not (args.select or args.inspect):
        parser.error("--serving-condition is only used for offline --select / --inspect")
    config = load_config(args.config)
    root = Path(args.output_root).expanduser().resolve()
    if args.characterization_root and not (args.select or args.inspect):
        parser.error("--characterization-root is only used for offline --select / --inspect")
    if args.select or args.inspect:
        source = Path(args.characterization_root or config["characterization_root"]).expanduser()
        if not source.is_absolute():
            source = REPO_ROOT / source
        if args.inspect and (root / "selection.json").is_file():
            selection = load_selection(root, config)
            if args.serving_condition is not None and args.serving_condition != (selection.get("serving_condition") or "legacy"):
                raise ValueError("frozen characterization serving condition differs from --serving-condition; use a fresh output root")
        else:
            selection = select(config, source, serving_condition=args.serving_condition)
        if args.select:
            gap._persist(root / "selection.json", selection)
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    selection = load_selection(root, config)
    if args.aggregate:
        return aggregate(root, config, selection)
    resolved = dict(config, candidate_ops=selection["candidate_ops"])
    mixtures = config["mixtures"]
    if args.mixture:
        requested = set(args.mixture.split(","))
        mixtures = [item for item in mixtures if item["name"] in requested]
        if len(mixtures) != len(requested):
            parser.error("unknown mixture selector")
    bank = args.bank
    if bank and bank.endswith("-neutral"):
        if args.mapping not in (None, "neutral"):
            parser.error("neutral bank conflicts with --mapping")
        bank, args.mapping = bank.removesuffix("-neutral"), "neutral"
    selected_banks = banks(resolved, bank, args.op_indices)
    repetitions = args.repetitions if args.repetitions is not None else config["repetitions"]
    _positive_int(repetitions, "repetitions")
    if args.repetition is not None and not 1 <= args.repetition <= repetitions:
        parser.error("--repetition must be within 1..repetitions")
    if args.list:
        for mixture, bank, index, run_dir in planned(config, mixtures, selected_banks, repetitions, root, args.repetition, args.mapping):
            print("%s ops=%s mapping=%s" % (run_dir, [selection["candidate_ops"][item] for item in bank["ops"]], bank["mapping"]))
        return 0
    return run_mode(config, selection, root, mixtures, selected_banks, repetitions, args.dry_run, args.repetition, args.mapping)


if __name__ == "__main__":
    sys.exit(main())
