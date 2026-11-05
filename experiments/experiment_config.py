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

import itertools
import json
from pathlib import Path

from src.serving_profiles import merge_serving_configs, normalize_serving_config


SWEEP_FORMAT = "parsa-benchmark-sweep-v1"


def _validate_flat_experiment(exp):
    if not isinstance(exp, dict):
        raise TypeError("experiment entries must be objects")
    name = str(exp.get("name") or "").strip()
    model = str(exp.get("model") or "").strip()
    tp = int(exp.get("tp"))
    lanes = exp.get("lanes")
    total_num_reqs = int(exp.get("total_num_reqs"))
    if not name:
        raise ValueError("experiment name must be non-empty")
    if not model:
        raise ValueError("experiment model must be non-empty")
    if tp < 1:
        raise ValueError("experiment tp must be >= 1")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("experiment lanes must be a non-empty list")
    if total_num_reqs < 0:
        raise ValueError("experiment total_num_reqs must be >= 0")
    result = {
        "name": name,
        "model": model,
        "tp": tp,
        "lanes": lanes,
        "total_num_reqs": total_num_reqs,
        "routing_policy": exp.get("routing_policy", exp.get("routing-policy")),
        "routing_kwargs": exp.get("routing_kwargs"),
        "client": dict(exp.get("client") or {}),
        "tuning": dict(exp.get("tuning") or {}),
        "workload": exp.get("workload"),
        "role": exp.get("role"),
        "repetition": exp.get("repetition"),
        "metadata": dict(exp.get("metadata") or {}),
    }
    if "serving" in exp:
        result["serving"] = normalize_serving_config(exp["serving"])
    return result


WORKLOAD_CLIENT_SKIP_KEYS = frozenset({"suffix", "name", "client", "total_num_reqs", "notes", "metadata", "serving"})


def _client_extra_args(client):
    args = []
    mapping = {
        "prompt_source": "--prompt-source",
        "azure_conv_csv": "--azure-conv-csv",
        "azure_code_csv": "--azure-code-csv",
        "azure_timestamp_col": "--azure-timestamp-col",
        "sharegpt52k": "--sharegpt52k",
        "arxiv_summarization": "--arxiv-summarization",
        "openai_humaneval": "--openai-humaneval",
        "active_pools": "--active-pools",
        "pool_limit": "--pool-limit",
        "max_tokens": "--max-tokens",
        "max_in_flight": "--max-in-flight",
        "request_timeout_seconds": "--request-timeout-seconds",
        "max_prompts": "--max-prompts",
        "azure_mix_schedule_csv": "--azure-mix-schedule-csv",
        "synthetic_prompt_tokens": "--synthetic-prompt-tokens",
        "min_tokens": "--min-tokens",
    }
    for key, flag in mapping.items():
        if key not in client or client[key] is None:
            continue
        value = client[key]
        args.extend([flag, str(value)])
    if client.get("stream_metrics") is False:
        args.append("--no-stream-metrics")
    if client.get("rebuild_azure_mix_schedule"):
        args.append("--rebuild-azure-mix-schedule")
    if client.get("ignore_eos"):
        args.append("--ignore-eos")
    return args


def _expand_lane_sweep(lane_spec, defaults):
    role = str(lane_spec.get("role") or lane_spec.get("preset_id") or "lane")
    preset_id = str(lane_spec["preset_id"])
    backend_url = str(lane_spec["backend_url"])
    lane_type = str(lane_spec.get("lane_type") or "generic")
    sweep = dict(lane_spec.get("sweep") or {})
    sweep_keys = ("max_num_batched_tokens", "max_num_seqs", "frequency_mhz")
    values = {}
    for key in sweep_keys:
        raw = sweep.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError("lane %s sweep.%s must be a non-empty list" % (preset_id, key))
        values[key] = [int(item) for item in raw]
    workloads = lane_spec.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("lane %s workloads must be a non-empty list" % preset_id)

    routing = dict(defaults.get("routing") or {})
    client_defaults = dict(defaults.get("client_defaults") or {})
    serving_present = "serving" in defaults or "serving" in lane_spec
    serving = merge_serving_configs(defaults.get("serving"), lane_spec.get("serving"))
    model = str(defaults["model"])
    tp = int(defaults["tp"])
    default_capabilities = dict(lane_spec.get("capabilities") or {})
    if lane_spec.get("kv_projection") is True:
        default_capabilities.setdefault("kv_projection", True)
    if not default_capabilities and defaults.get("capabilities"):
        default_capabilities = dict(defaults.get("capabilities") or {})
    if not default_capabilities:
        default_capabilities = {"kv_projection": True}

    repetitions = int(defaults.get("repetitions", 1))
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    expanded = []
    for mbt, max_seqs, freq in itertools.product(
        values["max_num_batched_tokens"],
        values["max_num_seqs"],
        values["frequency_mhz"],
    ):
        for workload in workloads:
            suffix = str(workload.get("suffix") or workload.get("name") or "run")
            name = "profile-%s-%s-mbt%d-ms%d-f%d" % (role, suffix, mbt, max_seqs, freq)
            client = dict(client_defaults)
            client.update(workload.get("client") or {})
            for key, value in workload.items():
                if key in WORKLOAD_CLIENT_SKIP_KEYS:
                    continue
                client[key] = value
            total_num_reqs = int(workload.get("total_num_reqs", lane_spec.get("total_num_reqs", 0)))
            lane_entry = {
                "preset_id": preset_id,
                "backend_url": backend_url,
                "max_num_batched_tokens": mbt,
                "max_num_seqs": max_seqs,
                "frequency_mhz": freq,
                "lane_type": lane_type,
            }
            if default_capabilities:
                lane_entry["capabilities"] = dict(default_capabilities)
            for repetition in range(1, repetitions + 1):
                metadata = dict(defaults.get("metadata") or {})
                metadata.update(workload.get("metadata") or {})
                metadata["repetition"] = repetition
                expanded.append(
                    _validate_flat_experiment(
                        {
                            "name": "%s-r%d" % (name, repetition),
                            "model": model,
                            "tp": tp,
                            "lanes": [lane_entry],
                            "total_num_reqs": total_num_reqs,
                            "routing_policy": routing.get("routing_policy"),
                            "routing_kwargs": routing.get("routing_kwargs"),
                            "client": client,
                            "tuning": {
                                "max_num_batched_tokens": mbt,
                                "max_num_seqs": max_seqs,
                                "frequency_mhz": freq,
                            },
                            "workload": suffix,
                            "role": role,
                            "repetition": repetition,
                            "metadata": metadata,
                            **({"serving": merge_serving_configs(serving, workload.get("serving"))}
                               if serving_present or "serving" in workload else {}),
                        },
                    )
                )
    return expanded


def expand_sweep_config(config):
    if not isinstance(config, dict):
        raise TypeError("sweep config must be an object")
    if config.get("format") != SWEEP_FORMAT:
        raise ValueError("unsupported sweep format: %s" % config.get("format"))
    model = str(config.get("model") or "").strip()
    tp = int(config.get("tp", 0))
    lanes = config.get("lanes")
    if not model:
        raise ValueError("sweep config model must be non-empty")
    if tp < 1:
        raise ValueError("sweep config tp must be >= 1")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError("sweep config lanes must be a non-empty list")
    defaults = {
        "model": model,
        "tp": tp,
        "client_defaults": dict(config.get("client_defaults") or {}),
        "routing": dict(config.get("routing") or {}),
        "repetitions": int(config.get("repetitions", 1)),
        "metadata": dict(config.get("metadata") or {}),
    }
    if "serving" in config:
        defaults["serving"] = normalize_serving_config(config["serving"])
    expanded = []
    for lane_spec in lanes:
        expanded.extend(_expand_lane_sweep(lane_spec, defaults))
    return expanded


def load_experiments(path):
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [_validate_flat_experiment(item) for item in data]
    if isinstance(data, dict) and data.get("format") == SWEEP_FORMAT:
        return expand_sweep_config(data)
    raise TypeError(
        "experiments config must be a JSON array or a %s object" % SWEEP_FORMAT
    )


__all__ = [
    "SWEEP_FORMAT",
    "load_experiments",
    "expand_sweep_config",
    "_client_extra_args",
]
