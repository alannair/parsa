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

import asyncio
import hashlib
import logging
import json
import math
import re
import time
import uuid
from bisect import bisect_right
from collections import Counter, deque
from copy import deepcopy

from .profile_table import COEFF_KEYS


DEFAULT_DRIFT_CONFIG = {
    "mode": "off", "catalog_path": None, "runtime_context": None,
    "window_s": 120.0, "evaluate_every_s": 30.0, "min_arrivals": 100,
    "max_observations": 10000, "persist_windows": 3,
    "min_residence_s": 600.0, "cooldown_s": 600.0,
    "benefit_horizon_s": 900.0, "min_saving_fraction": 0.05, "margin_j": 0.0,
}
_POSITIVE_TIMES = ("window_s", "evaluate_every_s", "min_residence_s", "cooldown_s", "benefit_horizon_s")
_POSITIVE_COUNTS = ("min_arrivals", "max_observations", "persist_windows")
_COMPAT_KEYS = ("model", "model_revision", "versions", "tp", "precision", "max_num_batched_tokens", "device_cohort", "serving_condition", "serving_profile_hash", "predictor_id", "routing_policy", "slo_definition")


def load_drift_config(routing_kwargs):
    if routing_kwargs is None:
        routing_kwargs = {}
    if not isinstance(routing_kwargs, dict):
        raise ValueError("routing_kwargs must be an object")
    supplied = routing_kwargs.get("op_drift")
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("op_drift must be an object")
    unknown = set(supplied) - set(DEFAULT_DRIFT_CONFIG)
    if unknown:
        raise ValueError("unknown op_drift keys: %s" % sorted(unknown))
    config = dict(DEFAULT_DRIFT_CONFIG)
    config.update(supplied)
    if config["mode"] not in ("off", "shadow", "live"):
        raise ValueError("op_drift mode must be off, shadow, or live")
    if config["mode"] == "live":
        raise ValueError("live op_drift unavailable: calibrated supervision/drain stages are not implemented")
    for key in _POSITIVE_TIMES:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("op_drift %s must be a finite positive number" % key)
        config[key] = float(value)
    for key in _POSITIVE_COUNTS:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("op_drift %s must be a positive integer" % key)
    fraction = config["min_saving_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("op_drift min_saving_fraction must be finite and in [0, 1)")
    margin = config["margin_j"]
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("op_drift margin_j must be finite and nonnegative")
    if config["mode"] == "shadow":
        if not isinstance(config["catalog_path"], str) or not config["catalog_path"].strip():
            raise ValueError("shadow op_drift requires a nonempty catalog_path")
        if not isinstance(config["runtime_context"], dict):
            raise ValueError("shadow op_drift requires a runtime_context object")
    return config


def _finite_number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("%s must be finite numeric" % label)
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError("%s is outside its valid range" % label)
    return float(value)


def _integer(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise ValueError("%s must be an%s integer" % (label, " positive" if positive else ""))
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("invalid %s" % label)
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _no_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: %s" % key)
        result[key] = value
    return result


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("catalog contains a nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _validate_context_value(value, label):
    if isinstance(value, bool) or value is None:
        raise ValueError("%s contains a boolean or null identity" % label)
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("%s contains an empty identity" % label)
    elif isinstance(value, (int, float)):
        _finite_number(value, label)
    elif isinstance(value, list):
        if not value:
            raise ValueError("%s contains an empty identity list" % label)
        for item in value:
            _validate_context_value(item, label)
    elif isinstance(value, dict):
        if not value:
            raise ValueError("%s must not be empty" % label)
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("%s contains an invalid key" % label)
            _validate_context_value(item, label)
    else:
        raise ValueError("%s contains a non-JSON identity" % label)


def _ordered_bounds(value, label, minimum=0.0):
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise ValueError("%s must have min and max" % label)
    low = _finite_number(value["min"], label + ".min", nonnegative=minimum == 0)
    high = _finite_number(value["max"], label + ".max", nonnegative=minimum == 0)
    if low < minimum or high < low:
        raise ValueError("invalid %s bounds" % label)


def load_catalog(path, runtime_context):
    """Load parsa-op-catalog-v1: operator-declared compatibility, finite OP profiles,
    length/SLO feature bins, lane banks with optional measured regimes, and optional
    directed transitions. Hashes identify canonical content; they are not attestations.
    """
    if not isinstance(runtime_context, dict):
        raise ValueError("runtime_context must be an object")
    _validate_context_value(runtime_context, "runtime_context")
    with open(path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle, object_pairs_hook=_no_duplicate_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON number: " + value)))
    if not isinstance(catalog, dict):
        raise ValueError("catalog must be an object")
    _finite_tree(catalog)
    required_top = {"schema_version", "compatibility", "feature_schema", "operating_points", "initial_bank", "banks", "transitions"}
    if set(catalog) != required_top or catalog.get("schema_version") != "parsa-op-catalog-v1":
        raise ValueError("invalid catalog schema")
    compatibility = catalog["compatibility"]
    if not isinstance(compatibility, dict) or set(compatibility) != set(_COMPAT_KEYS):
        raise ValueError("catalog compatibility does not exactly match runtime_context")
    _validate_context_value(compatibility, "compatibility")
    if _hash(compatibility) != _hash(runtime_context):
        raise ValueError("catalog compatibility does not exactly match runtime_context")
    for key in ("model", "model_revision", "precision", "device_cohort", "serving_condition", "serving_profile_hash", "predictor_id", "routing_policy"):
        if not isinstance(compatibility[key], str) or not compatibility[key].strip():
            raise ValueError("compatibility %s must be nonempty" % key)
    versions = compatibility["versions"]
    if not isinstance(versions, dict) or set(versions) != {"vllm", "vllm_ascend"} or any(not isinstance(v, str) or not v.strip() for v in versions.values()):
        raise ValueError("compatibility versions must identify vllm and vllm_ascend")
    _integer(compatibility["tp"], "compatibility tp", positive=True)
    _integer(compatibility["max_num_batched_tokens"], "compatibility max_num_batched_tokens", positive=True)
    if not isinstance(compatibility["slo_definition"], dict) or not compatibility["slo_definition"]:
        raise ValueError("compatibility slo_definition must be a nonempty identity object")
    features = catalog["feature_schema"]
    if not isinstance(features, dict) or set(features) != {"revision", "prompt_edges", "output_edges", "slo_classes", "slo_targets"} or features["revision"] != "length-bins-v1":
        raise ValueError("invalid feature_schema")
    for name in ("prompt_edges", "output_edges"):
        edges = features[name]
        if not isinstance(edges, list) or not edges:
            raise ValueError("%s must be a nonempty edge list" % name)
        values = [_finite_number(edge, name, positive=True) for edge in edges]
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("%s must be strictly increasing" % name)
    classes = features["slo_classes"]
    if not isinstance(classes, list) or not classes or any(not isinstance(name, str) for name in classes) or len(set(classes)) != len(classes):
        raise ValueError("slo_classes must be unique and nonempty")
    for name in classes:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid SLO class")
    targets = features["slo_targets"]
    if not isinstance(targets, dict) or set(targets) != set(classes):
        raise ValueError("slo_targets must exactly match slo_classes")
    for value in targets.values():
        if not 0 < _finite_number(value, "SLO target") <= 1:
            raise ValueError("SLO targets must be in (0, 1]")
    ops = catalog["operating_points"]
    if not isinstance(ops, dict) or not ops:
        raise ValueError("operating_points must be nonempty")
    for op_id, op in ops.items():
        _identifier(op_id, "operating-point ID")
        if not isinstance(op, dict) or set(op) != {"frequency_mhz", "max_num_seqs", "profile"}:
            raise ValueError("invalid operating point %s" % op_id)
        _finite_number(op["frequency_mhz"], "frequency_mhz", positive=True)
        _integer(op["max_num_seqs"], "max_num_seqs", positive=True)
        profile = op["profile"]
        if not isinstance(profile, dict) or set(profile) != set(COEFF_KEYS):
            raise ValueError("profile must contain exactly COEFF_KEYS")
        for value in profile.values():
            _finite_number(value, "profile coefficient")
    initial = catalog["initial_bank"]
    banks = catalog["banks"]
    if not isinstance(initial, dict) or not initial or not isinstance(banks, dict) or not banks:
        raise ValueError("initial bank and banks must be nonempty objects")
    for lane, op in initial.items():
        _identifier(lane, "lane ID")
        if op not in ops:
            raise ValueError("initial bank references unknown OP")
    initial_sets = set(initial)
    seen_lanes = set()
    for bank_id, bank in banks.items():
        _identifier(bank_id, "bank ID")
        if not isinstance(bank, dict) or set(bank) != {"lanes", "regimes"}:
            raise ValueError("invalid bank")
        lanes = bank["lanes"]
        if not isinstance(lanes, dict) or set(lanes) != initial_sets:
            raise ValueError("bank lane IDs must match initial bank")
        if any(op not in ops for op in lanes.values()):
            raise ValueError("bank references unknown OP")
        lane_identity = tuple(sorted(lanes.items()))
        if lane_identity in seen_lanes:
            raise ValueError("duplicate lane assignment map")
        seen_lanes.add(lane_identity)
        regimes = bank["regimes"]
        if not isinstance(regimes, list):
            raise ValueError("bank regimes must be a list")
        regime_ids = set()
        for regime in regimes:
            _validate_regime(regime, classes, features)
            if regime["regime_id"] in regime_ids:
                raise ValueError("duplicate regime_id within bank")
            regime_ids.add(regime["regime_id"])
    if sum(bank["lanes"] == initial for bank in banks.values()) != 1:
        raise ValueError("exactly one bank must match initial_bank")
    transitions = catalog["transitions"]
    if not isinstance(transitions, list):
        raise ValueError("transitions must be a list")
    # Task 1 accepts only transitions that are structurally consistent. Empty is
    # valid and intentionally conveys no selectable calibrated neighbor.
    for transition in transitions:
        _validate_transition(transition, banks)
    profile_hashes = {op_id: _hash(op["profile"]) for op_id, op in ops.items()}
    return {**catalog, "identity_source": "operator_declared", "profile_hashes": profile_hashes,
            "catalog_hash": _hash(catalog)}


def _validate_regime(regime, classes, features):
    required = {"regime_id", "arrival_rps", "class_mix", "power_w", "capacity_rps_lo", "slo_success_lo", "calibration_revision", "source_run_ids"}
    if not isinstance(regime, dict) or set(regime) != required:
        raise ValueError("invalid regime fields")
    _identifier(regime["regime_id"], "regime ID")
    _ordered_bounds(regime["arrival_rps"], "arrival_rps")
    power = regime["power_w"]
    if not isinstance(power, dict) or set(power) != {"lo", "hi"}:
        raise ValueError("power_w must have lo and hi")
    power_lo = _finite_number(power["lo"], "power_w.lo", nonnegative=True)
    power_hi = _finite_number(power["hi"], "power_w.hi", nonnegative=True)
    if power_hi < power_lo:
        raise ValueError("invalid power_w bounds")
    _finite_number(regime["capacity_rps_lo"], "capacity_rps_lo", nonnegative=True)
    mix = regime["class_mix"]
    if not isinstance(mix, dict):
        raise ValueError("class_mix must be an object")
    known = {"unknown"}
    for pi in range(len(features["prompt_edges"]) + 1):
        for oi in range(len(features["output_edges"]) + 1):
            known.update("p%d:o%d:%s" % (pi, oi, name) for name in classes)
    if not set(mix) <= known:
        raise ValueError("unknown class_mix feature ID")
    for bounds in mix.values():
        _ordered_bounds(bounds, "class_mix", minimum=0)
        if bounds["max"] > 1:
            raise ValueError("class_mix exceeds one")
    if sum(bounds["min"] for bounds in mix.values()) > 1 + 1e-12 or sum(bounds["max"] for bounds in mix.values()) < 1 - 1e-12:
        raise ValueError("class_mix bounds cannot sum to one")
    success = regime["slo_success_lo"]
    if not isinstance(success, dict) or set(success) != set(classes):
        raise ValueError("slo_success_lo must exactly match SLO classes")
    for value in success.values():
        if not 0 <= _finite_number(value, "SLO success") <= 1:
            raise ValueError("SLO success must be in [0,1]")
    if not isinstance(regime["calibration_revision"], str) or not regime["calibration_revision"].strip() or not isinstance(regime["source_run_ids"], list) or not regime["source_run_ids"] or any(not isinstance(x, str) or not x.strip() for x in regime["source_run_ids"]):
        raise ValueError("regime provenance must be nonempty")


def _validate_transition(item, banks):
    required = {"source_bank", "target_bank", "lane_id", "duration_hi_s", "extra_energy_hi_j", "remaining_capacity_rps_lo", "max_backlog_requests", "calibration_revision", "source_run_ids"}
    if not isinstance(item, dict) or set(item) != required:
        raise ValueError("invalid transition fields")
    source, target = item["source_bank"], item["target_bank"]
    if source not in banks or target not in banks or source == target:
        raise ValueError("invalid transition bank references")
    a, b = banks[source]["lanes"], banks[target]["lanes"]
    changed = [lane for lane in a if a[lane] != b[lane]]
    if changed != [item["lane_id"]] and not (len(changed) == 1 and changed[0] == item["lane_id"]):
        raise ValueError("transition must change exactly its named lane")
    _finite_number(item["duration_hi_s"], "duration_hi_s", positive=True)
    _finite_number(item["extra_energy_hi_j"], "extra_energy_hi_j", nonnegative=True)
    _finite_number(item["remaining_capacity_rps_lo"], "remaining_capacity_rps_lo", positive=True)
    _integer(item["max_backlog_requests"], "max_backlog_requests")
    if item["max_backlog_requests"] < 0 or not isinstance(item["calibration_revision"], str) or not item["calibration_revision"].strip() or not isinstance(item["source_run_ids"], list) or not item["source_run_ids"] or any(not isinstance(x, str) or not x.strip() for x in item["source_run_ids"]):
        raise ValueError("invalid transition provenance/backlog")


def _nonnegative(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


class _ObservationContext:
    """Caller-owned, opaque request token; never retained by DemandWindow."""

    __slots__ = ("owner", "request_id", "bucket_generation", "arrival_at_s",
                 "enriched", "completed", "slo_class")

    def __init__(self, window, request_id, generation, at_s, slo_class):
        self.owner = window.window_id
        self.request_id = request_id
        self.bucket_generation = generation
        self.arrival_at_s = at_s
        self.enriched = False
        self.completed = False
        self.slo_class = slo_class


class DemandWindow:
    """Single-owner bounded offered-demand counters, with no request registry.

    N full buckets plus the open bucket cover [exposure_start_s, at_s].
    After warmup exposure is window_s .. window_s + bucket_width_s, not an
    exact sliding window. max_observations caps numeric diagnostics only.
    Explicit timestamps and clock() must share one monotonic domain.
    """

    __slots__ = ("config", "feature_schema", "feature_schema_id", "clock", "start_s",
                 "last_s", "window_id", "full_bucket_count", "bucket_width_s",
                 "buckets", "diagnostics")

    def __init__(self, config, feature_schema, clock=time.monotonic):
        self.config = load_drift_config({"op_drift": config})
        # feature_schema is the validated Task 1 producer's schema.
        self.feature_schema = deepcopy(feature_schema)
        self.feature_schema_id = _hash(self.feature_schema)
        self.clock = clock
        self.start_s = _finite_number(clock(), "clock")
        self.last_s = self.start_s
        self.window_id = uuid.uuid4().hex
        ratio = self.config["window_s"] / self.config["evaluate_every_s"]
        self.full_bucket_count = max(1, math.ceil(min(1023, ratio)))
        self.bucket_width_s = self.config["window_s"] / self.full_bucket_count
        self.buckets = [None] * (self.full_bucket_count + 1)
        self.diagnostics = deque(maxlen=self.config["max_observations"])

    def _time(self, at_s):
        at_s = _finite_number(self.clock() if at_s is None else at_s, "clock")
        if at_s < self.last_s or not math.isfinite((at_s - self.start_s) / self.bucket_width_s):
            raise ValueError("clock must be monotonic and within finite bucket range")
        self.last_s = at_s
        return at_s

    def _generation(self, at_s):
        return math.floor((at_s - self.start_s) / self.bucket_width_s)

    def _bucket(self, generation):
        index = generation % len(self.buckets)
        bucket = self.buckets[index]
        if bucket is None or bucket["generation"] != generation:
            bucket = {"generation": generation, "classes": Counter(), "slos": Counter(),
                      "outcomes": Counter(), "slo_outcomes": Counter()}
            self.buckets[index] = bucket
        return bucket

    def _context(self, context, request_id):
        if (not isinstance(context, _ObservationContext) or context.owner != self.window_id
                or context.request_id != request_id):
            raise ValueError("original request/window observation context is required")

    def _slo(self, slo_class):
        if isinstance(slo_class, str) and slo_class in self.feature_schema["slo_classes"]:
            return slo_class
        return None

    def _enrich(self, context, at_s, request_features, slo_class):
        if request_features is None or context.enriched or context.completed:
            return
        context.enriched = True
        generation = context.bucket_generation
        if generation < self._generation(at_s) - self.full_bucket_count:
            return
        bucket = self.buckets[generation % len(self.buckets)]
        if bucket is None or bucket["generation"] != generation:
            return
        slo = self._slo(slo_class)
        bucket["slos"][context.slo_class] -= 1
        bucket["slos"][slo] += 1
        context.slo_class = slo
        if isinstance(request_features, dict):
            l_in, l_out = request_features.get("l_in"), request_features.get("l_out_raw")
        else:
            l_in = getattr(request_features, "l_in", None)
            l_out = getattr(request_features, "l_out_raw", None)
        if slo is None or not _nonnegative(l_in) or not _nonnegative(l_out):
            return
        pi = bisect_right(self.feature_schema["prompt_edges"], l_in)
        oi = bisect_right(self.feature_schema["output_edges"], l_out)
        bucket["classes"]["unknown"] -= 1
        bucket["classes"]["p%d:o%d:%s" % (pi, oi, slo)] += 1
        self.diagnostics.append((generation, float(l_in), float(l_out)))


def observe_arrival(window, request_id, at_s, request_features=None, slo_class="default", context=None):
    """Count before admission; pass the SAME returned context on later replans.

    Without a context this is a new offered arrival, even for a reused ID.
    The first non-None classification finalizes enrichment, valid or unknown.
    Only l_in/l_out_raw are read; text and future output lengths are ignored.
    """
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a nonempty string")
    if context is not None:
        window._context(context, request_id)
    at_s = window._time(at_s)
    if context is None:
        generation = window._generation(at_s)
        slo = window._slo(slo_class)
        context = _ObservationContext(window, request_id, generation, at_s, slo)
        bucket = window._bucket(generation)
        bucket["classes"]["unknown"] += 1
        bucket["slos"][slo] += 1
    window._enrich(context, at_s, request_features, slo_class)
    return context


def observe_completion(window, request_id, at_s, outcome, context=None):
    """Count one terminal outcome at completion time, never rewrite arrivals.

    outcome: status string or {status, slo_met: bool | None}. Non-completed
    statuses cannot be SLO successes. Missing SLO knowledge stays unknown.
    A context is mandatory: there is deliberately no ID-based lookup.
    """
    window._context(context, request_id)
    status = outcome.get("status") if isinstance(outcome, dict) else outcome
    if status not in ("completed", "failed", "rejected", "cancelled"):
        raise ValueError("outcome must name a terminal status")
    at_s = window._time(at_s)
    if context.completed:
        return False
    bucket = window._bucket(window._generation(at_s))
    bucket["outcomes"][status] += 1
    slo_met = outcome.get("slo_met") if isinstance(outcome, dict) else None
    if isinstance(slo_met, bool):
        bucket["slo_outcomes"]["known"] += 1
        key = "success" if status == "completed" and slo_met else "failure"
        bucket["slo_outcomes"][key] += 1
    else:
        bucket["slo_outcomes"]["unknown"] += 1
    context.completed = True
    return True


def window_snapshot(window, now_s):
    """Detached numeric snapshot; headroom is supplied externally, never invented.

    demand_bound_rps is the peak bucket rate (open bucket uses elapsed time).
    None means a nonempty zero-exposure bucket has no finite rate bound.
    Future persistence of this empirical bound is an explicit selector assumption.
    """
    now_s = window._time(now_s)
    generation = window._generation(now_s)
    oldest = max(0, generation - window.full_bucket_count)
    start = window.start_s + oldest * window.bucket_width_s
    exposure = now_s - start
    classes, slos, outcomes, slo_outcomes = Counter(), Counter(), Counter(), Counter()
    peak = 0.0
    for bucket in window.buckets:
        if bucket is None or not oldest <= bucket["generation"] <= generation:
            continue
        classes.update(bucket["classes"])
        slos.update(bucket["slos"])
        outcomes.update(bucket["outcomes"])
        slo_outcomes.update(bucket["slo_outcomes"])
        count = sum(bucket["classes"].values())
        duration = min(window.bucket_width_s, now_s - (window.start_s + bucket["generation"] * window.bucket_width_s))
        if count and duration <= 0:
            peak = None
        elif count and peak is not None:
            peak = max(peak, count / duration)
    count = sum(classes.values())
    snapshot = {
        "window_id": window.window_id,
        "evaluation_index": math.floor((now_s - window.start_s) / window.config["evaluate_every_s"]),
        "feature_revision": window.feature_schema["revision"], "feature_schema_id": window.feature_schema_id,
        "at_s": now_s, "exposure_start_s": start, "exposure_s": exposure,
        "bucket_width_s": window.bucket_width_s,
        "ready": exposure >= window.config["window_s"] and count >= window.config["min_arrivals"],
        "arrival_count": count, "arrival_rps": count / exposure if exposure > 0 else 0.0,
        "class_mix": {key: value / count for key, value in classes.items() if value > 0},
        "slo_counts": {key: value for key, value in slos.items() if key is not None and value > 0},
        "slo_unknown_count": slos[None],
        "unknown_count": classes["unknown"], "demand_bound_rps": peak,
        "outcome_counts": {key: outcomes[key] for key in ("completed", "failed", "rejected", "cancelled")},
        "slo_outcome_counts": {key: slo_outcomes[key] for key in ("known", "unknown", "success", "failure")},
        "diagnostic_count": sum(oldest <= item[0] <= generation for item in window.diagnostics),
    }
    snapshot["observation_id"] = _hash(snapshot)
    return snapshot


def _snapshot_identity(observation):
    """Headroom is external and cannot change the offered-work identity."""
    if not isinstance(observation, dict) or not observation.get("observation_id"):
        return False
    try:
        return observation["observation_id"] == _hash({
            key: value for key, value in observation.items() if key not in ("observation_id", "headroom")
        })
    except (ValueError, TypeError):
        return False


def _coverage_id(catalog_hash, feature_schema_id, observation_id):
    return _hash([catalog_hash, feature_schema_id, observation_id])


def estimate_bank(catalog, bank_id, observation):
    """Conservative intersection of every applicable measured regime; no refits.

    Takes a validated catalog and an unmodified DemandWindow snapshot (optionally
    with external headroom). Returns None for unsupported work/feature coverage.
    A supported estimate can still be infeasible for capacity or active SLOs.
    """
    features = catalog["feature_schema"]
    bank = catalog["banks"].get(bank_id)
    if bank is None or not bank["regimes"] or not _snapshot_identity(observation):
        return None
    if (observation.get("feature_revision") != features["revision"]
            or observation.get("feature_schema_id") != _hash(features)):
        return None
    count = observation.get("arrival_count")
    exposure, rate = observation.get("exposure_s"), observation.get("arrival_rps")
    if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or not _nonnegative(exposure) or exposure == 0 or not _nonnegative(rate)
            or not math.isclose(rate, count / exposure, rel_tol=1e-12)):
        return None
    known = {"unknown"}
    for pi in range(len(features["prompt_edges"]) + 1):
        for oi in range(len(features["output_edges"]) + 1):
            known.update("p%d:o%d:%s" % (pi, oi, name) for name in features["slo_classes"])
    mix, slos = observation.get("class_mix"), observation.get("slo_counts")
    unknown_slo = observation.get("slo_unknown_count")
    if (not isinstance(mix, dict) or not set(mix) <= known
            or any(not _nonnegative(value) or value > 1 for value in mix.values())
            or not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-12)
            or not isinstance(slos, dict) or not set(slos) <= set(features["slo_classes"])
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in slos.values())
            or isinstance(unknown_slo, bool) or not isinstance(unknown_slo, int) or unknown_slo < 0
            or sum(slos.values()) + unknown_slo != count
            or not math.isclose(mix.get("unknown", 0) * count, observation.get("unknown_count", -1), abs_tol=1e-9)):
        return None
    matches = []
    for regime in bank["regimes"]:
        if not regime["arrival_rps"]["min"] <= rate <= regime["arrival_rps"]["max"]:
            continue
        bounds = regime["class_mix"]
        if any(mass > 0 and (name not in bounds or bounds[name]["max"] == 0) for name, mass in mix.items()):
            continue
        if any(not limits["min"] <= mix.get(name, 0) <= limits["max"] for name, limits in bounds.items()):
            continue
        matches.append(regime)
    if not matches:
        return None
    active = sorted(features["slo_classes"] if unknown_slo else
                    (name for name in features["slo_classes"] if slos.get(name, 0)))
    power = {"lo": min(row["power_w"]["lo"] for row in matches),
             "hi": max(row["power_w"]["hi"] for row in matches)}
    capacity = min(row["capacity_rps_lo"] for row in matches)
    success = {name: min(row["slo_success_lo"][name] for row in matches) for name in features["slo_classes"]}
    reason = "capacity" if rate > capacity else "slo" if any(success[name] < features["slo_targets"][name] for name in active) else None
    return {
        "bank_id": bank_id, "lanes": dict(bank["lanes"]),
        "observation_id": observation["observation_id"],
        "coverage_id": _coverage_id(catalog["catalog_hash"], observation["feature_schema_id"], observation["observation_id"]),
        "catalog_hash": catalog["catalog_hash"], "feature_schema_id": observation["feature_schema_id"],
        "identity_source": catalog["identity_source"],
        "power_w": power, "capacity_rps_lo": capacity, "slo_success_lo": success,
        "slo_targets": dict(features["slo_targets"]), "active_slo_classes": active,
        "steady_feasible": reason is None, "feasibility_reason": reason,
        "regime_ids": sorted(row["regime_id"] for row in matches),
        "calibration_revisions": sorted({row["calibration_revision"] for row in matches}),
        "source_run_ids": sorted({source for row in matches for source in row["source_run_ids"]}),
    }


def make_policy_state(config, catalog, current_bank, generations, *, clock=time.monotonic, now_s=None):
    """Construct Task 3's single-owner shadow state. See task-2-report.md.

    Residence starts conservatively on entry, without inventing a commit.
    choose_retune returns an updated state; it never mutates this input.
    """
    config = load_drift_config({"op_drift": config})
    if current_bank not in catalog["banks"]:
        raise ValueError("current_bank must be an exact catalog bank ID")
    lanes = catalog["banks"][current_bank]["lanes"]
    if not isinstance(generations, dict) or set(generations) != set(lanes):
        raise ValueError("generations must exactly cover current lanes")
    for value in generations.values():
        if _integer(value, "generation") < 0:
            raise ValueError("generations must be nonnegative")
    now_s = _finite_number(clock() if now_s is None else now_s, "clock")
    return {
        "config": config, "clock": clock, "current_bank": current_bank,
        "bank_lanes": dict(lanes), "generations": dict(generations),
        "catalog_hash": catalog["catalog_hash"], "feature_schema_id": _hash(catalog["feature_schema"]),
        "observation": None, "entered_bank_s": now_s,
        "last_commit_s": None, "last_attempt_s": None, "last_proposal_s": None,
        "last_evaluated_s": now_s, "active_transition": False,
        "pending_candidate": None, "winner_windows": 0,
        "last_window_id": None, "last_window_index": None,
    }


def choose_retune(current_bank, estimates, transitions, policy_state, now_s):
    """Pure local shadow assessment. Caller retains decision['policy_state'].

    Conditional assumption: the observed peak bucket demand persists as an upper
    demand bound throughout the calibrated transition. This is not a live safety
    certificate. Proposals update only persistence/evaluation/proposal metadata.
    """
    state = policy_state
    now_s = _finite_number(state["clock"]() if now_s is None else now_s, "clock")
    for key in ("entered_bank_s", "last_evaluated_s", "last_commit_s", "last_attempt_s", "last_proposal_s"):
        value = state[key]
        if value is not None and _finite_number(value, key) > now_s:
            raise ValueError("policy timestamps must share a monotonic clock")
    next_state = {**state, "config": dict(state["config"]), "generations": dict(state["generations"]),
                  "bank_lanes": dict(state["bank_lanes"]), "observation": deepcopy(state["observation"]),
                  "last_evaluated_s": now_s}
    config, observation = state["config"], state["observation"]
    result = {
        "action": "NO_CHANGE", "reason": None, "current_bank": current_bank, "target_bank": current_bank,
        "lane_id": None, "expected_generation": None,
        "observation_id": observation.get("observation_id") if isinstance(observation, dict) else None,
        "coverage_id": None, "identity_source": "operator_declared", "shadow": True,
        "benefit_lo_j": None, "extra_energy_hi_j": None, "required_j": None,
        "required_saving_j": None, "useful_horizon_s": None, "duration_hi_s": None,
        "expected_backlog_requests": None, "transition_source_run_ids": [],
        "transition_calibration_revisions": [], "policy_state": next_state,
    }

    def finish(reason, details=None, *, reset=True):
        if details:
            result.update(details)
        result["reason"] = reason
        if reset:
            next_state.update(pending_candidate=None, winner_windows=0)
        return result

    if current_bank != state["current_bank"]:
        return finish("current_bank_mismatch")
    if state["active_transition"]:
        return finish("transition_active")
    if observation is None:
        return finish("no_observation")
    if not _snapshot_identity(observation) or observation.get("feature_schema_id") != state["feature_schema_id"]:
        return finish("coverage_mismatch")
    if observation["at_s"] > now_s or now_s - observation["at_s"] > config["evaluate_every_s"]:
        return finish("stale_observation")
    window_id, index = observation["window_id"], observation["evaluation_index"]
    previous_index = state["last_window_index"]
    same_collector = state["last_window_id"] == window_id
    new_window = not same_collector or previous_index is None or index > previous_index
    consecutive = same_collector and previous_index is not None and index == previous_index + 1
    if new_window:
        next_state.update(last_window_id=window_id, last_window_index=index)
    if observation["exposure_s"] < config["window_s"]:
        return finish("cold_start")
    if observation["arrival_count"] < config["min_arrivals"]:
        return finish("low_samples")
    if not observation["ready"]:
        return finish("cold_start")
    baseline = estimates.get(current_bank)
    if baseline is None:
        return finish("no_calibration")
    coverage = _coverage_id(state["catalog_hash"], state["feature_schema_id"], observation["observation_id"])
    result["coverage_id"] = coverage

    def comparable(bank_id, estimate):
        return (estimate["bank_id"] == bank_id and estimate["observation_id"] == observation["observation_id"]
                and estimate["coverage_id"] == coverage and estimate["catalog_hash"] == state["catalog_hash"]
                and estimate["feature_schema_id"] == state["feature_schema_id"]
                and estimate["identity_source"] == baseline["identity_source"]
                and estimate["slo_targets"] == baseline["slo_targets"]
                and estimate["active_slo_classes"] == baseline["active_slo_classes"])

    if not comparable(current_bank, baseline) or baseline["lanes"] != state["bank_lanes"]:
        return finish("coverage_mismatch")
    if not baseline["steady_feasible"]:
        return finish("baseline_" + baseline["feasibility_reason"])
    headroom = observation.get("headroom")
    if headroom is None:
        return finish("headroom_missing")
    if (not isinstance(headroom, dict) or not isinstance(headroom.get("fresh"), bool)
            or not _nonnegative(headroom.get("at_s"))
            or not _nonnegative(headroom.get("backlog_requests"))
            or not isinstance(headroom.get("healthy_lane_ids"), (list, tuple))
            or any(not isinstance(lane, str) for lane in headroom["healthy_lane_ids"])
            or len(set(headroom["healthy_lane_ids"])) != len(headroom["healthy_lane_ids"])
            or not set(headroom["healthy_lane_ids"]) <= set(baseline["lanes"])):
        return finish("headroom_invalid")
    if not headroom["fresh"] or not 0 <= now_s - headroom["at_s"] <= config["evaluate_every_s"]:
        return finish("stale_headroom")
    demand = observation.get("demand_bound_rps")
    if not _nonnegative(demand) or demand < observation["arrival_rps"]:
        return finish("demand_bound_unavailable")
    residence_start = max(state["entered_bank_s"], state["last_commit_s"] if state["last_commit_s"] is not None else state["entered_bank_s"])
    if now_s - residence_start < config["min_residence_s"]:
        return finish("min_residence")
    if any(state[key] is not None and now_s - state[key] < config["cooldown_s"] for key in ("last_attempt_s", "last_commit_s")):
        return finish("cooldown")
    if state["last_proposal_s"] is not None and now_s - state["last_proposal_s"] < config["cooldown_s"]:
        return finish("proposal_cooldown")

    winner, best_net = None, -math.inf
    failure, failure_details, furthest = "no_calibration", None, 0
    for bank_id in sorted(estimates):
        candidate = estimates[bank_id]
        if bank_id == current_bank or candidate is None:
            continue
        details = {"target_bank": bank_id}
        if not comparable(bank_id, candidate):
            reason, stage = "coverage_mismatch", 1
        else:
            lanes = baseline["lanes"]
            changed = [lane for lane in lanes if candidate["lanes"].get(lane) != lanes[lane]]
            if set(candidate["lanes"]) != set(lanes) or len(changed) != 1:
                reason, stage = "no_neighbor", 2
            else:
                lane = changed[0]
                details.update(lane_id=lane, expected_generation=state["generations"][lane])
                records = [edge for edge in transitions if edge["source_bank"] == current_bank
                           and edge["target_bank"] == bank_id and edge["lane_id"] == lane]
                if not records:
                    reason, stage = "missing_transition", 3
                elif not candidate["steady_feasible"]:
                    reason, stage = "candidate_" + candidate["feasibility_reason"], 4
                elif len(lanes) < 2 or not set(lanes) <= set(headroom["healthy_lane_ids"]):
                    reason, stage = "unhealthy_residual_lanes", 5
                else:
                    duration = max(edge["duration_hi_s"] for edge in records)
                    energy = max(edge["extra_energy_hi_j"] for edge in records)
                    remaining = min(edge["remaining_capacity_rps_lo"] for edge in records)
                    backlog_limit = min(edge["max_backlog_requests"] for edge in records)
                    backlog = headroom["backlog_requests"] + max(0, demand - remaining) * duration
                    horizon = config["benefit_horizon_s"]
                    useful = max(0.0, horizon - duration)
                    benefit = (baseline["power_w"]["lo"] - candidate["power_w"]["hi"]) * useful
                    required = energy + config["margin_j"]
                    saving = config["min_saving_fraction"] * baseline["power_w"]["lo"] * horizon
                    details.update(duration_hi_s=duration, extra_energy_hi_j=energy, benefit_lo_j=benefit,
                                   useful_horizon_s=useful, required_j=required, required_saving_j=saving,
                                   expected_backlog_requests=backlog,
                                   transition_source_run_ids=sorted({source for edge in records for source in edge["source_run_ids"]}),
                                   transition_calibration_revisions=sorted({edge["calibration_revision"] for edge in records}))
                    if remaining < observation["arrival_rps"]:
                        reason, stage = "residual_capacity", 6
                    elif not math.isfinite(backlog) or backlog > backlog_limit:
                        reason, stage = "transition_backlog", 7
                    elif (not all(math.isfinite(value) for value in (benefit, required, saving))
                          or not benefit > required or not benefit > saving):
                        reason, stage = "insufficient_savings", 8
                    else:
                        # Maximize a conservative net energy bound, then bank ID.
                        net = benefit - energy
                        if net > best_net:
                            winner, best_net = details, net
                        continue
        if stage > furthest:
            failure, failure_details, furthest = reason, details, stage
    if winner is None:
        return finish(failure, failure_details)
    if not new_window:
        return finish("duplicate_window", winner, reset=state["pending_candidate"] != winner["target_bank"])
    count = state["winner_windows"] + 1 if consecutive and state["pending_candidate"] == winner["target_bank"] else 1
    next_state.update(pending_candidate=winner["target_bank"], winner_windows=min(count, config["persist_windows"]))
    if count < config["persist_windows"]:
        return finish("persistence", winner, reset=False)
    next_state["last_proposal_s"] = now_s
    result["action"] = "RETUNE"
    return finish("shadow_recommendation", winner, reset=False)


class ShadowObserver:
    """Single-event-loop owner of observations and conditional recommendations.

    No runtime profile, lane lifecycle, process, or hardware capabilities are held.
    Request contexts belong to callers, not to an observer-side request registry.
    """

    __slots__ = ("config", "catalog", "window", "policy_state", "clock", "metrics", "config_hash")

    def __init__(self, config, presets, routing_policy, metrics, *, clock=time.monotonic):
        self.config = load_drift_config({"op_drift": config})
        if self.config["mode"] != "shadow":
            raise ValueError("ShadowObserver requires shadow mode")
        self.catalog = load_catalog(self.config["catalog_path"], self.config["runtime_context"])
        self.validate_runtime(presets, routing_policy)
        self.clock, self.metrics = clock, metrics
        self.config_hash = _hash(self.config)
        self.window = DemandWindow(self.config, self.catalog["feature_schema"], clock=clock)
        initial = self.catalog["initial_bank"]
        bank = next(name for name, value in self.catalog["banks"].items() if value["lanes"] == initial)
        # Generation zero is a static shadow label, never an attested backend epoch.
        self.policy_state = make_policy_state(self.config, self.catalog, bank,
                                              {lane: 0 for lane in initial}, clock=clock)

    def validate_runtime(self, presets, routing_policy):
        initial = self.catalog["initial_bank"]
        actual = {preset.preset_id: preset for preset in presets}
        if len(actual) != len(presets) or set(actual) != set(initial):
            raise ValueError("initial_bank lane IDs must exactly match the registry")
        if self.catalog["compatibility"]["routing_policy"] != routing_policy:
            raise ValueError("op_drift routing_policy does not match the controller")
        for lane, op_id in initial.items():
            op, preset = self.catalog["operating_points"][op_id], actual[lane]
            for key, expected in (("frequency_mhz", op["frequency_mhz"]),
                                  ("max_num_seqs", op["max_num_seqs"]),
                                  ("max_num_batched_tokens", self.catalog["compatibility"]["max_num_batched_tokens"])):
                value = getattr(preset, key, None)
                if not _nonnegative(value) or value != expected:
                    raise ValueError("initial_bank %s %s does not match the preset" % (lane, key))

    def _event(self, decision, observation=None, estimates=None, **diagnostic):
        self.metrics.record_shadow_event({
            "identity_source": "operator_declared", "runtime_context": self.catalog["compatibility"],
            "catalog_hash": self.catalog["catalog_hash"], "config_hash": self.config_hash,
            "observation": observation, "estimates": estimates,
            "decision": {key: value for key, value in decision.items() if key != "policy_state"},
            **diagnostic,
        })

    def error(self, stage, exc):
        self.policy_state.update(pending_candidate=None, winner_windows=0,
                                 last_window_id=None, last_window_index=None, observation=None)
        decision = {"action": "NO_CHANGE", "reason": "shadow_error",
                    "current_bank": self.policy_state["current_bank"],
                    "target_bank": self.policy_state["current_bank"]}
        try:
            # Exception messages may contain request data. Record only the type.
            self._event(decision, stage=stage, error_type=type(exc).__name__)
        except Exception:
            # Recording must never replace a routing error or a valid response.
            try:
                logging.getLogger("src.router_app").warning("op_drift shadow_error: diagnostic_recording_failed")
            except Exception:
                pass
        return decision

    def begin(self, request_id):
        try:
            return observe_arrival(self.window, str(request_id or uuid.uuid4().hex), None, slo_class=None)
        except Exception as exc:
            self.error("arrival", exc)
            return None

    def enrich(self, context, features, options):
        if context is None:
            return
        try:
            observe_arrival(self.window, context.request_id, None, features,
                            options.get("slo_class"), context=context)
        except Exception as exc:
            self.error("enrichment", exc)

    def finish(self, context, status):
        if context is None:
            return
        try:
            return observe_completion(self.window, context.request_id, None,
                                      {"status": status, "slo_met": None}, context=context)
        except Exception as exc:
            self.error("completion", exc)

    def snapshot(self):
        return window_snapshot(self.window, None)

    def tick(self, headroom=None):
        try:
            observation = self.snapshot()
            if headroom is not None:
                observation["headroom"] = deepcopy(headroom)
            self.policy_state["observation"] = observation
            estimates = {bank: estimate_bank(self.catalog, bank, observation) for bank in self.catalog["banks"]}
            result = choose_retune(self.policy_state["current_bank"], estimates,
                                   self.catalog["transitions"], self.policy_state, self.clock())
            self.policy_state = result["policy_state"]
            self._event(result, observation, estimates)
            return {key: value for key, value in result.items() if key != "policy_state"}
        except Exception as exc:
            return self.error("tick", exc)

    async def run(self, headroom_producer, *, sleep=asyncio.sleep):
        while True:
            try:
                headroom = await headroom_producer()
                self.tick(headroom)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error("collection", exc)
            await sleep(self.config["evaluate_every_s"])
