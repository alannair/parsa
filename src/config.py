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

DEFAULT_LANE_CLASS_PROFILES = {
    "prefill": {
        "prefill_affinity": 1.35,
        "decode_affinity": 0.15,
        "batch_affinity": 0.10,
        "concurrency_affinity": 0.05,
        "latency_weight": 0.20,
        "energy_weight": 0.04,
        "queue_weight": 1.00,
        "power_weight": 0.20,
        "rescue_penalty": 0.10,
    },
    "decode": {
        "prefill_affinity": 0.15,
        "decode_affinity": 1.35,
        "batch_affinity": 0.35,
        "concurrency_affinity": 0.30,
        "latency_weight": 0.20,
        "energy_weight": 0.04,
        "queue_weight": 1.00,
        "power_weight": 0.20,
        "rescue_penalty": 0.10,
    },
    "generic": {
        "prefill_affinity": 0.80,
        "decode_affinity": 0.80,
        "batch_affinity": 0.20,
        "concurrency_affinity": 0.20,
        "latency_weight": 0.16,
        "energy_weight": 0.06,
        "queue_weight": 1.00,
        "power_weight": 0.20,
        "rescue_penalty": 0.15,
    },
    "rescue": {
        "prefill_affinity": 0.75,
        "decode_affinity": 0.75,
        "batch_affinity": 0.30,
        "concurrency_affinity": 0.30,
        "latency_weight": 0.24,
        "energy_weight": 0.08,
        "queue_weight": 0.80,
        "power_weight": 0.15,
        "rescue_penalty": 1.25,
    },
}

DEFAULT_ADMISSION = {
    "max_in_flight_multiplier": 1.0,
    "max_queue_delay_s": None,
    "reject_on_overload": False,
}

DEFAULT_POWER_POLICY = {
    "enabled": False,
    "budget_w": None,
    "reserve_w": 0.0,
    "block_over_budget": True,
    "over_budget_penalty": 1000.0,
}

DEFAULT_REBALANCING = {
    "enabled": True,
    "imbalance_threshold": 0.25,
    "in_flight_weight": 0.15,
    "queue_delay_weight": 1.00,
    "ewma_weight": 0.25,
}

DEFAULT_SOFT_ADMISSION = {
    "enabled": False,
    "planner_policy": "greedy_slack",
    "replan_interval_s": 0.25,
    "new_request_threshold": 1,
    "finished_request_threshold": 1,
    "max_pending_requests": 1024,
    "lookahead_s": 15.0,
    "force_admit_running": True,
    "default_request_value": 1,
    "admit_only_if_feasible": True,
    "best_effort_enabled": True,
    "best_effort_queue_limit": 4096,
    "best_effort_dispatch_fraction": 0.15,
    "max_route_attempts": 3,
    "route_strategy": "score_then_sequential",
    "planner_objective": "value_density",
    "slo_classes": {
        "default": {
            "request_value": 1,
            "allow_best_effort": True,
        }
    },
}

DEFAULT_QUEUE = {
    "enabled": False,
    "tick_ms": 50,
    "max_pending_requests": 1024,
    "request_timeout_s": 5.0,
    "planner_window": 128,
    "max_dispatch_per_tick": 32,
    "idle_flush": True,
}

DEFAULT_FEEDBACK = {
    "enabled": False,
    "observe_streaming": True,
    "ttft_alpha": 0.2,
    "tpot_alpha": 0.2,
    "service_rate_alpha": 0.2,
    "slack_alpha": 0.1,
    "quantile_default": 0.95,
    "quantile_overrides": {},
    "min_samples": 1,
    "ttft_weight": 1.0,
    "tpot_weight": 1.0,
    "strict_ttft": False,
    "strict_tpot": False,
    "slo_classes": {},
}

DEFAULT_DEPON = {
    "enabled": False,
    "block_on_missing_parents": True,
    "max_blocked_requests": 2048,
    "critical_path_bonus": 2.0,
    "stage_slack_weight": 1.0,
    "inherit_request_deadline": True,
}

DEFAULT_KV_PROJECT = {
    "enabled": False,
    "strict": True,
    "poll_interval_ms": 50,
    "stale_after_ms": 250,
    "startup_probe_timeout_ms": 500,
    "request_timeout_ms": 50,
    "fail_closed": True,
    "current_util_weight": 0.5,
    "peak_util_weight": 4.0,
    "recovery_weight": 0.75,
    "batch_weight": 0.10,
    "reject_peak_utilization": 0.98,
    "warn_peak_utilization": 0.90,
    "soft_cap_utilization": 0.85,
    "sim_quantile": "p95",
}


def _copy_mapping(value):
    if not isinstance(value, dict):
        return {}
    copied = {}
    for key, item in value.items():
        if isinstance(item, dict):
            copied[key] = _copy_mapping(item)
        else:
            copied[key] = item
    return copied


def _merge_mapping(base, override):
    merged = _copy_mapping(base)
    if not isinstance(override, dict):
        return merged
    for key, item in override.items():
        if isinstance(item, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged.get(key), item)
        else:
            merged[key] = item
    return merged


class ControllerConfig:

    __slots__ = (
        "lane_class_profiles",
        "admission",
        "power_policy",
        "rebalancing",
        "soft_admission",
        "queue",
        "feedback",
        "depon",
        "kv_project",
        "op_drift",
        "random_seed",
    )

    def __init__(
        self,
        lane_class_profiles=None,
        admission=None,
        power_policy=None,
        rebalancing=None,
        queue=None,
        feedback=None,
        depon=None,
        kv_project=None,
        random_seed=0,
        op_drift=None,
    ):
        self.lane_class_profiles = _merge_mapping(DEFAULT_LANE_CLASS_PROFILES, lane_class_profiles or {})
        self.admission = _merge_mapping(DEFAULT_ADMISSION, admission or {})
        self.power_policy = _merge_mapping(DEFAULT_POWER_POLICY, power_policy or {})
        self.rebalancing = _merge_mapping(DEFAULT_REBALANCING, rebalancing or {})
        self.soft_admission = _merge_mapping(DEFAULT_SOFT_ADMISSION, {})
        self.queue = _merge_mapping(DEFAULT_QUEUE, queue or {})
        self.feedback = _merge_mapping(DEFAULT_FEEDBACK, feedback or {})
        self.depon = _merge_mapping(DEFAULT_DEPON, depon or {})
        self.kv_project = _merge_mapping(DEFAULT_KV_PROJECT, kv_project or {})
        from .op_drift import load_drift_config
        self.op_drift = load_drift_config({"op_drift": op_drift})
        self.random_seed = random_seed

    def apply_soft_admission(self, soft_admission):
        self.soft_admission = _merge_mapping(DEFAULT_SOFT_ADMISSION, soft_admission or {})

    def apply_queue(self, queue):
        self.queue = _merge_mapping(DEFAULT_QUEUE, queue or {})

    def apply_feedback(self, feedback):
        self.feedback = _merge_mapping(DEFAULT_FEEDBACK, feedback or {})

    def apply_depon(self, depon):
        self.depon = _merge_mapping(DEFAULT_DEPON, depon or {})

    def apply_kv_project(self, kv_project):
        self.kv_project = _merge_mapping(DEFAULT_KV_PROJECT, kv_project or {})

    @classmethod
    def from_routing_kwargs(cls, routing_kwargs):
        if routing_kwargs is None:
            routing_kwargs = {}
        if not isinstance(routing_kwargs, dict):
            raise ValueError("routing_kwargs must decode to an object")
        cfg = cls(
            lane_class_profiles=routing_kwargs.get("lane_class_profiles"),
            admission=routing_kwargs.get("admission"),
            power_policy=routing_kwargs.get("power_policy"),
            rebalancing=routing_kwargs.get("rebalancing"),
            queue=routing_kwargs.get("queue"),
            feedback=routing_kwargs.get("feedback"),
            depon=routing_kwargs.get("depon"),
            kv_project=routing_kwargs.get("kv_project"),
            op_drift=routing_kwargs.get("op_drift"),
            random_seed=routing_kwargs.get("random_seed", 0),
        )
        cfg.apply_soft_admission(routing_kwargs.get("soft_admission"))
        cfg.apply_queue(routing_kwargs.get("queue"))
        cfg.apply_feedback(routing_kwargs.get("feedback"))
        cfg.apply_depon(routing_kwargs.get("depon"))
        cfg.apply_kv_project(routing_kwargs.get("kv_project"))
        return cfg

    def lane_profile(self, preset):
        lane_type = getattr(preset, "lane_type", "generic") or "generic"
        base = self.lane_class_profiles.get(lane_type) or self.lane_class_profiles.get("generic", {})
        overrides = getattr(preset, "policy_weights", None) or {}
        return _merge_mapping(base, overrides)

    def to_dict(self):
        return {
            "lane_class_profiles": _copy_mapping(self.lane_class_profiles),
            "admission": _copy_mapping(self.admission),
            "power_policy": _copy_mapping(self.power_policy),
            "rebalancing": _copy_mapping(self.rebalancing),
            "soft_admission": _copy_mapping(self.soft_admission),
            "queue": _copy_mapping(self.queue),
            "feedback": _copy_mapping(self.feedback),
            "depon": _copy_mapping(self.depon),
            "kv_project": _copy_mapping(self.kv_project),
            "random_seed": self.random_seed,
            **({"op_drift": _copy_mapping(self.op_drift)} if self.op_drift["mode"] != "off" else {}),
        }
