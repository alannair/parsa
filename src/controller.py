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
import math
import random
import time
import uuid

from .classifier import RequestClassifier
from .config import ControllerConfig
from .dispatch_queue import DispatchQueue, QueuedDispatchRequest
from .feedback import FeedbackModel
from .kv_projection import KVProjectionStore
from .lane_registry import LaneRegistry
from .lane_state import QueueManager
from .metrics import MetricsPipeline
from .policies import AdmissionController, AutoscalingHook, LaneScore, PolicyEngine, QueueBalancer, SchedulingLayer, SteeringContext
from .power_monitor import NullPowerMonitor
from .request_state import RequestAdmissionRecord, RequestStateStore
from .soft_admission import SoftAdmissionPlanner
from .workflow_state import WorkflowStateStore


ROUTING_POLICIES = (
    "legacy_ratio",
    "best_feasible_latency",
    "best_feasible_energy",
    "random",
    "round_robin",
    "oracle_class_affinity",
)


class OverloadedError(RuntimeError):
    pass


class SLO:

    __slots__ = ("max_latency_s",)

    def __init__(self, max_latency_s):
        value = float(max_latency_s)
        if value <= 0:
            raise ValueError("max_latency_s must be > 0")
        self.max_latency_s = value


class Choice:

    __slots__ = (
        "preset",
        "predicted",
        "predicted_latency_s",
        "predicted_energy_j",
        "feasible",
        "used_rescue",
        "routing_l_in",
        "routing_l_out_predicted_raw",
        "routing_l_out_effective",
        "queue_delay_s",
        "admitted",
        "admission_reason",
        "projected_power_w",
        "policy_score",
        "affinity_score",
        "request_id",
        "admission_tier",
        "route_attempts",
        "slo_class",
        "requested_max_latency_s",
        "requested_ttft_s",
        "requested_tpot_s",
        "request_value",
        "predicted_output_quantiles",
        "workflow_id",
        "stage_id",
        "critical_path",
        "request_kind",
        "_observation_context",
    )

    def __init__(
        self,
        preset,
        predicted,
        predicted_latency_s,
        predicted_energy_j,
        feasible,
        used_rescue,
        routing_l_in,
        routing_l_out_predicted_raw,
        routing_l_out_effective,
        queue_delay_s,
        admitted,
        admission_reason,
        projected_power_w,
        policy_score,
        affinity_score,
        request_id="",
        admission_tier="",
        route_attempts=0,
        slo_class="default",
        requested_max_latency_s=None,
        requested_ttft_s=None,
        requested_tpot_s=None,
        request_value=1,
        predicted_output_quantiles=None,
        workflow_id="",
        stage_id=None,
        critical_path=False,
        request_kind="default",
    ):
        self.preset = preset
        self.predicted = predicted
        self.predicted_latency_s = float(predicted_latency_s or 0.0)
        self.predicted_energy_j = float(predicted_energy_j or 0.0)
        self.feasible = bool(feasible)
        self.used_rescue = bool(used_rescue)
        self.routing_l_in = float(routing_l_in or 0.0)
        self.routing_l_out_predicted_raw = float(routing_l_out_predicted_raw or 0.0)
        self.routing_l_out_effective = float(routing_l_out_effective or 0.0)
        self.queue_delay_s = float(queue_delay_s or 0.0)
        self.admitted = bool(admitted)
        self.admission_reason = str(admission_reason or "")
        self.projected_power_w = float(projected_power_w or 0.0)
        self.policy_score = float(policy_score or 0.0)
        self.affinity_score = float(affinity_score or 0.0)
        self.request_id = str(request_id or "")
        self.admission_tier = str(admission_tier or "")
        self.route_attempts = int(route_attempts or 0)
        self.slo_class = str(slo_class or "default")
        self.requested_max_latency_s = None if requested_max_latency_s is None else float(requested_max_latency_s)
        self.requested_ttft_s = None if requested_ttft_s is None else float(requested_ttft_s)
        self.requested_tpot_s = None if requested_tpot_s is None else float(requested_tpot_s)
        self.request_value = int(request_value or 1)
        self.predicted_output_quantiles = dict(predicted_output_quantiles or {})
        self.workflow_id = str(workflow_id or "")
        self.stage_id = None if stage_id is None else str(stage_id)
        self.critical_path = bool(critical_path)
        self.request_kind = str(request_kind or "default")
        self._observation_context = None


class EnergyAwareController:

    __slots__ = (
        "profile",
        "slo",
        "rescue_preset_id",
        "classifier",
        "registry",
        "queue_manager",
        "metrics",
        "power_monitor",
        "autoscaling_hook",
        "config",
        "scheduler",
        "policy_engine",
        "admission_controller",
        "rebalancer",
        "request_state_store",
        "soft_admission_planner",
        "dispatch_queue",
        "feedback_model",
        "kv_projection_store",
        "workflow_state_store",
        "random_source",
        "_routing_policy",
        "random_lb",
        "specialization_gap",
        "shadow",
    )

    def __init__(
        self,
        presets,
        profile,
        slo,
        rescue_preset_id=None,
        token_counter=None,
        output_len_predictor=None,
        out_len_safety_factor=1.2,
        random_lb=False,
        routing_policy="legacy_ratio",
        routing_kwargs=None,
        queue_manager=None,
        registry=None,
        metrics=None,
        power_monitor=None,
        autoscaling_hook=None,
        *,
        shadow_clock=time.monotonic,
    ):
        self.profile = profile
        self.slo = slo
        self.rescue_preset_id = rescue_preset_id
        self.classifier = RequestClassifier(
            token_counter=token_counter,
            output_len_predictor=output_len_predictor,
            out_len_safety_factor=out_len_safety_factor,
        )
        self.registry = registry or LaneRegistry(presets)
        self.queue_manager = queue_manager or QueueManager([preset.preset_id for preset in list(presets or [])])
        self.metrics = metrics or MetricsPipeline()
        self.config = ControllerConfig.from_routing_kwargs(routing_kwargs)
        self.power_monitor = power_monitor or NullPowerMonitor(budget_w=self.config.power_policy.get("budget_w"))
        self.autoscaling_hook = autoscaling_hook or AutoscalingHook()
        self.scheduler = SchedulingLayer()
        self.policy_engine = PolicyEngine(self.scheduler)
        self.admission_controller = AdmissionController(self.config.admission)
        self.rebalancer = QueueBalancer(self.config.rebalancing)
        self.request_state_store = RequestStateStore()
        self.soft_admission_planner = SoftAdmissionPlanner(
            self.config.soft_admission,
            self.request_state_store,
            metrics=self.metrics,
        )
        self.dispatch_queue = DispatchQueue()
        self.feedback_model = FeedbackModel(self.config.feedback)
        self.kv_projection_store = KVProjectionStore(self.config.kv_project)
        self.workflow_state_store = WorkflowStateStore()
        self.random_source = random.Random(self.config.random_seed)
        self.random_lb = bool(random_lb)
        self._routing_policy = "random" if self.random_lb else str(routing_policy or "legacy_ratio")
        self.specialization_gap = (routing_kwargs or {}).get("specialization_gap")
        op_conflict = (routing_kwargs or {}).get("op_conflict")
        if self.config.op_drift["mode"] != "off" and (self.specialization_gap is not None or op_conflict is not None):
            raise ValueError("op_drift cannot be enabled in experiment modes")
        if op_conflict is not None:
            if self.specialization_gap is not None or not isinstance(op_conflict, dict):
                raise ValueError("op_conflict must be an object and cannot also configure specialization_gap")
            # Reuse the experiment gate so the proxy also suppresses prediction headers.
            self.specialization_gap = dict(op_conflict, label_key="op_conflict")
        if self.specialization_gap is not None:
            options = self.specialization_gap
            if not isinstance(options, dict) or self._routing_policy not in ("oracle_class_affinity", "round_robin"):
                raise ValueError("specialization_gap requires oracle_class_affinity or round_robin")
            lane_ids = options.get("lane_ids")
            pools = options.get("active_pools")
            mapping = options.get("pool_to_lane")
            if not isinstance(lane_ids, list) or len(lane_ids) != 2 or len(set(lane_ids)) != 2 or lane_ids != self.registry.ids():
                raise ValueError("specialization_gap requires exactly two ordered lane IDs matching the presets")
            label_key = options.get("label_key", "specialization_gap")
            if label_key not in ("specialization_gap", "op_conflict"):
                raise ValueError("unknown experiment label key")
            known_pools = {"prefill-heavy", "decode-heavy"} if label_key == "op_conflict" else {"summarization", "conversation"}
            if not isinstance(pools, list) or not pools or len(set(pools)) != len(pools) or not set(pools) <= known_pools:
                raise ValueError("specialization_gap requires known experiment source pools")
            if not isinstance(mapping, dict):
                raise ValueError("specialization_gap pool_to_lane must be an object")
            if self._routing_policy == "oracle_class_affinity":
                if len(pools) != 2 or set(mapping) != set(pools) or set(mapping.values()) != set(lane_ids):
                    raise ValueError("oracle_class_affinity requires both pools mapped to distinct lanes")
            elif mapping:
                raise ValueError("neutral round_robin must not specify a class mapping")
            if any((self.soft_admission_enabled, self.dispatch_loop_enabled, self.feedback_enabled,
                    self.kv_project_enabled, self.config.power_policy.get("enabled"))):
                raise ValueError("specialization_gap cannot use prediction, admission planning, feedback, or power routing")
            if label_key == "op_conflict" and (token_counter is not None or output_len_predictor is not None):
                raise ValueError("op_conflict cannot configure token or output-length predictors")
        elif self._routing_policy == "oracle_class_affinity":
            raise ValueError("oracle_class_affinity is experiment-only; specialization_gap kwargs are required")
        if self._routing_policy != "oracle_class_affinity" and self._routing_policy not in self.policy_engine.names():
            raise ValueError("unknown routing policy: %s" % self._routing_policy)
        self.queue_manager.ensure_from_presets(self.registry.snapshot())
        self.shadow = None
        if self.config.op_drift["mode"] == "shadow":
            from .op_drift import ShadowObserver
            self.shadow = ShadowObserver(self.config.op_drift, self.registry.snapshot(),
                                         self.routing_policy, self.metrics, clock=shadow_clock)

    @property
    def routing_policy(self):
        return self._routing_policy

    @property
    def soft_admission_enabled(self):
        return bool(self.config.soft_admission.get("enabled"))

    @property
    def queue_enabled(self):
        return bool(self.config.queue.get("enabled"))

    @property
    def feedback_enabled(self):
        return bool(self.config.feedback.get("enabled"))

    @property
    def depon_enabled(self):
        return bool(self.config.depon.get("enabled"))

    @property
    def dispatch_loop_enabled(self):
        return bool(self.queue_enabled or self.depon_enabled)

    @property
    def kv_project_enabled(self):
        return bool(self.config.kv_project.get("enabled"))

    def estimate_l_in(self, request_json):
        return self.classifier.estimate_l_in(request_json)

    def estimate_l_out_raw(self, request_json, force_body_only=False):
        return self.classifier.estimate_l_out_raw(request_json, force_body_only=force_body_only)

    def estimate_l_out(self, request_json, raw=None):
        return self.classifier.estimate_l_out(request_json, raw=raw)

    def add_preset(self, preset):
        self.registry.add(preset)
        self.queue_manager.ensure_lane(preset.preset_id)
        self.metrics.record_registry_change("add", preset.preset_id)
        self.autoscaling_hook.on_lane_added(preset)

    def remove_preset(self, preset_id):
        self.registry.remove(preset_id)
        self.queue_manager.remove_lane(preset_id)
        self.metrics.record_registry_change("remove", preset_id)
        self.autoscaling_hook.on_lane_removed(preset_id)

    def replace_presets(self, presets):
        self.registry.replace(presets)
        self.queue_manager.ensure_from_presets(self.registry.snapshot())
        self.metrics.record_registry_change("replace", ",".join(self.registry.ids()))

    def _capacity_normalizers(self, presets):
        max_batch = 1.0
        max_seqs = 1.0
        for preset in presets:
            max_batch = max(max_batch, float(getattr(preset, "max_num_batched_tokens", 1) or 1))
            seqs = getattr(preset, "max_num_seqs", None)
            if seqs is not None:
                max_seqs = max(max_seqs, float(seqs or 1))
        return {"max_batch": max_batch, "max_seqs": max_seqs}

    def _affinity_score(self, preset, request_features, normalizers):
        weights = self.config.lane_profile(preset)
        batch_capacity = float(getattr(preset, "max_num_batched_tokens", 0) or 0.0) / normalizers.get("max_batch", 1.0)
        seq_capacity = 0.0
        if getattr(preset, "max_num_seqs", None) is not None:
            seq_capacity = float(preset.max_num_seqs or 0.0) / normalizers.get("max_seqs", 1.0)
        affinity = request_features.prefill_share * float(weights.get("prefill_affinity", 0.5) or 0.0)
        affinity += request_features.decode_share * float(weights.get("decode_affinity", 0.5) or 0.0)
        affinity += batch_capacity * float(weights.get("batch_affinity", 0.0) or 0.0)
        affinity += seq_capacity * float(weights.get("concurrency_affinity", 0.0) or 0.0)
        return affinity, weights

    def _queue_snapshot_with_delay(self, presets, request_features):
        snapshot = {}
        for preset in presets:
            state = self.queue_manager.get(preset.preset_id)
            estimate = self.profile.estimate(preset_id=preset.preset_id, l_in=request_features.l_in, l_out=request_features.l_out)
            fallback_service = estimate.latency_s(l_out=request_features.l_out, queue_delay_s=0.0)
            lane_snapshot = state.snapshot()
            lane_snapshot["queue_delay_s"] = state.estimate_queue_delay_s(fallback_service)
            snapshot[preset.preset_id] = lane_snapshot
        return snapshot

    def _power_penalty(self, preset, projected_power_w, power_snapshot, weights):
        policy = self.config.power_policy
        if not policy.get("enabled"):
            return 0.0, True, ""
        budget_w = power_snapshot.budget_w
        if budget_w is None:
            budget_w = policy.get("budget_w")
        if budget_w is None:
            return 0.0, True, ""
        reserve_w = float(policy.get("reserve_w", 0.0) or 0.0)
        lane_now_w = power_snapshot.lane_power(preset.preset_id, 0.0)
        projected_total = max(0.0, power_snapshot.cluster_power_w - lane_now_w + max(lane_now_w, projected_power_w))
        over_budget_w = projected_total + reserve_w - float(budget_w)
        if over_budget_w <= 0:
            return 0.0, True, ""
        penalty = over_budget_w * float(policy.get("over_budget_penalty", 1000.0) or 0.0)
        penalty *= float(weights.get("power_weight", 1.0) or 0.0)
        allowed = not bool(policy.get("block_over_budget", True)) or getattr(preset, "lane_type", "") == "rescue"
        return penalty, allowed, "power_budget"

    def _kv_projection_penalty(self, preset, request_features, request_options):
        if not self.kv_project_enabled:
            return 0.0, True, "", None
        policy = self.config.kv_project
        snapshot = self.kv_projection_store.get_snapshot(preset.preset_id)
        if snapshot is None:
            if bool(policy.get("fail_closed", True)):
                return 1.0e6, False, "npu_kv_projection_unavailable", None
            return 0.0, True, "", None
        quantiles = dict((request_options or {}).get("predicted_output_quantiles") or {})
        sim_quantile = str(policy.get("sim_quantile", "p95") or "p95")
        output_tokens = float(quantiles.get(sim_quantile, quantiles.get("p95", request_features.l_out_raw)) or 0.0)
        total_tokens = max(0.0, float(request_features.l_in or 0.0)) + max(0.0, output_tokens)
        delta_blocks = snapshot.delta_blocks_for_tokens(total_tokens)
        current_util = snapshot.current_utilization()
        peak_util = snapshot.peak_util_if_admitted(delta_blocks=delta_blocks, quantile=sim_quantile)
        recovery_s = snapshot.recovery_seconds("0.70")
        peak_batch = snapshot.peak_batch(sim_quantile) + 1
        penalty = current_util * float(policy.get("current_util_weight", 0.5) or 0.0)
        penalty += peak_util * float(policy.get("peak_util_weight", 4.0) or 0.0)
        penalty += max(0.0, float(recovery_s or 0.0)) * float(policy.get("recovery_weight", 0.75) or 0.0)
        penalty += max(0, int(peak_batch or 0)) * float(policy.get("batch_weight", 0.10) or 0.0)
        soft_cap = float(policy.get("soft_cap_utilization", 0.85) or 0.85)
        warn_cap = float(policy.get("warn_peak_utilization", 0.90) or 0.90)
        reject_cap = float(policy.get("reject_peak_utilization", 0.98) or 0.98)
        if peak_util >= soft_cap:
            penalty += (peak_util - soft_cap) * 10.0
        if peak_util >= warn_cap:
            penalty += (peak_util - warn_cap) * 100.0
        if snapshot.host_used_blocks > 0:
            return penalty + 1.0e6, False, "npu_kv_projection_host_swap", snapshot
        if peak_util >= reject_cap:
            return penalty + 1.0e6, False, "npu_kv_projection_peak", snapshot
        return penalty, True, "", snapshot

    def _score_preset(self, preset, request_features, request_options, normalizers, queue_snapshot, power_snapshot):
        est = self.profile.estimate(preset_id=preset.preset_id, l_in=request_features.l_in, l_out=request_features.l_out)
        if self.feedback_enabled:
            est = self.feedback_model.adjust_estimate(preset.preset_id, est)
        state = self.queue_manager.get(preset.preset_id)
        fallback_service = est.latency_s(l_out=request_features.l_out, queue_delay_s=0.0)
        queue_delay_s = state.estimate_queue_delay_s(fallback_service_s=fallback_service)
        predicted_latency_s = est.latency_s(l_out=request_features.l_out, queue_delay_s=queue_delay_s)
        predicted_energy_j = est.energy_j(l_out=request_features.l_out)
        projected_power_w = est.power_w(l_out=request_features.l_out, queue_delay_s=queue_delay_s)
        latency_limit = self.slo.max_latency_s if request_options is None else float(request_options.get("max_latency_s", self.slo.max_latency_s))
        target_ttft_s = None if request_options is None else request_options.get("target_ttft_s")
        target_tpot_s = None if request_options is None else request_options.get("target_tpot_s")
        feasible = predicted_latency_s <= latency_limit
        ttft_penalty_s = 0.0
        tpot_penalty_s = 0.0
        if target_ttft_s is not None:
            target_ttft_s = max(0.0, float(target_ttft_s or 0.0))
            ttft_penalty_s = max(0.0, float(est.ttft_s or 0.0) - target_ttft_s)
            if ttft_penalty_s > 0 and bool(self.config.feedback.get("strict_ttft", False)):
                feasible = False
        if target_tpot_s is not None:
            target_tpot_s = max(0.0, float(target_tpot_s or 0.0))
            tpot_penalty_s = max(0.0, float(est.tbt_s or 0.0) - target_tpot_s)
            if tpot_penalty_s > 0 and bool(self.config.feedback.get("strict_tpot", False)):
                feasible = False
        if self.feedback_enabled:
            predicted_latency_s += ttft_penalty_s
            predicted_latency_s += tpot_penalty_s * max(1.0, float(request_features.l_out or 0.0))
            feasible = feasible and predicted_latency_s <= latency_limit
        admission = self.admission_controller.assess(preset, state, queue_delay_s)
        affinity_score, weights = self._affinity_score(preset, request_features, normalizers)
        balance_penalty = self.rebalancer.penalty(preset.preset_id, queue_snapshot)
        power_penalty, power_allowed, power_reason = self._power_penalty(preset, projected_power_w, power_snapshot, weights)
        kv_penalty, kv_allowed, kv_reason, _kv_snapshot = self._kv_projection_penalty(preset, request_features, request_options)
        admitted = admission.accepted and power_allowed and kv_allowed
        admission_reason = admission.reason or power_reason or kv_reason
        policy_score = -affinity_score
        policy_score += float(weights.get("latency_weight", 0.0) or 0.0) * predicted_latency_s
        policy_score += float(weights.get("energy_weight", 0.0) or 0.0) * predicted_energy_j
        policy_score += float(weights.get("queue_weight", 0.0) or 0.0) * queue_delay_s
        policy_score += balance_penalty + power_penalty + kv_penalty
        if self.feedback_enabled:
            policy_score += ttft_penalty_s * float(self.config.feedback.get("ttft_weight", 1.0) or 0.0)
            policy_score += tpot_penalty_s * float(self.config.feedback.get("tpot_weight", 1.0) or 0.0)
        if getattr(preset, "lane_type", "") == "rescue":
            policy_score += float(weights.get("rescue_penalty", 0.0) or 0.0)
        return LaneScore(
            preset=preset,
            predicted=est,
            predicted_latency_s=predicted_latency_s,
            predicted_energy_j=predicted_energy_j,
            feasible=feasible,
            admitted=admitted,
            admission_reason=admission_reason,
            queue_delay_s=queue_delay_s,
            projected_power_w=projected_power_w,
            policy_score=policy_score,
            affinity_score=affinity_score,
        )

    def _fallback_choice(self, scored):
        prefer = "energy" if self._routing_policy == "best_feasible_energy" else "latency"
        return self.scheduler.pick_fallback(scored, prefer=prefer)

    def _build_choice(
        self,
        score,
        request_features,
        request_options=None,
        request_id="",
        admission_tier="",
        route_attempts=0,
        slo_class="default",
        requested_max_latency_s=None,
        admission_reason=None,
    ):
        return Choice(
            preset=score.preset,
            predicted=score.predicted,
            predicted_latency_s=score.predicted_latency_s,
            predicted_energy_j=score.predicted_energy_j,
            feasible=score.feasible,
            used_rescue=score.used_rescue,
            routing_l_in=request_features.l_in,
            routing_l_out_predicted_raw=request_features.l_out_raw,
            routing_l_out_effective=request_features.l_out,
            queue_delay_s=score.queue_delay_s,
            admitted=score.admitted,
            admission_reason=score.admission_reason if admission_reason is None else admission_reason,
            projected_power_w=score.projected_power_w,
            policy_score=score.policy_score,
            affinity_score=score.affinity_score,
            request_id=request_id,
            admission_tier=admission_tier,
            route_attempts=route_attempts,
            slo_class=slo_class,
            requested_max_latency_s=requested_max_latency_s,
            requested_ttft_s=None if request_options is None else request_options.get("target_ttft_s"),
            requested_tpot_s=None if request_options is None else request_options.get("target_tpot_s"),
            request_value=1 if request_options is None else request_options.get("request_value", 1),
            predicted_output_quantiles={} if request_options is None else request_options.get("predicted_output_quantiles", {}),
            workflow_id="" if request_options is None else request_options.get("workflow_id", ""),
            stage_id=None if request_options is None else request_options.get("stage_id"),
            critical_path=False if request_options is None else request_options.get("critical_path", False),
            request_kind="default" if request_options is None else request_options.get("request_kind", "default"),
        )

    def _predict_output_quantiles(self, request_json, request_features):
        predictor = getattr(self.classifier, "output_len_predictor", None)
        base_raw = max(0.0, float(getattr(request_features, "l_out_raw", 0.0) or 0.0))
        base_effective = max(0.0, float(getattr(request_features, "l_out", 0.0) or 0.0))
        quantiles = {}
        for quantile, name in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95")):
            value = None
            if predictor is not None and hasattr(predictor, "predict_quantile"):
                try:
                    value = float(predictor.predict_quantile(request_json, quantile))
                except Exception:
                    value = None
            if value is None:
                if name == "p50":
                    value = base_raw
                elif name == "p90":
                    value = max(base_raw, base_effective)
                else:
                    value = max(base_raw, base_effective * 1.05)
            if value < 0 or math.isnan(value) or math.isinf(value):
                value = max(base_raw, base_effective)
            quantiles[name] = float(value)
        return quantiles

    def _classify_request(self, request_json):
        request_options = self._request_options(request_json)
        force_body_only = self._routing_policy == "random"
        quantile = None
        if self.feedback_enabled:
            quantile = self.feedback_model.quantile_for_request(request_options)
        request_features = self.classifier.classify(
            request_json,
            force_body_only=force_body_only,
            quantile=quantile,
        )
        request_options["predicted_output_quantiles"] = self._predict_output_quantiles(request_json, request_features)
        return request_features, request_options

    def _request_energy_aware(self, request_json):
        data = request_json.get("energy_aware") or {}
        if not isinstance(data, dict):
            return {}
        return data

    def _request_options(self, request_json):
        soft = self.config.soft_admission
        feedback = self.config.feedback
        depon = self.config.depon
        classes = feedback.get("slo_classes") or soft.get("slo_classes") or {}
        data = self._request_energy_aware(request_json)
        slo_class = str(data.get("slo_class") or "default")
        defaults = classes.get(slo_class) or classes.get("default") or {}
        max_latency_s = data.get("max_latency_s", defaults.get("max_latency_s", self.slo.max_latency_s))
        if max_latency_s is None:
            max_latency_s = self.slo.max_latency_s
        stage_deadline_s = data.get("stage_deadline_s")
        if stage_deadline_s is not None:
            try:
                stage_deadline_s = float(stage_deadline_s)
            except Exception:
                stage_deadline_s = None
        if stage_deadline_s is not None and bool(depon.get("inherit_request_deadline", True)):
            max_latency_s = min(float(max_latency_s), float(stage_deadline_s))
        request_value = data.get("request_value", defaults.get("request_value", soft.get("default_request_value", 1)))
        allow_best_effort = data.get("allow_best_effort", defaults.get("allow_best_effort", soft.get("best_effort_enabled", True)))
        max_route_attempts = data.get("max_route_attempts", soft.get("max_route_attempts", 3))
        priority = data.get("priority", request_json.get("priority", 0))
        parent_request_ids = data.get("parent_request_ids") or []
        if isinstance(parent_request_ids, str):
            parent_request_ids = [parent_request_ids]
        elif not isinstance(parent_request_ids, (list, tuple, set)):
            parent_request_ids = []
        try:
            priority = int(priority)
        except Exception:
            priority = 0
        return {
            "slo_class": slo_class,
            "max_latency_s": float(max_latency_s),
            "target_ttft_s": data.get("target_ttft_s", defaults.get("target_ttft_s")),
            "target_tpot_s": data.get("target_tpot_s", defaults.get("target_tpot_s")),
            "request_value": int(request_value or 1),
            "allow_best_effort": bool(allow_best_effort),
            "max_route_attempts": int(max_route_attempts or 0),
            "priority": priority,
            "workflow_id": str(data.get("workflow_id") or ""),
            "stage_id": None if data.get("stage_id") is None else str(data.get("stage_id")),
            "parent_request_ids": [str(item) for item in parent_request_ids if str(item or "").strip()],
            "stage_deadline_s": stage_deadline_s,
            "critical_path": bool(data.get("critical_path", False)),
            "request_kind": str(data.get("request_kind") or defaults.get("request_kind") or "default"),
        }

    def _register_request_record(self, request_id, request_json, request_features, request_options):
        record = RequestAdmissionRecord(
            request_id=request_id,
            request_json=request_json,
            l_in=request_features.l_in,
            l_out_raw=request_features.l_out_raw,
            l_out=request_features.l_out,
            priority=request_options.get("priority", 0),
            request_value=request_options.get("request_value", 1),
            allow_best_effort=request_options.get("allow_best_effort", True),
            max_route_attempts=request_options.get("max_route_attempts", 3),
            requested_max_latency_s=request_options.get("max_latency_s"),
            requested_ttft_s=request_options.get("target_ttft_s"),
            requested_tpot_s=request_options.get("target_tpot_s"),
            slo_class=request_options.get("slo_class", "default"),
            workflow_id=request_options.get("workflow_id", ""),
            stage_id=request_options.get("stage_id"),
            parent_request_ids=request_options.get("parent_request_ids"),
            stage_deadline_s=request_options.get("stage_deadline_s"),
            critical_path=request_options.get("critical_path", False),
            request_kind=request_options.get("request_kind", "default"),
        )
        registered = self.request_state_store.register_request(record)
        self.workflow_state_store.register_request(registered)
        return registered

    def _score_request(self, request_features, request_options, presets=None):
        presets = list(self.registry.snapshot() if presets is None else presets)
        if not presets:
            raise ValueError("no presets configured")
        power_snapshot = self.power_monitor.snapshot()
        normalizers = self._capacity_normalizers(presets)
        queue_snapshot = self._queue_snapshot_with_delay(presets, request_features)
        scored = [
            self._score_preset(
                preset=preset,
                request_features=request_features,
                request_options=request_options,
                normalizers=normalizers,
                queue_snapshot=queue_snapshot,
                power_snapshot=power_snapshot,
            )
            for preset in presets
        ]
        return scored

    def _plan_with_soft_admission(self, request_id, request_json, request_features, request_options, scored):
        record = self._register_request_record(request_id, request_json, request_features, request_options)
        plan = self.soft_admission_planner.plan(
            record=record,
            scored=scored,
            routing_policy=self._routing_policy,
            policy_engine=self.policy_engine,
            registry=self.registry,
            rescue_preset_id=self.rescue_preset_id,
            random_source=self.random_source,
            admission_config=self.config.admission,
        )
        if plan.tier == "rejected" or plan.score is None:
            self.metrics.record_admission_decision(
                request_id,
                "rejected",
                reason=plan.reason,
                route_attempts=plan.route_attempts,
                preset_id=plan.selected_preset_id,
                planner_epoch=plan.planner_epoch,
            )
            self.metrics.record_rejection(plan.reason, tier="rejected")
            self.request_state_store.mark_finished(request_id, status="rejected")
            self.request_state_store.remove_request(request_id)
            raise OverloadedError(plan.reason or "soft admission rejected request")
        self.request_state_store.mark_planned(
            request_id,
            tier=plan.tier,
            selected_preset_id=plan.selected_preset_id,
            predicted_latency_s=plan.score.predicted_latency_s,
            predicted_energy_j=plan.score.predicted_energy_j,
            route_attempts=plan.route_attempts,
            planner_epoch=plan.planner_epoch,
            admission_reason=plan.reason,
        )
        self.metrics.record_admission_decision(
            request_id,
            plan.tier,
            reason=plan.reason,
            route_attempts=plan.route_attempts,
            preset_id=plan.selected_preset_id,
            planner_epoch=plan.planner_epoch,
        )
        choice = self._build_choice(
            plan.score,
            request_features,
            request_options=request_options,
            request_id=request_id,
            admission_tier=plan.tier,
            route_attempts=plan.route_attempts,
            slo_class=record.slo_class,
            requested_max_latency_s=record.requested_max_latency_s,
            admission_reason=plan.reason,
        )
        self.metrics.record_choice(choice)
        return choice

    def _choose_specialization_gap(self, request_json, request_id):
        # Experiment-only: provenance and cohort order, never predicted energy or admission scores.
        options = self.specialization_gap
        label_key = options.get("label_key", "specialization_gap")
        metadata = self._request_energy_aware(request_json).get(label_key)
        if not isinstance(metadata, dict) or metadata.get("pool") not in options["active_pools"]:
            raise ValueError("missing or invalid specialization_gap source-pool label")
        index = metadata.get("request_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("specialization_gap request_index must be a positive integer")
        if self.registry.ids() != options["lane_ids"]:
            raise ValueError("specialization_gap lanes changed during the experiment")
        if self._routing_policy == "oracle_class_affinity":
            preset_id = options["pool_to_lane"][metadata["pool"]]
        else:
            # Cohort-index round-robin is unaffected by HTTP arrival order or lane occupancy.
            preset_id = options["lane_ids"][(index - 1) % 2]
        choice = Choice(
            preset=self.registry.get(preset_id), predicted=None,
            predicted_latency_s=0, predicted_energy_j=0, feasible=True, used_rescue=False,
            routing_l_in=0, routing_l_out_predicted_raw=0, routing_l_out_effective=0,
            queue_delay_s=0, admitted=True, admission_reason=label_key + "_experiment",
            projected_power_w=0, policy_score=0, affinity_score=0, request_id=request_id or "",
        )
        self.metrics.record_choice(choice)
        if label_key == "op_conflict":
            # Consume this opt-in label before the existing proxy's backend-body preparation.
            # Copy the nested object so callers retaining it keep their provenance record.
            request_json["energy_aware"] = dict(self._request_energy_aware(request_json))
            request_json["energy_aware"].pop("op_conflict")
        return choice

    def _begin_observation(self, request_id=None):
        return None if self.shadow is None else self.shadow.begin(request_id)

    def _finish_observation(self, context, status):
        if self.shadow is not None:
            self.shadow.finish(context, status)

    def choose(self, request_json, request_id=None, *, _observation_context=None):
        context = _observation_context
        if context is None:
            context = self._begin_observation(request_id)
        try:
            choice = self._choose(request_json, request_id, _observation_context=context)
            choice._observation_context = context
            return choice
        except asyncio.CancelledError:
            self._finish_observation(context, "cancelled")
            raise
        except Exception as exc:
            self._finish_observation(context, "rejected" if isinstance(exc, OverloadedError) else "failed")
            raise

    def _choose(self, request_json, request_id=None, *, _observation_context=None):
        if self.specialization_gap is not None:
            return self._choose_specialization_gap(request_json, request_id)
        presets = self.registry.snapshot()
        if not presets:
            raise ValueError("no presets configured")
        request_features, request_options = self._classify_request(request_json)
        if self.shadow is not None:
            self.shadow.enrich(_observation_context, request_features, request_options)
        scored = self._score_request(request_features, request_options, presets=presets)
        if self.soft_admission_enabled:
            return self._plan_with_soft_admission(
                request_id or uuid.uuid4().hex,
                request_json,
                request_features,
                request_options,
                scored,
            )
        context = SteeringContext(
            scored=scored,
            registry=self.registry,
            rescue_preset_id=self.rescue_preset_id,
            random_source=self.random_source,
        )
        selected = self.policy_engine.choose(self._routing_policy, context)
        if selected is None:
            selected = self._fallback_choice(scored)
        if selected is None:
            self.metrics.record_rejection("no_candidates")
            self.autoscaling_hook.on_overload({"reason": "no_candidates"})
            raise OverloadedError("no routable presets available")
        if self.kv_project_enabled and bool(self.config.kv_project.get("fail_closed", True)) and not any(item.admitted for item in scored):
            self.metrics.record_rejection("npu_kv_projection_no_capacity")
            raise OverloadedError("npu_kv_projection_no_capacity")
        if not selected.admitted and self.config.admission.get("reject_on_overload") and not selected.used_rescue:
            self.metrics.record_rejection(selected.admission_reason or "overload")
            details = {"reason": selected.admission_reason or "overload", "policy": self._routing_policy}
            self.autoscaling_hook.on_overload(details)
            raise OverloadedError(selected.admission_reason or "all lanes overloaded")
        choice = self._build_choice(
            selected,
            request_features,
            request_options=request_options,
            request_id=request_id or "",
            requested_max_latency_s=request_options.get("max_latency_s"),
            slo_class=request_options.get("slo_class", "default"),
        )
        self.metrics.record_choice(choice)
        if not any(item.admitted for item in scored):
            self.autoscaling_hook.on_overload(
                {
                    "reason": "all_not_admitted",
                    "policy": self._routing_policy,
                    "selected_preset_id": choice.preset.preset_id,
                }
            )
        return choice

    def suggested_priority(self, choice, request_json):
        if "priority" in request_json:
            try:
                return int(request_json.get("priority"))
            except Exception:
                return 0
        if choice.used_rescue:
            return -1
        return 0

    def _dependency_blockers(self, record):
        if not self.depon_enabled:
            return []
        if not bool(self.config.depon.get("block_on_missing_parents", True)):
            return []
        return self.workflow_state_store.missing_parents(record.request_id)

    async def _reject_queued_request(self, request_id, reason, status="rejected"):
        item = await self.dispatch_queue.get(request_id)
        if item is None:
            return False
        self._finish_observation(item.observation_context, "cancelled" if status == "cancelled" else "rejected")
        await self.dispatch_queue.set_state(request_id, status, reason=reason)
        self.request_state_store.mark_finished(request_id, status=status)
        self.workflow_state_store.mark_status(request_id, status)
        self.metrics.record_rejection(reason, tier=status)
        removed = await self.dispatch_queue.remove(request_id)
        record = self.request_state_store.get(request_id)
        route_attempts = getattr(record, "route_attempts", 0) if record is not None else 0
        planner_epoch = getattr(record, "planner_epoch", 0) if record is not None else 0
        self.metrics.record_admission_decision(
            request_id,
            status,
            reason=reason,
            route_attempts=route_attempts,
            planner_epoch=planner_epoch,
        )
        self.metrics.record_queue_event(request_id, status, reason=reason)
        self.request_state_store.remove_request(request_id)
        self.workflow_state_store.remove_request(request_id)
        if removed is not None and not removed.future.done():
            removed.future.set_exception(OverloadedError(str(reason or status)))
        return True

    async def choose_or_wait(self, request_json, request_id=None, *, _observation_context=None):
        context = _observation_context
        if context is None:
            context = self._begin_observation(request_id)
        try:
            choice = await self._choose_or_wait(request_json, request_id, _observation_context=context)
            choice._observation_context = context
            return choice
        except asyncio.CancelledError:
            self._finish_observation(context, "cancelled")
            raise
        except Exception as exc:
            self._finish_observation(context, "rejected" if isinstance(exc, OverloadedError) else "failed")
            raise

    async def _choose_or_wait(self, request_json, request_id=None, *, _observation_context=None):
        if not self.dispatch_loop_enabled:
            return self.choose(request_json, request_id=request_id, _observation_context=_observation_context)
        request_id = str(request_id or uuid.uuid4().hex)
        queue_snapshot = await self.dispatch_queue.snapshot()
        max_pending_requests = int(self.config.queue.get("max_pending_requests", 1024) or 1024)
        if int(queue_snapshot.get("size", 0) or 0) >= max_pending_requests:
            raise OverloadedError("dispatch_queue_full")
        request_features, request_options = self._classify_request(request_json)
        if self.shadow is not None:
            self.shadow.enrich(_observation_context, request_features, request_options)
        record = self._register_request_record(request_id, request_json, request_features, request_options)
        timeout_s = self.config.queue.get("request_timeout_s")
        loop = asyncio.get_running_loop()
        item = QueuedDispatchRequest(
            request_id=request_id,
            body=request_json,
            request_features=request_features,
            request_options=request_options,
            record=record,
            future=loop.create_future(),
            timeout_s=timeout_s,
            observation_context=_observation_context,
        )
        blockers = self._dependency_blockers(record)
        if blockers:
            max_blocked_requests = int(self.config.depon.get("max_blocked_requests", 2048) or 2048)
            if int(queue_snapshot.get("counts", {}).get("blocked", 0) or 0) >= max_blocked_requests:
                self.request_state_store.remove_request(request_id)
                self.workflow_state_store.remove_request(request_id)
                raise OverloadedError("dependency_queue_full")
            item.state = "blocked"
            item.reason = "dependency_blocked"
            item.blocked_on = list(blockers)
            self.request_state_store.mark_blocked(request_id, blockers, admission_reason="dependency_blocked")
            self.workflow_state_store.mark_status(request_id, "blocked")
        else:
            self.workflow_state_store.mark_status(request_id, "pending")
        await self.dispatch_queue.enqueue(item)
        self.metrics.record_queue_event(request_id, item.state, reason=item.reason, blocked_on=item.blocked_on)
        await self.plan_pending_dispatches()
        try:
            if item.expires_at_s is None:
                return await item.future
            timeout_left_s = max(0.0, float(item.expires_at_s) - time.time()) + 0.05
            return await asyncio.wait_for(item.future, timeout=timeout_left_s)
        except asyncio.TimeoutError:
            await self._reject_queued_request(request_id, "dispatch_queue_timeout", status="timed_out")
            raise OverloadedError("dispatch_queue_timeout")

    async def plan_pending_dispatches(self, now_s=None):
        now_value = time.time() if now_s is None else float(now_s)
        timed_out = await self.dispatch_queue.timed_out_items(now_value)
        for item in timed_out:
            await self._reject_queued_request(item.request_id, "dispatch_queue_timeout", status="timed_out")

        if self.depon_enabled:
            blocked_items = await self.dispatch_queue.blocked_items()
            for item in blocked_items:
                blockers = self._dependency_blockers(item.record)
                if blockers:
                    continue
                await self.dispatch_queue.set_state(item.request_id, "pending", blocked_on=[])
                self.request_state_store.update_request(item.request_id, status="pending", blocked_on=[])
                self.workflow_state_store.mark_status(item.request_id, "pending")
                self.metrics.record_queue_event(item.request_id, "unblocked")

        pending_limit = int(self.config.queue.get("planner_window", 128) or 128)
        pending_items = await self.dispatch_queue.pending_items(limit=pending_limit)
        if not pending_items:
            return 0

        max_dispatch = max(1, int(self.config.queue.get("max_dispatch_per_tick", 32) or 32))
        if not self.queue_enabled:
            released = 0
            for item in pending_items:
                if released >= max_dispatch:
                    break
                try:
                    choice = self.choose(item.body, request_id=item.request_id,
                                         _observation_context=item.observation_context)
                except OverloadedError as exc:
                    await self._reject_queued_request(item.request_id, str(exc), status="rejected")
                    continue
                await self.dispatch_queue.set_state(item.request_id, "released")
                removed = await self.dispatch_queue.remove(item.request_id)
                self.metrics.record_queue_event(item.request_id, "released")
                if removed is not None and not removed.future.done():
                    removed.future.set_result(choice)
                released += 1
            return released

        candidates = []
        for item in pending_items:
            scored = self._score_request(item.request_features, item.request_options)
            candidates.append(
                {
                    "request_id": item.request_id,
                    "record": item.record,
                    "request_features": item.request_features,
                    "request_options": item.request_options,
                    "scored": scored,
                }
            )
        plans = self.soft_admission_planner.plan_batch(
            candidates=candidates,
            routing_policy=self._routing_policy,
            policy_engine=self.policy_engine,
            registry=self.registry,
            rescue_preset_id=self.rescue_preset_id,
            random_source=self.random_source,
            admission_config=self.config.admission,
            max_dispatch=max_dispatch,
            depon_enabled=self.depon_enabled,
            feedback_enabled=self.feedback_enabled,
        )

        released = 0
        for plan in plans:
            item = await self.dispatch_queue.get(plan.request_id)
            if item is None:
                continue
            if plan.tier == "deferred":
                continue
            if plan.tier == "rejected" or plan.score is None:
                await self._reject_queued_request(plan.request_id, plan.reason or "soft admission rejected request", status="rejected")
                continue
            self.request_state_store.mark_planned(
                plan.request_id,
                tier=plan.tier,
                selected_preset_id=plan.selected_preset_id,
                predicted_latency_s=plan.score.predicted_latency_s,
                predicted_energy_j=plan.score.predicted_energy_j,
                route_attempts=plan.route_attempts,
                planner_epoch=plan.planner_epoch,
                admission_reason=plan.reason,
            )
            self.workflow_state_store.mark_status(plan.request_id, "planned")
            self.metrics.record_admission_decision(
                plan.request_id,
                plan.tier,
                reason=plan.reason,
                route_attempts=plan.route_attempts,
                preset_id=plan.selected_preset_id,
                planner_epoch=plan.planner_epoch,
            )
            choice = self._build_choice(
                plan.score,
                item.request_features,
                request_options=item.request_options,
                request_id=plan.request_id,
                admission_tier=plan.tier,
                route_attempts=plan.route_attempts,
                slo_class=item.record.slo_class,
                requested_max_latency_s=item.record.requested_max_latency_s,
                admission_reason=plan.reason,
            )
            choice._observation_context = item.observation_context
            self.metrics.record_choice(choice)
            await self.dispatch_queue.set_state(plan.request_id, "released", reason=plan.reason)
            removed = await self.dispatch_queue.remove(plan.request_id)
            self.metrics.record_queue_event(plan.request_id, "released", reason=plan.reason)
            if removed is not None and not removed.future.done():
                removed.future.set_result(choice)
            released += 1
        return released

    def on_request_dispatch(self, request_id, choice):
        if not request_id:
            return
        record = self.request_state_store.get(request_id)
        if record is not None:
            self.request_state_store.mark_running(request_id)
            self.workflow_state_store.mark_status(request_id, "running")
        if self.feedback_enabled:
            queue_wait_s = 0.0 if record is None else float(getattr(record, "queue_wait_s", 0.0) or 0.0)
            self.feedback_model.on_dispatch(request_id, choice, queue_wait_s=queue_wait_s)

    def on_request_finish(self, request_id, choice, elapsed_s, status_code=None):
        self._finish_observation(getattr(choice, "_observation_context", None),
                                 "completed" if status_code is not None and 200 <= status_code < 300 else "failed")
        if not request_id:
            return
        observation = None
        if self.feedback_enabled:
            observation = self.feedback_model.on_finish(request_id, choice, elapsed_s, status_code=status_code)
            if observation is not None:
                self.metrics.record_feedback_observation(request_id, observation)
        record = self.request_state_store.get(request_id)
        if record is not None:
            self.request_state_store.mark_finished(request_id)
            self.request_state_store.remove_request(request_id)
        self.workflow_state_store.mark_finished(request_id)
        self.workflow_state_store.remove_request(request_id)
        if self.soft_admission_enabled or self.queue_enabled:
            self.soft_admission_planner.notify_completion()

    async def collect_shadow_headroom(self):
        if self.shadow is None or not self.kv_project_enabled:
            return None
        self.shadow.validate_runtime(self.registry.snapshot(), self.routing_policy)
        items = await self.dispatch_queue.items()
        records = self.request_state_store.records()
        queues = self.queue_manager.snapshot()
        lanes = self.shadow.catalog["initial_bank"]
        if set(queues) != set(lanes):
            return None
        unbound = {item.request_id for item in items if item.state in ("pending", "blocked")}
        planned = dict.fromkeys(lanes, 0)
        running = dict.fromkeys(lanes, 0)
        for record in records:
            if record.status in ("pending", "blocked"):
                unbound.add(record.request_id)
            elif record.status in ("planned", "released", "running"):
                if record.selected_preset_id not in lanes:
                    return None
                counts = running if record.status == "running" else planned
                counts[record.selected_preset_id] += 1
                unbound.discard(record.request_id)
            elif record.status not in ("finished", "rejected", "cancelled", "timed_out"):
                return None
        backlog = len(unbound)
        for lane in lanes:
            # The KV store owns epoch freshness. Never feed it monotonic seconds.
            snapshot = self.kv_projection_store.get_snapshot(lane)
            if snapshot is None or snapshot.engine_id != lane:
                return None
            values = (queues[lane].get("in_flight"), snapshot.active_requests, snapshot.npu_blocks,
                      snapshot.npu_used_blocks, snapshot.npu_free_blocks, snapshot.host_used_blocks)
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                return None
            if (not getattr(snapshot, "host_used_blocks_available", False)
                    or not math.isfinite(snapshot.ts_s) or snapshot.ts_s > time.time()
                    or snapshot.npu_blocks <= 0 or snapshot.host_used_blocks > 0
                    or snapshot.npu_used_blocks + snapshot.npu_free_blocks != snapshot.npu_blocks):
                return None
            # Planned reservations are not yet backend work. Running sources overlap.
            backlog += planned[lane] + max(running[lane], queues[lane]["in_flight"], snapshot.active_requests)
        return {"fresh": True, "at_s": self.shadow.clock(), "healthy_lane_ids": list(lanes),
                "backlog_requests": backlog}
