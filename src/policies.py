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

class AdmissionDecision:

    __slots__ = ("accepted", "reason")

    def __init__(self, accepted=True, reason=""):
        self.accepted = bool(accepted)
        self.reason = str(reason or "")


class LaneScore:

    __slots__ = (
        "preset",
        "predicted",
        "predicted_latency_s",
        "predicted_energy_j",
        "feasible",
        "admitted",
        "admission_reason",
        "queue_delay_s",
        "projected_power_w",
        "policy_score",
        "affinity_score",
        "used_rescue",
    )

    def __init__(
        self,
        preset,
        predicted,
        predicted_latency_s,
        predicted_energy_j,
        feasible,
        admitted,
        admission_reason,
        queue_delay_s,
        projected_power_w,
        policy_score,
        affinity_score,
        used_rescue=False,
    ):
        self.preset = preset
        self.predicted = predicted
        self.predicted_latency_s = float(predicted_latency_s or 0.0)
        self.predicted_energy_j = float(predicted_energy_j or 0.0)
        self.feasible = bool(feasible)
        self.admitted = bool(admitted)
        self.admission_reason = str(admission_reason or "")
        self.queue_delay_s = float(queue_delay_s or 0.0)
        self.projected_power_w = float(projected_power_w or 0.0)
        self.policy_score = float(policy_score or 0.0)
        self.affinity_score = float(affinity_score or 0.0)
        self.used_rescue = bool(used_rescue)

    def clone(self):
        return LaneScore(
            preset=self.preset,
            predicted=self.predicted,
            predicted_latency_s=self.predicted_latency_s,
            predicted_energy_j=self.predicted_energy_j,
            feasible=self.feasible,
            admitted=self.admitted,
            admission_reason=self.admission_reason,
            queue_delay_s=self.queue_delay_s,
            projected_power_w=self.projected_power_w,
            policy_score=self.policy_score,
            affinity_score=self.affinity_score,
            used_rescue=self.used_rescue,
        )


class SteeringContext:

    __slots__ = ("scored", "registry", "rescue_preset_id", "random_source")

    def __init__(self, scored, registry, rescue_preset_id, random_source):
        self.scored = list(scored or [])
        self.registry = registry
        self.rescue_preset_id = rescue_preset_id
        self.random_source = random_source


class AdmissionController:

    __slots__ = ("config",)

    def __init__(self, config=None):
        self.config = dict(config or {})

    def assess(self, preset, lane_state, queue_delay_s):
        max_num_seqs = getattr(preset, "max_num_seqs", None)
        multiplier = float(self.config.get("max_in_flight_multiplier", 1.0) or 1.0)
        if max_num_seqs is not None:
            hard_limit = max(1, int(max_num_seqs * multiplier))
            if lane_state.in_flight >= hard_limit:
                return AdmissionDecision(False, "max_in_flight")
        max_queue_delay_s = self.config.get("max_queue_delay_s")
        if max_queue_delay_s is not None and float(queue_delay_s or 0.0) > float(max_queue_delay_s):
            return AdmissionDecision(False, "queue_delay")
        return AdmissionDecision(True, "")


class QueueBalancer:

    __slots__ = ("config",)

    def __init__(self, config=None):
        self.config = dict(config or {})

    def penalty(self, preset_id, queue_snapshot):
        if not self.config.get("enabled", True):
            return 0.0
        if not queue_snapshot:
            return 0.0
        in_flights = [float(item.get("in_flight", 0) or 0.0) for item in queue_snapshot.values()]
        if not in_flights:
            return 0.0
        mean_in_flight = sum(in_flights) / max(1.0, float(len(in_flights)))
        current = queue_snapshot.get(preset_id, {})
        current_in_flight = float(current.get("in_flight", 0) or 0.0)
        current_queue_delay = float(current.get("queue_delay_s", 0.0) or 0.0)
        current_ewma = float(current.get("ewma_service_s", 0.0) or 0.0)
        threshold = 1.0 + float(self.config.get("imbalance_threshold", 0.25) or 0.0)
        overload = max(0.0, current_in_flight - mean_in_flight * threshold)
        penalty = overload * float(self.config.get("in_flight_weight", 0.15) or 0.0)
        penalty += current_queue_delay * float(self.config.get("queue_delay_weight", 1.0) or 0.0)
        penalty += current_ewma * float(self.config.get("ewma_weight", 0.25) or 0.0)
        return penalty


class SchedulingLayer:

    def _sort_key(self, prefer, item):
        if prefer == "energy":
            return (
                item.predicted_energy_j,
                item.predicted_latency_s,
                item.projected_power_w,
                item.preset.preset_id,
            )
        return (
            item.predicted_latency_s,
            item.predicted_energy_j,
            item.projected_power_w,
            item.preset.preset_id,
        )

    def _rescue_score(self, scored, rescue_preset_id=None):
        rescue = None
        if rescue_preset_id:
            for item in scored:
                if item.preset.preset_id == rescue_preset_id:
                    rescue = item
                    break
        if rescue is None:
            for item in scored:
                if getattr(item.preset, "lane_type", "") == "rescue":
                    rescue = item
                    break
        if rescue is None:
            return None
        cloned = rescue.clone()
        cloned.used_rescue = True
        cloned.admitted = True
        cloned.admission_reason = ""
        return cloned

    def pick_best(self, scored, prefer="latency", rescue_preset_id=None):
        admitted = [item for item in scored if item.admitted]
        feasible = [item for item in admitted if item.feasible]
        if feasible:
            return min(feasible, key=lambda item: self._sort_key(prefer, item))
        rescue = self._rescue_score(scored, rescue_preset_id=rescue_preset_id)
        if rescue is not None:
            return rescue
        if admitted:
            return min(admitted, key=lambda item: self._sort_key(prefer, item))
        return None

    def pick_fallback(self, scored, prefer="latency"):
        if not scored:
            return None
        return min(scored, key=lambda item: self._sort_key(prefer, item))


class AutoscalingHook:

    def on_overload(self, details):
        return details

    def on_lane_added(self, preset):
        return preset

    def on_lane_removed(self, preset_id):
        return preset_id


class PolicyEngine:

    __slots__ = ("scheduler", "_policies")

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._policies = {}
        self._register_builtin_policies()

    def register(self, name, fn):
        self._policies[str(name)] = fn

    def names(self):
        return tuple(self._policies)

    def choose(self, name, context):
        if name not in self._policies:
            raise ValueError("unknown routing policy: %s" % name)
        return self._policies[name](context)

    def _register_builtin_policies(self):
        self.register("legacy_ratio", self._legacy_ratio)
        self.register("best_feasible_latency", self._best_feasible_latency)
        self.register("best_feasible_energy", self._best_feasible_energy)
        self.register("random", self._random)
        self.register("round_robin", self._round_robin)

    def _candidate_scores(self, context):
        admitted = [item for item in context.scored if item.admitted]
        return admitted or list(context.scored)

    def _legacy_ratio(self, context):
        candidates = self._candidate_scores(context)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item.policy_score,
                item.predicted_latency_s,
                item.predicted_energy_j,
                item.preset.preset_id,
            ),
        )

    def _best_feasible_latency(self, context):
        return self.scheduler.pick_best(context.scored, prefer="latency", rescue_preset_id=context.rescue_preset_id)

    def _best_feasible_energy(self, context):
        return self.scheduler.pick_best(context.scored, prefer="energy", rescue_preset_id=context.rescue_preset_id)

    def _random(self, context):
        candidates = self._candidate_scores(context)
        if not candidates:
            return None
        return context.random_source.choice(candidates)

    def _round_robin(self, context):
        candidates = [item.preset.preset_id for item in self._candidate_scores(context)]
        preset = context.registry.next_round_robin(candidates)
        if preset is None:
            return None
        for item in context.scored:
            if item.preset.preset_id == preset.preset_id:
                return item
        return None
