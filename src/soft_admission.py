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

import time

from .policies import SteeringContext


class AdmissionCandidate:

    __slots__ = ("score", "guaranteed_limit", "hard_limit")

    def __init__(self, score, guaranteed_limit, hard_limit):
        self.score = score
        self.guaranteed_limit = guaranteed_limit
        self.hard_limit = hard_limit


class AdmissionPlan:

    __slots__ = (
        "request_id",
        "tier",
        "score",
        "preferred_preset_id",
        "selected_preset_id",
        "route_attempts",
        "planner_epoch",
        "reason",
    )

    def __init__(self, request_id, tier, score, preferred_preset_id, selected_preset_id, route_attempts, planner_epoch, reason):
        self.request_id = str(request_id)
        self.tier = str(tier)
        self.score = score
        self.preferred_preset_id = str(preferred_preset_id or "")
        self.selected_preset_id = str(selected_preset_id or "")
        self.route_attempts = int(route_attempts or 0)
        self.planner_epoch = int(planner_epoch or 0)
        self.reason = str(reason or "")


class AdmissionSnapshot:

    __slots__ = ("active_guaranteed_by_lane", "active_total_by_lane", "pending_count", "best_effort_queue_length")

    def __init__(self, active_guaranteed_by_lane, active_total_by_lane, pending_count, best_effort_queue_length):
        self.active_guaranteed_by_lane = dict(active_guaranteed_by_lane or {})
        self.active_total_by_lane = dict(active_total_by_lane or {})
        self.pending_count = int(pending_count or 0)
        self.best_effort_queue_length = int(best_effort_queue_length or 0)


class SoftAdmissionPlanner:

    __slots__ = ("config", "request_state_store", "metrics", "_epoch", "_last_run_s", "_completed_since_run")

    def __init__(self, config, request_state_store, metrics=None):
        self.config = dict(config or {})
        self.request_state_store = request_state_store
        self.metrics = metrics
        self._epoch = 0
        self._last_run_s = 0.0
        self._completed_since_run = 0

    def enabled(self):
        return bool(self.config.get("enabled"))

    def notify_completion(self):
        self._completed_since_run += 1

    def snapshot(self):
        return AdmissionSnapshot(
            active_guaranteed_by_lane=self.request_state_store.counts_by_lane(
                tiers=("guaranteed",),
                statuses=("planned", "running"),
            ),
            active_total_by_lane=self.request_state_store.counts_by_lane(
                tiers=("guaranteed", "best_effort"),
                statuses=("planned", "running"),
            ),
            pending_count=len(self.request_state_store.pending_records()),
            best_effort_queue_length=self.request_state_store.best_effort_queue_length(),
        )

    def _hard_limit(self, preset, admission_config):
        max_num_seqs = getattr(preset, "max_num_seqs", None)
        if max_num_seqs is None:
            return None
        multiplier = float(admission_config.get("max_in_flight_multiplier", 1.0) or 1.0)
        return max(1, int(float(max_num_seqs) * multiplier))

    def _guaranteed_limit(self, hard_limit):
        if hard_limit is None:
            return None
        fraction = float(self.config.get("best_effort_dispatch_fraction", 0.15) or 0.0)
        fraction = min(0.95, max(0.0, fraction))
        return max(1, int(float(hard_limit) * (1.0 - fraction)))

    def _under_limit(self, count, limit):
        return True if limit is None else int(count or 0) < int(limit)

    def _sort_prefer(self, routing_policy):
        return "energy" if routing_policy == "best_feasible_energy" else "latency"

    def _policy_pick(self, scores, routing_policy, policy_engine, registry, rescue_preset_id, random_source):
        if not scores:
            return None
        context = SteeringContext(
            scored=scores,
            registry=registry,
            rescue_preset_id=rescue_preset_id,
            random_source=random_source,
        )
        selected = policy_engine.choose(routing_policy, context)
        if selected is not None:
            return selected
        return policy_engine.scheduler.pick_fallback(scores, prefer=self._sort_prefer(routing_policy))

    def _preferred_score(self, scored, routing_policy, policy_engine, registry, rescue_preset_id, random_source):
        return self._policy_pick(scored, routing_policy, policy_engine, registry, rescue_preset_id, random_source)

    def _filter_guaranteed(self, scored, snapshot, admission_config):
        result = []
        require_feasible = bool(self.config.get("admit_only_if_feasible", True))
        for score in list(scored or []):
            hard_limit = self._hard_limit(score.preset, admission_config)
            guaranteed_limit = self._guaranteed_limit(hard_limit)
            active_total = snapshot.active_total_by_lane.get(score.preset.preset_id, 0)
            active_guaranteed = snapshot.active_guaranteed_by_lane.get(score.preset.preset_id, 0)
            if not self._under_limit(active_total, hard_limit):
                continue
            if not self._under_limit(active_guaranteed, guaranteed_limit):
                continue
            if not score.admitted:
                continue
            if require_feasible and not score.feasible:
                continue
            result.append(AdmissionCandidate(score, guaranteed_limit, hard_limit))
        return result

    def _filter_best_effort(self, scored, snapshot, admission_config):
        result = []
        if not bool(self.config.get("best_effort_enabled", True)):
            return result
        if snapshot.best_effort_queue_length >= int(self.config.get("best_effort_queue_limit", 4096) or 4096):
            return result
        for score in list(scored or []):
            hard_limit = self._hard_limit(score.preset, admission_config)
            active_total = snapshot.active_total_by_lane.get(score.preset.preset_id, 0)
            if not self._under_limit(active_total, hard_limit):
                continue
            if not score.admitted:
                continue
            result.append(AdmissionCandidate(score, self._guaranteed_limit(hard_limit), hard_limit))
        return result

    def _extract_scores(self, candidates):
        return [candidate.score for candidate in list(candidates or [])]

    def _route_attempts(self, record, preferred, selected):
        route_attempts = int(record.route_attempts or 0)
        preferred_id = preferred.preset.preset_id if preferred is not None else ""
        selected_id = selected.preset.preset_id if selected is not None else ""
        if preferred_id and selected_id and preferred_id != selected_id:
            route_attempts += 1
        return route_attempts

    def _accept(self, record, tier, selected, preferred, reason):
        route_attempts = self._route_attempts(record, preferred, selected)
        if route_attempts > int(record.max_route_attempts or 0):
            return AdmissionPlan(
                request_id=record.request_id,
                tier="rejected",
                score=None,
                preferred_preset_id=preferred.preset.preset_id if preferred is not None else "",
                selected_preset_id="",
                route_attempts=route_attempts,
                planner_epoch=self._epoch,
                reason="soft_admission_route_limit",
            )
        return AdmissionPlan(
            request_id=record.request_id,
            tier=tier,
            score=selected,
            preferred_preset_id=preferred.preset.preset_id if preferred is not None else "",
            selected_preset_id=selected.preset.preset_id,
            route_attempts=route_attempts,
            planner_epoch=self._epoch,
            reason=reason,
        )

    def _should_replan(self, snapshot, now_s):
        if self._last_run_s <= 0:
            return True
        if (now_s - self._last_run_s) >= float(self.config.get("replan_interval_s", 0.25) or 0.25):
            return True
        if snapshot.pending_count >= int(self.config.get("new_request_threshold", 1) or 1):
            return True
        if self._completed_since_run >= int(self.config.get("finished_request_threshold", 1) or 1):
            return True
        return False

    def _candidate_sort_key(self, candidate, depon_enabled=False):
        record = candidate.get("record")
        scored = list(candidate.get("scored") or [])
        admitted = [item for item in scored if item.admitted]
        pool = admitted or scored
        if pool:
            best_latency_s = min(float(item.predicted_latency_s or 0.0) for item in pool)
        else:
            best_latency_s = float("inf")
        latency_limit_s = getattr(record, "requested_max_latency_s", None)
        slack_s = float("inf") if latency_limit_s is None else float(latency_limit_s) - float(best_latency_s)
        density = float(getattr(record, "request_value", 1) or 1) / max(1e-6, float(best_latency_s or 0.0))
        priority = int(getattr(record, "priority", 0) or 0)
        critical = 1 if depon_enabled and bool(getattr(record, "critical_path", False)) else 0
        return (
            -priority,
            -critical,
            slack_s,
            -density,
            float(getattr(record, "arrival_s", 0.0) or 0.0),
            str(getattr(record, "request_id", "")),
        )

    def _reserve_snapshot(self, snapshot, tier, preset_id):
        if tier == "guaranteed":
            snapshot.active_guaranteed_by_lane[preset_id] = snapshot.active_guaranteed_by_lane.get(preset_id, 0) + 1
        snapshot.active_total_by_lane[preset_id] = snapshot.active_total_by_lane.get(preset_id, 0) + 1
        if tier == "best_effort":
            snapshot.best_effort_queue_length += 1

    def plan_batch(
        self,
        candidates,
        routing_policy,
        policy_engine,
        registry,
        rescue_preset_id,
        random_source,
        admission_config,
        max_dispatch=32,
        depon_enabled=False,
        feedback_enabled=False,
    ):
        _ = feedback_enabled
        started_s = time.time()
        snapshot = self.snapshot()
        if self._should_replan(snapshot, started_s):
            self._epoch += 1
            self._last_run_s = started_s
            self._completed_since_run = 0
        dynamic_snapshot = AdmissionSnapshot(
            active_guaranteed_by_lane=snapshot.active_guaranteed_by_lane,
            active_total_by_lane=snapshot.active_total_by_lane,
            pending_count=snapshot.pending_count,
            best_effort_queue_length=snapshot.best_effort_queue_length,
        )
        ordered = sorted(list(candidates or []), key=lambda candidate: self._candidate_sort_key(candidate, depon_enabled=depon_enabled))
        plans = []
        accepted = 0
        guaranteed_count = 0
        best_effort_count = 0
        for candidate in ordered:
            record = candidate.get("record")
            scored = list(candidate.get("scored") or [])
            preferred = self._preferred_score(scored, routing_policy, policy_engine, registry, rescue_preset_id, random_source)
            if accepted >= max(1, int(max_dispatch or 1)):
                plans.append(
                    AdmissionPlan(
                        request_id=record.request_id,
                        tier="deferred",
                        score=None,
                        preferred_preset_id=preferred.preset.preset_id if preferred is not None else "",
                        selected_preset_id="",
                        route_attempts=int(record.route_attempts or 0),
                        planner_epoch=self._epoch,
                        reason="queue_batch_limit",
                    )
                )
                continue
            guaranteed = self._filter_guaranteed(scored, dynamic_snapshot, admission_config)
            selected = self._policy_pick(
                self._extract_scores(guaranteed),
                routing_policy,
                policy_engine,
                registry,
                rescue_preset_id,
                random_source,
            )
            if selected is not None:
                reason = "soft_admission_guaranteed"
                if preferred is not None and preferred.preset.preset_id != selected.preset.preset_id:
                    reason = "soft_admission_guaranteed_reroute"
                plan = self._accept(record, "guaranteed", selected, preferred, reason)
                plans.append(plan)
                if plan.tier != "rejected" and plan.score is not None:
                    self._reserve_snapshot(dynamic_snapshot, "guaranteed", plan.selected_preset_id)
                    accepted += 1
                    guaranteed_count += 1
                continue

            best_effort = self._filter_best_effort(scored, dynamic_snapshot, admission_config)
            selected = self._policy_pick(
                self._extract_scores(best_effort),
                routing_policy,
                policy_engine,
                registry,
                rescue_preset_id,
                random_source,
            )
            if selected is not None and record.allow_best_effort:
                reason = "soft_admission_best_effort"
                if preferred is not None and preferred.preset.preset_id != selected.preset.preset_id:
                    reason = "soft_admission_best_effort_reroute"
                plan = self._accept(record, "best_effort", selected, preferred, reason)
                plans.append(plan)
                if plan.tier != "rejected" and plan.score is not None:
                    self._reserve_snapshot(dynamic_snapshot, "best_effort", plan.selected_preset_id)
                    accepted += 1
                    best_effort_count += 1
                continue

            plans.append(
                AdmissionPlan(
                    request_id=record.request_id,
                    tier="deferred",
                    score=None,
                    preferred_preset_id=preferred.preset.preset_id if preferred is not None else "",
                    selected_preset_id="",
                    route_attempts=int(record.route_attempts or 0),
                    planner_epoch=self._epoch,
                    reason="soft_admission_wait",
                )
            )

        if self.metrics is not None:
            self.metrics.record_planner_run(
                planner_epoch=self._epoch,
                pending_count=len(ordered),
                guaranteed_count=guaranteed_count,
                best_effort_count=best_effort_count,
                duration_s=time.time() - started_s,
            )
        return plans

    def plan(self, record, scored, routing_policy, policy_engine, registry, rescue_preset_id, random_source, admission_config):
        started_s = time.time()
        snapshot = self.snapshot()
        if self._should_replan(snapshot, started_s):
            self._epoch += 1
            self._last_run_s = started_s
            self._completed_since_run = 0
        preferred = self._preferred_score(scored, routing_policy, policy_engine, registry, rescue_preset_id, random_source)
        guaranteed = self._filter_guaranteed(scored, snapshot, admission_config)
        selected = self._policy_pick(
            self._extract_scores(guaranteed),
            routing_policy,
            policy_engine,
            registry,
            rescue_preset_id,
            random_source,
        )
        if selected is not None:
            reason = "soft_admission_guaranteed"
            if preferred is not None and preferred.preset.preset_id != selected.preset.preset_id:
                reason = "soft_admission_guaranteed_reroute"
            plan = self._accept(record, "guaranteed", selected, preferred, reason)
        else:
            best_effort = self._filter_best_effort(scored, snapshot, admission_config)
            selected = self._policy_pick(
                self._extract_scores(best_effort),
                routing_policy,
                policy_engine,
                registry,
                rescue_preset_id,
                random_source,
            )
            if selected is not None and record.allow_best_effort:
                reason = "soft_admission_best_effort"
                if preferred is not None and preferred.preset.preset_id != selected.preset.preset_id:
                    reason = "soft_admission_best_effort_reroute"
                plan = self._accept(record, "best_effort", selected, preferred, reason)
            else:
                plan = AdmissionPlan(
                    request_id=record.request_id,
                    tier="rejected",
                    score=None,
                    preferred_preset_id=preferred.preset.preset_id if preferred is not None else "",
                    selected_preset_id="",
                    route_attempts=int(record.route_attempts or 0),
                    planner_epoch=self._epoch,
                    reason="soft_admission_no_capacity",
                )
        if self.metrics is not None:
            self.metrics.record_planner_run(
                planner_epoch=self._epoch,
                pending_count=snapshot.pending_count,
                guaranteed_count=len(guaranteed),
                best_effort_count=len(self._filter_best_effort(scored, snapshot, admission_config)),
                duration_s=time.time() - started_s,
            )
        return plan
