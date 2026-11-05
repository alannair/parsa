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

import threading
import time


class RequestAdmissionRecord:

    __slots__ = (
        "request_id",
        "arrival_s",
        "request_json",
        "l_in",
        "l_out_raw",
        "l_out",
        "priority",
        "request_value",
        "allow_best_effort",
        "max_route_attempts",
        "requested_max_latency_s",
        "requested_ttft_s",
        "requested_tpot_s",
        "slo_class",
        "workflow_id",
        "stage_id",
        "parent_request_ids",
        "stage_deadline_s",
        "critical_path",
        "request_kind",
        "tier",
        "status",
        "blocked_on",
        "selected_preset_id",
        "predicted_latency_s",
        "predicted_energy_j",
        "route_attempts",
        "planner_epoch",
        "admission_reason",
        "queue_wait_s",
        "planned_s",
        "released_s",
        "started_s",
        "finished_s",
    )

    def __init__(
        self,
        request_id,
        request_json,
        l_in,
        l_out_raw,
        l_out,
        priority=0,
        request_value=1,
        allow_best_effort=True,
        max_route_attempts=3,
        requested_max_latency_s=None,
        requested_ttft_s=None,
        requested_tpot_s=None,
        slo_class="default",
        workflow_id="",
        stage_id=None,
        parent_request_ids=None,
        stage_deadline_s=None,
        critical_path=False,
        request_kind="default",
        arrival_s=None,
    ):
        self.request_id = str(request_id)
        self.arrival_s = time.time() if arrival_s is None else float(arrival_s)
        self.request_json = dict(request_json or {})
        self.l_in = float(l_in or 0.0)
        self.l_out_raw = float(l_out_raw or 0.0)
        self.l_out = float(l_out or 0.0)
        self.priority = int(priority or 0)
        self.request_value = int(request_value or 1)
        self.allow_best_effort = bool(allow_best_effort)
        self.max_route_attempts = int(max_route_attempts or 0)
        self.requested_max_latency_s = None if requested_max_latency_s is None else float(requested_max_latency_s)
        self.requested_ttft_s = None if requested_ttft_s is None else float(requested_ttft_s)
        self.requested_tpot_s = None if requested_tpot_s is None else float(requested_tpot_s)
        self.slo_class = str(slo_class or "default")
        self.workflow_id = str(workflow_id or "")
        self.stage_id = None if stage_id is None else str(stage_id)
        self.parent_request_ids = [str(item) for item in list(parent_request_ids or []) if str(item or "").strip()]
        self.stage_deadline_s = None if stage_deadline_s is None else float(stage_deadline_s)
        self.critical_path = bool(critical_path)
        self.request_kind = str(request_kind or "default")
        self.tier = "pending"
        self.status = "pending"
        self.blocked_on = []
        self.selected_preset_id = ""
        self.predicted_latency_s = 0.0
        self.predicted_energy_j = 0.0
        self.route_attempts = 0
        self.planner_epoch = 0
        self.admission_reason = ""
        self.queue_wait_s = 0.0
        self.planned_s = None
        self.released_s = None
        self.started_s = None
        self.finished_s = None

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "arrival_s": self.arrival_s,
            "l_in": self.l_in,
            "l_out_raw": self.l_out_raw,
            "l_out": self.l_out,
            "priority": self.priority,
            "request_value": self.request_value,
            "allow_best_effort": self.allow_best_effort,
            "max_route_attempts": self.max_route_attempts,
            "requested_max_latency_s": self.requested_max_latency_s,
            "requested_ttft_s": self.requested_ttft_s,
            "requested_tpot_s": self.requested_tpot_s,
            "slo_class": self.slo_class,
            "workflow_id": self.workflow_id,
            "stage_id": self.stage_id,
            "parent_request_ids": list(self.parent_request_ids),
            "stage_deadline_s": self.stage_deadline_s,
            "critical_path": self.critical_path,
            "request_kind": self.request_kind,
            "tier": self.tier,
            "status": self.status,
            "blocked_on": list(self.blocked_on),
            "selected_preset_id": self.selected_preset_id,
            "predicted_latency_s": self.predicted_latency_s,
            "predicted_energy_j": self.predicted_energy_j,
            "route_attempts": self.route_attempts,
            "planner_epoch": self.planner_epoch,
            "admission_reason": self.admission_reason,
            "queue_wait_s": self.queue_wait_s,
            "planned_s": self.planned_s,
            "released_s": self.released_s,
            "started_s": self.started_s,
            "finished_s": self.finished_s,
        }


class RequestStateStore:

    __slots__ = ("_lock", "_records", "_order")

    def __init__(self):
        self._lock = threading.RLock()
        self._records = {}
        self._order = []

    def register_request(self, record):
        with self._lock:
            existing = self._records.get(record.request_id)
            if existing is not None:
                return existing
            self._records[record.request_id] = record
            self._order.append(record.request_id)
            return record

    def get(self, request_id):
        with self._lock:
            return self._records.get(str(request_id))

    def update_request(self, request_id, **changes):
        with self._lock:
            record = self._records.get(str(request_id))
            if record is None:
                return None
            for key, value in changes.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            return record

    def mark_planned(self, request_id, tier, selected_preset_id, predicted_latency_s, predicted_energy_j, route_attempts, planner_epoch, admission_reason):
        return self.update_request(
            request_id,
            tier=str(tier),
            status="planned",
            blocked_on=[],
            selected_preset_id=str(selected_preset_id or ""),
            predicted_latency_s=float(predicted_latency_s or 0.0),
            predicted_energy_j=float(predicted_energy_j or 0.0),
            route_attempts=int(route_attempts or 0),
            planner_epoch=int(planner_epoch or 0),
            admission_reason=str(admission_reason or ""),
            planned_s=time.time(),
        )

    def mark_blocked(self, request_id, blocked_on=None, admission_reason="dependency_blocked"):
        return self.update_request(
            request_id,
            status="blocked",
            blocked_on=list(blocked_on or []),
            admission_reason=str(admission_reason or "dependency_blocked"),
        )

    def mark_released(self, request_id, queue_wait_s=0.0):
        return self.update_request(
            request_id,
            status="released",
            blocked_on=[],
            queue_wait_s=float(queue_wait_s or 0.0),
            released_s=time.time(),
        )

    def mark_running(self, request_id):
        record = self.get(request_id)
        now_s = time.time()
        queue_wait_s = 0.0
        if record is not None:
            queue_wait_s = max(0.0, now_s - float(record.arrival_s or now_s))
        return self.update_request(request_id, status="running", started_s=now_s, queue_wait_s=queue_wait_s)

    def mark_finished(self, request_id, status="finished"):
        return self.update_request(request_id, status=str(status), finished_s=time.time())

    def remove_request(self, request_id):
        with self._lock:
            key = str(request_id)
            record = self._records.pop(key, None)
            if record is not None:
                self._order = [item for item in self._order if item != key]
            return record

    def _iter_records(self):
        for request_id in list(self._order):
            record = self._records.get(request_id)
            if record is not None:
                yield record

    def records(self):
        with self._lock:
            return list(self._iter_records())

    def pending_records(self, limit=None):
        with self._lock:
            records = [record for record in self._iter_records() if record.status == "pending"]
            if limit is not None:
                records = records[: int(limit)]
            return records

    def active_records(self, tiers=None, statuses=None):
        with self._lock:
            result = []
            allowed_tiers = None if tiers is None else set(tiers)
            allowed_statuses = None if statuses is None else set(statuses)
            for record in self._iter_records():
                if allowed_tiers is not None and record.tier not in allowed_tiers:
                    continue
                if allowed_statuses is not None and record.status not in allowed_statuses:
                    continue
                result.append(record)
            return result

    def counts_by_lane(self, tiers=None, statuses=None):
        counts = {}
        for record in self.active_records(tiers=tiers, statuses=statuses):
            preset_id = record.selected_preset_id
            if not preset_id:
                continue
            counts[preset_id] = counts.get(preset_id, 0) + 1
        return counts

    def best_effort_queue_length(self):
        return len(self.active_records(tiers=("best_effort",), statuses=("planned", "running", "pending")))

    def snapshot(self):
        with self._lock:
            records = [record.to_dict() for record in self._iter_records()]
        return {
            "pending": len([item for item in records if item.get("status") == "pending"]),
            "active_guaranteed": len([item for item in records if item.get("tier") == "guaranteed" and item.get("status") in ("planned", "running")]),
            "active_best_effort": len([item for item in records if item.get("tier") == "best_effort" and item.get("status") in ("planned", "running")]),
            "records": records,
        }
