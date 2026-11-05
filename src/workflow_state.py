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


class WorkflowRecord:

    __slots__ = (
        "workflow_id",
        "request_id",
        "stage_id",
        "parent_request_ids",
        "critical_path",
        "stage_deadline_s",
        "request_kind",
        "status",
    )

    def __init__(
        self,
        workflow_id,
        request_id,
        stage_id=None,
        parent_request_ids=None,
        critical_path=False,
        stage_deadline_s=None,
        request_kind="default",
        status="pending",
    ):
        self.workflow_id = str(workflow_id or "")
        self.request_id = str(request_id or "")
        self.stage_id = None if stage_id is None else str(stage_id)
        self.parent_request_ids = [str(item) for item in list(parent_request_ids or []) if str(item or "").strip()]
        self.critical_path = bool(critical_path)
        self.stage_deadline_s = None if stage_deadline_s is None else float(stage_deadline_s)
        self.request_kind = str(request_kind or "default")
        self.status = str(status or "pending")

    def to_dict(self):
        return {
            "workflow_id": self.workflow_id,
            "request_id": self.request_id,
            "stage_id": self.stage_id,
            "parent_request_ids": list(self.parent_request_ids),
            "critical_path": self.critical_path,
            "stage_deadline_s": self.stage_deadline_s,
            "request_kind": self.request_kind,
            "status": self.status,
        }


class WorkflowStateStore:

    __slots__ = ("_lock", "_records", "_finished")

    def __init__(self):
        self._lock = threading.RLock()
        self._records = {}
        self._finished = set()

    def register_request(self, record):
        workflow_id = str(getattr(record, "workflow_id", "") or "")
        if not workflow_id:
            return None
        workflow_record = WorkflowRecord(
            workflow_id=workflow_id,
            request_id=record.request_id,
            stage_id=getattr(record, "stage_id", None),
            parent_request_ids=getattr(record, "parent_request_ids", ()),
            critical_path=getattr(record, "critical_path", False),
            stage_deadline_s=getattr(record, "stage_deadline_s", None),
            request_kind=getattr(record, "request_kind", "default"),
            status=getattr(record, "status", "pending"),
        )
        with self._lock:
            self._records[workflow_record.request_id] = workflow_record
        return workflow_record

    def get(self, request_id):
        with self._lock:
            return self._records.get(str(request_id))

    def mark_status(self, request_id, status):
        with self._lock:
            record = self._records.get(str(request_id))
            if record is None:
                return None
            record.status = str(status or record.status)
            return record

    def mark_finished(self, request_id):
        with self._lock:
            key = str(request_id)
            self._finished.add(key)
            record = self._records.get(key)
            if record is not None:
                record.status = "finished"
            return record

    def remove_request(self, request_id):
        with self._lock:
            return self._records.pop(str(request_id), None)

    def parents_satisfied(self, request_id=None, parent_request_ids=None):
        if parent_request_ids is None:
            record = self.get(request_id)
            if record is None:
                return True
            parent_request_ids = record.parent_request_ids
        with self._lock:
            return all(str(parent_id) in self._finished for parent_id in list(parent_request_ids or []))

    def missing_parents(self, request_id=None, parent_request_ids=None):
        if parent_request_ids is None:
            record = self.get(request_id)
            if record is None:
                return []
            parent_request_ids = record.parent_request_ids
        with self._lock:
            return [str(parent_id) for parent_id in list(parent_request_ids or []) if str(parent_id) not in self._finished]

    def unblock_ready(self):
        with self._lock:
            return [
                record.request_id
                for record in self._records.values()
                if record.status in ("pending", "blocked") and self.parents_satisfied(parent_request_ids=record.parent_request_ids)
            ]

    def snapshot(self):
        with self._lock:
            return {
                "finished": sorted(self._finished),
                "records": [record.to_dict() for record in self._records.values()],
            }
