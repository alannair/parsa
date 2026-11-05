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
import time


class QueuedDispatchRequest:

    __slots__ = (
        "request_id",
        "body",
        "request_features",
        "request_options",
        "record",
        "future",
        "arrival_s",
        "expires_at_s",
        "state",
        "reason",
        "blocked_on",
        "observation_context",
    )

    def __init__(self, request_id, body, request_features, request_options, record, future, timeout_s=None, observation_context=None):
        self.request_id = str(request_id)
        self.body = dict(body or {})
        self.request_features = request_features
        self.request_options = dict(request_options or {})
        self.record = record
        self.future = future
        self.arrival_s = time.time()
        timeout_value = None if timeout_s is None else max(0.0, float(timeout_s))
        self.expires_at_s = None if timeout_value is None else self.arrival_s + timeout_value
        self.state = "pending"
        self.reason = ""
        self.blocked_on = []
        self.observation_context = observation_context

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "arrival_s": self.arrival_s,
            "expires_at_s": self.expires_at_s,
            "state": self.state,
            "reason": self.reason,
            "blocked_on": list(self.blocked_on),
            "slo_class": self.request_options.get("slo_class", "default"),
        }


class DispatchQueue:

    __slots__ = ("_items", "_order", "_lock")

    def __init__(self):
        self._items = {}
        self._order = []
        self._lock = asyncio.Lock()

    async def enqueue(self, item):
        async with self._lock:
            existing = self._items.get(item.request_id)
            if existing is not None:
                return existing
            self._items[item.request_id] = item
            self._order.append(item.request_id)
            return item

    async def get(self, request_id):
        async with self._lock:
            return self._items.get(str(request_id))

    async def remove(self, request_id):
        async with self._lock:
            key = str(request_id)
            item = self._items.pop(key, None)
            if item is not None:
                self._order = [entry for entry in self._order if entry != key]
            return item

    async def set_state(self, request_id, state, reason="", blocked_on=None):
        async with self._lock:
            item = self._items.get(str(request_id))
            if item is None:
                return None
            item.state = str(state or item.state)
            item.reason = str(reason or "")
            item.blocked_on = list(blocked_on or [])
            return item

    async def items(self, states=None, limit=None):
        async with self._lock:
            allowed = None if states is None else set(states)
            result = []
            for request_id in list(self._order):
                item = self._items.get(request_id)
                if item is None:
                    continue
                if allowed is not None and item.state not in allowed:
                    continue
                result.append(item)
                if limit is not None and len(result) >= int(limit):
                    break
            return result

    async def pending_items(self, limit=None):
        return await self.items(states=("pending",), limit=limit)

    async def blocked_items(self, limit=None):
        return await self.items(states=("blocked",), limit=limit)

    async def timed_out_items(self, now_s=None):
        now_value = time.time() if now_s is None else float(now_s)
        async with self._lock:
            result = []
            for request_id in list(self._order):
                item = self._items.get(request_id)
                if item is None:
                    continue
                if item.state not in ("pending", "blocked"):
                    continue
                if item.expires_at_s is not None and now_value >= float(item.expires_at_s):
                    result.append(item)
            return result

    async def snapshot(self):
        async with self._lock:
            items = [item.to_dict() for item in self._items.values()]
        counts = {}
        for item in items:
            state = item.get("state", "pending")
            counts[state] = counts.get(state, 0) + 1
        return {
            "counts": counts,
            "size": len(items),
            "items": sorted(items, key=lambda item: (item.get("arrival_s", 0.0), item.get("request_id", ""))),
        }
