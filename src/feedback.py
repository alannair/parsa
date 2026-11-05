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


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


class LaneFeedbackState:

    __slots__ = (
        "ttft_scale",
        "tpot_scale",
        "service_scale",
        "queue_wait_ewma",
        "slack_hit_rate",
        "sample_count",
    )

    def __init__(self):
        self.ttft_scale = 1.0
        self.tpot_scale = 1.0
        self.service_scale = 1.0
        self.queue_wait_ewma = 0.0
        self.slack_hit_rate = 1.0
        self.sample_count = 0

    def to_dict(self):
        return {
            "ttft_scale": self.ttft_scale,
            "tpot_scale": self.tpot_scale,
            "service_scale": self.service_scale,
            "queue_wait_ewma": self.queue_wait_ewma,
            "slack_hit_rate": self.slack_hit_rate,
            "sample_count": self.sample_count,
        }


class _RequestFeedbackState:
    __slots__ = (
        "preset_id",
        "dispatch_s",
        "predicted_ttft_s",
        "predicted_tpot_s",
        "predicted_latency_s",
        "queue_wait_s",
        "first_token_s",
        "last_chunk_s",
        "gap_total_s",
        "gap_count",
        "requested_max_latency_s",
    )

    def __init__(self, choice, queue_wait_s=0.0):
        predicted = getattr(choice, "predicted", None)
        self.preset_id = choice.preset.preset_id
        self.dispatch_s = time.time()
        self.predicted_ttft_s = float(getattr(predicted, "ttft_s", 0.0) or 0.0)
        self.predicted_tpot_s = float(getattr(predicted, "tbt_s", 0.0) or 0.0)
        self.predicted_latency_s = float(getattr(choice, "predicted_latency_s", 0.0) or 0.0)
        self.queue_wait_s = float(queue_wait_s or 0.0)
        self.first_token_s = None
        self.last_chunk_s = None
        self.gap_total_s = 0.0
        self.gap_count = 0
        self.requested_max_latency_s = getattr(choice, "requested_max_latency_s", None)


class FeedbackModel:

    __slots__ = ("config", "_lock", "_lanes", "_requests")

    def __init__(self, config=None):
        self.config = dict(config or {})
        self._lock = threading.RLock()
        self._lanes = {}
        self._requests = {}

    def enabled(self):
        return bool(self.config.get("enabled"))

    def _lane_state(self, preset_id):
        lane = self._lanes.get(str(preset_id))
        if lane is None:
            lane = LaneFeedbackState()
            self._lanes[str(preset_id)] = lane
        return lane

    def quantile_for_request(self, request_options):
        request_options = dict(request_options or {})
        slo_class = str(request_options.get("slo_class") or "default")
        overrides = self.config.get("quantile_overrides") or {}
        value = overrides.get(slo_class, self.config.get("quantile_default", 0.95))
        return _clip(value, 0.5, 0.999)

    def adjust_estimate(self, preset_id, estimate):
        from .profile_table import ProfileEstimate

        with self._lock:
            lane = self._lanes.get(str(preset_id))
            if lane is None:
                return estimate
            min_samples = max(1, int(self.config.get("min_samples", 1) or 1))
            blend = min(1.0, float(lane.sample_count) / float(min_samples))
            ttft_scale = 1.0 + (float(lane.ttft_scale) - 1.0) * blend
            tpot_scale = 1.0 + (float(lane.tpot_scale) - 1.0) * blend
            return ProfileEstimate(
                ttft_s=float(estimate.ttft_s) * ttft_scale,
                tbt_s=float(estimate.tbt_s) * tpot_scale,
                e_prefill_j=estimate.e_prefill_j,
                e_decode_per_tok_j=estimate.e_decode_per_tok_j,
            )

    def on_dispatch(self, request_id, choice, queue_wait_s=0.0):
        if not self.enabled() or not request_id:
            return
        with self._lock:
            self._requests[str(request_id)] = _RequestFeedbackState(choice, queue_wait_s=queue_wait_s)

    def observe_stream_chunk(self, request_id, at_s=None):
        if not self.enabled() or not request_id or not bool(self.config.get("observe_streaming", True)):
            return None
        now_s = time.time() if at_s is None else float(at_s)
        with self._lock:
            state = self._requests.get(str(request_id))
            if state is None:
                return None
            if state.first_token_s is None:
                state.first_token_s = now_s
                state.last_chunk_s = now_s
                return {"ttft_s": max(0.0, now_s - state.dispatch_s)}
            gap_s = max(0.0, now_s - float(state.last_chunk_s or now_s))
            state.last_chunk_s = now_s
            if gap_s > 0:
                state.gap_total_s += gap_s
                state.gap_count += 1
            return {"gap_s": gap_s}

    def on_finish(self, request_id, choice, elapsed_s, status_code=None):
        if not self.enabled() or not request_id:
            return None
        with self._lock:
            state = self._requests.pop(str(request_id), None)
            if state is None:
                return None
            lane = self._lane_state(state.preset_id)

            ttft_s = None
            if state.first_token_s is not None:
                ttft_s = max(0.0, float(state.first_token_s) - float(state.dispatch_s))
            tpot_s = None
            if state.gap_count > 0:
                tpot_s = max(0.0, float(state.gap_total_s) / float(state.gap_count))
            service_s = max(0.0, float(elapsed_s or 0.0) - float(state.queue_wait_s or 0.0))

            ttft_alpha = float(self.config.get("ttft_alpha", 0.2) or 0.2)
            tpot_alpha = float(self.config.get("tpot_alpha", 0.2) or 0.2)
            service_alpha = float(self.config.get("service_rate_alpha", 0.2) or 0.2)
            slack_alpha = float(self.config.get("slack_alpha", 0.1) or 0.1)

            if ttft_s is not None and state.predicted_ttft_s > 0:
                ratio = _clip(ttft_s / state.predicted_ttft_s, 0.5, 2.5)
                lane.ttft_scale = (1.0 - ttft_alpha) * lane.ttft_scale + ttft_alpha * ratio
            if tpot_s is not None and state.predicted_tpot_s > 0:
                ratio = _clip(tpot_s / state.predicted_tpot_s, 0.5, 2.5)
                lane.tpot_scale = (1.0 - tpot_alpha) * lane.tpot_scale + tpot_alpha * ratio
            if service_s > 0 and state.predicted_latency_s > 0:
                ratio = _clip(service_s / state.predicted_latency_s, 0.5, 2.5)
                lane.service_scale = (1.0 - service_alpha) * lane.service_scale + service_alpha * ratio
            lane.queue_wait_ewma = (1.0 - service_alpha) * lane.queue_wait_ewma + service_alpha * float(
                state.queue_wait_s or 0.0
            )
            success = status_code is None or int(status_code) < 400
            if state.requested_max_latency_s is not None and success:
                hit_value = 1.0 if float(elapsed_s or 0.0) <= float(state.requested_max_latency_s) else 0.0
                lane.slack_hit_rate = (1.0 - slack_alpha) * lane.slack_hit_rate + slack_alpha * hit_value
            lane.sample_count += 1
            return {
                "preset_id": state.preset_id,
                "ttft_s": ttft_s,
                "tpot_s": tpot_s,
                "queue_wait_s": state.queue_wait_s,
                "service_s": service_s,
                "sample_count": lane.sample_count,
            }

    def snapshot(self):
        with self._lock:
            return {
                "lanes": {preset_id: lane.to_dict() for preset_id, lane in self._lanes.items()},
                "active_requests": len(self._requests),
            }
