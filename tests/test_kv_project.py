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

import os
import time
import unittest

from src.backends import build_backend_adapter
from src.cli import build_controller, build_parser
from src.controller import EnergyAwareController, OverloadedError, SLO
from src.kv_projection import SCHEMA_VERSION, KVProjectionSnapshot, fetch_kv_projection_snapshot
from src.presets import Preset
from src.profile_table import DictProfileTable
from src.router_app import EnergyAwareRouter


def make_projection_payload(engine_id, peak_p95, peak_blocks_p95=None, used_blocks=20, capacity_blocks=100):
    peak_blocks_p95 = peak_blocks_p95 if peak_blocks_p95 is not None else int(round(float(peak_p95) * float(capacity_blocks)))
    return {
        "schema_version": SCHEMA_VERSION,
        "ts_s": time.time(),
        "engine_id": str(engine_id),
        "block_size_tokens": 16,
        "capacity": {"npu_blocks": int(capacity_blocks), "host_blocks": 0},
        "current": {
            "npu_used_blocks": int(used_blocks),
            "npu_free_blocks": max(0, int(capacity_blocks) - int(used_blocks)),
            "host_used_blocks": 0,
            "active_requests": 4,
        },
        "projection_basis": {"mode": "router_hints_plus_runtime", "horizon_iters": 32, "quantiles": [0.5, 0.9, 0.95]},
        "projection": {
            "p50": [{"iter_offset": 0, "batch_size": 4, "npu_kv_blocks": int(used_blocks)}],
            "p90": [{"iter_offset": 0, "batch_size": 4, "npu_kv_blocks": int(used_blocks)}],
            "p95": [{"iter_offset": 0, "batch_size": 4, "npu_kv_blocks": int(peak_blocks_p95)}],
        },
        "summary": {
            "peak_npu_blocks": {"p50": int(used_blocks), "p90": int(used_blocks), "p95": int(peak_blocks_p95)},
            "peak_utilization": {"p50": float(used_blocks) / float(capacity_blocks), "p90": float(used_blocks) / float(capacity_blocks), "p95": float(peak_p95)},
            "peak_batch_size": {"p50": 4, "p90": 4, "p95": 4},
            "time_to_below_utilization_ms": {"0.80": 0, "0.70": 150, "0.60": 300},
        },
    }


def make_profile(mapping=None):
    if mapping is None:
        mapping = {
            "lane_hot": {"a_ttft": 0.10, "a_tbt": 0.01, "a_ep": 2.0, "a_ed": 0.05},
            "lane_cool": {"a_ttft": 0.20, "a_tbt": 0.01, "a_ep": 2.5, "a_ed": 0.05},
        }
    return DictProfileTable.from_mapping(mapping)


def make_controller(presets, profile=None, routing_kwargs=None):
    return EnergyAwareController(
        presets=presets,
        profile=make_profile() if profile is None else profile,
        slo=SLO(20.0),
        rescue_preset_id=None,
        routing_policy="best_feasible_latency",
        routing_kwargs=routing_kwargs,
    )


def request_body():
    return {"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 10}


class KVProjectTests(unittest.IsolatedAsyncioTestCase):
    def test_cli_flag_enables_kv_project(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--presets",
                os.path.join("src", "presets.example.json"),
                "--profile",
                os.path.join("src", "profile.example.json"),
                "--slo-max-latency-s",
                "120",
                "--kv-project",
            ]
        )
        controller, _ = build_controller(args)
        self.assertTrue(controller.kv_project_enabled)

    def test_backend_adapter_supports_kv_projection(self):
        adapter = build_backend_adapter("vllm", "http://127.0.0.1:8001", capabilities={"kv_projection": True})
        self.assertTrue(adapter.supports_kv_projection())
        self.assertEqual(adapter.admin_kv_projection_url(), "http://127.0.0.1:8001/admin/parsa/kv-projection")

    def test_projection_parser_accepts_npu_contract(self):
        snapshot = KVProjectionSnapshot.from_payload(make_projection_payload("lane_hot", 0.75), expected_engine_id="lane_hot")
        self.assertEqual(snapshot.engine_id, "lane_hot")
        self.assertEqual(snapshot.npu_blocks, 100)
        self.assertAlmostEqual(snapshot.peak_util("p95"), 0.75)

    async def test_fetch_projection_snapshot_over_http(self):
        class _FakeResponse:
            def __init__(self, payload, status=200):
                self._payload = payload
                self.status = status

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def json(self, content_type=None):
                _ = content_type
                return dict(self._payload)

            async def text(self):
                return "ok"

        class _FakeSession:
            def __init__(self, payload):
                self._payload = payload

            def get(self, url):
                _ = url
                return _FakeResponse(self._payload)

        snapshot = await fetch_kv_projection_snapshot(
            _FakeSession(make_projection_payload("lane_hot", 0.72)),
            "http://unit-test/admin/parsa/kv-projection",
            timeout_s=0.5,
            expected_engine_id="lane_hot",
        )
        self.assertEqual(snapshot.engine_id, "lane_hot")
        self.assertAlmostEqual(snapshot.peak_util("p95"), 0.72)

    def test_controller_prefers_lane_with_lower_npu_pressure(self):
        presets = [
            Preset("lane_hot", "http://hot", 4096, max_num_seqs=2, lane_type="generic", capabilities={"kv_projection": True}),
            Preset("lane_cool", "http://cool", 4096, max_num_seqs=2, lane_type="generic", capabilities={"kv_projection": True}),
        ]
        controller = make_controller(
            presets,
            routing_kwargs={
                "kv_project": {
                    "enabled": True,
                    "fail_closed": True,
                    "peak_util_weight": 10.0,
                    "current_util_weight": 0.0,
                    "recovery_weight": 0.0,
                    "batch_weight": 0.0,
                    "reject_peak_utilization": 0.98,
                    "sim_quantile": "p95",
                }
            },
        )
        controller.kv_projection_store.update("lane_hot", KVProjectionSnapshot.from_payload(make_projection_payload("lane_hot", 0.97), expected_engine_id="lane_hot"))
        controller.kv_projection_store.update("lane_cool", KVProjectionSnapshot.from_payload(make_projection_payload("lane_cool", 0.30), expected_engine_id="lane_cool"))
        choice = controller.choose(request_body())
        self.assertEqual(choice.preset.preset_id, "lane_cool")

    def test_controller_rejects_when_projection_missing_and_fail_closed(self):
        presets = [Preset("lane_hot", "http://hot", 4096, max_num_seqs=2, lane_type="generic", capabilities={"kv_projection": True})]
        controller = make_controller(
            presets,
            routing_kwargs={"kv_project": {"enabled": True, "fail_closed": True}},
        )
        with self.assertRaises(OverloadedError):
            controller.choose(request_body())

    def test_router_backend_metadata_headers_include_projection_quantiles(self):
        presets = [Preset("lane_hot", "http://hot", 4096, max_num_seqs=2, lane_type="generic", capabilities={"kv_projection": True})]
        controller = make_controller(
            presets,
            routing_kwargs={"kv_project": {"enabled": True, "fail_closed": False}},
        )
        controller.kv_projection_store.update("lane_hot", KVProjectionSnapshot.from_payload(make_projection_payload("lane_hot", 0.30), expected_engine_id="lane_hot"))
        choice = controller.choose(request_body(), request_id="req-1")
        headers = EnergyAwareRouter(controller)._backend_metadata_headers(choice)
        self.assertEqual(headers["X-Energy-Aware-Request-Id"], "req-1")
        self.assertIn("X-Energy-Aware-Predicted-Output-Tokens-P50", headers)
        self.assertIn("X-Energy-Aware-Predicted-Output-Tokens-P90", headers)
        self.assertIn("X-Energy-Aware-Predicted-Output-Tokens-P95", headers)


if __name__ == "__main__":
    unittest.main()
