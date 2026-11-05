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
import os
import json
import time
import unittest
from unittest.mock import patch

from src.cli import build_controller, build_parser
from src.controller import EnergyAwareController, SLO
from src.presets import Preset
from src.profile_table import DictProfileTable


def make_profile(mapping=None):
    if mapping is None:
        mapping = {
            "lane_a": {"a_ttft": 0.10, "a_tbt": 0.01, "a_ep": 2.0, "a_ed": 0.05},
            "lane_b": {"a_ttft": 0.30, "a_tbt": 0.02, "a_ep": 2.5, "a_ed": 0.06},
        }
    return DictProfileTable.from_mapping(mapping)


def make_controller(presets, profile=None, routing_kwargs=None, routing_policy="best_feasible_latency"):
    return EnergyAwareController(
        presets=presets,
        profile=make_profile() if profile is None else profile,
        slo=SLO(20.0),
        rescue_preset_id=None,
        routing_policy=routing_policy,
        routing_kwargs=routing_kwargs,
    )


def request_body(**energy_aware):
    body = {"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 10}
    if energy_aware:
        body["energy_aware"] = dict(energy_aware)
    return body


class RouterModesTests(unittest.IsolatedAsyncioTestCase):
    def test_cli_rejects_shadow_experiments_before_loading_or_prediction(self):
        for key in ("specialization_gap", "op_conflict"):
            args = build_parser().parse_args([
                "--presets", "unused.json", "--profile", "unused.json", "--slo-max-latency-s", "20",
                "--routing-kwargs", json.dumps({key: {}, "op_drift": {
                    "mode": "shadow", "catalog_path": "unused-catalog.json", "runtime_context": {"declared": "only"}}}),
            ])
            with self.subTest(key=key), patch("src.cli.load_presets_json", side_effect=AssertionError("loaded presets")), \
                    patch("src.cli.build_output_len_predictor", side_effect=AssertionError("built predictor")):
                with self.assertRaisesRegex(ValueError, "op_drift.*experiment"):
                    build_controller(args)

    async def test_off_router_object_preserves_payload_headers_and_metrics_shape(self):
        from src.router_app import EnergyAwareRouter
        controller = make_controller([Preset("lane_a", "http://a", 4096, max_num_seqs=2)])
        router = EnergyAwareRouter(controller)
        body = request_body()
        _, choice, payload = await router._apply_routing(body)
        self.assertEqual(payload, {"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 10})
        self.assertEqual(router._routing_headers(choice), {
            "X-Energy-Aware-Preset-Id": "lane_a", "X-Energy-Aware-Rescue": "0",
            "X-Energy-Aware-Routing-Policy": "best_feasible_latency", "X-Energy-Aware-Random-LB": "0",
            "X-Energy-Aware-L-In": "2.0", "X-Energy-Aware-L-Out-Predicted-Raw": "10.0",
            "X-Energy-Aware-L-Out-Effective": "12.0",
        })
        snapshot = controller.metrics.snapshot()
        self.assertEqual(set(snapshot), {"route_counts", "rejections", "completions", "admission_tiers",
            "planner_runs", "reroutes", "queue_events", "feedback_events", "events"})
        self.assertEqual(snapshot["route_counts"], {"lane_a": 1})
        self.assertEqual([event["event"] for event in snapshot["events"]], ["choice"])

    def test_cli_flags_enable_optional_modes(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--presets",
                os.path.join("src", "presets.example.json"),
                "--profile",
                os.path.join("src", "profile.example.json"),
                "--slo-max-latency-s",
                "120",
                "--queue",
                "--feedback",
                "--depon",
            ]
        )
        controller, _ = build_controller(args)
        self.assertTrue(controller.queue_enabled)
        self.assertTrue(controller.feedback_enabled)
        self.assertTrue(controller.depon_enabled)

    async def test_queue_mode_choose_or_wait_releases_request(self):
        presets = [Preset("lane_a", "http://a", 4096, max_num_seqs=2, lane_type="generic")]
        controller = make_controller(presets, routing_kwargs={"queue": {"enabled": True}})
        choice = await asyncio.wait_for(controller.choose_or_wait(request_body(), request_id="queued-1"), timeout=1.0)
        self.assertEqual(choice.request_id, "queued-1")
        self.assertEqual(choice.preset.preset_id, "lane_a")
        self.assertEqual(choice.admission_tier, "guaranteed")

    async def test_queue_mode_waits_for_capacity(self):
        presets = [Preset("lane_a", "http://a", 4096, max_num_seqs=1, lane_type="generic")]
        controller = make_controller(
            presets,
            routing_kwargs={"queue": {"enabled": True, "request_timeout_s": 1.0}},
        )
        first = await asyncio.wait_for(controller.choose_or_wait(request_body(), request_id="req-1"), timeout=1.0)
        controller.on_request_dispatch("req-1", first)
        controller.queue_manager.get("lane_a").on_start()

        second_task = asyncio.create_task(controller.choose_or_wait(request_body(), request_id="req-2"))
        await asyncio.sleep(0.05)
        self.assertFalse(second_task.done())

        controller.queue_manager.get("lane_a").on_finish(0.1)
        controller.on_request_finish("req-1", first, 0.1, status_code=200)
        await controller.plan_pending_dispatches()

        second = await asyncio.wait_for(second_task, timeout=1.0)
        self.assertEqual(second.request_id, "req-2")
        self.assertEqual(second.preset.preset_id, "lane_a")

    def test_feedback_mode_prefers_lower_ttft_lane_for_interactive_request(self):
        presets = [
            Preset("fast_ttft", "http://a", 4096, max_num_seqs=2, lane_type="generic"),
            Preset("slow_ttft", "http://b", 4096, max_num_seqs=2, lane_type="generic"),
        ]
        profile = DictProfileTable.from_mapping(
            {
                "fast_ttft": {"a_ttft": 0.10, "a_tbt": 0.08, "a_ep": 2.0, "a_ed": 0.05},
                "slow_ttft": {"a_ttft": 0.60, "a_tbt": 0.02, "a_ep": 2.0, "a_ed": 0.05},
            }
        )
        baseline = make_controller(presets, profile=profile)
        feedback = make_controller(presets, profile=profile, routing_kwargs={"feedback": {"enabled": True}})
        body = request_body(target_ttft_s=0.20)
        self.assertEqual(baseline.choose(body).preset.preset_id, "slow_ttft")
        self.assertEqual(feedback.choose(body).preset.preset_id, "fast_ttft")

    def test_feedback_mode_records_observations(self):
        presets = [Preset("lane_a", "http://a", 4096, max_num_seqs=2, lane_type="generic")]
        controller = make_controller(presets, routing_kwargs={"feedback": {"enabled": True}})
        choice = controller.choose(request_body(max_latency_s=5.0), request_id="fb-1")
        controller.on_request_dispatch("fb-1", choice)
        time.sleep(0.01)
        controller.on_request_finish("fb-1", choice, 0.2, status_code=200)
        snapshot = controller.feedback_model.snapshot()
        self.assertEqual(snapshot["lanes"]["lane_a"]["sample_count"], 1)

    async def test_depon_blocks_child_until_parent_finishes(self):
        presets = [Preset("lane_a", "http://a", 4096, max_num_seqs=2, lane_type="generic")]
        controller = make_controller(
            presets,
            routing_kwargs={
                "depon": {"enabled": True},
                "queue": {"request_timeout_s": 1.0},
            },
        )
        parent_body = request_body(workflow_id="wf-1", stage_id="root")
        parent_choice = await asyncio.wait_for(controller.choose_or_wait(parent_body, request_id="parent"), timeout=1.0)
        controller.on_request_dispatch("parent", parent_choice)
        controller.queue_manager.get("lane_a").on_start()

        child_task = asyncio.create_task(
            controller.choose_or_wait(
                request_body(
                    workflow_id="wf-1",
                    stage_id="child",
                    parent_request_ids=["parent"],
                    critical_path=True,
                    request_kind="compound",
                ),
                request_id="child",
            )
        )
        await asyncio.sleep(0.05)
        self.assertFalse(child_task.done())

        controller.queue_manager.get("lane_a").on_finish(0.1)
        controller.on_request_finish("parent", parent_choice, 0.1, status_code=200)
        await controller.plan_pending_dispatches()

        child_choice = await asyncio.wait_for(child_task, timeout=1.0)
        self.assertEqual(child_choice.request_id, "child")
        self.assertEqual(child_choice.workflow_id, "wf-1")
        self.assertTrue(child_choice.critical_path)


if __name__ == "__main__":
    unittest.main()
