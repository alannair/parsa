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
import unittest

from src.cli import build_controller, build_parser
from src.controller import EnergyAwareController, OverloadedError, SLO
from src.presets import Preset
from src.profile_table import DictProfileTable


def make_profile(mapping=None):
    if mapping is None:
        mapping = {
            "lane_a": {"a_ttft": 0.10, "a_tbt": 0.01, "a_ep": 2.0, "a_ed": 0.05},
            "lane_b": {"a_ttft": 0.30, "a_tbt": 0.02, "a_ep": 2.5, "a_ed": 0.06},
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


class SoftAdmissionTests(unittest.TestCase):
    def test_cli_flag_enables_soft_admission(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--presets",
                os.path.join("src", "presets.example.json"),
                "--profile",
                os.path.join("src", "profile.example.json"),
                "--slo-max-latency-s",
                "120",
                "--enable-soft-admission",
            ]
        )
        controller, _ = build_controller(args)
        self.assertTrue(controller.soft_admission_enabled)

    def test_second_request_becomes_best_effort(self):
        presets = [Preset("lane_a", "http://a", 4096, max_num_seqs=2, lane_type="generic")]
        controller = make_controller(
            presets,
            profile=DictProfileTable.from_mapping({"lane_a": {"a_ttft": 0.10, "a_tbt": 0.01, "a_ep": 2.0, "a_ed": 0.05}}),
            routing_kwargs={"soft_admission": {"enabled": True, "best_effort_dispatch_fraction": 0.50}},
        )
        first = controller.choose(request_body(), request_id="req-1")
        controller.on_request_dispatch("req-1", first)
        controller.queue_manager.get("lane_a").on_start()
        second = controller.choose(request_body(), request_id="req-2")
        self.assertEqual(first.admission_tier, "guaranteed")
        self.assertEqual(second.admission_tier, "best_effort")
        self.assertEqual(second.preset.preset_id, "lane_a")

    def test_soft_admission_reroutes_to_alternate_lane(self):
        presets = [
            Preset("lane_a", "http://a", 4096, max_num_seqs=2, lane_type="generic"),
            Preset("lane_b", "http://b", 4096, max_num_seqs=2, lane_type="generic"),
        ]
        profile = DictProfileTable.from_mapping(
            {
                "lane_a": {"a_ttft": 0.05, "a_tbt": 0.01, "a_ep": 2.0, "a_ed": 0.05},
                "lane_b": {"a_ttft": 0.30, "a_tbt": 0.02, "a_ep": 2.5, "a_ed": 0.06},
            }
        )
        controller = make_controller(
            presets,
            profile=profile,
            routing_kwargs={"soft_admission": {"enabled": True, "best_effort_dispatch_fraction": 0.50}},
        )
        first = controller.choose(request_body(), request_id="req-1")
        controller.on_request_dispatch("req-1", first)
        controller.queue_manager.get("lane_a").on_start()
        second = controller.choose(request_body(), request_id="req-2")
        self.assertEqual(first.preset.preset_id, "lane_a")
        self.assertEqual(second.preset.preset_id, "lane_b")
        self.assertEqual(second.admission_tier, "guaranteed")
        self.assertEqual(second.route_attempts, 1)

    def test_soft_admission_can_reject_when_disabled_best_effort(self):
        presets = [Preset("lane_a", "http://a", 4096, max_num_seqs=1, lane_type="generic")]
        controller = make_controller(
            presets,
            profile=DictProfileTable.from_mapping({"lane_a": {"a_ttft": 0.10, "a_tbt": 0.01, "a_ep": 2.0, "a_ed": 0.05}}),
            routing_kwargs={"soft_admission": {"enabled": True, "best_effort_enabled": False}},
        )
        first = controller.choose(request_body(), request_id="req-1")
        controller.on_request_dispatch("req-1", first)
        controller.queue_manager.get("lane_a").on_start()
        with self.assertRaises(OverloadedError):
            controller.choose(request_body(), request_id="req-2")


if __name__ == "__main__":
    unittest.main()
