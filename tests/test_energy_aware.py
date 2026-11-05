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

import unittest

from src.controller import EnergyAwareController, SLO
from src.power_monitor import StaticPowerMonitor
from src.presets import Preset
from src.profile_table import DictProfileTable, LinearProfile


def make_presets():
    return [
        Preset("lane_p", "http://p", 2048, max_num_seqs=1, lane_type="prefill"),
        Preset("lane_balanced", "http://b", 4096, max_num_seqs=1, lane_type="generic"),
        Preset("lane_d", "http://d", 8192, max_num_seqs=1, lane_type="decode"),
        Preset("rescue", "http://r", 16384, max_num_seqs=8, lane_type="rescue"),
    ]


def make_profile_table(mapping=None):
    if mapping is None:
        mapping = {
            "lane_p": {
                "a_ttft": 0.20,
                "b_ttft": 0.004,
                "a_tbt": 0.020,
                "b_tbt": 0.0003,
                "a_ep": 6.0,
                "b_ep": 0.030,
                "a_ed": 0.25,
                "b_ed": 0.0010,
            },
            "lane_balanced": {
                "a_ttft": 0.35,
                "b_ttft": 0.002,
                "a_tbt": 0.012,
                "b_tbt": 0.00015,
                "a_ep": 5.0,
                "b_ep": 0.025,
                "a_ed": 0.16,
                "b_ed": 0.0008,
            },
            "lane_d": {
                "a_ttft": 0.60,
                "b_ttft": 0.001,
                "a_tbt": 0.005,
                "b_tbt": 0.00005,
                "a_ep": 4.0,
                "b_ep": 0.020,
                "a_ed": 0.08,
                "b_ed": 0.0003,
            },
            "rescue": {
                "a_ttft": 0.90,
                "b_ttft": 0.001,
                "a_tbt": 0.015,
                "b_tbt": 0.0001,
                "a_ep": 4.5,
                "b_ep": 0.018,
                "a_ed": 0.10,
                "b_ed": 0.0004,
            },
        }
    return DictProfileTable.from_mapping(mapping)


def make_controller(
    routing_policy="legacy_ratio",
    routing_kwargs=None,
    power_monitor=None,
    rescue_preset_id="rescue",
    profile=None,
    presets=None,
    slo_max_latency_s=100.0,
):
    return EnergyAwareController(
        presets=make_presets() if presets is None else presets,
        profile=make_profile_table() if profile is None else profile,
        slo=SLO(slo_max_latency_s),
        rescue_preset_id=rescue_preset_id,
        routing_policy=routing_policy,
        routing_kwargs=routing_kwargs,
        power_monitor=power_monitor,
    )


class EnergyAwareControllerTests(unittest.TestCase):
    def test_legacy_ratio_generalizes_to_multiple_lanes(self):
        controller = make_controller(routing_policy="legacy_ratio")
        short_prompt = {"messages": [{"role": "user", "content": "hi there"}], "max_tokens": 400}
        long_prompt = {"messages": [{"role": "user", "content": "word " * 300}], "max_tokens": 10}
        short_choice = controller.choose(short_prompt)
        long_choice = controller.choose(long_prompt)
        self.assertEqual(short_choice.preset.preset_id, "lane_d")
        self.assertEqual(long_choice.preset.preset_id, "lane_p")

    def test_best_feasible_latency_prefers_fastest_feasible_lane(self):
        profile = make_profile_table(
            {
                "lane_p": {"a_ttft": 10.0, "a_tbt": 0.30, "a_ep": 3.0, "a_ed": 0.10},
                "lane_balanced": {"a_ttft": 1.0, "a_tbt": 0.01, "a_ep": 4.0, "a_ed": 0.10},
                "lane_d": {"a_ttft": 2.0, "a_tbt": 0.02, "a_ep": 2.5, "a_ed": 0.08},
                "rescue": {"a_ttft": 5.0, "a_tbt": 0.05, "a_ep": 2.0, "a_ed": 0.07},
            }
        )
        controller = make_controller(
            routing_policy="best_feasible_latency",
            profile=profile,
            slo_max_latency_s=6.0,
        )
        choice = controller.choose({"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 20})
        self.assertEqual(choice.preset.preset_id, "lane_balanced")

    def test_best_feasible_energy_prefers_lowest_energy_lane(self):
        profile = make_profile_table(
            {
                "lane_p": {"a_ttft": 0.6, "a_tbt": 0.02, "a_ep": 6.0, "a_ed": 0.20},
                "lane_balanced": {"a_ttft": 0.8, "a_tbt": 0.02, "a_ep": 4.0, "a_ed": 0.10},
                "lane_d": {"a_ttft": 0.9, "a_tbt": 0.03, "a_ep": 2.0, "a_ed": 0.05},
                "rescue": {"a_ttft": 1.5, "a_tbt": 0.04, "a_ep": 8.0, "a_ed": 0.30},
            }
        )
        controller = make_controller(routing_policy="best_feasible_energy", profile=profile, slo_max_latency_s=50.0)
        choice = controller.choose({"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 30})
        self.assertEqual(choice.preset.preset_id, "lane_d")

    def test_round_robin_cycles_across_more_than_two_lanes(self):
        controller = make_controller(routing_policy="round_robin")
        request = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5}
        sequence = [controller.choose(request).preset.preset_id for _ in range(4)]
        self.assertEqual(sequence, ["lane_p", "lane_balanced", "lane_d", "rescue"])

    def test_queue_balancing_avoids_loaded_lane(self):
        profile = make_profile_table(
            {
                "lane_p": {"a_ttft": 3.0, "a_tbt": 0.10, "a_ep": 4.0, "a_ed": 0.10},
                "lane_balanced": {"a_ttft": 0.3, "a_tbt": 0.01, "a_ep": 4.0, "a_ed": 0.10},
                "lane_d": {"a_ttft": 0.5, "a_tbt": 0.02, "a_ep": 4.0, "a_ed": 0.10},
                "rescue": {"a_ttft": 2.0, "a_tbt": 0.03, "a_ep": 4.0, "a_ed": 0.10},
            }
        )
        controller = make_controller(routing_policy="best_feasible_latency", profile=profile, slo_max_latency_s=20.0)
        busy = controller.queue_manager.get("lane_balanced")
        for _ in range(4):
            busy.on_start()
        busy.ewma_service_s = 3.0
        choice = controller.choose({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10})
        self.assertEqual(choice.preset.preset_id, "lane_d")

    def test_overload_uses_rescue_lane(self):
        profile = make_profile_table(
            {
                "lane_p": {"a_ttft": 2.5, "a_tbt": 0.10, "a_ep": 4.0, "a_ed": 0.10},
                "lane_balanced": {"a_ttft": 2.0, "a_tbt": 0.10, "a_ep": 4.0, "a_ed": 0.10},
                "lane_d": {"a_ttft": 2.2, "a_tbt": 0.10, "a_ep": 4.0, "a_ed": 0.10},
                "rescue": {"a_ttft": 3.0, "a_tbt": 0.20, "a_ep": 4.0, "a_ed": 0.10},
            }
        )
        controller = make_controller(
            routing_policy="best_feasible_latency",
            profile=profile,
            slo_max_latency_s=1.0,
        )
        controller.queue_manager.get("lane_p").on_start()
        controller.queue_manager.get("lane_balanced").on_start()
        controller.queue_manager.get("lane_d").on_start()
        choice = controller.choose({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10})
        self.assertEqual(choice.preset.preset_id, "rescue")
        self.assertTrue(choice.used_rescue)
        self.assertEqual(controller.suggested_priority(choice, {}), -1)

    def test_dynamic_lane_addition_and_removal(self):
        controller = make_controller(routing_policy="best_feasible_latency")
        request = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 10}
        base_choice = controller.choose(request)
        controller.profile.profiles["lane_ultra"] = LinearProfile(a_ttft=0.05, a_tbt=0.002, a_ep=3.0, a_ed=0.03)
        controller.add_preset(Preset("lane_ultra", "http://u", 6000, max_num_seqs=4, lane_type="generic"))
        added_choice = controller.choose(request)
        controller.remove_preset("lane_ultra")
        removed_choice = controller.choose(request)
        self.assertNotEqual(base_choice.preset.preset_id, "lane_ultra")
        self.assertEqual(added_choice.preset.preset_id, "lane_ultra")
        self.assertNotEqual(removed_choice.preset.preset_id, "lane_ultra")

    def test_power_threshold_prefers_lower_power_lane(self):
        presets = [
            Preset("low_power", "http://l", 4096, max_num_seqs=4, lane_type="generic"),
            Preset("high_power", "http://h", 4096, max_num_seqs=4, lane_type="generic"),
            Preset("rescue", "http://r", 8192, max_num_seqs=8, lane_type="rescue"),
        ]
        profile = DictProfileTable.from_mapping(
            {
                "low_power": {"a_ttft": 0.8, "a_tbt": 0.02, "a_ep": 1.0, "a_ed": 0.05},
                "high_power": {"a_ttft": 0.8, "a_tbt": 0.02, "a_ep": 5.0, "a_ed": 0.40},
                "rescue": {"a_ttft": 2.0, "a_tbt": 0.06, "a_ep": 2.0, "a_ed": 0.08},
            }
        )
        power_monitor = StaticPowerMonitor(cluster_power_w=95.0, budget_w=100.0)
        controller = make_controller(
            routing_policy="best_feasible_latency",
            routing_kwargs={"power_policy": {"enabled": True, "budget_w": 100.0}},
            power_monitor=power_monitor,
            presets=presets,
            profile=profile,
            slo_max_latency_s=50.0,
        )
        choice = controller.choose({"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 10})
        self.assertEqual(choice.preset.preset_id, "low_power")


if __name__ == "__main__":
    unittest.main()
