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

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from plots.request_shape_characterization import load_runs, main, plot_request_shapes, summarize_runs


def constant_power_series(watts):
    return [
        {"timestamp": timestamp, "per_npu": {"0": {"power_sample": watts, "frequency": 1300}}}
        for timestamp in (0, 2)
    ]


class RequestShapeCharacterizationTests(unittest.TestCase):
    def test_summary_uses_trimmed_measurement_window(self):
        series = [
            {"timestamp": timestamp, "per_npu": {
                "2": {"power_sample": p2, "frequency": f2},
                "3": {"power_sample": p3, "frequency": f3},
            }}
            for timestamp, p2, p3, f2, f3 in (
                (0, 100, 100, 800, 800),
                (2, 4, 6, 800, 900),
                (5, 6, 8, 800, 800),
                (9, 8, 10, 900, 800),
                (12, 100, 100, 800, 800),
            )
        ]
        frame = pd.DataFrame([{
            "request_shape": "prefill-heavy", "frequency_mhz": 900,
            "max_num_seqs": 8, "shape_valid": True, "num_requests": 2,
            "energy_joules": 999, "power_samples_series": series,
        }])

        total = summarize_runs(frame, total_energy=True)
        corrected = summarize_runs(frame, static_power_per_npu_watts=2)

        # Include both idle brackets: 210 + 36 + 64 + 327 = 637 J over 12 s.
        self.assertEqual(total.iloc[0]["energy_per_request_mean"], 318.5)
        self.assertEqual(corrected.iloc[0]["energy_per_request_mean"], 294.5)
        self.assertEqual(frame.iloc[0]["energy_joules"], 999)
        self.assertEqual(len(frame.iloc[0]["power_samples_series"]), 5)

    def test_single_active_sample_integrates_with_its_idle_brackets(self):
        series = [
            {"timestamp": index, "per_npu": {"0": {"power_sample": power, "frequency": frequency}}}
            for index, (power, frequency) in enumerate(((10, 800), (30, 900), (10, 800)))
        ]
        frame = pd.DataFrame([{
            "request_shape": "prefill-heavy", "frequency_mhz": 900,
            "max_num_seqs": 8, "shape_valid": True, "num_requests": 1,
            "power_samples_series": series,
        }])
        self.assertEqual(summarize_runs(frame, total_energy=True).iloc[0]["energy_per_request_mean"], 40)
        self.assertEqual(summarize_runs(frame).iloc[0]["energy_per_request_mean"], 20)

    def test_plot_request_shapes_uses_independent_bar_axis_limits(self):
        summary = pd.DataFrame([
            {"request_shape": "prefill-heavy", "frequency_mhz": 900, "max_num_seqs": 1,
             "energy_per_request_mean": 10, "energy_per_request_std": 1},
            {"request_shape": "decode-heavy", "frequency_mhz": 900, "max_num_seqs": 1,
             "energy_per_request_mean": 20, "energy_per_request_std": 1},
        ])
        figure = plot_request_shapes(summary)
        self.assertEqual([patch.get_height() for patch in figure.axes[0].patches], [10])
        self.assertEqual([patch.get_height() for patch in figure.axes[1].patches], [20])
        self.assertEqual(figure.axes[0].get_ylim()[1], 12)
        self.assertEqual(figure.axes[1].get_ylim()[1], 24)

    def test_plot_request_shapes_adds_version_title(self):
        summary = pd.DataFrame([
            {"request_shape": "prefill-heavy", "frequency_mhz": 900, "max_num_seqs": 1,
             "energy_per_request_mean": 10, "energy_per_request_std": 1},
            {"request_shape": "decode-heavy", "frequency_mhz": 900, "max_num_seqs": 1,
             "energy_per_request_mean": 10, "energy_per_request_std": 1},
        ])

        figure = plot_request_shapes(summary, version="0.23")

        self.assertEqual(figure._suptitle.get_text(), "vLLM 0.23")

    def test_main_accepts_version_and_titles_the_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape in ("prefill-heavy", "decode-heavy"):
                run_dir = root / ("profile-request-shape-" + shape)
                run_dir.mkdir()
                row = {"request_shape": shape, "frequency_mhz": 900, "max_num_seqs": 1,
                       "energy_joules": 10, "num_requests": 2, "shape_valid": True,
                       "power_samples_series": constant_power_series(5)}
                (run_dir / "result.json").write_text(json.dumps(row), encoding="utf-8")

            self.assertEqual(main(["--runs-root", str(root), "--total-energy", "--version", "0.23",
                                   "--output", str(root / "version.png")]), 0)

        self.assertEqual(plt.gcf()._suptitle.get_text(), "vLLM 0.23")

    def test_main_uses_legacy_npu_zero_baseline_for_two_npu_runs(self):
        def samples(power_by_npu, frequency):
            return [
                {"timestamp": timestamp, "per_npu": {
                    device: {"power_sample": power, "frequency": frequency}
                    for device, power in power_by_npu.items()
                }}
                for timestamp in (0, 4)
            ]

        condition = "sp-" + "1" * 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape in ("prefill-heavy", "decode-heavy"):
                legacy = {"request_shape": shape, "frequency_mhz": 800, "max_num_seqs": 1,
                          "energy_joules": 60, "num_requests": 2, "shape_valid": True,
                          "power_samples_series": samples({"0": 5, "1": 10}, 800)}
                selected = {"request_shape": shape, "frequency_mhz": 900, "max_num_seqs": 1,
                            "energy_joules": 120, "num_requests": 2, "shape_valid": True,
                            "serving_condition": condition,
                            "power_samples_series": samples({"2": 15, "3": 15}, 900)}
                for suffix, row in (("legacy", legacy), (condition, selected)):
                    run_dir = root / ("profile-request-shape-" + shape + "-" + suffix)
                    run_dir.mkdir()
                    (run_dir / "result.json").write_text(json.dumps(row), encoding="utf-8")

            self.assertEqual(main(["--runs-root", str(root), "--serving-condition", condition,
                                   "--output", str(root / "dynamic.png")]), 0)

        self.assertEqual([patch.get_height() for patch in plt.gcf().axes[0].patches], [40])
        self.assertEqual([patch.get_height() for patch in plt.gcf().axes[1].patches], [40])

    def test_main_rejects_missing_legacy_static_calibration(self):
        condition = "sp-" + "1" * 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / ("profile-request-shape-prefill-heavy-" + condition)
            run_dir.mkdir()
            (run_dir / "result.json").write_text(json.dumps({
                "request_shape": "prefill-heavy", "frequency_mhz": 1300,
                "max_num_seqs": 1, "num_requests": 2, "shape_valid": True,
                "serving_condition": condition, "power_samples_series": constant_power_series(5),
            }), encoding="utf-8")
            output = root / "corrected.png"

            with self.assertRaisesRegex(ValueError, "no legacy NPU 0 power samples at 800 MHz"):
                main(["--runs-root", str(root), "--serving-condition", condition, "--output", str(output)])

            self.assertFalse(output.exists())

    def test_summarize_runs_uses_valid_repetitions_and_sample_stddev(self):
        rows = pd.DataFrame(
            [
                {"request_shape": "prefill-heavy", "frequency_mhz": 1300, "max_num_seqs": 5, "energy_joules": 10, "num_requests": 2, "shape_valid": True,
                 "power_samples_series": constant_power_series(5)},
                {"request_shape": "prefill-heavy", "frequency_mhz": 1300, "max_num_seqs": 5, "energy_joules": 14, "num_requests": 2, "shape_valid": True,
                 "power_samples_series": constant_power_series(7)},
                {"request_shape": "prefill-heavy", "frequency_mhz": 1300, "max_num_seqs": 5, "energy_joules": 99, "num_requests": 2, "shape_valid": False},
            ]
        )
        summary = summarize_runs(rows, total_energy=True)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "runs"], 2)
        self.assertEqual(summary.loc[0, "energy_per_request_mean"], 6.0)
        self.assertAlmostEqual(summary.loc[0, "energy_per_request_std"], 2**0.5)

    def test_summarize_runs_subtracts_power_measured_at_800_mhz(self):
        def samples(duration, watts, frequency):
            return [
                {"timestamp": 10, "per_npu": {"0": {"power_sample": watts / 2, "frequency": frequency}, "1": {"power_sample": watts / 2, "frequency": frequency}}},
                {"timestamp": 10 + duration, "per_npu": {"0": {"power_sample": watts / 2, "frequency": frequency}, "1": {"power_sample": watts / 2, "frequency": frequency}}},
            ]

        rows = pd.DataFrame([
            {"request_shape": "prefill-heavy", "frequency_mhz": 900, "max_num_seqs": 5,
             "energy_joules": 100, "num_requests": 2, "shape_valid": True, "power_samples_series": [
                 {"timestamp": 0, "per_npu": {
                     "0": {"power_sample": 2.5, "frequency": 800},
                     "1": {"power_sample": 2.5, "frequency": 800},
                 }},
                 *samples(10, 10, 900),
             ]},
            {"request_shape": "prefill-heavy", "frequency_mhz": 1300, "max_num_seqs": 5,
             "energy_joules": 30, "num_requests": 2, "shape_valid": True, "power_samples_series": samples(4, 7.5, 1300)},
        ])

        corrected = summarize_runs(rows)
        total = summarize_runs(rows, total_energy=True)
        corrected_1300 = corrected[corrected["frequency_mhz"] == 1300].iloc[0]
        total_1300 = total[total["frequency_mhz"] == 1300].iloc[0]
        self.assertEqual(corrected_1300["static_power_watts"], 5.0)
        self.assertEqual(corrected_1300["energy_per_request_mean"], 5.0)
        self.assertEqual(total_1300["energy_per_request_mean"], 15.0)

    def test_mixed_legacy_and_two_version_conditions_are_rejected_or_filtered(self):
        first, second = "sp-" + "1" * 24, "sp-" + "2" * 24
        base = {"request_shape": "decode-heavy", "frequency_mhz": 1300, "max_num_seqs": 5,
                "num_requests": 2, "shape_valid": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, row in (
                ("legacy", {**base, "energy_joules": 10}),
                (first, {**base, "energy_joules": 20, "serving_condition": first,
                         "serving_profile": {"versions": {"vllm": "0.11.0", "vllm_ascend": "0.11.0"}}}),
                (second, {**base, "energy_joules": 40, "serving_condition": second,
                          "serving_profile": {"versions": {"vllm": "0.23.0", "vllm_ascend": "0.23.0"}}}),
            ):
                run_dir = root / ("profile-request-shape-decode-heavy-" + name)
                run_dir.mkdir()
                row["power_samples_series"] = constant_power_series(row["energy_joules"] / 2)
                (run_dir / "result.json").write_text(json.dumps(row), encoding="utf-8")
            frame = load_runs(root)
            for mixed in (frame, frame[frame["serving_condition"].notna()]):
                with self.assertRaises(ValueError) as error:
                    summarize_runs(mixed, total_energy=True)
                self.assertIn(first, str(error.exception))
                self.assertIn(second, str(error.exception))
                self.assertIn("--serving-condition", str(error.exception))
            with self.assertRaisesRegex(ValueError, "legacy"):
                summarize_runs(frame, total_energy=True)
            for condition, energy in (("legacy", 5), (first, 10), (second, 20)):
                summary = summarize_runs(frame, total_energy=True, serving_condition=condition)
                self.assertEqual(summary.loc[0, "energy_per_request_mean"], energy)
                self.assertEqual(summary.loc[0, "runs"], 1)
            output = root / "selected.png"
            self.assertEqual(main(["--runs-root", str(root), "--total-energy", "--serving-condition", first,
                                   "--output", str(output)]), 0)
            self.assertTrue(output.is_file())

    def test_condition_guard_ignores_invalid_shapes_and_nonpositive_requests(self):
        base = {"request_shape": "decode-heavy", "frequency_mhz": 1300, "max_num_seqs": 5,
                "energy_joules": 10, "num_requests": 2, "shape_valid": True,
                "power_samples_series": constant_power_series(5)}
        frame = pd.DataFrame([
            {**base, "serving_condition": "sp-" + "1" * 24},
            {**base, "shape_valid": False, "serving_condition": "sp-" + "2" * 24},
            {**base, "num_requests": 0},
        ])
        summary = summarize_runs(frame, total_energy=True)
        self.assertEqual(summary.loc[0, "runs"], 1)
        self.assertEqual(summary.loc[0, "energy_per_request_mean"], 5)
        legacy = summarize_runs(pd.DataFrame([base]), total_energy=True, serving_condition="legacy")
        self.assertEqual(legacy.loc[0, "energy_per_request_mean"], 5)

    def test_condition_filter_keeps_static_power_samples_within_selected_condition(self):
        def samples(watts, frequency):
            return [{"timestamp": timestamp, "per_npu": {"0": {"power_sample": watts, "frequency": frequency}}}
                    for timestamp in (0, 10)]

        condition = "sp-" + "1" * 24
        base = {"request_shape": "decode-heavy", "max_num_seqs": 5, "num_requests": 2, "shape_valid": True}
        frame = pd.DataFrame([
            {**base, "frequency_mhz": 900, "energy_joules": 100, "serving_condition": condition,
             "power_samples_series": [
                 {"timestamp": -1, "per_npu": {"0": {"power_sample": 5, "frequency": 800}}},
                 *samples(10, 900),
             ]},
            {**base, "frequency_mhz": 1300, "energy_joules": 70, "serving_condition": condition,
             "power_samples_series": samples(7, 1300)},
            {**base, "frequency_mhz": 900, "energy_joules": 1000,
             "power_samples_series": [
                 {"timestamp": -1, "per_npu": {"0": {"power_sample": 90, "frequency": 800}}},
                 *samples(100, 900),
             ]},
        ])
        with self.assertRaisesRegex(ValueError, "--serving-condition"):
            summarize_runs(frame)
        summary = summarize_runs(frame, serving_condition=condition)
        row = summary[summary["frequency_mhz"] == 1300].iloc[0]
        self.assertEqual(row["static_power_watts"], 5)
        self.assertEqual(row["energy_per_request_mean"], 10)


if __name__ == "__main__":
    unittest.main()
