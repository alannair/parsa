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

from plots.dataload import trim_idle_edges
from plots.request_shape_power_trace import _power_trace, plot_power_traces


class RequestShapePowerTraceTests(unittest.TestCase):
    def test_trace_trims_idle_edges_but_keeps_internal_idle_samples(self):
        series = [
            {"timestamp": timestamp, "per_npu": {
                "2": {"power_sample": p2, "frequency": f2},
                "3": {"power_sample": p3, "frequency": f3},
            }}
            for timestamp, p2, p3, f2, f3 in (
                (-2, 100, 100, 800, 800),
                (0, 100, 100, 800, 800),
                (2, 4, 6, 800, 900),
                (5, 6, 8, 800, 800),
                (9, 8, 10, 900, 800),
                (12, 100, 100, 800, 800),
                (14, 100, 100, 800, 800),
            )
        ]
        original = json.loads(json.dumps(series))

        times, powers = _power_trace(series)

        self.assertEqual(times, [0, 2, 5, 9, 12])
        self.assertEqual(powers, [200, 10, 14, 18, 200])
        self.assertEqual(series, original)

    def test_trim_keeps_exactly_one_idle_sample_at_each_boundary(self):
        series = [
            {"timestamp": index, "per_npu": {"0": {"power_sample": 10, "frequency": frequency}}}
            for index, frequency in enumerate((800, 800, 900, 900, 800, 900, 800, 800))
        ]
        self.assertEqual(trim_idle_edges(series), series[1:7])

    def test_single_active_sample_keeps_only_available_neighbors(self):
        for frequencies, expected_powers in (
            ((800, 800, 900, 800, 800), [2, 3, 4]),
            ((900, 800, 800), [1, 2]),
            ((800, 800, 900), [2, 3]),
            ((900, 900), [1, 2]),
        ):
            with self.subTest(frequencies=frequencies):
                series = [
                    {"timestamp": index, "per_npu": {"0": {
                        "power_sample": index + 1, "frequency": frequency,
                    }}}
                    for index, frequency in enumerate(frequencies)
                ]
                times, powers = _power_trace(series)
                self.assertEqual(times, list(range(len(expected_powers))))
                self.assertEqual(powers, expected_powers)

    def test_trace_rejects_unusable_frequency_windows(self):
        valid = [
            {"timestamp": timestamp, "per_npu": {
                "0": {"power_sample": 10, "frequency": 900},
            }}
            for timestamp in (0, 1)
        ]
        cases = {"empty": [], "one sample with no neighbors": valid[:1]}
        for name, field, value in (
            ("missing frequency", "frequency", None),
            ("nonfinite frequency", "frequency", float("nan")),
            ("nonfinite power", "power_sample", float("inf")),
        ):
            series = json.loads(json.dumps(valid))
            series[0]["per_npu"]["0"][field] = value
            cases[name] = series
        for name, timestamps in (
            ("duplicate timestamp", (0, 0)),
            ("reversed timestamps", (1, 0)),
            ("nonfinite timestamp", (0, float("inf"))),
        ):
            series = json.loads(json.dumps(valid))
            for sample, timestamp in zip(series, timestamps):
                sample["timestamp"] = timestamp
            cases[name] = series
        for name, frequencies in (("all idle", (800, 800)),):
            series = json.loads(json.dumps(valid))
            for sample, frequency in zip(series, frequencies):
                sample["per_npu"]["0"]["frequency"] = frequency
            cases[name] = series
        series = json.loads(json.dumps(valid))
        series[1]["per_npu"]["1"] = {"power_sample": 10, "frequency": 900}
        cases["inconsistent devices"] = series
        cases["invalid sample outside retained boundaries"] = [
            {"timestamp": -2, "per_npu": {"0": {"power_sample": float("nan"), "frequency": 800}}},
            {"timestamp": -1, "per_npu": {"0": {"power_sample": 10, "frequency": 800}}},
            *valid,
        ]

        for name, series in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                _power_trace(series)

    def test_plots_available_matching_repetitions(self):
        condition = "sp-" + "2" * 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape in ("prefill-heavy", "decode-heavy"):
                for repetition in range(1, 4):
                    run = root / ("profile-request-shape-%s-mbt16384-ms5-f900-r%d-%s" % (shape, repetition, condition))
                    run.mkdir()
                    (run / "result.json").write_text(json.dumps({
                        "request_shape": shape,
                        "frequency_mhz": 900,
                        "max_num_seqs": 5,
                        "power_samples_series": [
                            {"timestamp": 10, "per_npu": {"0": {"power_sample": 1, "frequency": 900}}},
                            {"timestamp": 12, "per_npu": {"0": {"power_sample": 1, "frequency": 900}}},
                        ],
                    }), encoding="utf-8")

            figure = plot_power_traces(root, 900, 5, condition, "0.23")

        self.assertEqual(len(figure.axes), 6)
        self.assertEqual(figure.axes[0].get_title(), "Prefill-heavy")
        self.assertEqual(figure.axes[1].get_title(), "Decode-heavy")

    def test_plots_five_runs_per_shape_with_shared_time_axis(self):
        condition = "sp-" + "1" * 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for shape in ("prefill-heavy", "decode-heavy"):
                for repetition in range(1, 6):
                    run = root / ("profile-request-shape-%s-mbt16384-ms5-f900-r%d-%s" % (shape, repetition, condition))
                    run.mkdir()
                    samples = [
                        {"timestamp": timestamp, "per_npu": {
                            "0": {"power_sample": repetition, "frequency": 900},
                            "1": {"power_sample": 2, "frequency": 900},
                        }}
                        for timestamp in (10, 12)
                    ]
                    (run / "result.json").write_text(json.dumps({
                        "request_shape": shape,
                        "frequency_mhz": 900,
                        "max_num_seqs": 5,
                        "power_samples_series": samples,
                    }), encoding="utf-8")

            figure = plot_power_traces(root, 900, 5, condition, "0.23")

        self.assertEqual(len(figure.axes), 10)
        self.assertTrue(figure.axes[0].get_shared_x_axes().joined(figure.axes[0], figure.axes[-1]))
        self.assertEqual(list(figure.axes[0].lines[0].get_xdata()), [0, 2])
        self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [3, 3])
        self.assertEqual(figure.axes[0].get_title(), "Prefill-heavy")
        self.assertEqual(figure.axes[1].get_title(), "Decode-heavy")


if __name__ == "__main__":
    unittest.main()
