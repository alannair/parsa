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

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from plots import plot_prompt_generated_tokens, plot_request_shape_distribution, plot_token_histogram


class PlotCommandTests(unittest.TestCase):
    def test_token_histogram_joins_results_to_schedule_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            schedule = Path(directory) / "schedule.csv"
            schedule.write_text(
                "arrival_sec,type,pool,dataset_idx,prompt\n"
                "0,0,conversation,0,hello\n"
                "1,0,summarization,0,summary\n"
                "2,1,code,0,code\n",
                encoding="utf-8",
            )
            results = Path(directory) / "results.csv"
            results.write_text(
                "index,input_length,output_length\n"
                "0,10,20\n"
                "1,30,40\n"
                "2,50,60\n",
                encoding="utf-8",
            )
            frame = plot_token_histogram.load_scheduled_result_tokens(schedule, results)
        self.assertEqual(frame["type"].tolist(), [0, 2, 1])
        self.assertEqual(frame["ContextTokens"].tolist(), [10, 30, 50])
        self.assertEqual(frame["GeneratedTokens"].tolist(), [20, 40, 60])

    def test_token_histogram_writes_png(self):
        figure = Mock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "histogram.png"
            with patch.object(plot_token_histogram, "load_scheduled_result_tokens", return_value=Mock()) as load, patch.object(
                plot_token_histogram, "plot_token_distribution_panels", return_value=(figure, Mock())
            ) as plot:
                self.assertEqual(
                    plot_token_histogram.main(
                        [
                            "--schedule", "schedule.csv",
                            "--result-csv", "results.csv",
                            "--output", str(output),
                        ]
                    ),
                    0,
                )
        load.assert_called_once_with("schedule.csv", "results.csv")
        plot.assert_called_once_with(load.return_value, bins=50, show=False)
        figure.savefig.assert_called_once_with(output, bbox_inches="tight")

    def test_prompt_generated_scatter_writes_png(self):
        figure = Mock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scatter.png"
            with patch.object(plot_prompt_generated_tokens, "load_azure_conv_and_code", return_value=Mock()) as load, patch.object(
                plot_prompt_generated_tokens, "plot_token_dist", return_value=(figure, Mock())
            ) as plot:
                self.assertEqual(
                    plot_prompt_generated_tokens.main(
                        [
                            "--azure-conv-csv", "conv.csv",
                            "--azure-code-csv", "code.csv",
                            "--output", str(output),
                        ]
                    ),
                    0,
                )
        load.assert_called_once_with("conv.csv", "code.csv")
        plot.assert_called_once_with(load.return_value, show=False, show_regression_line=True)
        figure.savefig.assert_called_once_with(output, bbox_inches="tight")

    def test_request_shape_distribution_discards_nonpositive_token_counts(self):
        frame = pd.DataFrame({
            "ContextTokens": [1, 0, 10, -1],
            "GeneratedTokens": [100, 5, 0, 2],
        })
        pairs, excluded = plot_request_shape_distribution.positive_token_pairs(frame)
        self.assertEqual(excluded, 3)
        self.assertEqual(pairs[["ContextTokens", "GeneratedTokens"]].values.tolist(), [[1, 100]])

    def test_request_shape_cluster_outline_closes_hull(self):
        outline = plot_request_shape_distribution.cluster_outline([
            (0, 0), (1, 0), (1, 1), (0, 1),
        ])
        self.assertEqual(len(outline), 5)
        self.assertEqual(outline[0], outline[-1])

    def test_request_shape_smooth_outline_samples_closed_hull(self):
        outline = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        smooth = plot_request_shape_distribution.smooth_cluster_outline(outline)
        self.assertEqual(len(smooth), 128)
        self.assertAlmostEqual(smooth[0][0], smooth[-1][0])
        self.assertAlmostEqual(smooth[0][1], smooth[-1][1])

    def test_request_shape_contour_fits_two_log_space_centroids(self):
        frame = pd.DataFrame({
            "ContextTokens": [1, 2, 100, 200],
            "GeneratedTokens": [100, 200, 1, 2],
        })
        centroids = plot_request_shape_distribution.fit_log_centroids(frame)
        self.assertEqual(len(centroids), 2)
        self.assertTrue((centroids["ContextTokens"] > 0).all())
        self.assertTrue((centroids["GeneratedTokens"] > 0).all())

    def test_request_shape_distribution_writes_png(self):
        figure = Mock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figure11.png"
            with patch.object(plot_request_shape_distribution, "load_scheduled_result_tokens", return_value=Mock()) as load, patch.object(
                plot_request_shape_distribution, "plot_request_shape_distribution", return_value=(figure, Mock(), 0)
            ) as plot:
                self.assertEqual(
                    plot_request_shape_distribution.main(
                        [
                            "--schedule", "schedule.csv",
                            "--result-csv", "results.csv",
                            "--output", str(output),
                        ]
                    ),
                    0,
                )
        load.assert_called_once_with("schedule.csv", "results.csv")
        plot.assert_called_once_with(load.return_value, bins=50, centroids=None, centroid_contour=False)
        figure.savefig.assert_called_once_with(output, bbox_inches="tight")


if __name__ == "__main__":
    unittest.main()
