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

import importlib
import importlib.util
import unittest
import warnings
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from plots.style import tokens_per_energy_heatmap_cmap as paper_cmap


class EnergyHeatmapTests(unittest.TestCase):
    def render(self, frame, **kwargs):
        self.assertIsNotNone(
            importlib.util.find_spec("plots.plot_energy_heatmap"),
            "The energy heatmap must live in its own plots module",
        )
        module = importlib.import_module("plots.plot_energy_heatmap")
        with patch.object(plt, "show"):
            module.plot_energy_heatmap(frame, **kwargs)
        figure = plt.gcf()
        self.addCleanup(plt.close, figure)
        figure.canvas.draw()
        return figure.axes[0]

    def test_maximum_mean_tokens_per_joule_is_bordered_with_black_missing_cells(self):
        frame = pd.DataFrame({
            "batch_size": [64, 64, 64, 128, 256],
            "frequency": [1000, 900, 900, 1000, 900],
            "energy_joules": [None, 2, 10, 2, None],
            "total_num_tokens": [10, 20, 20, 8, 10],
        })
        original = frame.copy(deep=True)
        original_bad = paper_cmap.get_bad().copy()

        axis = self.render(frame)

        colors = axis.collections[0].get_facecolors()
        self.assertEqual(colors.shape, (6, 4))
        colors = colors.reshape(3, 2, 4)
        for row, column in ((0, 1), (1, 0), (2, 0), (2, 1)):
            np.testing.assert_allclose(colors[row, column], to_rgba("black"))
        # Mean of run ratios: (20/2 + 20/10)/2 = 6, not 20/mean(2, 10).
        self.assertEqual(sorted(text.get_text() for text in axis.texts), ["4.00", "6.00"])
        self.assertEqual(len(axis.patches), 1)
        border = axis.patches[0]
        self.assertEqual(border.get_xy(), (0, 0))
        self.assertEqual((border.get_width(), border.get_height()), (1, 1))
        self.assertFalse(border.get_fill())
        np.testing.assert_allclose(border.get_edgecolor(), to_rgba("red"))
        np.testing.assert_allclose(
            axis.collections[0].cmap(np.linspace(0, 1, 17)),
            paper_cmap(np.linspace(0, 1, 17)),
        )
        np.testing.assert_array_equal(paper_cmap.get_bad(), original_bad)
        self.assertEqual(axis.get_title(), "Tokens per joule")
        self.assertEqual(axis.collections[0].colorbar.ax.get_ylabel(), "Tokens per joule")
        pd.testing.assert_frame_equal(frame, original)

    def test_all_tied_maximum_cells_are_bordered(self):
        frame = pd.DataFrame({
            "batch_size": [4, 4, 8, 8],
            "frequency": [900, 1000, 900, 1000],
            "energy_joules": [2, 4, 4, 8],
            "total_num_tokens": [20, 40, 20, 16],
        })

        axis = self.render(frame)

        self.assertEqual({border.get_xy() for border in axis.patches}, {(0, 0), (1, 0)})

    def test_highlighting_and_numeric_annotations_are_independent(self):
        frame = pd.DataFrame({"cap": [4], "freq": [900], "energy": [2], "tokens": [20]})
        for annotate, highlight in ((True, False), (False, True)):
            with self.subTest(annotate=annotate, highlight=highlight):
                axis = self.render(
                    frame, batch_col="cap", freq_col="freq", value_col="energy",
                    tokens_col="tokens", annotate=annotate, highlight_optimum=highlight,
                )
                self.assertEqual(len(axis.patches), int(highlight))
                self.assertEqual(len(axis.texts), int(annotate))

    def test_all_missing_values_render_black_without_an_optimum(self):
        frame = pd.DataFrame({
            "batch_size": [4, 8], "frequency": [900, 1000],
            "energy_joules": [None, None],
            "total_num_tokens": [20, 40],
        })

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            axis = self.render(frame)

        np.testing.assert_allclose(
            axis.collections[0].get_facecolors(),
            np.tile(to_rgba("black"), (4, 1)),
        )
        self.assertEqual(len(axis.patches), 0)

    def test_invalid_energy_or_tokens_are_missing_not_optimum_candidates(self):
        frame = pd.DataFrame({
            "batch_size": [1, 4, 8, 16, 64, 128],
            "frequency": [900] * 6,
            "energy_joules": [0, -2, np.inf, 2, 2, 2],
            "total_num_tokens": [20, 20, 20, -1, np.inf, 10],
        })

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            axis = self.render(frame)

        np.testing.assert_allclose(
            axis.collections[0].get_facecolors()[:5],
            np.tile(to_rgba("black"), (5, 1)),
        )
        self.assertEqual([text.get_text() for text in axis.texts], ["5.00"])
        self.assertEqual([border.get_xy() for border in axis.patches], [(0, 5)])

    def test_token_counts_are_required(self):
        frame = pd.DataFrame({
            "batch_size": [4], "frequency": [900], "energy_joules": [2],
        })

        with self.assertRaisesRegex(KeyError, "total_num_tokens"):
            self.render(frame)


if __name__ == "__main__":
    unittest.main()
