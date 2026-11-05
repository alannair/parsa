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

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

CONCURRENT_REQUESTS_LABEL = "Concurrent Requests"

PLOT_PALETTE_COLORS = [
    "#1c8362",
    "#ebcb8b",
    "#424855",
    "#999999",
    "#cc79a7",
    "#a3be8c",
    "#8c6bb1",
    "#e69f00",
    "#d08770",
    "#88c0d0",
]

sns.set_palette(PLOT_PALETTE_COLORS)

cmap = LinearSegmentedColormap.from_list("plot_palette", PLOT_PALETTE_COLORS)

tokens_per_energy_heatmap_cmap = LinearSegmentedColormap.from_list(
    "tokens_per_energy",
    [
        PLOT_PALETTE_COLORS[2],
        PLOT_PALETTE_COLORS[0],
        PLOT_PALETTE_COLORS[1],
    ],
)


def plot_color(index):
    return PLOT_PALETTE_COLORS[int(index) % len(PLOT_PALETTE_COLORS)]

plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=PLOT_PALETTE_COLORS),
    "font.family": "sans-serif",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "legend.labelcolor": "black",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.left": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "legend.frameon": True,
    "legend.fancybox": True,
    "legend.edgecolor": "black",
    "legend.framealpha": 0,
    "legend.borderpad": 0.6,
})


def style_legend(legend):
    if legend is None:
        return None
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_alpha(1)
    frame.set_edgecolor("black")
    frame.set_linewidth(1)
    for text in legend.get_texts():
        text.set_color("black")
    title = legend.get_title()
    if title is not None:
        title.set_color("black")
    return legend


def global_legend(ax=None, **kwargs):
    if ax is None:
        ax = kwargs.pop("ax", None) or plt.gca()
    legend_kw = {
        "loc": kwargs.pop("loc", "upper left"),
        "bbox_to_anchor": kwargs.pop("bbox_to_anchor", (0.0, 1.28)),
        "fancybox": kwargs.pop("fancybox", True),
        "frameon": kwargs.pop("frameon", True),
        "borderpad": kwargs.pop("borderpad", 0.5),
        "framealpha": kwargs.pop("framealpha", 0),
        "edgecolor": kwargs.pop("edgecolor", "black"),
    }
    legend_kw.update(kwargs)
    return style_legend(ax.legend(**legend_kw))


def tick_style(ax=None, min_ticks=10):
    ax = ax or plt.gca()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(min_n_ticks=min_ticks))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(min_n_ticks=min_ticks))


def set_size(width, fraction=1, height=None):

    fig_width_pt = width * fraction


    inches_per_pt = 1 / 72.27



    golden_ratio = (5**.5 - 1) / 2


    fig_width_in = fig_width_pt * inches_per_pt

    if height is not None:
        fig_height_in = float(height)
    else:
        fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
