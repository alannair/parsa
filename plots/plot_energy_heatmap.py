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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from plots.style import (
    CONCURRENT_REQUESTS_LABEL, set_size, tokens_per_energy_heatmap_cmap,
)


def plot_energy_heatmap(
    df, batch_col="batch_size", freq_col="frequency", value_col="energy_joules",
    title=None, cmap=None, annotate=True, height=None, highlight_optimum=True,
    tokens_col="total_num_tokens",
):
    """Plot mean per-run tokens/joule; ``value_col`` contains each run's joules.

    Missing or invalid measurements are black; red borders mark all maxima.
    """
    required = {batch_col, freq_col, value_col, tokens_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame missing columns: {', '.join(sorted(missing))}")
    df_numeric = df.copy()
    for column in required:
        df_numeric[column] = pd.to_numeric(
            df_numeric[column], errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.dropna(subset=[batch_col, freq_col])
    if df_numeric.empty:
        raise ValueError("No numeric capacity/frequency coordinates to plot")
    df_numeric["tokens_per_joule"] = (
        df_numeric[tokens_col].where(df_numeric[tokens_col] >= 0)
        / df_numeric[value_col].where(df_numeric[value_col] > 0)
    ).replace([np.inf, -np.inf], np.nan)
    pivot = df_numeric.pivot_table(
        index=batch_col, columns=freq_col, values="tokens_per_joule", aggfunc="mean",
    ).reindex(
        index=sorted(df_numeric[batch_col].unique()),
        columns=sorted(df_numeric[freq_col].unique()),
    )
    cmap = (tokens_per_energy_heatmap_cmap if cmap is None else cmap).with_extremes(bad="black")
    values = pivot.to_numpy(dtype=float, na_value=np.nan)
    has_data = np.isfinite(values).any()
    fig, ax = plt.subplots(figsize=set_size(350, height=height))
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, linewidths=0.5, linecolor="white",
        annot=annotate, fmt=".2f" if annotate else None,
        vmin=None if has_data else 0, vmax=None if has_data else 1,
        cbar_kws={"label": "Tokens per joule"},
    )
    if highlight_optimum and has_data:
        for row, column in np.argwhere(values == np.nanmax(values)):
            ax.add_patch(Rectangle(
                (column, row), 1, 1, fill=False, edgecolor="red", linewidth=2,
            ))
    ax.invert_yaxis()
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(CONCURRENT_REQUESTS_LABEL)
    ax.set_title(title or "Tokens per joule")
    fig.tight_layout()
    plt.show()
