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

"""Grouped minimum-feasible shared/specialized energies; error bars are sample SD.

Incomplete sweeps are marked with *; absent feasible winners are shown as missing,
never zero-energy bars. Diagonal winners reuse the same measurement in both searches.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.style import global_legend, plot_color, set_size


def load_summary(root):
    with open(Path(root) / "summary.csv", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"mixture", "search_complete", "selection_hash"} | {
            prefix + "_" + metric for prefix in ("shared", "specialized")
            for metric in ("mean_energy_j", "std_energy_j", "valid_repetitions")}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("summary.csv is missing OP-conflict aggregation fields")
        rows = list(reader)
    if not rows or len({row["mixture"] for row in rows}) != len(rows):
        raise ValueError("summary.csv must contain unique nonempty mixture rows")
    if len({row["selection_hash"] for row in rows}) != 1:
        raise ValueError("cannot plot different frozen selections together")
    for row in rows:
        for prefix in ("shared", "specialized"):
            for metric in ("mean_energy_j", "std_energy_j"):
                key = prefix + "_" + metric
                value = float(row[key]) if row[key] else None
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError("invalid energy statistic: %s" % key)
                row[key] = value
            if row[prefix + "_mean_energy_j"] is not None:
                if int(row[prefix + "_valid_repetitions"]) < 2 or row[prefix + "_std_energy_j"] is None:
                    raise ValueError("at least two repetitions are required for sample-SD error bars")
        row["search_complete"] = row["search_complete"] == "True"
    return rows


def plot_op_conflict(rows):
    fig, ax = plt.subplots(figsize=(set_size(500)[0], set_size(500)[1]))
    width = 0.36
    for offset, prefix, label, color in ((-0.5, "shared", "Best shared", plot_color(0)),
                                        (0.5, "specialized", "Best specialized (incl. shared)", plot_color(1))):
        available = [(index, row) for index, row in enumerate(rows) if row[prefix + "_mean_energy_j"] is not None]
        if available:
            ax.bar([index + offset * width for index, _ in available],
                   [row[prefix + "_mean_energy_j"] for _, row in available], width=width,
                   yerr=[row[prefix + "_std_energy_j"] for _, row in available],
                   capsize=3, color=color, label=label)
        for index, row in enumerate(rows):
            if row[prefix + "_mean_energy_j"] is None:
                ax.text(index + offset * width, 0.02, "missing", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", rotation=90, fontsize=8)
    ax.set_xticks(range(len(rows)), [row["mixture"] + ("*" if not row["search_complete"] else "") for row in rows])
    ax.set_xlabel("Prefill-heavy / decode-heavy mixture (%)")
    ax.set_ylabel("Measured two-lane energy (J)")
    ax.set_ylim(bottom=0)
    if any(not row["search_complete"] for row in rows):
        ax.set_title("* Incomplete candidate sweep; best measured feasible banks")
    for index, row in enumerate(rows):
        shared = row["shared_mean_energy_j"]
        specialized = row["specialized_mean_energy_j"]
        if shared is not None and specialized is not None:
            savings = (shared - specialized) / shared * 100 if shared else 0
            top = max(shared + (row["shared_std_energy_j"] or 0),
                      specialized + (row["specialized_std_energy_j"] or 0))
            ax.annotate("%.1f%%" % savings, (index, top), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
    if ax.get_legend_handles_labels()[0]:
        global_legend(ax, ncol=2)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plots.op_conflict", description=__doc__)
    parser.add_argument("--runs-root", default="experiments/_op-conflict")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    figure = plot_op_conflict(load_summary(args.runs_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
