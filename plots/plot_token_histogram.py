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

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from plots.plots import plot_token_distribution_panels
from src.azure_mix import read_schedule_file


def _png_path(value):
    path = Path(value)
    if path.suffix.lower() != ".png":
        raise argparse.ArgumentTypeError("--output must end in .png")
    return path


def load_scheduled_result_tokens(schedule_path, result_csv):
    events = read_schedule_file(schedule_path)
    if events is None:
        raise FileNotFoundError("schedule does not exist: %s" % schedule_path)
    results = pd.read_csv(result_csv)
    required = {"index", "input_length", "output_length"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError("result CSV missing columns: %s" % ", ".join(sorted(missing)))
    indices = pd.to_numeric(results["index"], errors="coerce")
    if indices.isna().any() or (indices % 1 != 0).any() or (indices < 0).any():
        raise ValueError("result CSV index values must be nonnegative integers")
    results = results.assign(index=indices.astype(int))
    if results["index"].duplicated().any():
        raise ValueError("result CSV contains duplicate indices")
    if not results.empty and int(results["index"].max()) >= len(events):
        raise ValueError("result CSV index exceeds schedule length")
    type_by_pool = {"conversation": 0, "code": 1, "summarization": 2}
    rows = []
    for result in results.to_dict(orient="records"):
        pool = events[result["index"]].get("pool")
        if pool not in type_by_pool:
            raise ValueError("unknown schedule pool: %s" % pool)
        rows.append(
            {
                "ContextTokens": result["input_length"],
                "GeneratedTokens": result["output_length"],
                "type": type_by_pool[pool],
            }
        )
    frame = pd.DataFrame(rows, columns=("ContextTokens", "GeneratedTokens", "type"))
    frame["ContextTokens"] = pd.to_numeric(frame["ContextTokens"], errors="coerce")
    frame["GeneratedTokens"] = pd.to_numeric(frame["GeneratedTokens"], errors="coerce")
    return frame.dropna(subset=["ContextTokens", "GeneratedTokens"])


def build_parser():
    parser = argparse.ArgumentParser(
        prog="plots.plot_token_histogram",
        description="Write schedule-aligned actual token histograms as a PNG",
    )
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--result-csv", required=True)
    parser.add_argument("--output", required=True, type=_png_path)
    parser.add_argument("--bins", default=50, type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    df = load_scheduled_result_tokens(args.schedule, args.result_csv)
    fig, _ = plot_token_distribution_panels(df, bins=args.bins, show=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
