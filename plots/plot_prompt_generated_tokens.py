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

matplotlib.use("Agg")

from plots.dataload import load_azure_conv_and_code
from plots.plots import plot_token_dist


def _png_path(value):
    path = Path(value)
    if path.suffix.lower() != ".png":
        raise argparse.ArgumentTypeError("--output must end in .png")
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="plots.plot_prompt_generated_tokens",
        description="Write Azure prompt-to-generated-token scatter plots as a PNG",
    )
    parser.add_argument("--azure-conv-csv", required=True)
    parser.add_argument("--azure-code-csv", required=True)
    parser.add_argument("--output", required=True, type=_png_path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    df = load_azure_conv_and_code(args.azure_conv_csv, args.azure_code_csv)
    fig, _ = plot_token_dist(df, show=False, show_regression_line=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
