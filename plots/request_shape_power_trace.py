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
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots.style import set_size, tick_style
from plots.dataload import trim_idle_edges


_RUN_NAME = re.compile(
    r"^profile-request-shape-(prefill-heavy|decode-heavy)-mbt\d+-ms(\d+)-f(\d+)-r(\d+)(?:-(sp-[a-f0-9]{24}))?$"
)
_SHAPES = (("prefill-heavy", "Prefill-heavy"), ("decode-heavy", "Decode-heavy"))


def _pdf_path(value):
    path = Path(value)
    if path.suffix.lower() != ".pdf":
        raise argparse.ArgumentTypeError("--output must end in .pdf")
    return path


def _power_trace(series):
    samples = [
        (float(sample["timestamp"]), sum(float(values["power_sample"]) for values in sample["per_npu"].values()))
        for sample in trim_idle_edges(series)
    ]
    start = samples[0][0]
    return [timestamp - start for timestamp, _ in samples], [power for _, power in samples]


def load_power_traces(root, frequency_mhz, max_num_seqs, serving_condition):
    selected = {shape: {} for shape, _ in _SHAPES}
    root = Path(root)
    for path in root.glob("profile-request-shape-*/result.json"):
        match = _RUN_NAME.fullmatch(path.parent.name)
        if match is None:
            continue
        shape, cap, frequency, repetition, condition = match.groups()
        condition = condition or "legacy"
        if (int(frequency), int(cap), condition) != (frequency_mhz, max_num_seqs, serving_condition):
            continue
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("request_shape") != shape or int(row.get("frequency_mhz")) != frequency_mhz or int(row.get("max_num_seqs")) != max_num_seqs:
            raise ValueError("run metadata does not match its directory: " + str(path))
        repetition = int(repetition)
        if repetition in selected[shape]:
            raise ValueError("duplicate repetition %d for %s" % (repetition, shape))
        try:
            selected[shape][repetition] = _power_trace(row.get("power_samples_series"))
        except ValueError as exc:
            raise ValueError("%s: %s" % (path, exc)) from exc
    expected = None
    for shape, _ in _SHAPES:
        repetitions = set(selected[shape])
        if not repetitions or repetitions != set(range(1, max(repetitions) + 1)):
            raise ValueError("expected contiguous repetitions 1..N for %s" % shape)
        if expected is None:
            expected = repetitions
        elif repetitions != expected:
            raise ValueError("expected matching repetitions for %s" % shape)
    return {shape: [selected[shape][repetition] for repetition in sorted(expected)] for shape, _ in _SHAPES}


def plot_power_traces(runs_root, frequency_mhz, max_num_seqs, serving_condition, version):
    traces = load_power_traces(runs_root, frequency_mhz, max_num_seqs, serving_condition)
    repetitions = len(traces[_SHAPES[0][0]])
    figure, axes = plt.subplots(repetitions, 2, sharex=True, squeeze=False,
                                figsize=(set_size(350)[0] * 2, set_size(350)[1] * repetitions))
    for row, (shape, title) in enumerate(_SHAPES):
        for repetition, (times, powers) in enumerate(traces[shape]):
            axis = axes[repetition, row]
            axis.plot(times, powers)
            axis.set_ylabel("Run %d\nPower (W)" % (repetition + 1))
            tick_style(axis)
        axes[0, row].set_title(title)
        axes[-1, row].set_xlabel("Time since first retained sample (s)")
    figure.suptitle("vLLM %s — %d MHz, max sequences %d" % (version, frequency_mhz, max_num_seqs))
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plots.request_shape_power_trace")
    parser.add_argument("--runs-root", default="experiments/_runs")
    parser.add_argument("--frequency-mhz", required=True, type=int)
    parser.add_argument("--max-num-seqs", required=True, type=int)
    parser.add_argument("--serving-condition", required=True, help="Serving-condition hash or legacy")
    parser.add_argument("--version", required=True, help="Version label for the plot title")
    parser.add_argument("--output", required=True, type=_pdf_path)
    args = parser.parse_args(argv)
    figure = plot_power_traces(args.runs_root, args.frequency_mhz, args.max_num_seqs,
                               args.serving_condition, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
