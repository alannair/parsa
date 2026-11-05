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
import re
from pathlib import Path

import numpy as np
import pandas as pd

from predictors.common.features import FEATURE_COLS

EXPERIMENT_JSON_PATTERN = re.compile(
    r"freq_(?P<experiment>[\w-]+)_(?P<model>[\w-]+)_ilen(?P<input_size>\d+)_olen(?P<output_size>\d+)_bs(?P<batch_size>\d+)_f(?P<frequency>\d+)_tp(?P<tp>\d+)\.json"
)

BENCHMARK_CSV_COLUMNS = list(FEATURE_COLS) + ["tp", "energy_joules", "tokens_per_second"]
BENCHMARK_CSV_OPTIONAL_COLUMNS = ["experiment", "model", "source_json"]


def generate_experiment_filename(experiment, model, batch_size, input_size, output_size, frequency, tp):
    return (
        "freq_%s_%s_ilen%s_olen%s_bs%s_f%s_tp%s.json"
        % (experiment, model, input_size, output_size, batch_size, frequency, tp)
    )


def extract_parameters_from_filename(filename):
    match = EXPERIMENT_JSON_PATTERN.match(str(filename))
    if match is None:
        raise ValueError("Filename does not match expected pattern: %s" % filename)
    return match.groupdict()


def _as_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _measure_base_power(samples, window=4):
    if not samples:
        return 0.0
    window = max(1, min(window, len(samples)))
    front_avg = np.mean([row[1] for row in samples[:window]])
    back_avg = np.mean([row[1] for row in samples[-window:]])
    return float(min(front_avg, back_avg))


def energy_from_samples(samples):
    if len(samples) < 2:
        return 0.0
    energy_joules = 0.0
    threshold = _measure_base_power(samples, window=4)
    for prev, curr in zip(samples, samples[1:]):
        dt = curr[0] - prev[0]
        if dt > 0:
            energy_joules += (prev[1] * dt) - threshold
    return energy_joules


def _energy_from_power_samples_series(value):
    if not isinstance(value, dict):
        return None
    series = value.get("0")
    if isinstance(series, list):
        return energy_from_samples(series)
    return None


def _tokens_per_second_from_payload(payload):
    direct = _as_float(payload.get("tokens_per_second"))
    if direct is not None:
        return direct
    elapsed = _as_float(payload.get("total_elapsed_time"))
    output_tokens = _as_float(payload.get("total_output_tokens"))
    if elapsed and elapsed > 0 and output_tokens is not None:
        return output_tokens / elapsed
    return None


def _energy_joules_from_payload(payload):
    direct = _as_float(payload.get("energy_joules"))
    if direct is not None:
        return direct
    return _energy_from_power_samples_series(payload.get("power_samples_series"))


def load_experiment_json_row(file_path):
    path = Path(file_path)
    params = extract_parameters_from_filename(path.name)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("expected JSON object in %s, got %s" % (path, type(payload).__name__))

    row = {
        "input_size": int(params["input_size"]),
        "output_size": int(params["output_size"]),
        "batch_size": int(params["batch_size"]),
        "frequency": int(params["frequency"]),
        "experiment": params["experiment"],
        "model": params["model"],
        "tp": int(params["tp"]),
        "energy_joules": _energy_joules_from_payload(payload),
        "tokens_per_second": _tokens_per_second_from_payload(payload),
        "source_json": str(path.resolve()),
    }
    return row


def iter_experiment_json_paths(experiments_root, *, recursive=True):
    root = Path(experiments_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("experiments root is not a directory: %s" % root)

    iterator = root.rglob("*.json") if recursive else root.glob("*.json")
    for path in sorted(iterator, key=lambda item: str(item)):
        if EXPERIMENT_JSON_PATTERN.match(path.name):
            yield path


def load_experiments_dataframe(experiments_root, *, recursive=True, skip_invalid=False):
    rows = []
    errors = []
    for path in iter_experiment_json_paths(experiments_root, recursive=recursive):
        try:
            rows.append(load_experiment_json_row(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if skip_invalid:
                errors.append("%s: %s" % (path, exc))
                continue
            raise ValueError("failed to load experiment JSON %s: %s" % (path, exc)) from exc

    if not rows:
        raise ValueError("no experiment JSON files found under %s" % experiments_root)

    df = pd.DataFrame(rows)
    required_columns = FEATURE_COLS + ["tp", "energy_joules", "tokens_per_second"]
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "experiment JSON files under %s produced no usable rows with required metrics"
            % experiments_root
        )
    if errors:
        df.attrs["load_errors"] = errors
    return df


def dump_benchmark_csv(experiments_root, output_csv, *, recursive=True, skip_invalid=False):
    df = load_experiments_dataframe(
        experiments_root,
        recursive=recursive,
        skip_invalid=skip_invalid,
    )
    columns = BENCHMARK_CSV_COLUMNS + [
        column for column in BENCHMARK_CSV_OPTIONAL_COLUMNS if column in df.columns
    ]
    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(out_path, index=False, encoding="utf-8")
    return out_path, df
