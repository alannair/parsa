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

from pathlib import Path

from predictors.common.benchmark import BenchmarkPredictor
from predictors.performance.common import FEATURE_COLS, TARGET_COL
from predictors.performance.train import DEFAULT_MODEL_DIR, PREDICTOR_KIND

_PREDICTOR = BenchmarkPredictor(TARGET_COL, FEATURE_COLS, PREDICTOR_KIND)


def default_bundle_dir():
    return DEFAULT_MODEL_DIR


def load_predictor(bundle_path=None):
    path = Path(bundle_path) if bundle_path is not None else default_bundle_dir()
    return _PREDICTOR.load_predictor(path)


def load_predictor_cpu_safe(bundle_path=None):
    return load_predictor(bundle_path)


def predict_throughput(sample, bundle_path=None):
    path = Path(bundle_path) if bundle_path is not None else default_bundle_dir()
    return _PREDICTOR.predict(sample, path)


def save_train_result_bundle(train_result, model_dir, extra_metrics=None):
    return _PREDICTOR.save_train_result_bundle(train_result, model_dir, extra_metrics)


__all__ = [
    "FEATURE_COLS",
    "TARGET_COL",
    "DEFAULT_MODEL_DIR",
    "default_bundle_dir",
    "load_predictor",
    "load_predictor_cpu_safe",
    "predict_throughput",
    "save_train_result_bundle",
]
