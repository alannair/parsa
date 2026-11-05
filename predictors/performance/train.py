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

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "performance"
PREDICTOR_KIND = "performance"


_PREDICTOR = BenchmarkPredictor(TARGET_COL, FEATURE_COLS, PREDICTOR_KIND)


def train_lr(df, test_size=0.2, random_state=42, model_dir=DEFAULT_MODEL_DIR):
    return _PREDICTOR.train_lr(df, test_size, random_state, model_dir)


def train_gradient_boosting(df, test_size=0.2, random_state=42, model_dir=DEFAULT_MODEL_DIR):
    return _PREDICTOR.train_gradient_boosting(df, test_size, random_state, model_dir)


def train_gb(df, test_size=0.2, random_state=42, model_dir=DEFAULT_MODEL_DIR):
    return _PREDICTOR.train_gb(df, test_size, random_state, model_dir)


def train_nn(
    df,
    layer_sizes,
    loss_function="mean_squared_error",
    optimizer="adam",
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    test_size=0.2,
    random_state=42,
    learning_rate=None,
    verbose=0,
    npu_ids=None,
    model_dir=DEFAULT_MODEL_DIR,
):
    return _PREDICTOR.train_nn(
        df, model_dir=model_dir, layer_sizes=layer_sizes, loss_function=loss_function,
        optimizer=optimizer, epochs=epochs, batch_size=batch_size,
        validation_split=validation_split, test_size=test_size, random_state=random_state,
        learning_rate=learning_rate, verbose=verbose, npu_ids=npu_ids,
    )


def train(df, model_type="lr", test_size=0.2, random_state=42, model_dir=DEFAULT_MODEL_DIR):
    return _PREDICTOR.train(df, model_type, test_size, random_state, model_dir)
