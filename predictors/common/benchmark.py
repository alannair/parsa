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

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from predictors.common.bundle_io import (
    load_predictor_cpu_safe,
    maybe_save_train_result,
    save_predictor_bundle,
)
from predictors.common.features import ensure_array, samples_to_dataframe
from predictors.common.nn import train_benchmark_nn, validate_training_frame
from predictors.common.sklearn import (
    build_gradient_boosting_regressor,
    build_linear_regressor,
    build_numeric_preprocessor,
    build_polynomial_preprocessor,
)


class BenchmarkPredictor:
    def __init__(self, target_col, feature_cols, predictor_kind):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.predictor_kind = predictor_kind

    def _result(self, model, X_test, y_test, y_pred, predict, model_type):
        y_true = np.asarray(y_test, dtype=np.float64)
        return {
            "model": model,
            "X_test": X_test,
            "y_test": y_true,
            "y_true": y_true,
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "model_type": model_type,
            "predict": predict,
        }

    def _save(self, result, model_dir):
        return maybe_save_train_result(result, model_dir, predictor_kind=self.predictor_kind)

    def train_lr(self, df, test_size, random_state, model_dir):
        validate_training_frame(df, self.target_col, feature_cols=self.feature_cols)
        X, y = df[self.feature_cols], df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = Pipeline([("pre", clone(build_numeric_preprocessor(self.feature_cols))), ("reg", build_linear_regressor())])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        def predict(samples, fitted_model=model):
            preds = np.asarray(fitted_model.predict(samples_to_dataframe(samples, self.feature_cols)), dtype=np.float64).ravel()
            return [{self.target_col: float(pred)} for pred in preds]

        return self._save(self._result(model, X_test.to_numpy(dtype=np.float32), y_test, y_pred, predict, "lr"), model_dir)

    def train_gradient_boosting(self, df, test_size, random_state, model_dir):
        validate_training_frame(df, self.target_col, feature_cols=self.feature_cols)
        X, y = df[self.feature_cols], df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = Pipeline([("pre", clone(build_polynomial_preprocessor(self.feature_cols))), ("reg", build_gradient_boosting_regressor(random_state=random_state))])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        def predict(samples, fitted_model=model):
            preds = np.asarray(fitted_model.predict(samples_to_dataframe(samples, self.feature_cols)), dtype=np.float64).ravel()
            return [{self.target_col: float(pred)} for pred in preds]

        return self._save(self._result(model, X_test.to_numpy(dtype=np.float32), y_test, y_pred, predict, "gradient_boosting"), model_dir)

    def train_gb(self, df, test_size, random_state, model_dir):
        validate_training_frame(df, self.target_col, feature_cols=self.feature_cols)
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        y = df[self.target_col].to_numpy(dtype=np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = build_gradient_boosting_regressor(random_state=random_state, max_depth=5, n_estimators=200, learning_rate=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        def predict(samples, fitted_model=model):
            if isinstance(samples, list) and samples and isinstance(samples[0], dict):
                samples = np.array([[sample[k] for k in self.feature_cols] for sample in samples], dtype=np.float32)
            else:
                samples = ensure_array(samples, self.feature_cols)
            preds = np.asarray(fitted_model.predict(samples), dtype=np.float64).ravel()
            return [{self.target_col: float(pred)} for pred in preds]

        return self._save(self._result(model, X_test, y_test, y_pred, predict, "gb"), model_dir)

    def train_nn(self, df, *, model_dir, **kwargs):
        return self._save(
            train_benchmark_nn(df, self.target_col, feature_cols=self.feature_cols, **kwargs),
            model_dir,
        )

    def train(self, df, model_type, test_size, random_state, model_dir):
        if model_type in ("lr", "linear_regression"):
            return self.train_lr(df, test_size, random_state, model_dir)
        if model_type == "gradient_boosting":
            return self.train_gradient_boosting(df, test_size, random_state, model_dir)
        if model_type in ("gb", "single_model"):
            return self.train_gb(df, test_size, random_state, model_dir)
        raise ValueError("model_type must be 'lr', 'linear_regression', 'gradient_boosting', 'gb', or 'single_model', got %r" % model_type)

    def load_predictor(self, bundle_path):
        return load_predictor_cpu_safe(bundle_path)

    def predict(self, sample, bundle_path):
        rows = self.load_predictor(bundle_path).predict(sample)
        if not rows:
            raise ValueError("predict returned no rows")
        return float(rows[0][self.target_col])

    def save_train_result_bundle(self, train_result, model_dir, extra_metrics=None):
        return save_predictor_bundle(train_result, model_dir, predictor_kind=self.predictor_kind, extra_metrics=extra_metrics)
