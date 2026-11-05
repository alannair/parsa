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

from predictors.common.features import ensure_array, samples_to_dataframe


class BenchmarkPredictor:

    __slots__ = (
        "model",
        "feature_cols",
        "target_col",
        "model_type",
        "backend",
        "scaler",
        "layer_sizes",
        "predictor_kind",
    )

    def __init__(
        self,
        model,
        feature_cols,
        target_col,
        *,
        model_type,
        backend,
        scaler=None,
        layer_sizes=None,
        predictor_kind=None,
    ):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.target_col = str(target_col)
        self.model_type = str(model_type)
        self.backend = str(backend)
        self.scaler = scaler
        self.layer_sizes = None if layer_sizes is None else [int(x) for x in layer_sizes]
        self.predictor_kind = predictor_kind

    def predict(self, samples):
        if isinstance(samples, dict):
            samples = [samples]
        if isinstance(samples, list) and samples and isinstance(samples[0], dict):
            if self.backend == "sklearn":
                X_in = samples_to_dataframe(samples, self.feature_cols)
                preds = np.asarray(self.model.predict(X_in), dtype=np.float64).ravel()
            else:
                arr = np.array(
                    [[sample[key] for key in self.feature_cols] for sample in samples],
                    dtype=np.float32,
                )
                preds = self._predict_array(arr)
        else:
            arr = ensure_array(samples, self.feature_cols)
            preds = self._predict_array(arr)
        return [{self.target_col: float(preds[index])} for index in range(len(preds))]

    def predict_value(self, sample):
        rows = self.predict(sample)
        if not rows:
            raise ValueError("predict returned no rows")
        return float(rows[0][self.target_col])

    def _predict_array(self, arr):
        if self.backend == "sklearn":
            return np.asarray(self.model.predict(arr), dtype=np.float64).ravel()
        if self.scaler is not None:
            arr = self.scaler.transform(arr).astype(np.float32)
        if self.backend == "keras":
            return np.asarray(self.model.predict(arr, verbose=0), dtype=np.float64).ravel()
        if self.backend == "torch":
            import torch

            module = self.model
            x = torch.from_numpy(np.asarray(arr, dtype=np.float32))
            module.eval()
            with torch.no_grad():
                out = module(x).squeeze(-1).detach().cpu().numpy()
            return np.asarray(out, dtype=np.float64).ravel()
        raise ValueError("unsupported backend %r" % self.backend)
