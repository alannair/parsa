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
import pandas as pd

FEATURE_COLS = ["input_size", "output_size", "batch_size", "frequency"]


def ensure_array(sample, feature_cols):
    if isinstance(sample, dict):
        return np.array([[sample[k] for k in feature_cols]], dtype=np.float32)
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    return sample


def samples_to_dataframe(samples, feature_cols):
    if isinstance(samples, pd.DataFrame):
        if set(feature_cols).issubset(samples.columns):
            return samples[feature_cols]
        return samples
    if isinstance(samples, list) and samples and isinstance(samples[0], dict):
        return pd.DataFrame([[sample[k] for k in feature_cols] for sample in samples], columns=feature_cols)
    arr = ensure_array(samples, feature_cols)
    return pd.DataFrame(arr, columns=feature_cols)
