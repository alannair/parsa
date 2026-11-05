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

import pandas as pd

from predictors.common.features import FEATURE_COLS


def load_benchmark_csv(csv_path, target_col, feature_cols=None):
    feature_cols = FEATURE_COLS if feature_cols is None else feature_cols
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError("benchmark CSV not found: %s" % path)

    df = pd.read_csv(path, encoding="utf-8")
    required = set(feature_cols + [target_col])
    missing = required - set(df.columns)
    if missing:
        raise KeyError("benchmark CSV missing columns: %s" % sorted(missing))
    if df.empty:
        raise ValueError("benchmark CSV is empty: %s" % path)

    out = df[list(feature_cols) + [target_col]].copy()
    for column in feature_cols + [target_col]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=list(feature_cols) + [target_col]).reset_index(drop=True)
    if out.empty:
        raise ValueError("benchmark CSV has no usable rows after numeric coercion: %s" % path)
    return out
