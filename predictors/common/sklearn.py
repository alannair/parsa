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

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SKLEARN_GB_LOSS_MSE = "squared_error"


def build_numeric_preprocessor(feature_cols):
    return ColumnTransformer([("num", StandardScaler(), feature_cols)], remainder="drop")


def build_polynomial_preprocessor(feature_cols):
    numeric_transformer = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ]
    )
    return ColumnTransformer([("num", numeric_transformer, feature_cols)], remainder="drop")


def build_gradient_boosting_regressor(random_state=42, **kwargs):
    params = {"random_state": random_state, "loss": SKLEARN_GB_LOSS_MSE}
    params.update(kwargs)
    return GradientBoostingRegressor(**params)


def build_linear_regressor(**kwargs):
    return LinearRegression(**kwargs)
