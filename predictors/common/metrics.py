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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def as_1d_float(values):
    return np.asarray(values, dtype=np.float64).ravel()


def reg_metrics(y_true, y_pred):
    y_true = as_1d_float(y_true)
    y_pred = as_1d_float(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def prediction_association_metrics(y_true, y_pred):
    from scipy.stats import kendalltau, pearsonr, spearmanr

    y_true = as_1d_float(y_true)
    y_pred = as_1d_float(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; got %s vs %s"
            % (y_true.shape, y_pred.shape)
        )
    n = y_true.size
    if n < 2:
        raise ValueError("Need at least two samples for correlation metrics.")

    pr, pr_p = pearsonr(y_true, y_pred)
    sr, sr_p = spearmanr(y_true, y_pred)
    kt, kt_p = kendalltau(y_true, y_pred)

    under = y_pred < y_true
    n_under = int(np.sum(under))
    underprediction_rate = float(np.mean(under))
    if n_under > 0:
        mean_underprediction = float(np.mean(y_true[under] - y_pred[under]))
    else:
        mean_underprediction = float("nan")

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pr_p),
        "spearman_rho": float(sr),
        "spearman_p": float(sr_p),
        "kendall_tau": float(kt),
        "kendall_p": float(kt_p),
        "underprediction_rate": underprediction_rate,
        "n_underprediction": n_under,
        "n": n,
        "mean_underprediction_shortfall": mean_underprediction,
    }


def y_true_y_pred_from_train_result(result, series="auto"):
    if series == "auto":
        if "y_true" in result and "y_pred" in result:
            return result["y_true"], result["y_pred"]
        if "y_test" in result and "y_pred" in result:
            series = "seq_pred"
        elif "energy_true" in result and "energy_pred" in result:
            series = "energy"
        elif "perf_true" in result and "perf_pred" in result:
            series = "perf"
        else:
            raise KeyError(
                "Could not infer series from result keys; expected y_true/y_pred, "
                "y_test/y_pred, energy_true/energy_pred, or perf_true/perf_pred."
            )
    if series == "seq_pred":
        return result["y_test"], result["y_pred"]
    if series == "energy":
        return result["energy_true"], result["energy_pred"]
    if series == "perf":
        return result["perf_true"], result["perf_pred"]
    raise ValueError("series must be 'auto', 'seq_pred', 'energy', or 'perf', got %r" % series)


def print_pearson_spearman_kendall_underpred(y_true, y_pred, label=""):
    metrics = prediction_association_metrics(y_true, y_pred)
    prefix = "%s — " % label if label else ""
    print("%sPearson r = %.6f (p = %.4g)" % (prefix, metrics["pearson_r"], metrics["pearson_p"]))
    print("%sSpearman rho = %.6f (p = %.4g)" % (prefix, metrics["spearman_rho"], metrics["spearman_p"]))
    print("%sKendall tau = %.6f (p = %.4g)" % (prefix, metrics["kendall_tau"], metrics["kendall_p"]))
    print(
        "%sUnderprediction: rate = %.4f (%d/%d), mean shortfall when under = %.6f"
        % (
            prefix,
            metrics["underprediction_rate"],
            metrics["n_underprediction"],
            metrics["n"],
            metrics["mean_underprediction_shortfall"],
        )
    )
    return metrics
