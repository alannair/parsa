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



def progress_iter(iterable, *, total=None, desc=""):
    from predictors.output_length.common import get_progress_iter

    return get_progress_iter(iterable, total=total, desc=desc)


def log_hybrid_metrics(metrics, log_fn=None):
    if log_fn is None:
        from predictors.output_length.common import _stdout as log_fn

    log_fn(
        "  bucket_acc=%.4f top2=%.4f mae=%.4f rmse=%.4f r2=%.4f spearman=%.4f "
        "input_output_spearman=%.4f"
        % (
            metrics["bucket_accuracy"],
            metrics["bucket_top2_adjacent_accuracy"],
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["spearman"],
            metrics.get("input_output_length_spearman", 0.0),
        )
    )


__all__ = ["progress_iter", "log_hybrid_metrics"]
