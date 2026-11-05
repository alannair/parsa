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


def train_energy(*args, **kwargs):
    from predictors.energy.train import train

    return train(*args, **kwargs)


def train_energy_lr(*args, **kwargs):
    from predictors.energy.train import train_lr

    return train_lr(*args, **kwargs)


def train_energy_gradient_boosting(*args, **kwargs):
    from predictors.energy.train import train_gradient_boosting

    return train_gradient_boosting(*args, **kwargs)


def train_energy_gb(*args, **kwargs):
    from predictors.energy.train import train_gb

    return train_gb(*args, **kwargs)


def train_energy_nn(*args, **kwargs):
    from predictors.energy.train import train_nn

    return train_nn(*args, **kwargs)


def load_predictor(*args, **kwargs):
    from predictors.energy.inference import load_predictor as _impl

    return _impl(*args, **kwargs)


def load_predictor_cpu_safe(*args, **kwargs):
    from predictors.energy.inference import load_predictor_cpu_safe as _impl

    return _impl(*args, **kwargs)


def predict_energy(*args, **kwargs):
    from predictors.energy.inference import predict_energy as _impl

    return _impl(*args, **kwargs)


def default_bundle_dir(*args, **kwargs):
    from predictors.energy.inference import default_bundle_dir as _impl

    return _impl(*args, **kwargs)


def save_train_result_bundle(*args, **kwargs):
    from predictors.energy.inference import save_train_result_bundle as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "train_energy",
    "train_energy_lr",
    "train_energy_gradient_boosting",
    "train_energy_gb",
    "train_energy_nn",
    "load_predictor",
    "load_predictor_cpu_safe",
    "predict_energy",
    "default_bundle_dir",
    "save_train_result_bundle",
]
