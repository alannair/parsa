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



def train_bert_point(*args, **kwargs):
    from predictors.output_length.trainers.bert_point import train_output_length_predictor

    return train_output_length_predictor(*args, **kwargs)


def train_deberta_lognormal(*args, **kwargs):
    from predictors.output_length.trainers.deberta_lognormal import train_output_length_deberta_predictor

    return train_output_length_deberta_predictor(*args, **kwargs)


def train_deberta_hybrid(*args, **kwargs):
    from predictors.output_length.trainers.deberta_hybrid import train_output_length_deberta_hybrid_predictor

    return train_output_length_deberta_hybrid_predictor(*args, **kwargs)


def train_output_length_predictor(*args, **kwargs):
    return train_bert_point(*args, **kwargs)


def train_output_length_deberta_predictor(*args, **kwargs):
    return train_deberta_lognormal(*args, **kwargs)


def train_output_length_deberta_hybrid_predictor(*args, **kwargs):
    return train_deberta_hybrid(*args, **kwargs)


def load_predictor(*args, **kwargs):
    from predictors.output_length.inference import load_inference_predictor

    return load_inference_predictor(*args, **kwargs)


def load_inference_predictor(*args, **kwargs):
    return load_predictor(*args, **kwargs)


def load_predictor_cpu_safe(*args, **kwargs):
    from predictors.output_length.inference import load_predictor_cpu_safe as _impl

    return _impl(*args, **kwargs)


def predict_output_tokens(*args, **kwargs):
    from predictors.output_length.inference import predict_output_tokens as _impl

    return _impl(*args, **kwargs)


def predict_labeled_bundle_csv(*args, **kwargs):
    from predictors.output_length.inference import predict_labeled_bundle_csv as _impl

    return _impl(*args, **kwargs)


def default_bundle_dir(*args, **kwargs):
    from predictors.output_length.inference import default_bundle_dir as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "train_bert_point",
    "train_deberta_lognormal",
    "train_deberta_hybrid",
    "train_output_length_predictor",
    "train_output_length_deberta_predictor",
    "train_output_length_deberta_hybrid_predictor",
    "load_predictor",
    "load_inference_predictor",
    "load_predictor_cpu_safe",
    "predict_output_tokens",
    "predict_labeled_bundle_csv",
    "default_bundle_dir",
]
