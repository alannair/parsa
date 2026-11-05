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

import json
import pickle
from pathlib import Path

import joblib
import numpy as np

from predictors.common.metrics import reg_metrics
from predictors.common.predictor import BenchmarkPredictor

PKL_NAME = "predictor.pkl"
INFERENCE_CONFIG_NAME = "inference_config.json"
BUNDLE_META_NAME = "bundle_meta.json"
SKLEARN_MODEL_NAME = "model.joblib"
SCALER_NAME = "scaler.joblib"
KERAS_MODEL_NAME = "keras_model.keras"
TORCH_WEIGHTS_NAME = "torch_model.pth"


def _stdout(*parts):
    print(*parts, flush=True)


def resolve_bundle_dir(bundle_path):
    path = Path(bundle_path)
    if path.is_dir():
        return path
    return path.parent


def read_bundle_inference_config(bundle_dir):
    path = Path(bundle_dir) / INFERENCE_CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError("missing %s in bundle %s" % (INFERENCE_CONFIG_NAME, bundle_dir))
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("%s must contain a JSON object" % INFERENCE_CONFIG_NAME)
    return config


def read_bundle_meta(bundle_dir):
    path = Path(bundle_dir) / BUNDLE_META_NAME
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def detect_backend(train_result):
    if train_result.get("torch_model") is not None:
        return "torch"
    model = train_result.get("model")
    if model is None:
        raise ValueError("train result missing model")
    if hasattr(model, "_module"):
        return "torch"
    try:
        from tensorflow.keras.models import Sequential

        if isinstance(model, Sequential):
            return "keras"
    except ImportError:
        pass
    return "sklearn"


def _build_torch_module(layer_sizes, in_dim):
    import torch

    layers = []
    prev = int(in_dim)
    for hidden in layer_sizes:
        hidden = int(hidden)
        layers.append(torch.nn.Linear(prev, hidden))
        layers.append(torch.nn.ReLU())
        prev = hidden
    layers.append(torch.nn.Linear(prev, 1))
    module = torch.nn.Sequential(*layers)
    module.eval()
    return module


def predictor_from_train_result(train_result, *, predictor_kind=None):
    backend = detect_backend(train_result)
    feature_cols = list(train_result["feature_cols"])
    target_col = str(train_result["target_col"])
    model_type = str(train_result.get("model_type") or "gb")
    scaler = train_result.get("scaler")
    layer_sizes = train_result.get("layer_sizes")

    if backend == "torch":
        torch_model = train_result.get("torch_model")
        if torch_model is None and hasattr(train_result.get("model"), "_module"):
            torch_model = train_result["model"]._module
        if torch_model is None:
            raise ValueError("torch train result missing torch_model")
        model = torch_model.cpu().float().eval()
    elif backend == "keras":
        model = train_result["model"]
    else:
        model = train_result["model"]

    return BenchmarkPredictor(
        model,
        feature_cols,
        target_col,
        model_type=model_type,
        backend=backend,
        scaler=scaler,
        layer_sizes=layer_sizes,
        predictor_kind=predictor_kind,
    )


def save_predictor_bundle(
    train_result,
    model_dir,
    *,
    predictor_kind,
    extra_metrics=None,
):
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = predictor_from_train_result(train_result, predictor_kind=predictor_kind)
    backend = predictor.backend
    feature_cols = predictor.feature_cols
    target_col = predictor.target_col
    model_type = predictor.model_type
    scaler = predictor.scaler
    layer_sizes = predictor.layer_sizes

    if backend == "sklearn":
        joblib.dump(predictor.model, out_dir / SKLEARN_MODEL_NAME)
    elif backend == "keras":
        predictor.model.save(str(out_dir / KERAS_MODEL_NAME))
    elif backend == "torch":
        import torch

        torch.save(predictor.model.state_dict(), str(out_dir / TORCH_WEIGHTS_NAME))

    if scaler is not None:
        joblib.dump(scaler, out_dir / SCALER_NAME)

    metrics = {
        "target_col": target_col,
        "model_type": model_type,
        "backend": backend,
        "feature_cols": feature_cols,
        "predictor_kind": predictor_kind,
        "test_metrics": reg_metrics(train_result["y_true"], train_result["y_pred"]),
    }
    if train_result.get("best_epoch") is not None:
        metrics["best_epoch"] = train_result["best_epoch"]
    if train_result.get("best_val_loss") is not None:
        metrics["best_val_loss"] = float(train_result["best_val_loss"])
    if extra_metrics:
        metrics.update(extra_metrics)

    inference_config = {
        "predictor_kind": predictor_kind,
        "target_col": target_col,
        "model_type": model_type,
        "backend": backend,
        "feature_cols": feature_cols,
        "layer_sizes": layer_sizes,
        "bundle_format_version": "1",
    }
    if backend == "torch":
        inference_config["input_dim"] = len(feature_cols)

    with open(out_dir / BUNDLE_META_NAME, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with open(out_dir / INFERENCE_CONFIG_NAME, "w", encoding="utf-8") as handle:
        json.dump(inference_config, handle, ensure_ascii=False, indent=2)

    bundle = {
        "predictor": predictor,
        "model": predictor.model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_type": model_type,
        "backend": backend,
        "scaler": scaler,
        "layer_sizes": layer_sizes,
        "predictor_kind": predictor_kind,
    }
    with open(out_dir / PKL_NAME, "wb") as handle:
        pickle.dump(bundle, handle)

    _stdout("saved bundle to %s" % out_dir)
    _stdout("  %s" % PKL_NAME)
    _stdout("  %s" % INFERENCE_CONFIG_NAME)
    _stdout("  %s" % BUNDLE_META_NAME)
    if backend == "sklearn":
        _stdout("  %s" % SKLEARN_MODEL_NAME)
    elif backend == "keras":
        _stdout("  %s" % KERAS_MODEL_NAME)
    elif backend == "torch":
        _stdout("  %s" % TORCH_WEIGHTS_NAME)
    if scaler is not None:
        _stdout("  %s" % SCALER_NAME)
    return out_dir


def _load_predictor_from_pickle(bundle_dir):
    pkl_path = Path(bundle_dir) / PKL_NAME
    if not pkl_path.is_file():
        return None
    with open(pkl_path, "rb") as handle:
        bundle = pickle.load(handle)
    if isinstance(bundle, dict):
        predictor = bundle.get("predictor")
    else:
        predictor = bundle
    if predictor is None or not hasattr(predictor, "predict"):
        return None
    return predictor


def _load_from_artifacts(bundle_dir, config):
    backend = str(config.get("backend") or "sklearn")
    feature_cols = list(config.get("feature_cols") or [])
    target_col = str(config.get("target_col") or "")
    model_type = str(config.get("model_type") or "gb")
    layer_sizes = config.get("layer_sizes")
    predictor_kind = config.get("predictor_kind")
    scaler_path = Path(bundle_dir) / SCALER_NAME
    scaler = joblib.load(scaler_path) if scaler_path.is_file() else None

    if backend == "sklearn":
        model = joblib.load(Path(bundle_dir) / SKLEARN_MODEL_NAME)
    elif backend == "keras":
        from tensorflow.keras.models import load_model

        model = load_model(str(Path(bundle_dir) / KERAS_MODEL_NAME))
    elif backend == "torch":
        import torch

        layer_sizes = layer_sizes or []
        in_dim = int(config.get("input_dim") or len(feature_cols))
        model = _build_torch_module(layer_sizes, in_dim)
        state = torch.load(str(Path(bundle_dir) / TORCH_WEIGHTS_NAME), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
    else:
        raise ValueError("unsupported backend %r in %s" % (backend, INFERENCE_CONFIG_NAME))

    return BenchmarkPredictor(
        model,
        feature_cols,
        target_col,
        model_type=model_type,
        backend=backend,
        scaler=scaler,
        layer_sizes=layer_sizes,
        predictor_kind=predictor_kind,
    )


def load_predictor_cpu_safe(bundle_path):
    bundle_dir = resolve_bundle_dir(bundle_path)
    predictor = _load_predictor_from_pickle(bundle_dir)
    if predictor is not None:
        if hasattr(predictor, "model") and hasattr(predictor.model, "cpu"):
            predictor.model = predictor.model.cpu().float().eval()
        _stdout("CPU load: restored from %s" % PKL_NAME)
        return predictor

    config = read_bundle_inference_config(bundle_dir)
    predictor = _load_from_artifacts(bundle_dir, config)
    _stdout("CPU load: rebuilt predictor from bundle artifacts")
    return predictor


def maybe_save_train_result(train_result, model_dir, *, predictor_kind, extra_metrics=None):
    if model_dir is None:
        return train_result
    bundle_dir = save_predictor_bundle(
        train_result,
        model_dir,
        predictor_kind=predictor_kind,
        extra_metrics=extra_metrics,
    )
    train_result = dict(train_result)
    train_result["bundle_dir"] = str(bundle_dir)
    train_result["predictor"] = predictor_from_train_result(train_result, predictor_kind=predictor_kind)
    return train_result
