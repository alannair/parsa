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

import importlib
import math
import os
import pickle
from pathlib import Path


def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
                continue
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                pieces.append(part.get("text"))
                continue
            if part.get("type") == "text" and isinstance(part.get("content"), str):
                pieces.append(part.get("content"))
        return "\n".join(piece for piece in pieces if piece)
    return ""


def extract_request_text(request_json):
    prompt = request_json.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(str(item) for item in prompt if item is not None)
    messages = request_json.get("messages") or []
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        parts.append(_message_text(message.get("content")))
    return "\n".join(part for part in parts if part)


def default_output_len_predictor(request_json):
    value = request_json.get("max_completion_tokens")
    if value is None:
        value = request_json.get("max_tokens")
    if value is None:
        return 128.0
    try:
        return float(value)
    except Exception:
        return 128.0


def resolve_qualname(spec):
    module_name, _, attr = str(spec).partition(":")
    if not module_name or not attr:
        raise ValueError("predictor must use module:attr syntax")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    if not callable(obj):
        raise TypeError("predictor target must be callable")
    return obj


def _repo_root():
    return Path(__file__).resolve().parents[1]


def default_ml_bundle_dir():
    try:
        from predictors.output_length.common import default_bundle_dir

        return str(default_bundle_dir())
    except Exception:
        return str(_repo_root() / "models" / "output_length_deberta")


def output_len_predictor_from_env_or_arg(spec):
    if spec is None or str(spec).strip() == "":
        env_path = os.environ.get("ENERGY_AWARE_SEQ_PRED_BERT_PKL")
        if env_path:
            return "bert:%s" % env_path
        return "bert:%s" % default_ml_bundle_dir()
    value = str(spec).strip()
    if value.lower() in ("none", "heuristic"):
        return None
    if value.lower() == "default":
        return "bert:%s" % default_ml_bundle_dir()
    return value


def build_output_len_predictor(spec):
    resolved = output_len_predictor_from_env_or_arg(spec)
    if resolved is None:
        return default_output_len_predictor
    if str(resolved).startswith("bert:"):
        return load_bert_pkl_predictor(str(resolved)[5:])
    return resolve_qualname(resolved)


def _resolve_bert_pkl_path(path):
    value = os.path.expanduser(str(path or ""))
    if os.path.isdir(value):
        return os.path.join(value, "bert.pkl")
    return value


def _ensure_repo_root_on_path():
    import sys as _sys

    root = str(_repo_root())
    if root not in _sys.path:
        _sys.path.insert(0, root)


def _load_bundle(path):
    _ensure_repo_root_on_path()
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            import torch

            return torch.load(path, map_location="cpu")
        except Exception:
            with open(path, "rb") as handle:
                return pickle.load(handle)
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _coerce_prediction(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        return _coerce_prediction(value[0])
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _invoke_callable(fn, text, request_json, quantile=None):
    if quantile is not None:
        try:
            return fn(request_json, quantile=quantile)
        except TypeError:
            try:
                return fn(text, quantile=quantile)
            except TypeError:
                try:
                    return fn(text, request_json, quantile=quantile)
                except TypeError:
                    pass

    try:
        return fn(request_json)
    except TypeError:
        try:
            return fn(text)
        except TypeError:
            return fn(text, request_json)


def _wrap_callable(fn):
    def predict(text, request_json, quantile=None):
        return _invoke_callable(fn, text, request_json, quantile=quantile)

    return predict


def _predictor_from_bundle(bundle):
    if callable(bundle):
        return _wrap_callable(bundle)
    if isinstance(bundle, dict):
        predictor = bundle.get("predictor")
        if predictor is not None and hasattr(predictor, "predict_generated_tokens"):
            return _wrap_callable(predictor.predict_generated_tokens)
    if hasattr(bundle, "predict_generated_tokens") and callable(bundle.predict_generated_tokens):
        return _wrap_callable(bundle.predict_generated_tokens)
    if hasattr(bundle, "predict") and callable(bundle.predict):
        def predict(text, request_json):
            try:
                value = bundle.predict([text])
            except Exception:
                value = bundle.predict(text)
            return _coerce_prediction(value)

        return predict
    if isinstance(bundle, dict):
        for key in ("predict_generated_tokens", "predictor", "model"):
            value = bundle.get(key)
            if callable(value):
                return _wrap_callable(value)
            if hasattr(value, "predict") and callable(value.predict):
                def predict(text, request_json, model=value):
                    try:
                        result = model.predict([text])
                    except Exception:
                        result = model.predict(text)
                    return _coerce_prediction(result)

                return predict
    return None


def load_bert_pkl_predictor(path):
    resolved = _resolve_bert_pkl_path(path)
    if not resolved or not os.path.exists(resolved):
        return default_output_len_predictor
    try:
        bundle = _load_bundle(resolved)
    except Exception:
        return default_output_len_predictor
    predictor = _predictor_from_bundle(bundle)
    if predictor is None:
        return default_output_len_predictor

    def wrapped(request_json):
        text = extract_request_text(request_json)
        if not str(text or "").strip():
            return default_output_len_predictor(request_json)
        try:
            value = predictor(text, request_json, quantile=None)
        except Exception:
            return default_output_len_predictor(request_json)
        numeric = _coerce_prediction(value)
        if numeric is None:
            return default_output_len_predictor(request_json)
        return numeric

    def predict_quantile(request_json, quantile):
        text = extract_request_text(request_json)
        if not str(text or "").strip():
            return default_output_len_predictor(request_json)
        try:
            value = predictor(text, request_json, quantile=quantile)
        except Exception:
            value = predictor(text, request_json, quantile=None)
        numeric = _coerce_prediction(value)
        if numeric is None:
            return default_output_len_predictor(request_json)
        return numeric

    wrapped.predict_quantile = predict_quantile

    return wrapped
