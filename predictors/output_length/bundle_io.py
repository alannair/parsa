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
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_TYPES = 3
PREDICTION_POINT = "point"
PREDICTION_LOGNORMAL = "lognormal"
PREDICTION_HYBRID_BUCKET = "hybrid_bucket"

PKL_NAME = "bert.pkl"
INFERENCE_CONFIG_NAME = "inference_config.json"
BACKBONE_DIR_NAME = "backbone"
WEIGHTS_NAME_LEGACY = "bert_regression_model.pth"
WEIGHTS_NAME_DEBERTA = "deberta_distribution_model.pth"
WEIGHTS_NAME_DEBERTA_HYBRID = "deberta_hybrid_bucket_model.pth"
DEFAULT_MAX_LENGTH = 512


def _stdout(*parts):
    print(*parts, flush=True)


def core_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def resolve_bundle_dir(bundle_path):
    path = Path(bundle_path)
    if path.is_dir():
        return path
    return path.parent


def read_bundle_inference_config(bundle_dir):
    bundle_dir = Path(bundle_dir)
    config = {}

    inference_path = bundle_dir / INFERENCE_CONFIG_NAME
    if inference_path.is_file():
        with open(inference_path, encoding="utf-8") as handle:
            config.update(json.load(handle))

    meta_path = bundle_dir / "bundle_meta.json"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as handle:
            config.update(json.load(handle))

    if (bundle_dir / WEIGHTS_NAME_DEBERTA_HYBRID).is_file():
        config.setdefault("predictor_type", "deberta_hybrid_bucket")
        config.setdefault("weights_name", WEIGHTS_NAME_DEBERTA_HYBRID)
        config.setdefault("prediction_mode", PREDICTION_HYBRID_BUCKET)
    elif (bundle_dir / WEIGHTS_NAME_DEBERTA).is_file():
        config.setdefault("predictor_type", "deberta_lognormal")
        config.setdefault("weights_name", WEIGHTS_NAME_DEBERTA)
        config.setdefault("prediction_mode", PREDICTION_LOGNORMAL)
    elif (bundle_dir / WEIGHTS_NAME_LEGACY).is_file():
        config.setdefault("predictor_type", "bert_point")
        config.setdefault("weights_name", WEIGHTS_NAME_LEGACY)
        config.setdefault("prediction_mode", PREDICTION_POINT)

    if not config.get("weights_name"):
        raise FileNotFoundError(
            "no weights file in bundle %s (expected %s, %s, or %s)"
            % (
                bundle_dir,
                WEIGHTS_NAME_DEBERTA_HYBRID,
                WEIGHTS_NAME_DEBERTA,
                WEIGHTS_NAME_LEGACY,
            )
        )
    return config


def _deberta_subset_from_checkpoint(state):
    subset = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        if name.startswith("deberta."):
            subset[name[len("deberta.") :]] = value
    return subset


def _head_subset_from_checkpoint(state):
    subset = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        if name.startswith("deberta."):
            continue
        subset[name] = value
    return subset


def load_checkpoint_cpu(weights_path):
    import torch

    kwargs = {"map_location": "cpu"}
    try:
        return torch.load(str(weights_path), weights_only=True, **kwargs)
    except TypeError:
        return torch.load(str(weights_path), **kwargs)


def write_deberta_backbone_weights_raw(deberta_module, backbone_dir):
    from predictors.output_length.deberta_distribution import _bundle_backbone_has_weights

    backbone_dir = Path(backbone_dir)
    backbone_dir.mkdir(parents=True, exist_ok=True)
    if _bundle_backbone_has_weights(backbone_dir):
        return backbone_dir

    cpu_state = {
        key: value.detach().cpu().contiguous()
        for key, value in deberta_module.state_dict().items()
    }
    safe_path = backbone_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(cpu_state, str(safe_path))
        _stdout("saved backbone weights -> %s (%d tensors)" % (safe_path.name, len(cpu_state)))
    except ImportError:
        import torch

        bin_path = backbone_dir / "pytorch_model.bin"
        torch.save(cpu_state, str(bin_path))
        _stdout("saved backbone weights -> %s (%d tensors)" % (bin_path.name, len(cpu_state)))
    return backbone_dir


def materialize_bundle_backbone_from_pth_raw(bundle_dir):
    from predictors.output_length.deberta_distribution import _bundle_backbone_has_weights

    bundle_dir = Path(bundle_dir)
    ensure_inference_config(bundle_dir)
    config = read_bundle_inference_config(bundle_dir)
    weights_path = bundle_dir / str(config["weights_name"])
    if not weights_path.is_file():
        raise FileNotFoundError("checkpoint not found: %s" % weights_path)

    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    if _bundle_backbone_has_weights(backbone_dir):
        _stdout("backbone weights already present in %s" % backbone_dir)
        return backbone_dir

    ensure_bundle_backbone(bundle_dir, config)

    _stdout("materialize backbone (raw): reading %s" % weights_path.name)
    state = load_checkpoint_cpu(weights_path)
    deberta_state = _deberta_subset_from_checkpoint(state)
    if not deberta_state:
        raise ValueError("no deberta.* keys in %s" % weights_path.name)

    cpu_state = {}
    for key, tensor in deberta_state.items():
        cpu_state[key] = tensor.detach().cpu().contiguous()

    backbone_dir.mkdir(parents=True, exist_ok=True)
    safe_path = backbone_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(cpu_state, str(safe_path))
        _stdout("materialize backbone (raw): wrote %s (%d tensors)" % (safe_path.name, len(cpu_state)))
    except ImportError:
        import torch

        bin_path = backbone_dir / "pytorch_model.bin"
        torch.save(cpu_state, str(bin_path))
        _stdout(
            "materialize backbone (raw): wrote %s (%d tensors); "
            "pip install safetensors for model.safetensors"
            % (bin_path.name, len(cpu_state))
        )

    if not _bundle_backbone_has_weights(backbone_dir):
        raise RuntimeError("weight file missing under %s after raw materialize" % backbone_dir)
    return backbone_dir


def materialize_bundle_backbone_from_pth(bundle_dir):
    return materialize_bundle_backbone_from_pth_raw(bundle_dir)


def materialize_bundle_backbone_weights(bundle_dir, use_training_builder=False):
    from predictors.output_length.deberta_distribution import _bundle_backbone_has_weights

    bundle_dir = Path(bundle_dir)
    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    if _bundle_backbone_has_weights(backbone_dir):
        _stdout("backbone weights already present")
        return backbone_dir

    if use_training_builder:
        return materialize_bundle_backbone_via_training_builder(bundle_dir)
    return materialize_bundle_backbone_from_pth_raw(bundle_dir)


def materialize_bundle_backbone_via_training_builder(bundle_dir):
    from predictors.output_length.deberta_distribution import (
        DEFAULT_DEBERTA_NAME,
        DEFAULT_HEAD_DIM,
        DEFAULT_HIDDEN_SIZE,
        DEFAULT_NUM_LAYERS,
        build_deberta_distribution_model,
    )

    bundle_dir = Path(bundle_dir)
    ensure_inference_config(bundle_dir)
    config = read_bundle_inference_config(bundle_dir)
    weights_path = bundle_dir / str(config["weights_name"])
    deberta_name = (
        config.get("deberta_pretrained_name")
        or config.get("backbone_name")
        or DEFAULT_DEBERTA_NAME
    )
    _stdout("materialize backbone: build %s (training path)" % deberta_name)
    model, _cfg = build_deberta_distribution_model(
        str(deberta_name),
        NUM_TYPES,
        int(config.get("hidden_dim", DEFAULT_HEAD_DIM)),
        num_layers=int(config.get("num_layers", DEFAULT_NUM_LAYERS)),
        hidden_size=int(config.get("hidden_size", DEFAULT_HIDDEN_SIZE)),
    )
    state = load_checkpoint_cpu(weights_path)
    model.load_state_dict(state, strict=True)
    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    backbone_dir.mkdir(parents=True, exist_ok=True)
    core = core_model(model)
    if hasattr(core, "deberta"):
        core.deberta.save_pretrained(str(backbone_dir))
    else:
        raise TypeError("expected DebertaDistributionModel for deberta bundle")
    _stdout("materialize backbone: wrote %s" % backbone_dir)
    return backbone_dir


def ensure_bundle_backbone(bundle_dir, config):
    from transformers import AutoConfig

    bundle_dir = Path(bundle_dir)
    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    if (backbone_dir / "config.json").is_file():
        return
    predictor_type = str(config.get("predictor_type") or "")
    if "deberta" in predictor_type:
        name = (
            config.get("deberta_pretrained_name")
            or config.get("backbone_name")
            or "microsoft/deberta-v3-base"
        )
    else:
        name = (
            config.get("bert_pretrained_name")
            or config.get("backbone_name")
            or "bert-base-uncased"
        )
    backbone_dir.mkdir(parents=True, exist_ok=True)
    _stdout("writing backbone config from %s (config only)" % name)
    AutoConfig.from_pretrained(str(name)).save_pretrained(str(backbone_dir))


def ensure_inference_config(bundle_dir):
    bundle_dir = Path(bundle_dir)
    path = bundle_dir / INFERENCE_CONFIG_NAME
    if path.is_file():
        return
    config = read_bundle_inference_config(bundle_dir)
    slim_keys = (
        "predictor_type",
        "prediction_mode",
        "max_length",
        "default_type",
        "hidden_dim",
        "backbone_name",
        "weights_name",
        "num_layers",
        "hidden_size",
        "deberta_pretrained_name",
        "bert_pretrained_name",
    )
    slim = {key: config[key] for key in slim_keys if key in config}
    slim.setdefault("max_length", DEFAULT_MAX_LENGTH)
    slim.setdefault("default_type", 0)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(slim, handle, ensure_ascii=False, indent=2)
    _stdout("wrote %s from existing bundle metadata" % path.name)


def _ensure_repo_root_on_path():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_predictor_from_pickle(bundle_dir):
    pkl_path = Path(bundle_dir) / PKL_NAME
    if not pkl_path.is_file():
        return None
    _ensure_repo_root_on_path()
    with open(pkl_path, "rb") as handle:
        bundle = pickle.load(handle)
    if isinstance(bundle, dict):
        predictor = bundle.get("predictor")
    else:
        predictor = bundle
    if predictor is None or not hasattr(predictor, "predict_generated_tokens"):
        return None
    return predictor


def _build_inference_model(bundle_dir, config):
    bundle_dir = Path(bundle_dir)
    predictor_type = str(config.get("predictor_type") or "")
    weights_name = str(config.get("weights_name") or "")

    ensure_bundle_backbone(bundle_dir, config)

    if "hybrid" in predictor_type or weights_name == WEIGHTS_NAME_DEBERTA_HYBRID:
        materialize_bundle_backbone_weights(bundle_dir)
        from predictors.output_length.deberta_hybrid import build_deberta_hybrid_model_from_bundle

        model, _cfg = build_deberta_hybrid_model_from_bundle(bundle_dir, config)
        prediction_mode = str(config.get("prediction_mode") or PREDICTION_HYBRID_BUCKET)
        return model, prediction_mode

    if "deberta" in predictor_type or weights_name == WEIGHTS_NAME_DEBERTA:
        materialize_bundle_backbone_weights(bundle_dir)
        from predictors.output_length.deberta_distribution import (
            build_deberta_distribution_model_from_bundle,
        )

        model, _cfg = build_deberta_distribution_model_from_bundle(bundle_dir, config)
        prediction_mode = str(config.get("prediction_mode") or PREDICTION_LOGNORMAL)
        return model, prediction_mode

    from predictors.output_length.train_output_length import build_bert_regression_model_from_bundle

    model, _cfg = build_bert_regression_model_from_bundle(bundle_dir, config)
    prediction_mode = str(config.get("prediction_mode") or PREDICTION_POINT)
    return model, prediction_mode


def _load_backbone_config_from_json(bundle_dir, config):
    from transformers import AutoConfig

    bundle_dir = Path(bundle_dir)
    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    config_path = backbone_dir / "config.json"
    if not config_path.is_file():
        ensure_bundle_backbone(bundle_dir, config)

    backbone_ref = str(backbone_dir)
    try:
        cfg = AutoConfig.from_pretrained(backbone_ref, local_files_only=True)
    except TypeError:
        cfg = AutoConfig.from_pretrained(backbone_ref)

    cfg.num_hidden_layers = int(config.get("num_layers", getattr(cfg, "num_hidden_layers", 12)))
    cfg.hidden_size = int(config.get("hidden_size", getattr(cfg, "hidden_size", 768)))
    return cfg


def save_predictor_bundle(
    predictor,
    model_dir,
    metrics,
    backbone_name,
    hidden_dim,
    predictor_type,
    num_layers=None,
    hidden_size=None,
    weights_name=WEIGHTS_NAME_LEGACY,
):
    import torch

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = out_dir / "model"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    predictor.tokenizer.save_pretrained(str(tokenizer_dir))
    core = core_model(predictor.model)
    core.to("cpu")
    core.eval()
    predictor.model = core
    predictor.device = torch.device("cpu")

    backbone_dir = out_dir / BACKBONE_DIR_NAME
    backbone_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(core, "deberta"):
        _stdout("saving DeBERTa backbone to %s/" % BACKBONE_DIR_NAME)
        backbone_meta = {
            "predictor_type": predictor_type,
            "deberta_pretrained_name": backbone_name,
            "backbone_name": backbone_name,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        }
        ensure_bundle_backbone(out_dir, backbone_meta)
        try:
            write_deberta_backbone_weights_raw(core.deberta, backbone_dir)
        except Exception as exc:
            _stdout("raw backbone save failed (%s); trying save_pretrained" % exc)
            core.deberta.save_pretrained(str(backbone_dir))
    elif hasattr(core, "bert"):
        _stdout("saving BERT backbone to %s/" % BACKBONE_DIR_NAME)
        core.bert.save_pretrained(str(backbone_dir))

    from predictors.output_length.deberta_distribution import _bundle_backbone_has_weights

    if hasattr(core, "deberta") and not _bundle_backbone_has_weights(backbone_dir):
        try:
            materialize_bundle_backbone_from_pth(out_dir)
        except Exception as exc:
            _stdout(
                "WARN: backbone/ has no weight files (%s) — run:\n"
                "  python -m predictors.output_length.train_output_length_deberta "
                "--materialize-backbone --model-dir %s"
                % (exc, out_dir)
            )

    weights_path = out_dir / weights_name
    torch.save(core.state_dict(), str(weights_path))

    pkl_path = out_dir / PKL_NAME
    bundle = {
        "predictor": predictor,
        "predict_generated_tokens": predictor.predict_generated_tokens,
        "model": predictor.model,
        "tokenizer": predictor.tokenizer,
        "device": str(predictor.device),
        "max_length": predictor.max_length,
        "backbone_name": backbone_name,
        "hidden_dim": hidden_dim,
        "num_types": NUM_TYPES,
        "default_type": predictor.default_type,
        "predictor_type": predictor_type,
        "prediction_mode": predictor.prediction_mode,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
    }
    with open(pkl_path, "wb") as handle:
        pickle.dump(bundle, handle)

    meta_path = out_dir / "bundle_meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    inference_config = {
        "predictor_type": predictor_type,
        "prediction_mode": predictor.prediction_mode,
        "max_length": int(predictor.max_length),
        "default_type": int(predictor.default_type),
        "hidden_dim": int(hidden_dim),
        "backbone_name": backbone_name,
        "weights_name": weights_name,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "bundle_format_version": "1",
    }
    if predictor_type == "deberta_lognormal":
        inference_config["deberta_pretrained_name"] = backbone_name
    elif predictor_type == "deberta_hybrid_bucket":
        from predictors.output_length.features import NUM_STRUCTURAL_FEATURES

        inference_config["deberta_pretrained_name"] = backbone_name
        inference_config["regression_loss"] = getattr(predictor, "regression_loss_type", "mse")
        inference_config["head_tail_truncation"] = bool(getattr(predictor, "head_tail_truncation", False))
        inference_config["use_structural_features"] = bool(getattr(predictor, "use_structural_features", False))
        inference_config["structural_feature_dim"] = (
            NUM_STRUCTURAL_FEATURES if getattr(predictor, "use_structural_features", False) else 0
        )
        use_length = bool(getattr(predictor, "use_input_length_feature", True))
        inference_config["use_input_length_feature"] = use_length
        inference_config["disable_input_length_feature"] = not use_length
    else:
        inference_config["bert_pretrained_name"] = backbone_name
    inference_path = out_dir / INFERENCE_CONFIG_NAME
    with open(inference_path, "w", encoding="utf-8") as handle:
        json.dump(inference_config, handle, ensure_ascii=False, indent=2)

    _stdout("saved bundle to %s" % out_dir)
    _stdout("  %s" % pkl_path.name)
    _stdout("  %s" % weights_path.name)
    _stdout("  %s" % inference_path.name)
    _stdout("  %s/" % tokenizer_dir.name)
    _stdout("  %s/" % backbone_dir.name)
    return out_dir


__all__ = [
    "REPO_ROOT",
    "NUM_TYPES",
    "PREDICTION_POINT",
    "PREDICTION_LOGNORMAL",
    "PREDICTION_HYBRID_BUCKET",
    "PKL_NAME",
    "INFERENCE_CONFIG_NAME",
    "BACKBONE_DIR_NAME",
    "WEIGHTS_NAME_LEGACY",
    "WEIGHTS_NAME_DEBERTA",
    "WEIGHTS_NAME_DEBERTA_HYBRID",
    "DEFAULT_MAX_LENGTH",
    "core_model",
    "resolve_bundle_dir",
    "read_bundle_inference_config",
    "load_checkpoint_cpu",
    "ensure_bundle_backbone",
    "ensure_inference_config",
    "materialize_bundle_backbone_from_pth_raw",
    "materialize_bundle_backbone_from_pth",
    "materialize_bundle_backbone_weights",
    "materialize_bundle_backbone_via_training_builder",
    "_deberta_subset_from_checkpoint",
    "_head_subset_from_checkpoint",
    "_build_inference_model",
    "_load_predictor_from_pickle",
    "_load_backbone_config_from_json",
    "save_predictor_bundle",
    "write_deberta_backbone_weights_raw",
]
