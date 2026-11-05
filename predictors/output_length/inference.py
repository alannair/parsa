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


import os
from pathlib import Path

from predictors.output_length.ascend import (
    configure_ascend_inference_env,
    setup_inference_device,
)
from predictors.output_length.bundle_io import (
    DEFAULT_MAX_LENGTH,
    PKL_NAME,
    WEIGHTS_NAME_DEBERTA,
    WEIGHTS_NAME_DEBERTA_HYBRID,
    core_model,
    ensure_bundle_backbone,
    ensure_inference_config,
    load_checkpoint_cpu,
    materialize_bundle_backbone_weights,
    read_bundle_inference_config,
    resolve_bundle_dir,
)
from predictors.output_length.dataset_io import (
    build_dataloader,
    load_labeled_csv,
    tokenize_context_lengths,
    write_predict_results_csv,
)
from predictors.output_length.tokenization import load_bundle_tokenizer

PREDICTION_POINT = "point"
PREDICTION_LOGNORMAL = "lognormal"
PREDICTION_HYBRID_BUCKET = "hybrid_bucket"
DEFAULT_DEBERTA_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "output_length_deberta"


def _stdout(*parts):
    print(*parts, flush=True)


def _checkpoint_has_input_length_encoder(state):
    for key in state:
        name = key[7:] if key.startswith("module.") else key
        if name.startswith("input_length_encoder."):
            return True
    return False


def _checkpoint_structural_feature_dim(state):
    for key in state:
        name = key[7:] if key.startswith("module.") else key
        if name == "struct_proj.weight":
            return int(state[key].shape[1])
    return 0


def resolve_use_input_length_feature(config, state=None, *, default=True):
    if bool(config.get("disable_input_length_feature", False)):
        return False
    use = config.get("use_input_length_feature")
    if use is None:
        use = default
    use = bool(use)
    if state is not None and use and not _checkpoint_has_input_length_encoder(state):
        return False
    return use


def _is_hybrid_bundle_config(config):
    predictor_type = str(config.get("predictor_type") or "")
    weights_name = str(config.get("weights_name") or "")
    return "hybrid" in predictor_type or weights_name == WEIGHTS_NAME_DEBERTA_HYBRID


def _build_deberta_from_state_dict_cpu(bundle_dir, config, state):
    from predictors.output_length.deberta_distribution import build_deberta_distribution_model_from_bundle

    bundle_dir = Path(bundle_dir)
    materialize_bundle_backbone_weights(bundle_dir)
    _stdout("[deberta] bundle backbone + full checkpoint load (CPU)")
    model, _cfg = build_deberta_distribution_model_from_bundle(bundle_dir, config)
    try:
        model.load_state_dict(state, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state, strict=True)
    model.cpu().float().eval()
    return model, str(config.get("prediction_mode") or PREDICTION_LOGNORMAL)


def _build_hybrid_from_state_dict_cpu(bundle_dir, config, state):
    from predictors.output_length.deberta_hybrid import build_deberta_hybrid_model_from_bundle

    bundle_dir = Path(bundle_dir)
    materialize_bundle_backbone_weights(bundle_dir)
    _stdout("[deberta-hybrid] bundle backbone + full checkpoint load (CPU)")
    load_config = dict(config)
    use_length = resolve_use_input_length_feature(load_config, state, default=True)
    if use_length != bool(load_config.get("use_input_length_feature", True)):
        _stdout(
            "[deberta-hybrid] checkpoint missing input_length_encoder weights; "
            "running without input-length feature"
        )
    load_config["use_input_length_feature"] = use_length
    struct_dim = int(load_config.get("structural_feature_dim", 0) or 0)
    checkpoint_struct_dim = _checkpoint_structural_feature_dim(state)
    if checkpoint_struct_dim and struct_dim != checkpoint_struct_dim:
        _stdout(
            "[deberta-hybrid] aligning structural_feature_dim=%d from checkpoint (config had %d)"
            % (checkpoint_struct_dim, struct_dim)
        )
        struct_dim = checkpoint_struct_dim
    elif not struct_dim and checkpoint_struct_dim:
        struct_dim = checkpoint_struct_dim
    load_config["structural_feature_dim"] = struct_dim
    load_config["use_structural_features"] = struct_dim > 0
    model, _cfg = build_deberta_hybrid_model_from_bundle(bundle_dir, load_config)
    strict = _checkpoint_has_input_length_encoder(state) == use_length
    try:
        model.load_state_dict(state, strict=strict, assign=True)
    except TypeError:
        model.load_state_dict(state, strict=strict)
    model.cpu().float().eval()
    return model, str(config.get("prediction_mode") or PREDICTION_HYBRID_BUCKET)


def _hybrid_warmup_model_kwargs(model, device, *, cap, default_type=0):
    import torch

    core = core_model(model)
    kwargs = {
        "return_dict": True,
        "prompt_type": torch.tensor([int(default_type)], dtype=torch.long, device=device),
    }
    if getattr(core, "use_input_length_feature", False):
        kwargs["original_input_token_count"] = torch.tensor(
            [float(cap)], device=device, dtype=torch.float32
        )
    struct_proj = getattr(core, "struct_proj", None)
    if struct_proj is not None:
        struct_dim = int(getattr(struct_proj, "in_features", 0) or 0)
        if struct_dim > 0:
            kwargs["structural_features"] = torch.zeros(
                1, struct_dim, device=device, dtype=torch.float32
            )
    return kwargs


def npu_warmup_forward(model, tokenizer, device, max_length, default_type=0):
    import torch

    if str(device).startswith("cpu") or not hasattr(torch, "npu"):
        return
    cap = max(8, min(int(max_length), 32))
    enc = tokenizer(
        ["warmup"],
        padding=True,
        truncation=True,
        max_length=cap,
        return_tensors="pt",
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    model.eval()
    with torch.no_grad():
        kwargs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask"),
        }
        if hasattr(core_model(model), "bucket_classifier"):
            kwargs.update(
                _hybrid_warmup_model_kwargs(
                    model, device, cap=cap, default_type=default_type
                )
            )
        else:
            kwargs["prompt_type"] = torch.tensor(
                [int(default_type)], dtype=torch.long, device=device
            )
        model(**kwargs)
    torch.npu.synchronize()


def load_predictor_cpu_safe(bundle_path):
    import torch

    from predictors.output_length.bundle_io import _load_predictor_from_pickle
    from predictors.output_length.common import OutputLengthPredictor

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    bundle_dir = resolve_bundle_dir(bundle_path)

    _stdout("CPU load: trying %s" % PKL_NAME)
    predictor = _load_predictor_from_pickle(bundle_dir)
    if predictor is not None:
        ensure_inference_config(bundle_dir)
        config = read_bundle_inference_config(bundle_dir)
        max_length = int(config.get("max_length", DEFAULT_MAX_LENGTH))
        core = core_model(predictor.model).cpu().float().eval()
        predictor.model = core
        predictor.device = torch.device("cpu")
        predictor.max_length = max_length
        if _is_hybrid_bundle_config(config):
            predictor.use_input_length_feature = resolve_use_input_length_feature(
                config, default=True
            )
            predictor.head_tail_truncation = bool(config.get("head_tail_truncation", False))
            predictor.use_structural_features = bool(config.get("use_structural_features", False))
            predictor.regression_loss_type = str(config.get("regression_loss") or "mse")
        if predictor.tokenizer is None:
            tokenizer_dir = bundle_dir / "model"
            predictor.tokenizer = load_bundle_tokenizer(tokenizer_dir, max_length=max_length)
        _stdout("CPU load: restored from %s (using bundled tokenizer)" % PKL_NAME)
        return predictor

    ensure_inference_config(bundle_dir)
    config = read_bundle_inference_config(bundle_dir)
    max_length = int(config.get("max_length", DEFAULT_MAX_LENGTH))
    default_type = int(config.get("default_type", 0))
    weights_path = bundle_dir / str(config["weights_name"])

    tokenizer_dir = bundle_dir / "model"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError("bundle tokenizer directory not found: %s" % tokenizer_dir)
    tokenizer = load_bundle_tokenizer(tokenizer_dir, max_length=max_length)

    _stdout("CPU load: loading checkpoint %s" % weights_path.name)
    state = load_checkpoint_cpu(weights_path)
    predictor_type = str(config.get("predictor_type") or "")
    if _is_hybrid_bundle_config(config):
        model, prediction_mode = _build_hybrid_from_state_dict_cpu(bundle_dir, config, state)
    elif "deberta" in predictor_type or str(config.get("weights_name")) == WEIGHTS_NAME_DEBERTA:
        model, prediction_mode = _build_deberta_from_state_dict_cpu(bundle_dir, config, state)
    else:
        from predictors.output_length.train_output_length import build_bert_regression_model_from_bundle

        model, prediction_mode = build_bert_regression_model_from_bundle(bundle_dir, config)
        model.load_state_dict(state, strict=True)
        model.cpu().float().eval()

    hybrid = _is_hybrid_bundle_config(config)
    return OutputLengthPredictor(
        model,
        tokenizer,
        torch.device("cpu"),
        max_length,
        default_type=default_type,
        prediction_mode=prediction_mode,
        regression_loss_type=str(config.get("regression_loss") or "mse"),
        head_tail_truncation=bool(config.get("head_tail_truncation", False)),
        use_structural_features=bool(config.get("use_structural_features", False)),
        use_input_length_feature=resolve_use_input_length_feature(config, state, default=hybrid)
        if hybrid
        else False,
    )


def load_inference_predictor(bundle_path, ascend_devices=None, device=None):
    import torch

    from predictors.output_length.common import OutputLengthPredictor

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device_str = str(device).strip() if device is not None else ""
    if device_str.lower() == "cpu":
        return load_predictor_cpu_safe(bundle_path)

    requested_device = torch.device(device_str) if device_str else None
    resolved_ascend_devices = ascend_devices
    if requested_device is not None and requested_device.type == "npu":
        resolved_ascend_devices = str(int(requested_device.index or 0))
    if requested_device is None or requested_device.type == "npu":
        configure_ascend_inference_env(resolved_ascend_devices)

    _stdout("load_inference_predictor: begin")
    bundle_dir = resolve_bundle_dir(bundle_path)
    ensure_inference_config(bundle_dir)
    config = read_bundle_inference_config(bundle_dir)
    weights_path = bundle_dir / str(config["weights_name"])
    if not weights_path.is_file():
        raise FileNotFoundError("checkpoint not found: %s" % weights_path)
    max_length = int(config.get("max_length", DEFAULT_MAX_LENGTH))
    default_type = int(config.get("default_type", 0))

    tokenizer_dir = bundle_dir / "model"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError("bundle tokenizer directory not found: %s" % tokenizer_dir)
    _stdout("loading bundle tokenizer (spm/tokenizers, not AutoTokenizer)")
    tokenizer = load_bundle_tokenizer(tokenizer_dir, max_length=max_length)

    ensure_bundle_backbone(bundle_dir, config)
    predictor_type = str(config.get("predictor_type") or "")
    if _is_hybrid_bundle_config(config) or (
        "deberta" in predictor_type or str(config.get("weights_name")) == WEIGHTS_NAME_DEBERTA
    ):
        materialize_bundle_backbone_weights(bundle_dir)
    _stdout("loading checkpoint %s on CPU (config + state_dict, no from_pretrained)" % weights_path.name)
    state = load_checkpoint_cpu(weights_path)
    if _is_hybrid_bundle_config(config):
        model, prediction_mode = _build_hybrid_from_state_dict_cpu(bundle_dir, config, state)
    elif "deberta" in predictor_type or str(config.get("weights_name")) == WEIGHTS_NAME_DEBERTA:
        model, prediction_mode = _build_deberta_from_state_dict_cpu(bundle_dir, config, state)
    else:
        from predictors.output_length.train_output_length import build_bert_regression_model_from_bundle

        _stdout("building BERT model from bundle/backbone config")
        model, prediction_mode = build_bert_regression_model_from_bundle(bundle_dir, config)
        model.load_state_dict(state, strict=True)
        model.cpu().float().eval()

    if requested_device is not None and requested_device.type != "npu":
        infer_device = requested_device
    else:
        infer_device, primary_id, _device_list = setup_inference_device(resolved_ascend_devices)
        if requested_device is not None:
            _stdout(
                "explicit device %s mapped to physical Ascend %s / logical %s"
                % (device_str, primary_id, infer_device)
            )

    _stdout("moving model to %s" % infer_device)
    model = model.to(infer_device)

    npu_warmup_forward(model, tokenizer, infer_device, max_length, default_type)

    hybrid = _is_hybrid_bundle_config(config)
    core = core_model(model)
    use_structural = getattr(core, "struct_proj", None) is not None
    predictor = OutputLengthPredictor(
        model,
        tokenizer,
        infer_device,
        max_length,
        default_type=default_type,
        prediction_mode=prediction_mode,
        regression_loss_type=str(config.get("regression_loss") or "mse"),
        head_tail_truncation=bool(config.get("head_tail_truncation", False)),
        use_structural_features=use_structural,
        use_input_length_feature=bool(getattr(core, "use_input_length_feature", False))
        if hybrid
        else False,
    )
    _stdout("load_inference_predictor: ready on %s" % infer_device)
    return predictor


def predict_labeled_bundle_csv(
    csv_path,
    bundle_path,
    output_csv,
    *,
    batch_size=16,
    max_length=None,
    device=None,
    ascend_devices=None,
    num_prompts=0,
    include_prompt=False,
    show_progress=True,
):
    import random

    import torch

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows = load_labeled_csv(csv_path)
    if num_prompts and int(num_prompts) > 0 and int(num_prompts) < len(rows):
        rows = random.Random(0).sample(rows, int(num_prompts))
        _stdout("predict: subsampled to %d rows" % len(rows))

    predictor = load_inference_predictor(
        bundle_path,
        ascend_devices=ascend_devices,
        device=device,
    )
    infer_device = predictor.device
    effective_max_length = int(getattr(predictor, "max_length", DEFAULT_MAX_LENGTH))
    if max_length is not None:
        effective_max_length = min(effective_max_length, max(1, int(max_length)))

    _stdout("predict: using inference device %s" % infer_device)
    rows = tokenize_context_lengths(rows, predictor.tokenizer, effective_max_length, show_progress)
    loader = build_dataloader(
        rows,
        predictor.tokenizer,
        max(1, int(batch_size)),
        effective_max_length,
        shuffle=False,
        head_tail_truncation=bool(getattr(predictor, "head_tail_truncation", False)),
        use_structural_features=bool(getattr(predictor, "use_structural_features", False)),
        use_input_length_feature=bool(getattr(predictor, "use_input_length_feature", False)),
    )

    preds = []
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    iterator = loader
    if show_progress:
        from predictors.output_length.common import get_progress_iter

        iterator = get_progress_iter(loader, total=total_batches, desc="predicting")

    from predictors.output_length.training_memory import (
        resolve_training_precision,
        training_autocast,
    )

    infer_precision = resolve_training_precision(infer_device, auto_on_accelerator=True)
    hybrid_mode = predictor.prediction_mode == PREDICTION_HYBRID_BUCKET
    pred_buckets = [] if hybrid_mode else None
    predictor.model.eval()
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(infer_device)
            attention_mask = batch["attention_mask"].to(infer_device)
            type_tensor = batch["type"].to(infer_device).long()
            struct = batch.get("structural_features")
            if struct is not None:
                struct = struct.to(infer_device)
            original_lengths = batch.get("original_input_token_count")
            if original_lengths is not None:
                original_lengths = original_lengths.to(infer_device)
            with training_autocast(infer_device, infer_precision):
                model_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "prompt_type": type_tensor,
                    "collect_metadata": hybrid_mode,
                }
                if struct is not None:
                    model_kwargs["structural_features"] = struct
                if original_lengths is not None:
                    model_kwargs["original_input_token_count"] = original_lengths
                if hybrid_mode:
                    model_kwargs["return_dict"] = True
                raw = predictor.model(**model_kwargs)
            if hybrid_mode:
                from predictors.output_length.deberta_hybrid import decode_hybrid_outputs

                decoded = decode_hybrid_outputs(raw, predictor.regression_loss_type)
                preds.extend(float(v) for v in decoded["predicted_tokens"].cpu().tolist())
                pred_buckets.extend(int(v) for v in decoded["predicted_bucket"].cpu().tolist())
            else:
                decoded = predictor._decode_predictions(raw).cpu().tolist()
                preds.extend(float(v) for v in decoded)

    if len(preds) != len(rows):
        raise RuntimeError("predictor returned %d values for %d rows" % (len(preds), len(rows)))
    if pred_buckets is not None and len(pred_buckets) != len(rows):
        raise RuntimeError(
            "predictor returned %d bucket ids for %d rows" % (len(pred_buckets), len(rows))
        )

    out = write_predict_results_csv(
        rows,
        preds,
        output_csv,
        include_prompt=include_prompt,
        pred_buckets=pred_buckets,
    )
    _stdout("predict: wrote %d rows to %s" % (len(rows), out))
    return out


def default_bundle_dir():
    return DEFAULT_DEBERTA_MODEL_DIR


def predict_output_tokens(
    prompt,
    prompt_type=0,
    bundle_path=None,
    *,
    device="cpu",
    ascend_devices=None,
):
    path = Path(bundle_path) if bundle_path is not None else default_bundle_dir()
    dev = str(device or "cpu").strip()

    if dev.lower() == "cpu":
        predictor = load_predictor_cpu_safe(path)
    else:
        predictor = load_inference_predictor(
            path,
            device=dev if dev else None,
            ascend_devices=ascend_devices,
        )
    sample = {"prompt": str(prompt), "type": int(prompt_type)}
    value = predictor.predict_generated_tokens(sample)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


__all__ = [
    "resolve_use_input_length_feature",
    "load_predictor_cpu_safe",
    "load_inference_predictor",
    "predict_labeled_bundle_csv",
    "predict_output_tokens",
    "default_bundle_dir",
    "npu_warmup_forward",
    "_checkpoint_has_input_length_encoder",
    "_is_hybrid_bundle_config",
    "_build_deberta_from_state_dict_cpu",
    "_build_hybrid_from_state_dict_cpu",
]
