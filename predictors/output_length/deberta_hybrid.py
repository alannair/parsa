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


import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from predictors.output_length.buckets import (
    NUM_BUCKETS,
    bucket_label,
    decode_token_prediction,
    token_length_to_bucket_id,
)
from predictors.output_length.deberta_distribution import (
    DEFAULT_DEBERTA_NAME,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    _bundle_backbone_has_weights,
    _materialize_deberta_backbone,
    expected_tokens_from_lognormal,
    lognormal_gaussian_nll,
)
from predictors.output_length.features import NUM_STRUCTURAL_FEATURES
from predictors.output_length.input_length_encoder import InputLengthEncoder, normalize_input_length

PREDICTOR_TYPE_HYBRID = "deberta_hybrid_bucket"
WEIGHTS_NAME_HYBRID = "deberta_hybrid_bucket_model.pth"
REGRESSION_LOSS_MSE = "mse"
REGRESSION_LOSS_HUBER = "huber"
REGRESSION_LOSS_LOGNORMAL = "lognormal"


class DebertaHybridBucketModel(nn.Module):

    def __init__(
        self,
        backbone,
        backbone_hidden_size,
        num_types,
        hidden_dim,
        num_buckets=NUM_BUCKETS,
        structural_feature_dim=0,
        regression_loss_type=REGRESSION_LOSS_MSE,
        dropout=0.1,
        use_input_length_feature=True,
    ):
        super().__init__()
        self.deberta = backbone
        self.num_buckets = int(num_buckets)
        self.hidden_dim = int(hidden_dim)
        self.regression_loss_type = str(regression_loss_type or REGRESSION_LOSS_MSE).lower()
        self.use_input_length_feature = bool(use_input_length_feature)
        self.type_embed = nn.Embedding(int(num_types), 16)
        self.bucket_embed = nn.Embedding(self.num_buckets, 16)
        struct_dim = int(structural_feature_dim)
        self.struct_proj = nn.Linear(struct_dim, 16) if struct_dim > 0 else None
        self.input_length_encoder = InputLengthEncoder(
            self.hidden_dim, enabled=self.use_input_length_feature
        )
        self.cls = nn.Linear(int(backbone_hidden_size), self.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        shared_dim = self._shared_feature_dim(struct_dim)
        self.bucket_classifier = nn.Sequential(
            nn.Linear(shared_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_buckets),
        )
        reg_in = shared_dim + 16
        self.reg_fc1 = nn.Linear(reg_in, self.hidden_dim)
        if self.regression_loss_type == REGRESSION_LOSS_LOGNORMAL:
            self.reg_out = nn.Linear(self.hidden_dim, 2)
        else:
            self.reg_out = nn.Linear(self.hidden_dim, 1)

    def _shared_feature_dim(self, struct_dim=None):
        struct_dim = int(struct_dim or 0)
        dim = self.hidden_dim + 16
        if self.use_input_length_feature:
            dim += self.hidden_dim
        if struct_dim > 0:
            dim += 16
        return dim

    def _struct_vec(self, structural_features):
        if self.struct_proj is None or structural_features is None:
            return None
        return self.act(self.struct_proj(structural_features.float()))

    def _backbone_requires_grad(self):
        return any(param.requires_grad for param in self.deberta.parameters())

    def _deberta_forward_cls(self, input_ids, attention_mask):
        import inspect

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
        }
        params = inspect.signature(self.deberta.forward).parameters
        outputs = self.deberta(**{k: v for k, v in kwargs.items() if k in params})
        pooled = outputs.last_hidden_state[:, 0, :]
        del outputs
        return pooled

    def _encode_backbone_cls(self, input_ids, attention_mask, *, autocast_ctx=None):
        if getattr(self, "_backbone_micro_batch_encode", False):
            return self._encode_backbone_cls_micro_batch(input_ids, attention_mask, autocast_ctx=autocast_ctx)

        target_device = input_ids.device
        backbone_device = next(self.deberta.parameters()).device
        ids = input_ids
        mask = attention_mask
        if backbone_device.type == "cpu":
            ids = ids.cpu()
            mask = mask.cpu()

        ctx = autocast_ctx
        if ctx is None:
            from contextlib import nullcontext

            ctx = nullcontext()

        if self._backbone_requires_grad() and torch.is_grad_enabled():
            with ctx:
                pooled = self._deberta_forward_cls(ids, mask)
            return pooled

        with torch.inference_mode():
            with ctx:
                pooled = self._deberta_forward_cls(ids, mask)
        if pooled.device != target_device:
            pooled = pooled.to(target_device)
        if pooled.dtype not in (torch.float32, torch.float64):
            pooled = pooled.float()
        return pooled

    def _encode_backbone_cls_micro_batch(self, input_ids, attention_mask, *, autocast_ctx=None):
        from contextlib import nullcontext

        batch_size = int(input_ids.size(0))
        target_device = input_ids.device
        backbone_device = next(self.deberta.parameters()).device
        ctx = autocast_ctx or nullcontext()
        pooled_rows = []
        for index in range(batch_size):
            mini_ids = input_ids[index : index + 1]
            mini_mask = attention_mask[index : index + 1]
            if backbone_device.type == "cpu":
                mini_ids = mini_ids.cpu()
                mini_mask = mini_mask.cpu()
            with torch.inference_mode():
                with ctx:
                    row = self._deberta_forward_cls(mini_ids, mini_mask)
            pooled_rows.append(row.to(target_device).float())
        return torch.cat(pooled_rows, dim=0)

    def build_shared_features(
        self,
        input_ids,
        attention_mask,
        prompt_type,
        *,
        original_input_token_count=None,
        structural_features=None,
        autocast_ctx=None,
    ):
        pooled = self._encode_backbone_cls(input_ids, attention_mask, autocast_ctx=autocast_ctx)
        cls_hidden = self.act(self.cls(pooled))
        type_vec = self.type_embed(prompt_type.long())
        parts = [cls_hidden, type_vec]
        if self.use_input_length_feature:
            if original_input_token_count is None:
                batch_size = int(cls_hidden.size(0))
                original_input_token_count = torch.zeros(
                    batch_size, device=cls_hidden.device, dtype=cls_hidden.dtype
                )
            length_emb = self.input_length_encoder(
                original_input_token_count, prompt_type=prompt_type
            )
            if length_emb is not None:
                parts.append(length_emb)
        struct_vec = self._struct_vec(structural_features)
        if struct_vec is not None:
            parts.append(struct_vec)
        return torch.cat(parts, dim=-1)

    def pooled_hidden(
        self,
        input_ids,
        attention_mask,
        prompt_type,
        structural_features=None,
        original_input_token_count=None,
    ):
        return self.build_shared_features(
            input_ids,
            attention_mask,
            prompt_type,
            original_input_token_count=original_input_token_count,
            structural_features=structural_features,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        prompt_type,
        bucket_ids=None,
        structural_features=None,
        original_input_token_count=None,
        return_dict=True,
        collect_metadata=None,
        autocast_ctx=None,
    ):
        if collect_metadata is None:
            collect_metadata = not self.training

        shared = self.build_shared_features(
            input_ids,
            attention_mask,
            prompt_type,
            original_input_token_count=original_input_token_count,
            structural_features=structural_features,
            autocast_ctx=autocast_ctx,
        )
        bucket_logits = self.bucket_classifier(shared)
        if bucket_ids is None:
            bucket_ids = bucket_logits.argmax(dim=-1)
        else:
            bucket_ids = bucket_ids.long()
        bucket_vec = self.bucket_embed(bucket_ids)
        reg_in = torch.cat([shared, bucket_vec], dim=-1)
        reg_hidden = self.dropout(self.act(self.reg_fc1(reg_in)))
        reg_out = self.reg_out(reg_hidden)
        if not return_dict:
            if self.regression_loss_type == REGRESSION_LOSS_LOGNORMAL:
                return reg_out
            return reg_out.squeeze(-1)
        out = {
            "bucket_logits": bucket_logits,
            "bucket_ids": bucket_ids,
            "regression": reg_out,
        }
        if collect_metadata and original_input_token_count is not None:
            if hasattr(original_input_token_count, "detach"):
                out["original_input_token_count"] = original_input_token_count.detach()
                out["normalized_input_length"] = torch.log1p(
                    original_input_token_count.float().clamp(min=0.0)
                ).detach()
            else:
                out["original_input_token_count"] = original_input_token_count
                out["normalized_input_length"] = normalize_input_length(
                    original_input_token_count
                )
        return out


def log1p_targets(token_lengths):
    return torch.log1p(token_lengths.clamp(min=0.0).float())


def regression_loss(reg_out, token_lengths, loss_type=REGRESSION_LOSS_MSE):
    target = log1p_targets(token_lengths)
    if loss_type == REGRESSION_LOSS_LOGNORMAL:
        if reg_out.dim() == 1:
            raise ValueError("lognormal regression expects [batch, 2] parameters")
        mu_log = reg_out[:, 0]
        log_var = reg_out[:, 1].clamp(min=-8.0, max=8.0)
        inv_var = torch.exp(-log_var)
        return 0.5 * (log_var + (target - mu_log).pow(2) * inv_var).mean()
    pred = reg_out.squeeze(-1)
    if loss_type == REGRESSION_LOSS_HUBER:
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


def hybrid_loss(
    outputs,
    token_lengths,
    *,
    bucket_targets=None,
    bucket_loss_weight=1.0,
    regression_loss_weight=1.0,
    regression_loss_type=REGRESSION_LOSS_MSE,
):
    if bucket_targets is None:
        bucket_targets = torch.tensor(
            [token_length_to_bucket_id(v) for v in token_lengths.detach().cpu().tolist()],
            device=token_lengths.device,
            dtype=torch.long,
        )
    else:
        bucket_targets = bucket_targets.long()
    cls_loss = F.cross_entropy(outputs["bucket_logits"], bucket_targets)
    reg_out = outputs["regression"]
    reg_loss = regression_loss(reg_out, token_lengths, loss_type=regression_loss_type)
    total = float(bucket_loss_weight) * cls_loss + float(regression_loss_weight) * reg_loss
    return total, {
        "loss": total.item(),
        "bucket_loss": cls_loss.item(),
        "regression_loss": reg_loss.item(),
    }


def _decode_regression_tensor(reg_out, regression_loss_type):
    if regression_loss_type == REGRESSION_LOSS_LOGNORMAL:
        mu = reg_out[:, 0]
        log_var = reg_out[:, 1].clamp(min=-8.0, max=8.0)

        return torch.expm1(mu + 0.5 * torch.exp(log_var))
    pred_log = reg_out.squeeze(-1)
    return torch.expm1(pred_log.clamp(min=0.0))


def decode_hybrid_outputs(outputs, regression_loss_type=REGRESSION_LOSS_MSE):
    import torch

    logits = outputs["bucket_logits"]
    probs = torch.softmax(logits, dim=-1)
    pred_bucket = logits.argmax(dim=-1)
    reg_out = outputs["regression"]
    pred_tokens = _decode_regression_tensor(reg_out, regression_loss_type)
    return {
        "bucket_logits": logits,
        "bucket_probs": probs,
        "predicted_bucket": pred_bucket,
        "predicted_tokens": pred_tokens,
    }


def _batch_original_lengths(batch, device):
    counts = batch.get("original_input_token_count")
    if counts is None:
        return None
    return counts.to(device).float()


def build_deberta_hybrid_model(
    pretrained_name,
    num_types,
    hidden_dim,
    num_layers=DEFAULT_NUM_LAYERS,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    structural_feature_dim=0,
    regression_loss_type=REGRESSION_LOSS_MSE,
    use_input_length_feature=True,
):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(pretrained_name)
    pretrained_hidden_size = int(getattr(config, "hidden_size", DEFAULT_HIDDEN_SIZE))
    pretrained_num_layers = int(getattr(config, "num_hidden_layers", DEFAULT_NUM_LAYERS))
    requested_hidden_size = int(hidden_size)
    requested_num_layers = int(num_layers)
    config.num_hidden_layers = requested_num_layers
    config.hidden_size = requested_hidden_size
    backbone = _materialize_deberta_backbone(
        pretrained_name,
        config,
        hidden_size_changed=requested_hidden_size != pretrained_hidden_size,
        num_layers_changed=requested_num_layers != pretrained_num_layers,
        requested_hidden_size=requested_hidden_size,
        pretrained_hidden_size=pretrained_hidden_size,
        requested_num_layers=requested_num_layers,
        pretrained_num_layers=pretrained_num_layers,
    )
    model = DebertaHybridBucketModel(
        backbone=backbone,
        backbone_hidden_size=int(config.hidden_size),
        num_types=int(num_types),
        hidden_dim=int(hidden_dim),
        structural_feature_dim=int(structural_feature_dim),
        regression_loss_type=regression_loss_type,
        use_input_length_feature=bool(use_input_length_feature),
    )
    return model, config


def build_deberta_hybrid_model_from_bundle(bundle_dir, config):
    from pathlib import Path

    from transformers import AutoModel

    from predictors.output_length.common import BACKBONE_DIR_NAME, NUM_TYPES, _stdout

    bundle_dir = Path(bundle_dir)
    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    hidden_dim = int(config.get("hidden_dim", DEFAULT_HEAD_DIM))
    num_layers = int(config.get("num_layers", DEFAULT_NUM_LAYERS))
    hidden_size = int(config.get("hidden_size", DEFAULT_HIDDEN_SIZE))
    struct_dim = int(config.get("structural_feature_dim", 0) or 0)
    reg_loss = str(config.get("regression_loss", REGRESSION_LOSS_MSE))
    use_length = config.get("use_input_length_feature", True)
    if use_length is None:
        use_length = not bool(config.get("disable_input_length_feature", False))
    use_length = bool(use_length)

    if _bundle_backbone_has_weights(backbone_dir):
        _stdout("[deberta-hybrid] loading backbone from %s" % backbone_dir)
        backbone = AutoModel.from_pretrained(str(backbone_dir), local_files_only=True)
        backbone_config = backbone.config
    else:
        pretrained_name = (
            config.get("deberta_pretrained_name")
            or config.get("backbone_name")
            or DEFAULT_DEBERTA_NAME
        )
        return build_deberta_hybrid_model(
            str(pretrained_name),
            NUM_TYPES,
            hidden_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            structural_feature_dim=struct_dim,
            regression_loss_type=reg_loss,
            use_input_length_feature=use_length,
        )

    model = DebertaHybridBucketModel(
        backbone=backbone,
        backbone_hidden_size=int(backbone_config.hidden_size),
        num_types=NUM_TYPES,
        hidden_dim=hidden_dim,
        structural_feature_dim=struct_dim,
        regression_loss_type=reg_loss,
        use_input_length_feature=use_length,
    )
    return model, backbone_config


def prepare_deberta_for_training(model):
    from predictors.output_length.common import core_model

    core = core_model(model)
    backbone = getattr(core, "deberta", None)
    if backbone is not None and hasattr(backbone, "config"):
        backbone.config.use_cache = False
        for attr in ("output_hidden_states", "output_attentions"):
            if hasattr(backbone.config, attr):
                setattr(backbone.config, attr, False)
    core._backbone_micro_batch_encode = False
    return core


def disable_deberta_gradient_checkpointing(model):
    from predictors.output_length.common import core_model

    core = core_model(model)
    backbone = getattr(core, "deberta", None)
    if backbone is not None and hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    return bool(backbone)


def freeze_hybrid_deberta_backbone(
    model,
    *,
    eval_mode=True,
    offload_to_cpu=False,
    micro_batch_encode=False,
):
    from predictors.output_length.common import core_model

    core = core_model(model)
    for param in core.deberta.parameters():
        param.requires_grad = False
    if eval_mode:
        core.deberta.eval()
    core._backbone_micro_batch_encode = bool(micro_batch_encode)
    disable_deberta_gradient_checkpointing(model)
    if offload_to_cpu:
        core.deberta.to("cpu")
    return core


def log_npu_memory(device, label=""):
    from predictors.output_length.training_memory import log_device_memory

    log_device_memory(device, label or "snapshot")


def enable_deberta_gradient_checkpointing(model, *, use_data_parallel=False):
    from predictors.output_length.common import _stdout, core_model

    if use_data_parallel:
        _stdout(
            "WARN: gradient checkpointing disabled with DataParallel "
            "(use --gradient-accumulation-steps and/or lower --batch-size)"
        )
        return False

    core = core_model(model)
    backbone = getattr(core, "deberta", None)
    if backbone is None:
        return False
    if not hasattr(backbone, "gradient_checkpointing_enable"):
        return False

    kwargs = {"use_reentrant": False}
    try:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    except TypeError:
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False
    return True


def release_training_batch_memory(device):
    import gc

    gc.collect()
    if hasattr(device, "type") and device.type == "npu":
        import torch

        if hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()


def _batch_size(batch):
    labels = batch.get("labels")
    if labels is not None and hasattr(labels, "size"):
        return int(labels.size(0))
    input_ids = batch.get("input_ids")
    if input_ids is not None and hasattr(input_ids, "size"):
        return int(input_ids.size(0))
    return 0


def _iter_micro_batches(batch, micro_batch_size):
    total = _batch_size(batch)
    if total <= 0:
        yield batch
        return
    size = int(micro_batch_size or 0)
    if size <= 0 or size >= total:
        yield batch
        return
    for start in range(0, total, size):
        end = min(total, start + size)
        chunk = {}
        for key, value in batch.items():
            if hasattr(value, "size") and int(value.size(0)) == total:
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        yield chunk


def _hybrid_forward_batch(model, batch, device, *, teacher_bucket_ids=None, collect_metadata=False):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    types = batch["type"].to(device, non_blocking=True).long()
    struct = batch.get("structural_features")
    if struct is not None:
        struct = struct.to(device, non_blocking=True)
    original_lengths = _batch_original_lengths(batch, device)
    model_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_type": types,
        "return_dict": True,
        "collect_metadata": bool(collect_metadata),
    }
    if struct is not None:
        model_kwargs["structural_features"] = struct
    if original_lengths is not None:
        model_kwargs["original_input_token_count"] = original_lengths
    if teacher_bucket_ids is not None:
        model_kwargs["bucket_ids"] = teacher_bucket_ids.long()
    outputs = model(**model_kwargs)
    return {
        "bucket_logits": outputs["bucket_logits"],
        "regression": outputs["regression"],
    }


def _forward_teacher_forced(model, batch, device):
    labels = batch["labels"].to(device, non_blocking=True).float()
    bucket_targets = batch.get("bucket_id")
    if bucket_targets is not None:
        bucket_targets = bucket_targets.to(device, non_blocking=True).long()
    else:
        bucket_targets = torch.tensor(
            [token_length_to_bucket_id(v) for v in labels.detach().cpu().tolist()],
            device=device,
            dtype=torch.long,
        )
    outputs = _hybrid_forward_batch(
        model, batch, device, teacher_bucket_ids=bucket_targets, collect_metadata=False
    )
    return outputs, labels, bucket_targets


def train_epoch_hybrid(
    model,
    loader,
    optimizer,
    device,
    train_mode,
    *,
    bucket_loss_weight=1.0,
    regression_loss_weight=1.0,
    regression_loss_type=REGRESSION_LOSS_MSE,
    gradient_accumulation_steps=1,
    train_micro_batch_size=None,
    precision=None,
    grad_scaler=None,
    low_memory=False,
    memory_diagnostics=None,
    progress_desc="",
    show_progress=True,
):
    from predictors.output_length.common import get_progress_iter
    from predictors.output_length.training_memory import (
        TrainingPrecision,
        training_autocast,
    )

    if precision is None:
        precision = TrainingPrecision("fp32", use_autocast=False, use_grad_scaler=False)

    if train_mode:
        model.train()
    else:
        model.eval()

    accum_steps = max(1, int(gradient_accumulation_steps))
    total = 0.0
    count = 0
    pending = 0
    cache_interval = 10 if low_memory else 0
    total_batches = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    iterator = get_progress_iter(loader, total=total_batches, desc=progress_desc)
    if not show_progress:
        iterator = loader
    if train_mode:
        optimizer.zero_grad(set_to_none=True)
    val_cache_interval = 1 if low_memory else 10
    for batch_idx, batch in enumerate(iterator):
        batch_items = _batch_size(batch)
        batch_loss_total = 0.0
        micro_batches = list(_iter_micro_batches(batch, train_micro_batch_size))
        forward_ctx = torch.inference_mode if not train_mode else torch.enable_grad
        for micro_idx, micro_batch in enumerate(micro_batches):
            with forward_ctx():
                with training_autocast(device, precision):
                    outputs, labels, bucket_targets = _forward_teacher_forced(model, micro_batch, device)
                    loss, _parts = hybrid_loss(
                        outputs,
                        labels,
                        bucket_targets=bucket_targets,
                        bucket_loss_weight=bucket_loss_weight,
                        regression_loss_weight=regression_loss_weight,
                        regression_loss_type=regression_loss_type,
                    )
                if memory_diagnostics is not None and batch_idx == 0 and micro_idx == 0:
                    memory_diagnostics.log("after_first_forward")
                batch_loss_total += loss.item() * labels.size(0)
                if train_mode:
                    micro_weight = float(labels.size(0)) / max(1, batch_items)
                    scaled = (loss * micro_weight) / accum_steps
                    if grad_scaler is not None:
                        grad_scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                    if memory_diagnostics is not None and batch_idx == 0 and micro_idx == 0:
                        memory_diagnostics.log("after_first_backward")
            del outputs, loss, labels, bucket_targets
            if low_memory and len(micro_batches) > 1 and micro_idx + 1 < len(micro_batches):
                release_training_batch_memory(device)
        if train_mode:
            pending += 1
            if pending >= accum_steps:
                if grad_scaler is not None:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                if memory_diagnostics is not None and batch_idx == 0:
                    memory_diagnostics.log("after_first_optimizer_step")
        del batch
        total += batch_loss_total
        count += batch_items
        if train_mode:
            if cache_interval and (batch_idx + 1) % cache_interval == 0:
                release_training_batch_memory(device)
        elif val_cache_interval and (batch_idx + 1) % val_cache_interval == 0:
            release_training_batch_memory(device)
        if hasattr(iterator, "set_postfix") and count:
            iterator.set_postfix(loss=total / count)
    if train_mode and pending > 0:
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total / max(count, 1)


def _accumulate_hybrid_metrics(model, loader, device, regression_loss_type):
    import numpy as np

    from predictors.output_length.buckets import BUCKETS
    from predictors.output_length.common import core_model

    was_training = model.training
    model.eval()
    micro_batch_size = int(getattr(core_model(model), "_train_micro_batch_size", 0) or 0)
    all_true = []
    all_pred = []
    all_true_bucket = []
    all_pred_bucket = []
    all_true_log = []
    all_pred_log = []
    all_input_lengths = []
    per_bucket_true = {i: [] for i in range(len(BUCKETS))}
    per_bucket_pred = {i: [] for i in range(len(BUCKETS))}
    top2_hits = 0
    total_rows = 0

    from predictors.output_length.training_memory import (
        resolve_training_precision,
        training_autocast,
    )

    eval_precision = resolve_training_precision(device, auto_on_accelerator=True)
    eval_cache_interval = 10
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch_original_lengths = batch.get("original_input_token_count")
            if batch_original_lengths is not None:
                all_input_lengths.extend(batch_original_lengths.detach().cpu().tolist())
            for micro_batch in _iter_micro_batches(batch, micro_batch_size):
                with training_autocast(device, eval_precision):
                    labels = micro_batch["labels"].to(device).float()
                    bucket_targets = micro_batch.get("bucket_id")
                    if bucket_targets is not None:
                        true_bucket = bucket_targets.to(device).long()
                    else:
                        true_bucket = torch.tensor(
                            [token_length_to_bucket_id(v) for v in labels.detach().cpu().tolist()],
                            device=device,
                            dtype=torch.long,
                        )
                    outputs = _hybrid_forward_batch(model, micro_batch, device, collect_metadata=False)
                    bucket_logits = outputs["bucket_logits"]
                    pred_bucket = bucket_logits.argmax(dim=-1)
                    top2 = bucket_logits.topk(min(2, bucket_logits.size(-1)), dim=-1).indices
                    top2_hits += int((top2 == true_bucket.unsqueeze(-1)).any(dim=-1).sum().item())
                    total_rows += labels.size(0)
                    pred_tokens = _decode_regression_tensor(outputs["regression"], regression_loss_type)
                    true_np = labels.detach().cpu().numpy()
                    pred_np = pred_tokens.detach().cpu().numpy()
                    true_bucket_np = true_bucket.detach().cpu().numpy()
                    pred_bucket_np = pred_bucket.detach().cpu().numpy()
                    del outputs, bucket_logits, pred_tokens
                all_true.extend(true_np.tolist())
                all_pred.extend(pred_np.tolist())
                all_true_bucket.extend(true_bucket_np.tolist())
                all_pred_bucket.extend(pred_bucket_np.tolist())
                all_true_log.extend(np.log1p(true_np).tolist())
                all_pred_log.extend(np.log1p(np.maximum(pred_np, 0.0)).tolist())
                for tb, y, p in zip(true_bucket_np.tolist(), true_np.tolist(), pred_np.tolist()):
                    per_bucket_true[tb].append(y)
                    per_bucket_pred[tb].append(p)
            del batch
            if eval_cache_interval and (batch_idx + 1) % eval_cache_interval == 0:
                release_training_batch_memory(device)

    if was_training:
        model.train()
    return {
        "all_true": all_true,
        "all_pred": all_pred,
        "all_true_bucket": all_true_bucket,
        "all_pred_bucket": all_pred_bucket,
        "all_true_log": all_true_log,
        "all_pred_log": all_pred_log,
        "all_input_lengths": all_input_lengths,
        "per_bucket_true": per_bucket_true,
        "per_bucket_pred": per_bucket_pred,
        "top2_hits": top2_hits,
        "total_rows": total_rows,
    }


def _spearman(x, y):
    try:
        from scipy.stats import spearmanr

        value, _p = spearmanr(x, y)
        return float(value) if value == value else 0.0
    except Exception:
        return 0.0


def _r2_score(y_true, y_pred):
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def summarize_hybrid_metrics(accumulated, *, log_confusion=False):
    import numpy as np

    y_true = np.asarray(accumulated["all_true"], dtype=np.float64)
    y_pred = np.asarray(accumulated["all_pred"], dtype=np.float64)
    true_b = np.asarray(accumulated["all_true_bucket"], dtype=np.int64)
    pred_b = np.asarray(accumulated["all_pred_bucket"], dtype=np.int64)
    n = max(1, y_true.size)
    bucket_acc = float(np.mean(true_b == pred_b))
    top2_acc = float(accumulated.get("top2_hits", 0) / max(1, accumulated.get("total_rows", n)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae_log = float(np.mean(np.abs(np.asarray(accumulated["all_true_log"]) - np.asarray(accumulated["all_pred_log"]))))
    spearman = _spearman(y_true.tolist(), y_pred.tolist())
    input_lengths = accumulated.get("all_input_lengths") or []
    input_output_spearman = (
        _spearman(input_lengths, y_true.tolist()) if len(input_lengths) == y_true.size else 0.0
    )
    metrics = {
        "bucket_accuracy": bucket_acc,
        "bucket_top2_adjacent_accuracy": top2_acc,
        "mae": mae,
        "rmse": rmse,
        "r2": _r2_score(y_true, y_pred),
        "mae_log": mae_log,
        "spearman": spearman,
        "input_output_length_spearman": input_output_spearman,
        "per_bucket": {},
    }
    for bucket_id, truths in accumulated["per_bucket_true"].items():
        preds = accumulated["per_bucket_pred"][bucket_id]
        if not truths:
            continue
        truths_np = np.asarray(truths, dtype=np.float64)
        preds_np = np.asarray(preds, dtype=np.float64)
        metrics["per_bucket"][str(bucket_id)] = {
            "label": bucket_label(bucket_id),
            "count": int(len(truths)),
            "mae": float(np.mean(np.abs(truths_np - preds_np))),
            "rmse": float(np.sqrt(np.mean((truths_np - preds_np) ** 2))),
        }
    if log_confusion:
        num_buckets = len(BUCKETS)
        matrix = np.zeros((num_buckets, num_buckets), dtype=np.int64)
        for t, p in zip(true_b.tolist(), pred_b.tolist()):
            matrix[t, p] += 1
        lines = ["bucket confusion (rows=true, cols=pred):"]
        header = "     " + " ".join("%5d" % i for i in range(num_buckets))
        lines.append(header)
        for i in range(num_buckets):
            lines.append("%4d |" % i + " ".join("%5d" % matrix[i, j] for j in range(num_buckets)))
        print("\n".join(lines), file=sys.stdout, flush=True)
    return metrics


def evaluate_hybrid(model, loader, device, *, regression_loss_type=REGRESSION_LOSS_MSE, log_confusion=False):
    accumulated = _accumulate_hybrid_metrics(model, loader, device, regression_loss_type)
    return summarize_hybrid_metrics(accumulated, log_confusion=log_confusion)


def evaluate_hybrid_ablation(
    model,
    loader,
    device,
    *,
    regression_loss_type=REGRESSION_LOSS_MSE,
    log_confusion=False,
):
    from predictors.output_length.common import core_model

    core = core_model(model)
    previous_encoder = bool(core.input_length_encoder.enabled)
    try:
        core.input_length_encoder.enabled = True
        with_metrics = evaluate_hybrid(
            model, loader, device, regression_loss_type=regression_loss_type, log_confusion=log_confusion
        )
        core.input_length_encoder.enabled = False
        without_metrics = evaluate_hybrid(
            model, loader, device, regression_loss_type=regression_loss_type, log_confusion=False
        )
    finally:
        core.input_length_encoder.enabled = previous_encoder
    return {
        "with_input_length_feature": with_metrics,
        "without_input_length_feature": without_metrics,
    }
