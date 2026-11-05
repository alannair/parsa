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


from pathlib import Path

from predictors.output_length.ascend import (
    distributed_barrier,
    is_main_training_process,
    setup_hybrid_training,
    unwrap_model_distributed,
    wrap_model_distributed,
)
from predictors.output_length.bundle_io import save_predictor_bundle
from predictors.output_length.common import (
    DEFAULT_CSV,
    DEFAULT_DEBERTA_MODEL_DIR,
    NUM_TYPES,
    OutputLengthPredictor,
    PREDICTION_HYBRID_BUCKET,
    WEIGHTS_NAME_DEBERTA_HYBRID,
    _stdout,
    core_model,
)
from predictors.output_length.dataset_io import build_dataloader, load_labeled_csv, split_rows
from predictors.output_length.deberta_distribution import (
    DEFAULT_DEBERTA_NAME,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
)
from predictors.output_length.deberta_hybrid import (
    PREDICTOR_TYPE_HYBRID,
    REGRESSION_LOSS_HUBER,
    REGRESSION_LOSS_LOGNORMAL,
    REGRESSION_LOSS_MSE,
    build_deberta_hybrid_model,
    enable_deberta_gradient_checkpointing,
    evaluate_hybrid,
    evaluate_hybrid_ablation,
    freeze_hybrid_deberta_backbone,
    log_npu_memory,
    prepare_deberta_for_training,
    train_epoch_hybrid,
)
from predictors.output_length.features import NUM_STRUCTURAL_FEATURES
from predictors.output_length.precision import (
    MemoryDiagnostics,
    create_grad_scaler,
    resolve_training_micro_batch_size,
    resolve_training_precision,
)
from predictors.output_length.trainer_utils import log_hybrid_metrics


def train_output_length_deberta_hybrid_predictor(
    csv_path=None,
    model_dir=None,
    deberta_pretrained_name=DEFAULT_DEBERTA_NAME,
    num_layers=DEFAULT_NUM_LAYERS,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    hidden_dim=DEFAULT_HEAD_DIM,
    num_epochs=6,
    batch_size=16,
    learning_rate=1e-5,
    freeze_backbone_after_epoch=None,
    max_length=512,
    test_size=0.2,
    val_size=0.2,
    seed=42,
    ascend_devices=None,
    *,
    bucket_loss_weight=1.0,
    regression_loss_weight=1.0,
    regression_loss_type=REGRESSION_LOSS_MSE,
    head_tail_truncation=False,
    use_structural_features=False,
    use_input_length_feature=True,
    ablation_input_length=False,
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    micro_batch_size=None,
    train_head_only=False,
    backbone_on_cpu=False,
    low_memory=False,
    low_memory_mode=False,
    bf16=False,
    fp16=False,
    fp32=False,
    memory_diagnostics=False,
    log_confusion=False,
):
    import torch
    from transformers import AutoTokenizer

    if low_memory_mode:
        low_memory = True
        memory_diagnostics = True
        gradient_checkpointing = True
        if not (bf16 or fp16 or fp32):
            bf16 = True

    csv_path = Path(csv_path or DEFAULT_CSV)
    model_dir = Path(model_dir or DEFAULT_DEBERTA_MODEL_DIR)
    struct_dim = NUM_STRUCTURAL_FEATURES if use_structural_features else 0

    rows = load_labeled_csv(csv_path)
    train_rows, val_rows, test_rows = split_rows(rows, test_size=test_size, val_size=val_size, seed=seed)
    _stdout(
        "hybrid splits: train=%d val=%d test=%d (total=%d)"
        % (len(train_rows), len(val_rows), len(test_rows), len(rows))
    )

    tokenizer = AutoTokenizer.from_pretrained(deberta_pretrained_name)
    model, _config = build_deberta_hybrid_model(
        deberta_pretrained_name,
        NUM_TYPES,
        hidden_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
        structural_feature_dim=struct_dim,
        regression_loss_type=regression_loss_type,
        use_input_length_feature=bool(use_input_length_feature),
    )

    train_ctx = setup_hybrid_training(ascend_devices)
    device = train_ctx["device"]
    npu_list = train_ctx["npu_list"]
    use_data_parallel = train_ctx["use_data_parallel"]
    world_size = int(train_ctx["world_size"])
    rank = int(train_ctx["rank"])
    main_process = is_main_training_process(train_ctx)
    precision = resolve_training_precision(
        device,
        bf16=bf16,
        fp16=fp16,
        fp32=fp32,
        low_memory_mode=low_memory_mode,
        auto_on_accelerator=not fp32,
    )
    grad_scaler = create_grad_scaler(device, precision)
    mem_diag = MemoryDiagnostics(device, enabled=memory_diagnostics, main_process=main_process)
    loader_workers = 0
    loader_pin_memory = False
    head_only = bool(train_head_only) or (
        freeze_backbone_after_epoch is not None and int(freeze_backbone_after_epoch) <= 0
    )
    if use_data_parallel and int(batch_size) < len(npu_list):
        _stdout(
            "WARN: --batch-size %d < %d NPUs; DataParallel leaves some devices idle. "
            "Prefer --batch-size >= %d (e.g. %d for ~2 samples/NPU)."
            % (int(batch_size), len(npu_list), len(npu_list), len(npu_list) * 2)
        )
    model = model.to(device)
    prepare_deberta_for_training(model)
    if head_only:
        freeze_hybrid_deberta_backbone(
            model,
            offload_to_cpu=bool(backbone_on_cpu),
            micro_batch_encode=bool(low_memory),
        )
        if main_process:
            if backbone_on_cpu:
                _stdout(
                    "DeBERTa frozen on CPU (--backbone-on-cpu; slower, use only if NPU OOM)"
                )
            elif low_memory:
                _stdout("DeBERTa frozen with micro-batch encode (--low-memory; slower)")
            else:
                _stdout("DeBERTa backbone frozen on device (head-only, full-batch encode)")
        log_npu_memory(device, "after backbone freeze")
    elif not gradient_checkpointing:
        gradient_checkpointing = True
        if main_process:
            _stdout("auto-enabling gradient checkpointing on DeBERTa backbone (trainable phase)")
    if gradient_checkpointing and not head_only:
        if enable_deberta_gradient_checkpointing(model, use_data_parallel=False):
            if main_process:
                _stdout("enabled DeBERTa gradient checkpointing (use_reentrant=False)")
        elif main_process:
            _stdout("WARN: gradient checkpointing not supported by backbone")


    model = wrap_model_distributed(
        model,
        train_ctx,
        find_unused_parameters=bool(head_only),
    )
    if main_process:
        _stdout(
            "hybrid precision=%s autocast=%s grad_scaler=%s"
            % (precision.name, precision.use_autocast, grad_scaler is not None)
        )
    mem_diag.reset_peak()
    mem_diag.log("after_model_init")
    log_npu_memory(device, "after model load")
    accum_steps = max(1, int(gradient_accumulation_steps))
    per_rank_batch = int(batch_size)
    train_micro_batch_size = resolve_training_micro_batch_size(
        per_rank_batch,
        device=device,
        max_length=max_length,
        train_head_only=head_only,
        low_memory=bool(low_memory or low_memory_mode),
        explicit_micro_batch_size=micro_batch_size,
    )
    effective_batch = per_rank_batch * accum_steps * world_size
    if train_ctx["is_distributed"]:
        if main_process:
            _stdout(
                "hybrid DDP on %d NPUs (per-rank batch=%d, micro-batch=%d, accum=%d, effective_global_batch=%d)"
                % (world_size, per_rank_batch, train_micro_batch_size, accum_steps, effective_batch)
            )
    elif npu_list:
        _stdout(
            "hybrid training on Ascend NPU %s (batch=%d, micro-batch=%d, accum=%d, effective_batch=%d)"
            % (npu_list[0], per_rank_batch, train_micro_batch_size, accum_steps, per_rank_batch * accum_steps)
        )
    if main_process and train_micro_batch_size < per_rank_batch:
        _stdout(
            "hybrid memory guard: splitting per-rank batch %d into micro-batches of %d"
            % (per_rank_batch, train_micro_batch_size)
        )
    core_model(model)._train_micro_batch_size = int(train_micro_batch_size)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    if head_only:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
        )

    loader_kwargs = dict(
        head_tail_truncation=head_tail_truncation,
        use_structural_features=use_structural_features,
        use_bucket_targets=True,
        use_input_length_feature=bool(use_input_length_feature),
    )
    loader_dist = dict(dist_rank=rank, dist_world_size=world_size)
    loader_extra = dict(num_workers=loader_workers, pin_memory=loader_pin_memory)
    train_loader = build_dataloader(
        train_rows,
        tokenizer,
        batch_size,
        max_length,
        True,
        drop_last=True,
        **loader_kwargs,
        **loader_dist,
        **loader_extra,
    )
    val_loader = build_dataloader(
        val_rows,
        tokenizer,
        batch_size,
        max_length,
        False,
        **loader_kwargs,
        **loader_dist,
        **loader_extra,
    )

    test_loader = build_dataloader(
        test_rows,
        tokenizer,
        batch_size,
        max_length,
        False,
        **loader_kwargs,
        dist_rank=None,
        dist_world_size=1,
        **loader_extra,
    )

    history = {"train_loss": [], "val_loss": []}
    _stdout(
        "hybrid training: epochs=%d batch_size=%d max_length=%d lr=%g "
        "bucket_w=%g regression_w=%g regression_loss=%s head_tail=%s structural=%s input_length=%s"
        % (
            int(num_epochs),
            int(batch_size),
            int(max_length),
            float(learning_rate),
            float(bucket_loss_weight),
            float(regression_loss_weight),
            regression_loss_type,
            head_tail_truncation,
            use_structural_features,
            use_input_length_feature,
        )
    )

    for epoch in range(num_epochs):
        if train_ctx["is_distributed"] and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        if freeze_backbone_after_epoch is not None and epoch == freeze_backbone_after_epoch:
            if int(freeze_backbone_after_epoch) > 0:
                if main_process:
                    _stdout("freezing DeBERTa backbone after epoch %d" % epoch)
                distributed_barrier(train_ctx)
                inner = unwrap_model_distributed(model, train_ctx)
                freeze_hybrid_deberta_backbone(
                    inner,
                    offload_to_cpu=bool(backbone_on_cpu),
                    micro_batch_encode=bool(low_memory),
                )
                if train_ctx["is_distributed"]:
                    model = wrap_model_distributed(
                        inner,
                        train_ctx,
                        find_unused_parameters=True,
                    )
                    core_model(model)._train_micro_batch_size = int(train_micro_batch_size)
                else:
                    model = inner
                distributed_barrier(train_ctx)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-4,
                )
                log_npu_memory(device, "after epoch freeze")

        train_loss = train_epoch_hybrid(
            model,
            train_loader,
            optimizer,
            device,
            train_mode=True,
            bucket_loss_weight=bucket_loss_weight,
            regression_loss_weight=regression_loss_weight,
            regression_loss_type=regression_loss_type,
            gradient_accumulation_steps=accum_steps,
            train_micro_batch_size=train_micro_batch_size,
            precision=precision,
            grad_scaler=grad_scaler,
            low_memory=low_memory or low_memory_mode,
            memory_diagnostics=mem_diag if memory_diagnostics else None,
            progress_desc="epoch %d/%d train" % (epoch + 1, num_epochs),
            show_progress=main_process,
        )
        val_loss = train_epoch_hybrid(
            model,
            val_loader,
            optimizer,
            device,
            train_mode=False,
            bucket_loss_weight=bucket_loss_weight,
            regression_loss_weight=regression_loss_weight,
            regression_loss_type=regression_loss_type,
            gradient_accumulation_steps=1,
            train_micro_batch_size=train_micro_batch_size,
            precision=precision,
            grad_scaler=None,
            low_memory=low_memory or low_memory_mode,
            memory_diagnostics=None,
            progress_desc="epoch %d/%d val" % (epoch + 1, num_epochs),
            show_progress=main_process,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if main_process:
            _stdout(
                "epoch %d/%d train_loss=%.6f val_loss=%.6f"
                % (epoch + 1, num_epochs, train_loss, val_loss)
            )

    distributed_barrier(train_ctx)
    bundle_dir_result = model_dir
    if main_process:
        _stdout("running hybrid test evaluation")
        if ablation_input_length and use_input_length_feature:
            ablation = evaluate_hybrid_ablation(
                model,
                test_loader,
                device,
                regression_loss_type=regression_loss_type,
                log_confusion=log_confusion,
            )
            test_metrics = ablation["with_input_length_feature"]
            _stdout("ablation (with input-length feature):")
            log_hybrid_metrics(test_metrics, log_fn=_stdout)
            _stdout("ablation (without input-length feature):")
            log_hybrid_metrics(ablation["without_input_length_feature"], log_fn=_stdout)
        else:
            test_metrics = evaluate_hybrid(
                model,
                test_loader,
                device,
                regression_loss_type=regression_loss_type,
                log_confusion=log_confusion,
            )
            log_hybrid_metrics(test_metrics, log_fn=_stdout)

        predictor = OutputLengthPredictor(
            model=core_model(model),
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            default_type=0,
            prediction_mode=PREDICTION_HYBRID_BUCKET,
            regression_loss_type=regression_loss_type,
            head_tail_truncation=head_tail_truncation,
            use_structural_features=use_structural_features,
            use_input_length_feature=bool(use_input_length_feature),
        )

        metrics = {
            "csv_path": str(csv_path),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "history": history,
            "test_metrics": test_metrics,
            "deberta_pretrained_name": deberta_pretrained_name,
            "num_layers": int(num_layers),
            "hidden_size": int(hidden_size),
            "hidden_dim": hidden_dim,
            "predictor_type": PREDICTOR_TYPE_HYBRID,
            "prediction_mode": PREDICTION_HYBRID_BUCKET,
            "ascend_devices": npu_list,
            "data_parallel": use_data_parallel,
            "distributed": train_ctx["is_distributed"],
            "world_size": world_size,
            "bucket_loss_weight": float(bucket_loss_weight),
            "regression_loss_weight": float(regression_loss_weight),
            "regression_loss": regression_loss_type,
            "head_tail_truncation": bool(head_tail_truncation),
            "use_structural_features": bool(use_structural_features),
            "structural_feature_dim": int(struct_dim),
            "use_input_length_feature": bool(use_input_length_feature),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "gradient_accumulation_steps": int(accum_steps),
            "train_micro_batch_size": int(train_micro_batch_size),
            "precision": precision.name,
            "low_memory": bool(low_memory),
            "low_memory_mode": bool(low_memory_mode),
            "backbone_on_cpu": bool(backbone_on_cpu),
            "memory_diagnostics": bool(memory_diagnostics),
            "freeze_backbone_after_epoch": freeze_backbone_after_epoch,
        }
        if ablation_input_length and use_input_length_feature:
            metrics["ablation_input_length"] = ablation

        bundle_dir_result = save_predictor_bundle(
            predictor,
            model_dir,
            metrics,
            backbone_name=deberta_pretrained_name,
            hidden_dim=hidden_dim,
            predictor_type=PREDICTOR_TYPE_HYBRID,
            num_layers=num_layers,
            hidden_size=hidden_size,
            weights_name=WEIGHTS_NAME_DEBERTA_HYBRID,
        )

    distributed_barrier(train_ctx)
    return bundle_dir_result


def _log_hybrid_metrics(metrics):
    return log_hybrid_metrics(metrics, log_fn=_stdout)


__all__ = [
    "train_output_length_deberta_hybrid_predictor",
    "_log_hybrid_metrics",
    "REGRESSION_LOSS_MSE",
    "REGRESSION_LOSS_HUBER",
    "REGRESSION_LOSS_LOGNORMAL",
]
