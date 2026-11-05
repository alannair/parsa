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


import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from predictors.output_length._ascend_env import early_bootstrap_ascend_from_argv

early_bootstrap_ascend_from_argv(sys.argv[1:])

from predictors.output_length._ascend_env import default_ascend_device_spec
from predictors.output_length.common import DEFAULT_CSV, DEFAULT_DEBERTA_MODEL_DIR, bootstrap_ascend_training_env
DEFAULT_DEBERTA_NAME = "microsoft/deberta-v3-base"
DEFAULT_NUM_LAYERS = 12
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_HEAD_DIM = 128
REGRESSION_LOSS_MSE = "mse"
REGRESSION_LOSS_HUBER = "huber"
REGRESSION_LOSS_LOGNORMAL = "lognormal"

PREDICTOR_TYPE = "deberta_lognormal"


def predict_labeled_csv(
    csv_path,
    model_dir,
    output_csv,
    *,
    batch_size=16,
    max_length=512,
    ascend_devices=None,
    num_prompts=0,
    include_prompt=False,
    show_progress=True,
):
    from predictors.output_length.api import predict_labeled_bundle_csv

    return predict_labeled_bundle_csv(
        csv_path,
        model_dir,
        output_csv,
        batch_size=batch_size,
        max_length=max_length,
        ascend_devices=ascend_devices,
        num_prompts=num_prompts,
        include_prompt=include_prompt,
        show_progress=show_progress,
    )


def train_output_length_deberta_predictor(
    csv_path=None,
    model_dir=None,
    deberta_pretrained_name=DEFAULT_DEBERTA_NAME,
    num_layers=DEFAULT_NUM_LAYERS,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    hidden_dim=DEFAULT_HEAD_DIM,
    num_epochs=6,
    batch_size=16,
    learning_rate=1e-5,
    freeze_backbone_after_epoch=3,
    max_length=512,
    test_size=0.2,
    val_size=0.2,
    seed=42,
    ascend_devices=None,
):
    from predictors.output_length.api import train_deberta_lognormal

    return train_deberta_lognormal(
        csv_path=csv_path,
        model_dir=model_dir,
        deberta_pretrained_name=deberta_pretrained_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_backbone_after_epoch=freeze_backbone_after_epoch,
        max_length=max_length,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        ascend_devices=ascend_devices,
    )


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
    from predictors.output_length.api import train_deberta_hybrid

    return train_deberta_hybrid(
        csv_path=csv_path,
        model_dir=model_dir,
        deberta_pretrained_name=deberta_pretrained_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_backbone_after_epoch=freeze_backbone_after_epoch,
        max_length=max_length,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        ascend_devices=ascend_devices,
        bucket_loss_weight=bucket_loss_weight,
        regression_loss_weight=regression_loss_weight,
        regression_loss_type=regression_loss_type,
        head_tail_truncation=head_tail_truncation,
        use_structural_features=use_structural_features,
        use_input_length_feature=use_input_length_feature,
        ablation_input_length=ablation_input_length,
        gradient_checkpointing=gradient_checkpointing,
        gradient_accumulation_steps=gradient_accumulation_steps,
        micro_batch_size=micro_batch_size,
        train_head_only=train_head_only,
        backbone_on_cpu=backbone_on_cpu,
        low_memory=low_memory,
        low_memory_mode=low_memory_mode,
        bf16=bf16,
        fp16=fp16,
        fp32=fp32,
        memory_diagnostics=memory_diagnostics,
        log_confusion=log_confusion,
    )


def _log_hybrid_metrics(metrics):
    from predictors.output_length.trainer_utils import log_hybrid_metrics

    return log_hybrid_metrics(metrics)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="predictors.output_length.train_output_length_deberta",
        description=(
            "Train DeBERTa-v3 output-length predictor. "
            "Default: log-normal regression. With --bucket: hybrid bucket classifier + log-space regression."
        ),
    )
    parser.add_argument(
        "--materialize-backbone",
        action="store_true",
        help="Extract bundle/backbone/ weights from .pth (no HF model build; prefer materialize_backbone_raw)",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run inference on --csv-path and write --output-csv (same NPU path as training)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Results CSV path (required with --predict-only)",
    )
    parser.add_argument("--num-prompts", default=0, type=int)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV))
    parser.add_argument("--model-dir", default=str(DEFAULT_DEBERTA_MODEL_DIR))
    parser.add_argument("--deberta-name", default=DEFAULT_DEBERTA_NAME)
    parser.add_argument("--num-layers", default=DEFAULT_NUM_LAYERS, type=int)
    parser.add_argument("--hidden-size", default=DEFAULT_HIDDEN_SIZE, type=int)
    parser.add_argument("--hidden-dim", default=DEFAULT_HEAD_DIM, type=int)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument(
        "--freeze-backbone-after-epoch",
        default=None,
        type=int,
        help="Freeze backbone at this epoch (hybrid default: 3; legacy log-normal default: 3)",
    )
    parser.add_argument("--max-length", default=512, type=int)
    parser.add_argument(
        "--ascend-devices",
        default=default_ascend_device_spec("0,1"),
        help=(
            "Physical Ascend NPU id(s). Single id: python -m training. "
            "Multiple ids: launch with torchrun --nproc_per_node=N (DDP; DataParallel disabled on NPU)."
        ),
    )
    parser.add_argument(
        "--bucket",
        action="store_true",
        help="Enable hybrid bucket classification + log-space regression (see --regression-loss)",
    )
    parser.add_argument("--bucket-loss-weight", default=1.0, type=float)
    parser.add_argument("--regression-loss-weight", default=1.0, type=float)
    parser.add_argument(
        "--regression-loss",
        default=REGRESSION_LOSS_MSE,
        choices=(REGRESSION_LOSS_MSE, REGRESSION_LOSS_HUBER, REGRESSION_LOSS_LOGNORMAL),
        help="Regression loss in log1p space (bucket mode only; default mse)",
    )
    parser.add_argument(
        "--head-tail-truncation",
        action="store_true",
        help="Use head+tail token truncation instead of truncating only the tail",
    )
    parser.add_argument(
        "--use-structural-features",
        action="store_true",
        help="Concatenate scalar prompt features (newlines, code blocks, messages)",
    )
    parser.add_argument(
        "--disable-input-length-feature",
        action="store_true",
        help="Disable pre-truncation input token count embedding (hybrid mode; default: enabled)",
    )
    parser.add_argument(
        "--ablation-input-length",
        action="store_true",
        help="Report test metrics with and without input-length embedding (hybrid mode)",
    )
    parser.add_argument(
        "--log-confusion",
        action="store_true",
        help="Print bucket confusion matrix to stdout after test evaluation (bucket mode)",
    )
    parser.add_argument(
        "--train-head-only",
        action="store_true",
        help="Freeze DeBERTa for all epochs (skip backbone fine-tune; use if OOM)",
    )
    parser.add_argument(
        "--backbone-on-cpu",
        action="store_true",
        help="Run frozen DeBERTa on CPU (slow; only if frozen backbone OOMs on NPU)",
    )
    parser.add_argument(
        "--no-backbone-on-cpu",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Micro-batch frozen backbone encode + periodic cache empty",
    )
    parser.add_argument(
        "--low-memory-mode",
        action="store_true",
        help="Safe mode: bf16/fp16 + gradient checkpointing + memory diagnostics",
    )
    parser.add_argument("--bf16", action="store_true", help="Train/val with bf16 autocast (preferred on NPU)")
    parser.add_argument("--fp16", action="store_true", help="Train/val with fp16 autocast + GradScaler")
    parser.add_argument("--fp32", action="store_true", help="Full fp32 (highest memory; debugging)")
    parser.add_argument(
        "--memory-diagnostics",
        action="store_true",
        help="Log allocated/reserved/peak device memory at key training steps",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable DeBERTa gradient checkpointing (auto-on when backbone trains)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        default=1,
        type=int,
        help="Micro-batches per optimizer step (hybrid mode)",
    )
    parser.add_argument(
        "--micro-batch-size",
        default=0,
        type=int,
        help=(
            "Maximum samples per forward/backward inside each per-rank batch. "
            "0 = auto memory-safe choice."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.materialize_backbone and not args.predict_only:
        bootstrap_ascend_training_env(args.ascend_devices)
    if args.materialize_backbone:
        from predictors.output_length.bundle_io import materialize_bundle_backbone_from_pth_raw

        materialize_bundle_backbone_from_pth_raw(args.model_dir)
        return 0
    if args.predict_only:
        if not args.output_csv:
            raise SystemExit("--output-csv is required with --predict-only")
        predict_labeled_csv(
            csv_path=args.csv_path,
            model_dir=args.model_dir,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            max_length=args.max_length,
            ascend_devices=args.ascend_devices,
            num_prompts=args.num_prompts,
            include_prompt=args.include_prompt,
            show_progress=not args.no_progress,
        )
        return 0

    freeze_epoch = args.freeze_backbone_after_epoch
    if args.bucket:
        if freeze_epoch is None:
            freeze_epoch = 3
        backbone_on_cpu = bool(args.backbone_on_cpu)
        if args.no_backbone_on_cpu:
            backbone_on_cpu = False
        train_output_length_deberta_hybrid_predictor(
            csv_path=args.csv_path,
            model_dir=args.model_dir,
            deberta_pretrained_name=args.deberta_name,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            hidden_dim=args.hidden_dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            freeze_backbone_after_epoch=freeze_epoch,
            max_length=args.max_length,
            ascend_devices=args.ascend_devices,
            bucket_loss_weight=args.bucket_loss_weight,
            regression_loss_weight=args.regression_loss_weight,
            regression_loss_type=args.regression_loss,
            head_tail_truncation=args.head_tail_truncation,
            use_structural_features=args.use_structural_features,
            use_input_length_feature=not args.disable_input_length_feature,
            ablation_input_length=args.ablation_input_length,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            micro_batch_size=args.micro_batch_size if int(args.micro_batch_size or 0) > 0 else None,
            train_head_only=args.train_head_only,
            backbone_on_cpu=backbone_on_cpu,
            low_memory=args.low_memory,
            low_memory_mode=args.low_memory_mode,
            bf16=args.bf16,
            fp16=args.fp16,
            fp32=args.fp32,
            memory_diagnostics=args.memory_diagnostics,
            log_confusion=args.log_confusion,
        )
        return 0

    if freeze_epoch is None:
        freeze_epoch = 3
    train_output_length_deberta_predictor(
        csv_path=args.csv_path,
        model_dir=args.model_dir,
        deberta_pretrained_name=args.deberta_name,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone_after_epoch=freeze_epoch,
        max_length=args.max_length,
        ascend_devices=args.ascend_devices,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
