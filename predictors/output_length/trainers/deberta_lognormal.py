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

from predictors.output_length.ascend import setup_training_devices, wrap_model_for_devices
from predictors.output_length.bundle_io import save_predictor_bundle
from predictors.output_length.common import (
    DEFAULT_CSV,
    DEFAULT_DEBERTA_MODEL_DIR,
    NUM_TYPES,
    OutputLengthPredictor,
    PREDICTION_LOGNORMAL,
    WEIGHTS_NAME_DEBERTA,
    _stdout,
    core_model,
)
from predictors.output_length.dataset_io import build_dataloader, load_labeled_csv, split_rows
from predictors.output_length.deberta_distribution import (
    DEFAULT_DEBERTA_NAME,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    build_deberta_distribution_model,
    evaluate_distribution,
    train_epoch_distribution,
)

PREDICTOR_TYPE = "deberta_lognormal"


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
    import torch
    from transformers import AutoTokenizer

    csv_path = Path(csv_path or DEFAULT_CSV)
    model_dir = Path(model_dir or DEFAULT_DEBERTA_MODEL_DIR)

    rows = load_labeled_csv(csv_path)
    train_rows, val_rows, test_rows = split_rows(rows, test_size=test_size, val_size=val_size, seed=seed)
    _stdout(
        "splits: train=%d val=%d test=%d (total=%d)"
        % (len(train_rows), len(val_rows), len(test_rows), len(rows))
    )

    _stdout("loading tokenizer %s" % deberta_pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(deberta_pretrained_name)
    _stdout(
        "building DeBERTa model layers=%d hidden_size=%d head_dim=%d"
        % (int(num_layers), int(hidden_size), int(hidden_dim))
    )
    model, _config = build_deberta_distribution_model(
        deberta_pretrained_name,
        NUM_TYPES,
        hidden_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )

    device, npu_list, use_data_parallel = setup_training_devices(ascend_devices)
    if use_data_parallel:
        raise RuntimeError(
            "Legacy DeBERTa log-normal training does not shard memory on Ascend multi-NPU. "
            "Use `--bucket` with torchrun DDP, or pass a single --ascend-devices id."
        )
    model = model.to(device)
    model = wrap_model_for_devices(model, npu_list)
    if npu_list:
        _stdout("training on Ascend NPU %s" % npu_list[0])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    train_loader = build_dataloader(train_rows, tokenizer, batch_size, max_length, shuffle=True)
    val_loader = build_dataloader(val_rows, tokenizer, batch_size, max_length, shuffle=False)
    test_loader = build_dataloader(test_rows, tokenizer, batch_size, max_length, shuffle=False)

    history = {"train_loss": [], "val_loss": []}

    _stdout(
        "starting training: epochs=%d batch_size=%d max_length=%d lr=%g"
        % (int(num_epochs), int(batch_size), int(max_length), float(learning_rate))
    )

    for epoch in range(num_epochs):
        if freeze_backbone_after_epoch is not None and epoch == freeze_backbone_after_epoch:
            _stdout("freezing DeBERTa backbone after epoch %d" % epoch)
            for param in core_model(model).deberta.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4,
            )

        train_loss = train_epoch_distribution(
            model,
            train_loader,
            optimizer,
            device,
            train_mode=True,
            progress_desc="epoch %d/%d train" % (epoch + 1, num_epochs),
        )
        val_loss = train_epoch_distribution(
            model,
            val_loader,
            optimizer,
            device,
            train_mode=False,
            progress_desc="epoch %d/%d val" % (epoch + 1, num_epochs),
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        _stdout("epoch %d/%d train_nll=%.6f val_nll=%.6f" % (epoch + 1, num_epochs, train_loss, val_loss))

    _stdout("running test evaluation")
    test_metrics = evaluate_distribution(model, test_loader, device, progress_desc="test")
    _stdout(
        "test metrics: l1=%.4f mse=%.4f nll=%.4f"
        % (test_metrics["l1"], test_metrics["mse"], test_metrics["nll"])
    )

    predictor = OutputLengthPredictor(
        model=core_model(model),
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        default_type=0,
        prediction_mode=PREDICTION_LOGNORMAL,
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
        "predictor_type": PREDICTOR_TYPE,
        "prediction_mode": PREDICTION_LOGNORMAL,
        "ascend_devices": npu_list,
        "data_parallel": use_data_parallel,
    }

    return save_predictor_bundle(
        predictor,
        model_dir,
        metrics,
        backbone_name=deberta_pretrained_name,
        hidden_dim=hidden_dim,
        predictor_type=PREDICTOR_TYPE,
        num_layers=num_layers,
        hidden_size=hidden_size,
        weights_name=WEIGHTS_NAME_DEBERTA,
    )


__all__ = [
    "PREDICTOR_TYPE",
    "train_output_length_deberta_predictor",
]
