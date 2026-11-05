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
    DEFAULT_BERT_MODEL_DIR,
    DEFAULT_CSV,
    NUM_TYPES,
    OutputLengthPredictor,
    PREDICTION_POINT,
    WEIGHTS_NAME_LEGACY,
    _stdout,
    core_model,
)
from predictors.output_length.dataset_io import build_dataloader, load_labeled_csv, split_rows
from predictors.output_length.trainer_utils import progress_iter

DEFAULT_BERT_NAME = "bert-base-uncased"
DEFAULT_HIDDEN_DIM = 128
PREDICTOR_TYPE = "bert_point"


def train_epoch_point(model, loader, criterion, optimizer, device, train_mode, progress_desc=""):
    import torch

    if train_mode:
        model.train()
    else:
        model.eval()

    total = 0.0
    count = 0
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    iterator = progress_iter(loader, total=total_batches, desc=progress_desc)
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()
        types = batch["type"].to(device).long()

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_type=types,
            )
            loss = criterion(pred, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

        total += loss.item() * labels.size(0)
        count += labels.size(0)
        if hasattr(iterator, "set_postfix") and count:
            iterator.set_postfix(loss=total / count)

    return total / max(count, 1)


def train_output_length_predictor(
    csv_path=None,
    model_dir=None,
    bert_pretrained_name=DEFAULT_BERT_NAME,
    hidden_dim=DEFAULT_HIDDEN_DIM,
    num_epochs=6,
    batch_size=16,
    learning_rate=1e-5,
    freeze_bert_after_epoch=3,
    max_length=512,
    test_size=0.2,
    val_size=0.2,
    seed=42,
    use_l1_loss=False,
    ascend_devices=None,
):
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer

    from predictors.output_length.train_output_length import build_bert_regression_model

    csv_path = Path(csv_path or DEFAULT_CSV)
    model_dir = Path(model_dir or DEFAULT_BERT_MODEL_DIR)

    rows = load_labeled_csv(csv_path)
    train_rows, val_rows, test_rows = split_rows(rows, test_size=test_size, val_size=val_size, seed=seed)
    _stdout(
        "splits: train=%d val=%d test=%d (total=%d)"
        % (len(train_rows), len(val_rows), len(test_rows), len(rows))
    )

    _stdout("loading tokenizer %s" % bert_pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_name)
    _stdout("building BERT model hidden_dim=%d" % int(hidden_dim))
    model, _config = build_bert_regression_model(bert_pretrained_name, hidden_dim, NUM_TYPES)

    device, npu_list, use_data_parallel = setup_training_devices(ascend_devices)
    model = model.to(device)
    model = wrap_model_for_devices(model, npu_list)
    if use_data_parallel:
        _stdout("training on Ascend NPUs %s with DataParallel" % ",".join(str(i) for i in npu_list))
    elif npu_list:
        _stdout("training on Ascend NPU %s" % npu_list[0])

    criterion = nn.L1Loss() if use_l1_loss else nn.MSELoss()
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
        if freeze_bert_after_epoch is not None and epoch == freeze_bert_after_epoch:
            _stdout("freezing BERT backbone after epoch %d" % epoch)
            for param in core_model(model).bert.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4,
            )

        train_loss = train_epoch_point(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            progress_desc="epoch %d/%d train" % (epoch + 1, num_epochs),
        )
        val_loss = train_epoch_point(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            progress_desc="epoch %d/%d val" % (epoch + 1, num_epochs),
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        _stdout("epoch %d/%d train_loss=%.6f val_loss=%.6f" % (epoch + 1, num_epochs, train_loss, val_loss))

    _stdout("running test evaluation")
    test_metrics = {"l1": 0.0, "mse": 0.0}
    model.eval()
    l1loss = nn.L1Loss(reduction="sum")
    mseloss = nn.MSELoss(reduction="sum")
    total_l1 = 0.0
    total_mse = 0.0
    count = 0
    try:
        total_test_batches = len(test_loader)
    except TypeError:
        total_test_batches = None
    iterator = progress_iter(test_loader, total=total_test_batches, desc="test")
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()
            types = batch["type"].to(device).long()
            pred = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_type=types,
            )
            total_l1 += l1loss(pred, labels).item()
            total_mse += mseloss(pred, labels).item()
            count += labels.size(0)
    if count:
        test_metrics = {"l1": total_l1 / count, "mse": total_mse / count}
    _stdout("test metrics: l1=%.4f mse=%.4f" % (test_metrics["l1"], test_metrics["mse"]))

    predictor = OutputLengthPredictor(
        model=core_model(model),
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        default_type=0,
        prediction_mode=PREDICTION_POINT,
    )

    metrics = {
        "csv_path": str(csv_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "history": history,
        "test_metrics": test_metrics,
        "bert_pretrained_name": bert_pretrained_name,
        "hidden_dim": hidden_dim,
        "predictor_type": PREDICTOR_TYPE,
        "prediction_mode": PREDICTION_POINT,
        "ascend_devices": npu_list,
        "data_parallel": use_data_parallel,
    }

    return save_predictor_bundle(
        predictor,
        model_dir,
        metrics,
        backbone_name=bert_pretrained_name,
        hidden_dim=hidden_dim,
        predictor_type=PREDICTOR_TYPE,
        weights_name=WEIGHTS_NAME_LEGACY,
    )


__all__ = [
    "DEFAULT_BERT_NAME",
    "DEFAULT_HIDDEN_DIM",
    "PREDICTOR_TYPE",
    "train_epoch_point",
    "train_output_length_predictor",
]
