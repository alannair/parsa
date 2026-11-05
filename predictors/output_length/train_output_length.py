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
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictors.output_length.common import (
    BACKBONE_DIR_NAME,
    DEFAULT_BERT_MODEL_DIR,
    DEFAULT_CSV,
    NUM_TYPES,
    _stdout,
)

DEFAULT_BERT_NAME = "bert-base-uncased"
DEFAULT_HIDDEN_DIM = 128
PREDICTOR_TYPE = "bert_point"


class BertRegressionModel(nn.Module):

    def __init__(self, backbone, backbone_hidden_size, num_types, hidden_dim):
        super().__init__()
        self.bert = backbone
        self.type_embed = nn.Embedding(int(num_types), 16)
        self.cls = nn.Linear(int(backbone_hidden_size), int(hidden_dim))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(hidden_dim) + 16, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), 1)

    def forward(self, input_ids, attention_mask, prompt_type):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.relu(self.cls(logits))
        type_vec = self.type_embed(prompt_type)
        output = self.relu(self.fc1(torch.cat((output, type_vec), dim=-1)))
        return self.fc2(output).squeeze(-1)


def build_bert_regression_model(bert_pretrained_name, hidden_dim, num_types):
    from transformers import AutoConfig, BertModel

    config = AutoConfig.from_pretrained(bert_pretrained_name)
    backbone = BertModel.from_pretrained(bert_pretrained_name)
    model = BertRegressionModel(
        backbone=backbone,
        backbone_hidden_size=int(config.hidden_size),
        num_types=int(num_types),
        hidden_dim=int(hidden_dim),
    )
    return model, config


def build_bert_regression_model_from_bundle(bundle_dir, config):
    from pathlib import Path

    from transformers import AutoConfig, BertModel

    bundle_dir = Path(bundle_dir)
    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    hidden_dim = int(config.get("hidden_dim", DEFAULT_HIDDEN_DIM))
    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.bin",
    )
    if backbone_dir.is_dir() and any((backbone_dir / name).is_file() for name in weight_names):
        _stdout("loading BERT backbone weights from %s" % backbone_dir)
        backbone = BertModel.from_pretrained(str(backbone_dir), local_files_only=True)
        backbone_config = backbone.config
    elif backbone_dir.is_dir() and (backbone_dir / "config.json").is_file():
        _stdout("loading BERT backbone config from %s" % backbone_dir)
        backbone_config = AutoConfig.from_pretrained(str(backbone_dir))
        backbone = BertModel.from_config(backbone_config)
    else:
        bert_name = (
            config.get("bert_pretrained_name")
            or config.get("backbone_name")
            or DEFAULT_BERT_NAME
        )
        _stdout("bundle backbone/ missing; falling back to HF build (%s)" % bert_name)
        return build_bert_regression_model(str(bert_name), hidden_dim, NUM_TYPES)

    model = BertRegressionModel(
        backbone=backbone,
        backbone_hidden_size=int(backbone_config.hidden_size),
        num_types=NUM_TYPES,
        hidden_dim=hidden_dim,
    )
    return model, backbone_config


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


def train_epoch_point(model, loader, criterion, optimizer, device, train_mode, progress_desc=""):
    from predictors.output_length.trainers.bert_point import train_epoch_point as _impl

    return _impl(model, loader, criterion, optimizer, device, train_mode, progress_desc=progress_desc)


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
    from predictors.output_length.api import train_bert_point

    return train_bert_point(
        csv_path=csv_path,
        model_dir=model_dir,
        bert_pretrained_name=bert_pretrained_name,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_bert_after_epoch=freeze_bert_after_epoch,
        max_length=max_length,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        use_l1_loss=use_l1_loss,
        ascend_devices=ascend_devices,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="predictors.output_length.train_output_length",
        description="Train legacy BERT point-regression output-length predictor",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run inference on --csv-path and write --output-csv",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--num-prompts", default=0, type=int)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV))
    parser.add_argument("--model-dir", default=str(DEFAULT_BERT_MODEL_DIR))
    parser.add_argument("--bert-name", default=DEFAULT_BERT_NAME)
    parser.add_argument("--hidden-dim", default=DEFAULT_HIDDEN_DIM, type=int)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--freeze-bert-after-epoch", default=3, type=int)
    parser.add_argument("--max-length", default=512, type=int)
    parser.add_argument("--l1-loss", action="store_true")
    parser.add_argument(
        "--ascend-devices",
        default=os.environ.get("ASCEND_DEVICES", "0,1"),
        help="Comma-separated Ascend NPU ids; DataParallel when len>1 (never uses CUDA)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
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
    train_output_length_predictor(
        csv_path=args.csv_path,
        model_dir=args.model_dir,
        bert_pretrained_name=args.bert_name,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_bert_after_epoch=args.freeze_bert_after_epoch,
        max_length=args.max_length,
        use_l1_loss=args.l1_loss,
        ascend_devices=args.ascend_devices,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
