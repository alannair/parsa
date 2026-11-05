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

DEFAULT_DEBERTA_NAME = "microsoft/deberta-v3-base"
DEFAULT_NUM_LAYERS = 12
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_HEAD_DIM = 128
LOG_TARGET_EPS = 1.0


class DebertaDistributionModel(nn.Module):

    def __init__(self, backbone, backbone_hidden_size, num_types, hidden_dim):
        super().__init__()
        self.deberta = backbone
        self.type_embed = nn.Embedding(int(num_types), 16)
        self.cls = nn.Linear(int(backbone_hidden_size), int(hidden_dim))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(int(hidden_dim) + 16, int(hidden_dim))
        self.fc_dist = nn.Linear(int(hidden_dim), 2)

    def encode(self, input_ids, attention_mask, prompt_type):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        hidden = self.relu(self.cls(pooled))
        type_vec = self.type_embed(prompt_type)
        hidden = self.relu(self.fc1(torch.cat((hidden, type_vec), dim=-1)))
        params = self.fc_dist(hidden)
        mu_log = params[:, 0]
        log_var = params[:, 1].clamp(min=-8.0, max=8.0)
        return torch.stack([mu_log, log_var], dim=-1)

    def forward(self, input_ids, attention_mask, prompt_type):
        return self.encode(input_ids, attention_mask, prompt_type)


def _materialize_deberta_backbone(
    pretrained_name,
    config,
    *,
    hidden_size_changed,
    num_layers_changed,
    requested_hidden_size,
    pretrained_hidden_size,
    requested_num_layers,
    pretrained_num_layers,
):
    from transformers import AutoModel

    if hidden_size_changed:
        print(
            "[deberta] hidden_size %d != pretrained %d; initializing backbone from config (no pretrained weights)"
            % (requested_hidden_size, pretrained_hidden_size),
            file=sys.stdout,
            flush=True,
        )
        return AutoModel.from_config(config)
    if num_layers_changed:
        print(
            "[deberta] num_hidden_layers %d != pretrained %d; loading with ignore_mismatched_sizes=True"
            % (requested_num_layers, pretrained_num_layers),
            file=sys.stdout,
            flush=True,
        )
        return AutoModel.from_pretrained(
            pretrained_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    return AutoModel.from_pretrained(pretrained_name, config=config)


def _bundle_backbone_has_weights(backbone_dir):
    from pathlib import Path

    root = Path(backbone_dir)
    if not root.is_dir():
        return False
    names = (
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.bin",
    )
    return any((root / name).is_file() for name in names)


def build_deberta_distribution_model_from_bundle(bundle_dir, config):
    from pathlib import Path

    from transformers import AutoModel

    from predictors.output_length.common import BACKBONE_DIR_NAME, NUM_TYPES, _stdout

    bundle_dir = Path(bundle_dir)
    backbone_dir = bundle_dir / BACKBONE_DIR_NAME
    hidden_dim = int(config.get("hidden_dim", DEFAULT_HEAD_DIM))
    num_layers = int(config.get("num_layers", DEFAULT_NUM_LAYERS))
    hidden_size = int(config.get("hidden_size", DEFAULT_HIDDEN_SIZE))

    if _bundle_backbone_has_weights(backbone_dir):
        _stdout("[deberta] AutoModel.from_pretrained(%s, local_files_only)" % backbone_dir)
        backbone = AutoModel.from_pretrained(str(backbone_dir), local_files_only=True)
        backbone_config = backbone.config
        _stdout("[deberta] DebertaDistributionModel head hidden_dim=%d" % hidden_dim)
        model = DebertaDistributionModel(
            backbone=backbone,
            backbone_hidden_size=int(backbone_config.hidden_size),
            num_types=NUM_TYPES,
            hidden_dim=hidden_dim,
        )
        return model, backbone_config

    pretrained_name = (
        config.get("deberta_pretrained_name")
        or config.get("backbone_name")
        or DEFAULT_DEBERTA_NAME
    )
    _stdout(
        "[deberta] no local backbone weights; using training builder for %s "
        "(layers=%d hidden_size=%d)"
        % (pretrained_name, num_layers, hidden_size)
    )
    return build_deberta_distribution_model(
        str(pretrained_name),
        NUM_TYPES,
        hidden_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
    )


def build_deberta_distribution_model(
    pretrained_name,
    num_types,
    hidden_dim,
    num_layers=DEFAULT_NUM_LAYERS,
    hidden_size=DEFAULT_HIDDEN_SIZE,
):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(pretrained_name)
    pretrained_hidden_size = int(getattr(config, "hidden_size", DEFAULT_HIDDEN_SIZE))
    pretrained_num_layers = int(getattr(config, "num_hidden_layers", DEFAULT_NUM_LAYERS))
    requested_hidden_size = int(hidden_size)
    requested_num_layers = int(num_layers)

    config.num_hidden_layers = requested_num_layers
    config.hidden_size = requested_hidden_size

    hidden_size_changed = requested_hidden_size != pretrained_hidden_size
    num_layers_changed = requested_num_layers != pretrained_num_layers

    backbone = _materialize_deberta_backbone(
        pretrained_name,
        config,
        hidden_size_changed=hidden_size_changed,
        num_layers_changed=num_layers_changed,
        requested_hidden_size=requested_hidden_size,
        pretrained_hidden_size=pretrained_hidden_size,
        requested_num_layers=requested_num_layers,
        pretrained_num_layers=pretrained_num_layers,
    )

    model = DebertaDistributionModel(
        backbone=backbone,
        backbone_hidden_size=int(config.hidden_size),
        num_types=int(num_types),
        hidden_dim=int(hidden_dim),
    )
    return model, config


def log_targets(token_lengths, eps=LOG_TARGET_EPS):
    import torch

    return torch.log(token_lengths.clamp(min=eps))


def lognormal_gaussian_nll(dist_params, token_lengths, eps=LOG_TARGET_EPS):
    import torch

    mu_log = dist_params[:, 0]
    log_var = dist_params[:, 1]
    log_y = log_targets(token_lengths, eps=eps)
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (log_y - mu_log).pow(2) * inv_var).mean()


def expected_tokens_from_lognormal(mu_log, log_var):
    import torch

    return torch.exp(mu_log + 0.5 * torch.exp(log_var))


def quantile_tokens_from_lognormal(mu_log, log_var, quantile):
    import math
    import statistics

    import torch

    q = min(0.9999, max(0.0001, float(quantile or 0.5)))
    z_value = statistics.NormalDist().inv_cdf(q)
    sigma = torch.exp(0.5 * log_var)
    return torch.exp(mu_log + sigma * float(z_value))


def point_metrics(dist_params, token_lengths):
    import torch

    pred = expected_tokens_from_lognormal(dist_params[:, 0], dist_params[:, 1])
    labels = token_lengths.float()
    l1 = torch.abs(pred - labels).mean().item()
    mse = torch.nn.functional.mse_loss(pred, labels).item()
    return {"l1": l1, "mse": mse}


def train_epoch_distribution(model, loader, optimizer, device, train_mode, progress_desc=""):
    import torch

    from .common import get_progress_iter

    if train_mode:
        model.train()
    else:
        model.eval()

    total = 0.0
    count = 0
    total_batches = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    iterator = get_progress_iter(loader, total=total_batches, desc=progress_desc)
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()
        types = batch["type"].to(device).long()

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            dist_params = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_type=types,
            )
            loss = lognormal_gaussian_nll(dist_params, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

        total += loss.item() * labels.size(0)
        count += labels.size(0)
        if hasattr(iterator, "set_postfix") and count:
            iterator.set_postfix(nll=total / count)

    return total / max(count, 1)


def evaluate_distribution(model, loader, device, progress_desc="test"):
    import torch

    from .common import get_progress_iter

    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_nll = 0.0
    count = 0
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    iterator = get_progress_iter(loader, total=total_batches, desc=progress_desc)
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()
            types = batch["type"].to(device).long()
            dist_params = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_type=types,
            )
            metrics = point_metrics(dist_params, labels)
            total_l1 += metrics["l1"] * labels.size(0)
            total_mse += metrics["mse"] * labels.size(0)
            total_nll += lognormal_gaussian_nll(dist_params, labels).item() * labels.size(0)
            count += labels.size(0)
    if not count:
        return {"l1": 0.0, "mse": 0.0, "nll": 0.0}
    return {
        "l1": total_l1 / count,
        "mse": total_mse / count,
        "nll": total_nll / count,
    }
