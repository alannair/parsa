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


import csv
import random
import sys
from pathlib import Path

NUM_TYPES = 3

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    _limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(_limit)
            break
        except OverflowError:
            _limit = _limit // 10
            if _limit < 2**20:
                csv.field_size_limit(2**20)
                break


def _stdout(*parts):
    print(*parts, flush=True)


def load_labeled_csv(csv_path):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError("labeled CSV not found: %s" % path)
    rows = []
    _stdout("loading labels from %s" % path)
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = str(row.get("prompt") or "").strip()
            if not prompt:
                continue
            try:
                prompt_type = int(row.get("type"))
                output_length = float(row.get("output_length"))
            except (TypeError, ValueError):
                continue
            if prompt_type < 0 or prompt_type >= NUM_TYPES:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "type": prompt_type,
                    "output_length": output_length,
                }
            )
    if len(rows) < 10:
        raise ValueError("need at least 10 labeled rows in %s" % path)
    _stdout("loaded %d labeled rows" % len(rows))
    return rows


def split_rows(rows, test_size=0.2, val_size=0.2, seed=42):
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int((n - n_test) * val_size))
    test_rows = shuffled[:n_test]
    val_rows = shuffled[n_test : n_test + n_val]
    train_rows = shuffled[n_test + n_val :]
    if not train_rows:
        train_rows = shuffled[n_test:]
        val_rows = shuffled[: max(1, len(shuffled) // 5)]
    return train_rows, val_rows, test_rows


def build_dataloader(
    rows,
    tokenizer,
    batch_size,
    max_length,
    shuffle,
    *,
    head_tail_truncation=False,
    use_structural_features=False,
    use_bucket_targets=False,
    use_input_length_feature=False,
    drop_last=False,
    dist_rank=None,
    dist_world_size=1,
    num_workers=0,
    pin_memory=False,
):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import DataCollatorWithPadding

    from predictors.output_length.buckets import token_length_to_bucket_id
    from predictors.output_length.features import (
        structural_feature_vector,
        tokenize_prompt_with_metadata,
    )

    pad_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    class PromptDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            row = self.items[index]
            prompt = row["prompt"]
            if use_input_length_feature or head_tail_truncation:
                enc = tokenize_prompt_with_metadata(
                    tokenizer,
                    prompt,
                    max_length,
                    head_tail_truncation=head_tail_truncation,
                )
            else:
                enc = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                enc = {
                    "input_ids": enc["input_ids"],
                    "original_input_token_count": len(enc["input_ids"]),
                    "truncated_input_token_count": len(enc["input_ids"]),
                }
            enc["labels"] = float(row["output_length"])
            enc["type"] = int(row["type"])
            if use_bucket_targets:
                enc["bucket_id"] = int(token_length_to_bucket_id(row["output_length"]))
            if use_structural_features:
                enc["structural_features"] = structural_feature_vector(prompt)
            return enc

    def collate(batch):
        labels = [item.pop("labels") for item in batch]
        types = [item.pop("type") for item in batch]
        bucket_ids = None
        if use_bucket_targets:
            bucket_ids = [item.pop("bucket_id") for item in batch]
        struct_feats = None
        if use_structural_features:
            struct_feats = [item.pop("structural_features") for item in batch]
        original_lengths = None
        if use_input_length_feature:
            original_lengths = [item.pop("original_input_token_count") for item in batch]
        else:
            for item in batch:
                item.pop("original_input_token_count", None)
        truncated_lengths = [item.pop("truncated_input_token_count", None) for item in batch]
        token_features = [{"input_ids": item.pop("input_ids")} for item in batch]
        padded = pad_collator(token_features)
        padded["labels"] = torch.tensor(labels, dtype=torch.float32)
        padded["type"] = torch.tensor(types, dtype=torch.long)
        if bucket_ids is not None:
            padded["bucket_id"] = torch.tensor(bucket_ids, dtype=torch.long)
        if struct_feats is not None:
            padded["structural_features"] = torch.tensor(struct_feats, dtype=torch.float32)
        if original_lengths is not None:
            padded["original_input_token_count"] = torch.tensor(
                original_lengths, dtype=torch.float32
            )
        if any(value is not None for value in truncated_lengths):
            padded["truncated_input_token_count"] = torch.tensor(
                [int(v or 0) for v in truncated_lengths], dtype=torch.long
            )
        return padded

    dataset = PromptDataset(rows)
    sampler = None
    loader_shuffle = shuffle
    if dist_rank is not None and int(dist_world_size) > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=int(dist_world_size),
            rank=int(dist_rank),
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
        )
        loader_shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate,
        drop_last=bool(drop_last) if sampler is None else False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers) > 0,
    )


def write_predict_results_csv(
    rows,
    preds,
    output_csv,
    include_prompt=False,
    *,
    pred_buckets=None,
):
    import csv as _csv

    from predictors.output_length.buckets import bucket_label, token_length_to_bucket_id

    try:
        _csv.field_size_limit(sys.maxsize)
    except OverflowError:
        _csv.field_size_limit(2**31 - 1)

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["type", "input_length", "output_length", "prediction"]
    if pred_buckets is not None:
        columns.extend(["true_bucket", "pred_bucket", "true_bucket_label", "pred_bucket_label"])
    if include_prompt:
        columns = ["prompt"] + columns
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=columns, quoting=_csv.QUOTE_MINIMAL)
        writer.writeheader()
        for index, (row, pred) in enumerate(zip(rows, preds)):
            true_bucket = int(token_length_to_bucket_id(row["output_length"]))
            record = {
                "type": int(row["type"]),
                "input_length": int(row["input_length"]),
                "output_length": int(float(row["output_length"])),
                "prediction": float(pred),
            }
            if pred_buckets is not None:
                pred_bucket = int(pred_buckets[index])
                record["true_bucket"] = true_bucket
                record["pred_bucket"] = pred_bucket
                record["true_bucket_label"] = bucket_label(true_bucket)
                record["pred_bucket_label"] = bucket_label(pred_bucket)
            if include_prompt:
                record["prompt"] = row["prompt"]
            writer.writerow(record)
    return out


def tokenize_context_lengths(rows, tokenizer, max_length, show_progress=True):
    items = list(rows)
    if show_progress:
        from predictors.output_length.common import get_progress_iter

        iterator = get_progress_iter(items, total=len(items), desc="tokenizing prompts")
    else:
        iterator = items
    cap = int(max_length)
    for row in iterator:
        ids = tokenizer(
            row["prompt"],
            truncation=True,
            max_length=cap,
            add_special_tokens=True,
        )["input_ids"]
        row["input_length"] = len(ids)
    return list(rows)


__all__ = [
    "NUM_TYPES",
    "load_labeled_csv",
    "split_rows",
    "build_dataloader",
    "write_predict_results_csv",
    "tokenize_context_lengths",
]
