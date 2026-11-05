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


def default_lmsys_dataset_dir():
    return os.path.expanduser(os.environ.get("LMSYS_DATASET_DIR", "~/Projects/workspace/datasets/lmsys-chat-1m"))


def default_seq_pred_lmsys_vicuna_dir():
    return os.path.expanduser(
        os.environ.get(
            "SEQ_PRED_LMSYS_DIR",
            "~/Projects/workspace/data/seq_pred/lmsys/data/lmsys_first_round_data_vicuna-13b_1000K",
        )
    )


def saved_dataset_dir_exists(path):
    target = os.path.expanduser(str(path or ""))
    return os.path.exists(os.path.join(target, "dataset_dict.json")) or os.path.exists(os.path.join(target, "state.json"))


def load_saved_lmsys(path=None):
    target = os.path.expanduser(path or default_lmsys_dataset_dir())
    from datasets import load_from_disk

    dataset = load_from_disk(target)
    if isinstance(dataset, dict):
        return dataset.get("train") or next(iter(dataset.values()))
    if hasattr(dataset, "keys"):
        if "train" in dataset:
            return dataset["train"]
    return dataset


def _message_content(message):
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part.get("text"))
        return "\n".join(piece.strip() for piece in pieces if str(piece).strip())
    value = message.get("value")
    if isinstance(value, str):
        return value.strip()
    return ""


def first_user_content(row):
    if not isinstance(row, dict):
        return None
    for key in ("conversation", "conversations", "messages"):
        items = row.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or item.get("from") or "").strip().lower()
            if role not in ("user", "human"):
                continue
            content = _message_content(item)
            if content:
                return content
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def _iter_dataset_rows(dataset):
    if isinstance(dataset, str):
        dataset = load_saved_lmsys(dataset)
    if hasattr(dataset, "__iter__"):
        for row in dataset:
            yield row


def iter_user_prompts(dataset=None, max_prompts=50, skip_rows=0):
    source = dataset or default_lmsys_dataset_dir()
    emitted = 0
    skipped = 0
    for row in _iter_dataset_rows(source):
        prompt = first_user_content(row)
        if not prompt:
            continue
        if skipped < int(skip_rows or 0):
            skipped += 1
            continue
        yield prompt
        emitted += 1
        if max_prompts and emitted >= int(max_prompts):
            break


def _iter_jsonish_rows(path):
    target = os.path.expanduser(path)
    if os.path.isdir(target):
        for entry in sorted(os.listdir(target)):
            child = os.path.join(target, entry)
            if os.path.isdir(child):
                continue
            if entry.endswith((".json", ".jsonl")):
                for row in _iter_jsonish_rows(child):
                    yield row
        return
    if target.endswith(".jsonl"):
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        for row in data:
            yield row
    elif isinstance(data, dict):
        for key in ("data", "rows", "items"):
            rows = data.get(key)
            if isinstance(rows, list):
                for row in rows:
                    yield row
                return
        yield data


def iter_decoded_seq_pred_prompts(data_dir=None, tokenizer_name="lmsys/vicuna-13b-v1.3", max_prompts=50, skip_rows=0):
    path = data_dir or default_seq_pred_lmsys_vicuna_dir()
    tokenizer = None
    emitted = 0
    skipped = 0
    for row in _iter_jsonish_rows(path):
        prompt = None
        if isinstance(row, dict):
            conversation = row.get("conversation")
            if isinstance(conversation, str) and conversation.strip():
                prompt = conversation.strip()
            elif isinstance(row.get("input_ids"), list):
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                prompt = tokenizer.decode(row.get("input_ids"), skip_special_tokens=True).strip()
        if not prompt:
            continue
        if skipped < int(skip_rows or 0):
            skipped += 1
            continue
        yield prompt
        emitted += 1
        if max_prompts and emitted >= int(max_prompts):
            break
