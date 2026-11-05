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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lmsys_data import first_user_content, saved_dataset_dir_exists

TYPE_CONVERSATION = 0
TYPE_CODE = 1
TYPE_SUMMARIZATION = 2

POOL_CONVERSATION = "conversation"
POOL_CODE = "code"
POOL_SUMMARIZATION = "summarization"
POOL_ALIASES = {
    "conv": POOL_CONVERSATION,
    "chat": POOL_CONVERSATION,
    "sum": POOL_SUMMARIZATION,
    "summarization": POOL_SUMMARIZATION,
    "arxiv": POOL_SUMMARIZATION,
    "code": POOL_CODE,
    "humaneval": POOL_CODE,
}

DEFAULT_SHAREGPT = "/datasets/sharegpt52k/"
DEFAULT_ARXIV = "/datasets/ccdv_arxiv-summarization/"
DEFAULT_HUMANEVAL = "/datasets/openai_openai_humaneval/"


def _expand(path):
    return os.path.expanduser(str(path or ""))


def _load_json_rows(path):
    target = _expand(path)
    if os.path.isdir(target):
        rows = []
        for entry in sorted(os.listdir(target)):
            child = os.path.join(target, entry)
            if os.path.isdir(child):
                continue
            if entry.endswith((".json", ".jsonl")):
                rows.extend(_load_json_rows(child))
        return rows
    if target.endswith(".jsonl"):
        rows = []
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "rows", "items", "problems"):
            rows = data.get(key)
            if isinstance(rows, list):
                return rows
        return [data]
    return []


def _limit_items(items, limit):
    items = list(items)
    if not limit or int(limit) <= 0:
        return items
    return items[: int(limit)]


def load_sharegpt_prompts(path, limit=None):
    target = _expand(path or DEFAULT_SHAREGPT)
    rows = []
    if os.path.isdir(target):
        for name in ("sg_90k_part1.json", "sg_90k_part2.json"):
            child = os.path.join(target, name)
            if not os.path.exists(child):
                raise FileNotFoundError(child)
            rows.extend(_load_json_rows(child))
    else:
        rows = _load_json_rows(target)
    prompts = []
    for row in rows:
        prompt = first_user_content(row)
        if prompt:
            prompts.append(prompt)
    return _limit_items(prompts, limit)


def _arxiv_article_text(row):
    if not isinstance(row, dict):
        return None
    text = row.get("article") or row.get("text") or row.get("abstract") or row.get("content")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _hf_dataset_split(dataset):
    split = dataset.get("train") if hasattr(dataset, "get") else None
    if split is None and isinstance(dataset, dict):
        split = next(iter(dataset.values()))
    if split is None:
        split = dataset
    return split


def load_arxiv_prompts(path, limit=None):
    target = _expand(path or DEFAULT_ARXIV)
    prompts = []
    prefix = "Summarize the following article:\n\n"
    if os.path.isdir(target) and saved_dataset_dir_exists(target):
        from datasets import load_from_disk

        dataset = load_from_disk(target)
        for row in _hf_dataset_split(dataset):
            text = _arxiv_article_text(row)
            if text:
                prompts.append(prefix + text)
    else:
        for row in _load_json_rows(target):
            text = _arxiv_article_text(row)
            if text:
                prompts.append(prefix + text)
    return _limit_items(prompts, limit)


def load_humaneval_prompts(path, limit=None):
    target = _expand(path or DEFAULT_HUMANEVAL)
    prompts = []
    if os.path.isdir(target) and saved_dataset_dir_exists(target):
        from datasets import load_from_disk

        dataset = load_from_disk(target)
        split = dataset.get("test") if hasattr(dataset, "get") else None
        if split is None and isinstance(dataset, dict):
            split = next(iter(dataset.values()))
        if split is None:
            split = dataset
        for row in split:
            prompt = row.get("prompt") if isinstance(row, dict) else None
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
    else:
        for row in _load_json_rows(target):
            if isinstance(row, dict) and isinstance(row.get("prompt"), str) and row.get("prompt").strip():
                prompts.append(row.get("prompt").strip())
    return _limit_items(prompts, limit)


def normalize_active_pools(active_pools):
    if active_pools is None:
        return {POOL_CONVERSATION, POOL_CODE, POOL_SUMMARIZATION}
    if isinstance(active_pools, str):
        parts = [item.strip().lower() for item in active_pools.split(",") if item.strip()]
    else:
        parts = [str(item).strip().lower() for item in active_pools if str(item).strip()]
    if not parts:
        return {POOL_CONVERSATION, POOL_CODE, POOL_SUMMARIZATION}
    normalized = set()
    for part in parts:
        name = POOL_ALIASES.get(part, part)
        if name not in (POOL_CONVERSATION, POOL_CODE, POOL_SUMMARIZATION):
            raise ValueError(
                "unknown pool %r; use conversation, code, and/or summarization" % part
            )
        normalized.add(name)
    return normalized


def load_prompt_pool(
    sharegpt_path=None,
    arxiv_path=None,
    humaneval_path=None,
    num_prompts=0,
    active_pools=None,
):
    pools = normalize_active_pools(active_pools)
    rows = []
    if POOL_CONVERSATION in pools:
        for prompt in load_sharegpt_prompts(sharegpt_path, limit=num_prompts):
            rows.append((prompt, TYPE_CONVERSATION))
    if POOL_SUMMARIZATION in pools:
        for prompt in load_arxiv_prompts(arxiv_path, limit=num_prompts):
            rows.append((prompt, TYPE_SUMMARIZATION))
    if POOL_CODE in pools:
        for prompt in load_humaneval_prompts(humaneval_path, limit=num_prompts):
            rows.append((prompt, TYPE_CODE))
    if not rows:
        raise ValueError("prompt pool is empty; check dataset paths and --pools")
    return rows
