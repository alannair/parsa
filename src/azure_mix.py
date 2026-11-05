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
import math
import os
import sys

DEFAULT_SCHEDULE_CSV = "/datasets/azure-mix/scheduled_prompts.csv"


def raise_csv_field_size_limit():
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                break
            except OverflowError:
                limit = limit // 10
                if limit < 2**20:
                    csv.field_size_limit(2**20)
                    break


raise_csv_field_size_limit()

SCHEDULE_FIELDS = ("arrival_sec", "type", "pool", "dataset_idx", "prompt")

POOL_ALIASES = {
    "all": "all",
    "conv": "conversation",
    "conversation": "conversation",
    "chat": "conversation",
    "sum": "summarization",
    "summary": "summarization",
    "summarization": "summarization",
    "summarisation": "summarization",
    "code": "code",
    "coding": "code",
}


def coerce_prompt(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, list):
        pieces = []
        for part in value:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str) and part.get("text").strip():
                    pieces.append(part.get("text").strip())
            else:
                text = coerce_prompt(part)
                if text:
                    pieces.append(text)
        text = "\n".join(pieces).strip()
        return text or None
    if isinstance(value, dict):
        if isinstance(value.get("text"), str) and value.get("text").strip():
            return value.get("text").strip()
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def normalize_pool_name(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return POOL_ALIASES.get(text, text)


def _pick_field(row, names):
    if not isinstance(row, dict):
        return None
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def normalize_schedule_row(row):
    if not isinstance(row, dict):
        return None
    prompt = coerce_prompt(_pick_field(row, ("prompt", "text", "content", "message")))
    if not prompt:
        return None
    pool_raw = _pick_field(row, ("pool", "prompt_pool", "prompt_type", "category", "workload"))
    pool = normalize_pool_name(pool_raw)
    if not pool:
        entry_type = _pick_field(row, ("type",))
        if entry_type in (0, "0", "conversation", "conv", "chat"):
            pool = "conversation"
        elif entry_type in (1, "1", "code", "coding"):
            pool = "code"
        elif entry_type in ("summarization", "summarisation", "sum", "summary"):
            pool = "summarization"
    arrival = _pick_field(row, ("arrival_sec", "arrival", "arrival_s", "timestamp"))
    try:
        arrival_sec = max(0.0, float(arrival or 0.0))
    except Exception:
        arrival_sec = 0.0
    entry_type = _pick_field(row, ("type",))
    try:
        type_value = int(float(entry_type or 0))
    except Exception:
        type_value = 0
    dataset_idx = _pick_field(row, ("dataset_idx", "dataset_index", "idx"))
    try:
        dataset_idx_value = int(float(dataset_idx or 0))
    except Exception:
        dataset_idx_value = 0
    return {
        "arrival_sec": arrival_sec,
        "type": type_value,
        "pool": pool,
        "dataset_idx": dataset_idx_value,
        "prompt": prompt,
    }


def normalize_schedule_events(rows):
    events = []
    for row in rows or []:
        normalized = normalize_schedule_row(row)
        if normalized is not None:
            events.append(normalized)
    return events


def filter_events_for_active_pools(events, active_pools):
    if not events:
        return []
    pools = normalize_pool_name(active_pools) or "all"
    if pools == "all":
        return list(events)
    filtered = []
    for event in events:
        pool = normalize_pool_name(event.get("pool"))
        if pool == pools:
            filtered.append(event)
    return filtered


def events_with_prompts(events):
    usable = []
    for event in events or []:
        prompt = coerce_prompt(event.get("prompt"))
        if not prompt:
            continue
        merged = dict(event)
        merged["prompt"] = prompt
        usable.append(merged)
    return usable


def can_build_azure_events(build_kwargs):
    kwargs = dict(build_kwargs or {})
    has_csv = bool(kwargs.get("azure_conv_csv") or kwargs.get("azure_code_csv"))
    if not has_csv:
        return False
    active_pools = str(kwargs.get("active_pools") or "all")
    if active_pools in ("all", "conversation") and not kwargs.get("sharegpt52k"):
        return False
    if active_pools in ("all", "summarization") and not kwargs.get("arxiv_summarization"):
        return False
    if active_pools in ("all", "code") and not kwargs.get("openai_humaneval"):
        return False
    return True


def _expand_path(path):
    return os.path.expanduser(str(path or ""))


def read_csv_dict_rows(path):
    target = _expand_path(path)
    with open(target, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_schedule_file(path):
    target = _expand_path(path)
    if not target or not os.path.exists(target):
        return None
    if target.endswith((".parquet", ".pq")):
        import pandas

        frame = pandas.read_parquet(target)
        return normalize_schedule_events(frame.to_dict(orient="records"))
    if target.endswith(".csv"):
        return normalize_schedule_events(read_csv_dict_rows(target))
    import json

    with open(target, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return normalize_schedule_events(data)
    return None


def load_schedule_events(path, *, active_pools="all", max_arrivals=0):
    schedule_path = _expand_path(path or DEFAULT_SCHEDULE_CSV)
    events = read_schedule_file(schedule_path) or []
    events = filter_events_for_active_pools(events, active_pools)
    events = events_with_prompts(events)
    if not events:
        raise ValueError(
            "no usable prompts for active_pools=%s in schedule %s"
            % (active_pools, schedule_path)
        )
    events = sorted(events, key=lambda item: (float(item.get("arrival_sec", 0.0)), int(item.get("type", 0))))
    if max_arrivals and int(max_arrivals) > 0:
        events = events[: int(max_arrivals)]
    return events
