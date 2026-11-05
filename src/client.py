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
import asyncio
import csv
import json
import math
import os
import sys
import time

import aiohttp

from .azure_mix import (
    DEFAULT_SCHEDULE_CSV,
    SCHEDULE_FIELDS,
    can_build_azure_events,
    coerce_prompt as _coerce_prompt,
    events_with_prompts as _events_with_prompts,
    filter_events_for_active_pools as _filter_events_for_active_pools,
    normalize_schedule_events,
    read_csv_dict_rows,
    read_schedule_file,
)
from .lmsys_data import (
    default_lmsys_dataset_dir,
    default_seq_pred_lmsys_vicuna_dir,
    first_user_content,
    iter_decoded_seq_pred_prompts,
    iter_user_prompts,
    saved_dataset_dir_exists,
)
from .stats_correlation import pearson, spearman


BUILTIN_PROMPTS = (
    "Write a haiku about energy-aware inference.",
    "Explain why request steering matters for multi-lane LLM serving.",
    "Summarize the benefits of queue-aware scheduling in three bullets.",
)

HTTP_TIMEOUT = aiohttp.ClientTimeout(connect=120, sock_read=600, total=720)
AZURE_MIX_SCHEDULE_FIELDS = SCHEDULE_FIELDS


def _stderr(*parts):
    print(*parts, file=sys.stderr)


def _expand(path):
    return os.path.expanduser(str(path or ""))


def _request_timeout(args):
    seconds = getattr(args, "request_timeout_seconds", None)
    if seconds is None:
        return HTTP_TIMEOUT
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("--request-timeout-seconds must be finite and positive")
    return aiohttp.ClientTimeout(connect=120, sock_read=seconds, total=seconds)


def _json_dump_atomic(path, payload):
    target = _expand(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = target + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, target)


def _percentile(values, percentile):
    values = sorted(float(item) for item in list(values or []))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * float(percentile)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return values[low]
    fraction = rank - low
    return values[low] * (1.0 - fraction) + values[high] * fraction


def _mean(values):
    values = list(values or [])
    if not values:
        return None
    return sum(float(item) for item in values) / float(len(values))


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _write_result_csv(path, results):
    target = _expand(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = target + ".tmp"
    ordered = sorted(results or [], key=lambda row: _safe_int(row.get("prompt_idx")) or 0)
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "input_length", "output_length"))
        writer.writeheader()
        for result in ordered:
            prompt_idx = _safe_int(result.get("prompt_idx"))
            if prompt_idx is None or prompt_idx <= 0:
                raise ValueError("result CSV requires positive prompt_idx values")
            writer.writerow(
                {
                    "index": prompt_idx - 1,
                    "input_length": _safe_int(result.get("input_num_tokens")),
                    "output_length": _safe_int(result.get("output_num_tokens")) if result.get("ok") else None,
                }
            )
    os.replace(tmp_path, target)


def _parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        pass
    try:
        from datetime import datetime

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _load_csv_rows(path):
    return read_csv_dict_rows(path)


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
    if not limit or int(limit) <= 0:
        return list(items)
    return list(items)[: int(limit)]


def _load_sharegpt_prompts(path, limit=None):
    target = _expand(path)
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


def _load_arxiv_prompts(path, limit=None):
    rows = _load_json_rows(path)
    prompts = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("article") or row.get("text") or row.get("abstract") or row.get("content")
        if isinstance(text, str) and text.strip():
            prompts.append("Summarize the following article:\n\n" + text.strip())
    return _limit_items(prompts, limit)


def _load_humaneval_prompts(path, limit=None):
    target = _expand(path)
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


def _read_schedule_cache(path):
    return read_schedule_file(path)


def _write_schedule_cache(path, events):
    target = _expand(path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if target.endswith((".parquet", ".pq")):
        import pandas

        frame = pandas.DataFrame(events)
        tmp_path = target + ".tmp"
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, target)
        return
    if target.endswith(".csv"):
        tmp_path = target + ".tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AZURE_MIX_SCHEDULE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for event in events:
                writer.writerow(event)
        os.replace(tmp_path, target)
        return
    _json_dump_atomic(target, list(events))


def _input_size_from_row(row):
    for key in ("ContextTokens", "input_size", "InputTokens", "input_tokens"):
        value = _safe_int(row.get(key))
        if value is not None and value >= 0:
            return value
    return None


def build_azure_events(
    *,
    azure_conv_csv="",
    azure_code_csv="",
    azure_timestamp_col="TIMESTAMP",
    sharegpt52k="",
    arxiv_summarization="",
    openai_humaneval="",
    active_pools="all",
    pool_limit=0,
):
    if not azure_conv_csv and not azure_code_csv:
        raise ValueError("azure-mix requires azure_conv_csv and/or azure_code_csv")
    conv_pool = []
    sum_pool = []
    code_pool = []
    pool_limit = None if not pool_limit or int(pool_limit) <= 0 else int(pool_limit)
    if active_pools in ("all", "conversation"):
        if not sharegpt52k:
            raise ValueError("azure-mix requires sharegpt52k for conversation prompts")
        conv_pool = _load_sharegpt_prompts(sharegpt52k, limit=pool_limit)
    if active_pools in ("all", "summarization"):
        if not arxiv_summarization:
            raise ValueError("azure-mix requires arxiv_summarization for summarization prompts")
        sum_pool = _load_arxiv_prompts(arxiv_summarization, limit=pool_limit)
    if active_pools in ("all", "code"):
        if not openai_humaneval:
            raise ValueError("azure-mix requires openai_humaneval for code prompts")
        code_pool = _load_humaneval_prompts(openai_humaneval, limit=pool_limit)
    rows = []
    if azure_conv_csv:
        for row in _load_csv_rows(azure_conv_csv):
            rows.append({"type": 0, "row": row})
    if azure_code_csv:
        for row in _load_csv_rows(azure_code_csv):
            rows.append({"type": 1, "row": row})
    stamped = []
    for item in rows:
        ts = _parse_timestamp(item["row"].get(azure_timestamp_col))
        if ts is None:
            continue
        stamped.append((ts, item["type"], item["row"]))
    stamped.sort(key=lambda entry: (entry[0], entry[1]))
    if not stamped:
        return []
    first_ts = stamped[0][0]
    events = []
    rr_type0 = 0
    code_index = 0
    for ts, entry_type, row in stamped:
        if entry_type == 0:
            if active_pools == "code":
                continue
            if active_pools == "conversation":
                pool_name = "conversation"
                pool = conv_pool
            elif active_pools == "summarization":
                pool_name = "summarization"
                pool = sum_pool
            else:
                if rr_type0 % 2 == 0:
                    pool_name = "conversation"
                    pool = conv_pool
                else:
                    pool_name = "summarization"
                    pool = sum_pool
                rr_type0 += 1
            if not pool:
                continue
            dataset_idx = len(events) % len(pool)
            prompt = pool[dataset_idx]
        else:
            if active_pools in ("conversation", "summarization"):
                continue
            if not code_pool:
                continue
            pool_name = "code"
            dataset_idx = code_index % len(code_pool)
            code_index += 1
            prompt = code_pool[dataset_idx]
        events.append(
            {
                "arrival_sec": max(0.0, ts - first_ts),
                "type": int(entry_type),
                "pool": pool_name,
                "dataset_idx": int(dataset_idx),
                "prompt": prompt,
                "input_size": _input_size_from_row(row),
            }
        )
    return events


def load_or_build_azure_events(
    *,
    azure_mix_schedule_csv="",
    rebuild_azure_mix_schedule=False,
    max_arrivals=0,
    **build_kwargs,
):
    active_pools = str(build_kwargs.get("active_pools") or "all")
    schedule_path = _expand(azure_mix_schedule_csv or DEFAULT_SCHEDULE_CSV)
    events = []
    if schedule_path and os.path.exists(schedule_path) and not rebuild_azure_mix_schedule:
        events = read_schedule_file(schedule_path) or []
    elif can_build_azure_events(build_kwargs):
        events = build_azure_events(**build_kwargs)
        if schedule_path:
            _write_schedule_cache(schedule_path, events)
    elif schedule_path and os.path.exists(schedule_path):
        events = read_schedule_file(schedule_path) or []
    else:
        raise ValueError(
            "no prompt schedule found at %s and azure-mix rebuild inputs are unavailable"
            % schedule_path
        )
    return _finalize_azure_events(events, active_pools, max_arrivals, schedule_path)


def _finalize_azure_events(events, active_pools, max_arrivals, schedule_path):
    events = _filter_events_for_active_pools(events, active_pools)
    events = _events_with_prompts(events)
    if not events:
        raise ValueError(
            "no usable prompts for active_pools=%s in schedule %s"
            % (active_pools, schedule_path)
        )
    events = sorted(events, key=lambda item: (float(item.get("arrival_sec", 0.0)), int(item.get("type", 0))))
    if max_arrivals and int(max_arrivals) > 0:
        events = events[: int(max_arrivals)]
    return events


def _build_azure_events(args):
    return build_azure_events(
        azure_conv_csv=args.azure_conv_csv,
        azure_code_csv=args.azure_code_csv,
        azure_timestamp_col=args.azure_timestamp_col,
        sharegpt52k=args.sharegpt52k,
        arxiv_summarization=args.arxiv_summarization,
        openai_humaneval=args.openai_humaneval,
        active_pools=args.active_pools,
        pool_limit=args.pool_limit,
    )


def _load_or_build_azure_events(args):
    return load_or_build_azure_events(
        azure_mix_schedule_csv=args.azure_mix_schedule_csv,
        rebuild_azure_mix_schedule=args.rebuild_azure_mix_schedule,
        max_arrivals=args.max_arrivals,
        azure_conv_csv=args.azure_conv_csv,
        azure_code_csv=args.azure_code_csv,
        azure_timestamp_col=args.azure_timestamp_col,
        sharegpt52k=args.sharegpt52k,
        arxiv_summarization=args.arxiv_summarization,
        openai_humaneval=args.openai_humaneval,
        active_pools=args.active_pools,
        pool_limit=args.pool_limit,
    )


def _request_payload(args, prompt, stream, event=None, prompt_idx=None):
    if getattr(args, "prompt_source", None) == "synthetic":
        if not isinstance(prompt, list) or not prompt or any(type(token) is not int for token in prompt):
            raise ValueError("synthetic prompts must be non-empty token ID lists")
        payload = {
            "model": args.model,
            "prompt": prompt,
            "temperature": args.temperature,
            "stream": bool(stream),
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if args.max_tokens is not None:
            payload["max_tokens"] = int(args.max_tokens)
        if args.min_tokens is not None:
            payload["min_tokens"] = int(args.min_tokens)
        if args.ignore_eos:
            payload["ignore_eos"] = True
        if getattr(args, "op_conflict", False):
            if not event or event.get("request_shape") not in ("prefill-heavy", "decode-heavy"):
                raise ValueError("op_conflict requires a synthetic request-shape label")
            output_tokens = event.get("output_tokens")
            if (type(output_tokens) is not int or output_tokens < 1
                    or event.get("prompt_tokens") != len(prompt)
                    or type(prompt_idx) is not int or prompt_idx < 1):
                raise ValueError("invalid op_conflict request shape or index")
            payload.update(max_tokens=output_tokens, min_tokens=output_tokens, ignore_eos=True)
            payload["energy_aware"] = {"op_conflict": {
                "pool": event["request_shape"], "request_index": prompt_idx,
            }}
        return payload
    prompt = _coerce_prompt(prompt)
    if not prompt:
        raise ValueError("prompt must be non-empty")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "stream": bool(stream),
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if args.max_tokens is not None:
        payload["max_tokens"] = int(args.max_tokens)
    if args.max_layers is not None:
        payload["vllm_xargs"] = {"max_layers": int(args.max_layers)}
    if getattr(args, "specialization_gap", False):
        if not event or event.get("pool") not in ("summarization", "conversation"):
            raise ValueError("specialization_gap requires an azure-mix source-pool label")
        payload["energy_aware"] = {"specialization_gap": {
            "pool": event["pool"], "request_index": prompt_idx,
        }}
    return payload


def _auth_headers(args, stream):
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = "Bearer " + str(args.api_key)
    if stream:
        headers["Accept"] = "text/event-stream"
        headers["Connection"] = "close"
    return headers


def _extract_response_text(data):
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    if isinstance(message.get("content"), str):
        return message.get("content")
    if isinstance(choice.get("text"), str):
        return choice.get("text")
    return ""


def _partial_stream_exceptions():
    exceptions = [aiohttp.ClientPayloadError, asyncio.IncompleteReadError]
    transfer_encoding = getattr(getattr(aiohttp, "http_exceptions", None), "TransferEncodingError", None)
    if transfer_encoding is not None:
        exceptions.append(transfer_encoding)
    return tuple(exceptions)


async def _consume_sse(response, strict=False):
    buffer = ""
    pieces = []
    prompt_tokens = None
    completion_tokens = None
    first_content_s = None
    last_content_s = None
    headers_time_s = time.time()
    chunk_times = []
    done = False
    finished = False
    partial_errors = _partial_stream_exceptions()
    try:
        async for chunk in response.content.iter_chunked(1024):
            buffer += chunk.decode("utf-8", "replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    continue
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except Exception:
                    if strict:
                        raise ValueError("invalid JSON in experiment stream")
                    continue
                if strict and (not isinstance(event, dict) or event.get("error")):
                    raise ValueError("error in experiment stream")
                usage = event.get("usage") or {}
                if prompt_tokens is None:
                    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
                completion_value = _safe_int(usage.get("completion_tokens"))
                if completion_value is not None:
                    completion_tokens = completion_value
                content_piece = ""
                for choice in event.get("choices") or []:
                    if not isinstance(choice, dict):
                        if strict:
                            raise ValueError("invalid choice in experiment stream")
                        continue
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        if strict and reason not in ("stop", "length"):
                            raise ValueError("unsuccessful experiment finish reason: %s" % reason)
                        finished = True
                    delta = choice.get("delta") or {}
                    if isinstance(delta.get("content"), str):
                        content_piece += delta.get("content")
                    elif isinstance(choice.get("text"), str):
                        content_piece += choice.get("text")
                if content_piece:
                    now = time.time()
                    pieces.append(content_piece)
                    if first_content_s is None:
                        first_content_s = now
                    else:
                        chunk_times.append(now - last_content_s)
                    last_content_s = now
    except partial_errors:
        if strict or not pieces:
            raise
        _stderr("warning: partial stream used")
    if strict and not (done and finished):
        raise ValueError("unfinished experiment stream: finish reason and [DONE] required")
    return {
        "text": "".join(pieces),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "headers_time_s": headers_time_s,
        "first_content_s": first_content_s,
        "last_content_s": last_content_s,
        "chunk_gaps": chunk_times,
        "complete": done and finished,
    }


def _request_result(prompt_idx, ok, lane="", rescue=False, input_num_tokens=None, output_num_tokens=None, predicted_output_num_tokens=None, ttft_s=None, tpot_mean_s=None, tpot_p999_s=None, duration_s=None, status_code=None, stream_complete=None):
    return {
        "prompt_idx": int(prompt_idx),
        "ok": bool(ok),
        "lane": lane,
        "rescue": bool(rescue),
        "input_num_tokens": input_num_tokens,
        "output_num_tokens": output_num_tokens,
        "predicted_output_num_tokens": predicted_output_num_tokens,
        "ttft_s": ttft_s,
        "tpot_mean_s": tpot_mean_s,
        "tpot_99.9th_s": tpot_p999_s,
        "duration_s": duration_s,
        "status_code": status_code,
        "stream_complete": stream_complete,
    }


async def run_one(session, semaphore, args, prompt, prompt_idx, event=None):
    async with semaphore:
        started_s = time.time()
        stream = bool(args.stream_metrics)
        try:
            payload = _request_payload(args, prompt, stream=stream, event=event, prompt_idx=prompt_idx)
        except ValueError as exc:
            _stderr("prompt_idx=%d ok=0 error=%s" % (prompt_idx, exc))
            return _request_result(prompt_idx, False)
        headers = _auth_headers(args, stream=stream)
        endpoint = "/completions" if args.prompt_source == "synthetic" else "/chat/completions"
        url = args.base_url.rstrip("/") + endpoint
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                lane = response.headers.get("X-Energy-Aware-Preset-Id", "")
                rescue = response.headers.get("X-Energy-Aware-Rescue", "0") == "1"
                predicted_output = _safe_int(response.headers.get("X-Energy-Aware-L-Out-Predicted-Raw"))
                if stream:
                    if response.status >= 400:
                        detail = await response.text()
                        _stderr("prompt_idx=%d status=%d lane:%s detail=%s" % (prompt_idx, response.status, lane, detail[:200]))
                        return _request_result(prompt_idx, False, lane=lane, status_code=response.status)
                    stream_data = await _consume_sse(response, strict=(getattr(args, "specialization_gap", False)
                                                                     or getattr(args, "op_conflict", False)))
                    ended_s = time.time()
                    output_text = stream_data.get("text") or ""
                    ttft_s = None
                    if stream_data.get("first_content_s") is not None:
                        ttft_s = stream_data.get("first_content_s") - started_s
                    tpot_mean_s = _mean(stream_data.get("chunk_gaps"))
                    tpot_p999_s = _percentile(stream_data.get("chunk_gaps"), 0.999)
                    prompt_tokens = stream_data.get("prompt_tokens")
                    completion_tokens = stream_data.get("completion_tokens")
                    if completion_tokens is None and output_text and not getattr(args, "op_conflict", False):
                        completion_tokens = len(output_text.split())
                    if args.full_output:
                        _stderr("PROMPT[%d]: %s" % (prompt_idx, prompt))
                        _stderr("OUTPUT[%d]: %s" % (prompt_idx, output_text))
                    _stderr(
                        "prompt_idx=%d ok=1 lane:%s rescue=%d ttft_s=%s duration_s=%.6f" % (
                            prompt_idx,
                            lane,
                            1 if rescue else 0,
                            "null" if ttft_s is None else "%.6f" % ttft_s,
                            ended_s - started_s,
                        )
                    )
                    return _request_result(
                        prompt_idx,
                        True,
                        lane=lane,
                        rescue=rescue,
                        input_num_tokens=prompt_tokens,
                        output_num_tokens=completion_tokens,
                        predicted_output_num_tokens=predicted_output,
                        ttft_s=ttft_s,
                        tpot_mean_s=tpot_mean_s,
                        tpot_p999_s=tpot_p999_s,
                        duration_s=ended_s - started_s,
                        status_code=response.status,
                        stream_complete=stream_data["complete"],
                    )
                body = await response.json(content_type=None)
                if response.status >= 400:
                    detail = await response.text()
                    _stderr(
                        "prompt_idx=%d status=%d lane:%s detail=%s"
                        % (prompt_idx, response.status, lane, detail[:500])
                    )
                    return _request_result(prompt_idx, False, lane=lane, status_code=response.status)
                output_text = _extract_response_text(body)
                usage = body.get("usage") or {}
                if args.full_output:
                    _stderr("PROMPT[%d]: %s" % (prompt_idx, prompt))
                    _stderr("OUTPUT[%d]: %s" % (prompt_idx, output_text))
                _stderr("prompt_idx=%d ok=1 lane:%s rescue=%d" % (prompt_idx, lane, 1 if rescue else 0))
                return _request_result(
                    prompt_idx,
                    True,
                    lane=lane,
                    rescue=rescue,
                    input_num_tokens=_safe_int(usage.get("prompt_tokens")),
                    output_num_tokens=_safe_int(usage.get("completion_tokens")) or (len(output_text.split()) if output_text else None),
                    predicted_output_num_tokens=predicted_output,
                    duration_s=time.time() - started_s,
                    status_code=response.status,
                )
        except Exception as exc:
            _stderr("prompt_idx=%d ok=0 error=%s" % (prompt_idx, exc))
            return _request_result(prompt_idx, False, stream_complete=False if (getattr(args, "specialization_gap", False)
                                                                              or getattr(args, "op_conflict", False)) else None)


async def _run_prompt_list(args, prompts):
    semaphore = asyncio.Semaphore(max(1, int(args.max_in_flight)))
    connector_limit = max(32, int(args.max_in_flight) + 8)
    connector = aiohttp.TCPConnector(
        limit=connector_limit,
        limit_per_host=connector_limit,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(timeout=_request_timeout(args), connector=connector) as session:
        tasks = []
        for index, prompt in enumerate(prompts, start=1):
            tasks.append(asyncio.create_task(run_one(session, semaphore, args, prompt, index)))
        if not tasks:
            return []
        return await asyncio.gather(*tasks)


async def _run_azure_mix(args, events):
    usable_events = _events_with_prompts(events)
    if not usable_events:
        raise ValueError("azure-mix produced no events with non-empty prompts")
    semaphore = asyncio.Semaphore(max(1, int(args.max_in_flight)))
    connector_limit = max(32, int(args.max_in_flight) + 8)
    connector = aiohttp.TCPConnector(
        limit=connector_limit,
        limit_per_host=connector_limit,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(timeout=_request_timeout(args), connector=connector) as session:
        tasks = []
        started_s = time.time()
        for index, event in enumerate(usable_events, start=1):
            arrival_sec = float(event.get("arrival_sec", 0.0) or 0.0)
            delay = arrival_sec - (time.time() - started_s)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(run_one(session, semaphore, args, event.get("prompt"), index, event=event)))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
        if getattr(args, "specialization_gap", False):
            for result, event in zip(results, usable_events):
                result["pool"] = event["pool"]
                result["dataset_idx"] = event.get("dataset_idx")
        return results


def _iter_prompts(args):
    if args.prompt_source == "synthetic":
        max_prompts = max(1, int(args.max_prompts))
        return [_synthetic_token_ids(args) for _ in range(max_prompts)]
    if args.prompt_source == "builtin":
        max_prompts = None if not args.max_prompts or int(args.max_prompts) <= 0 else int(args.max_prompts)
        prompts = list(BUILTIN_PROMPTS)
        if max_prompts is not None:
            prompts = prompts[:max_prompts]
        return prompts
    if args.prompt_source == "lmsys":
        max_prompts = None if not args.max_prompts or int(args.max_prompts) <= 0 else int(args.max_prompts)
        return list(iter_user_prompts(args.lmsys_data_dir, max_prompts=max_prompts, skip_rows=args.skip_rows))
    if args.prompt_source == "lmsys-seq-pred":
        max_prompts = None if not args.max_prompts or int(args.max_prompts) <= 0 else int(args.max_prompts)
        return list(
            iter_decoded_seq_pred_prompts(
                args.seq_pred_lmsys_dir,
                tokenizer_name=args.seq_pred_decode_tokenizer,
                max_prompts=max_prompts,
                skip_rows=args.skip_rows,
            )
        )
    raise ValueError("prompt_source does not use prompt list")


def _synthetic_token_ids(args):
    count = int(args.synthetic_prompt_tokens)
    if count < 1:
        raise ValueError("--synthetic-prompt-tokens must be >= 1")
    token_id = args.synthetic_token_id
    if token_id is None:
        from transformers import AutoTokenizer

        token_ids = AutoTokenizer.from_pretrained(args.model, use_fast=True).encode("a", add_special_tokens=False)
        if not token_ids:
            raise ValueError("tokenizer did not produce a usable synthetic token")
        token_id = token_ids[0]
    return [int(token_id)] * count


def _load_op_conflict_events(path):
    with open(_expand(path), encoding="utf-8") as handle:
        events = json.load(handle)
    if not isinstance(events, list) or not events:
        raise ValueError("op_conflict cohort must be a nonempty JSON list")
    previous_arrival = -1.0
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict) or event.get("request_shape") not in ("prefill-heavy", "decode-heavy"):
            raise ValueError("invalid op_conflict cohort label")
        if type(event.get("request_index")) is not int or event["request_index"] != index:
            raise ValueError("op_conflict cohort indices must be ordered and contiguous")
        for key in ("prompt_tokens", "output_tokens"):
            if type(event.get(key)) is not int or event[key] < 1:
                raise ValueError("op_conflict %s must be a positive integer" % key)
        if type(event.get("token_id")) is not int or event["token_id"] < 0:
            raise ValueError("op_conflict token_id must be a nonnegative integer")
        arrival = event.get("arrival_sec")
        if (type(arrival) not in (int, float) or not math.isfinite(arrival)
                or arrival < 0 or arrival < previous_arrival):
            raise ValueError("op_conflict arrivals must be finite, nonnegative and ordered")
        previous_arrival = arrival
    return events


async def _run_op_conflict(args, events):
    # Every offered arrival gets a task/connection; lane capacity never feeds back into the schedule.
    semaphore = asyncio.Semaphore(len(events))
    connector = aiohttp.TCPConnector(limit=len(events) + 8, limit_per_host=len(events) + 8)
    async with aiohttp.ClientSession(timeout=_request_timeout(args), connector=connector) as session:
        tasks = []
        started = time.monotonic()
        for event in events:
            delay = event["arrival_sec"] - (time.monotonic() - started)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(run_one(
                session, semaphore, args, [event["token_id"]] * event["prompt_tokens"],
                event["request_index"], event=event,
            )))
        results = await asyncio.gather(*tasks)
    for result, event in zip(results, events):
        result.update(request_shape=event["request_shape"], arrival_sec=event["arrival_sec"],
                      expected_input_tokens=event["prompt_tokens"], expected_output_tokens=event["output_tokens"])
        result["shape_valid"] = (result["ok"] and result["stream_complete"] is True
                                 and result["input_num_tokens"] == event["prompt_tokens"]
                                 and result["output_num_tokens"] == event["output_tokens"])
    return results


def _annotate_synthetic_results(args, results):
    if args.prompt_source != "synthetic" or getattr(args, "op_conflict", False):
        return
    expected_input = int(args.synthetic_prompt_tokens)
    expected_output = int(args.max_tokens)
    for result in results:
        result["expected_input_tokens"] = expected_input
        result["expected_output_tokens"] = expected_output
        result["shape_valid"] = bool(
            result.get("ok")
            and result.get("input_num_tokens") == expected_input
            and result.get("output_num_tokens") == expected_output
        )


def _summarize_results(args, results, total_elapsed_s):
    ordered = sorted(results, key=lambda item: item.get("prompt_idx", 0))
    ok_rows = [row for row in ordered if row.get("ok")]
    ttfts = [row.get("ttft_s") for row in ok_rows if row.get("ttft_s") is not None]
    tpot_means = [row.get("tpot_mean_s") for row in ok_rows if row.get("tpot_mean_s") is not None]
    total_input_tokens = sum(float(row.get("input_num_tokens") or 0) for row in ok_rows)
    total_output_tokens = sum(float(row.get("output_num_tokens") or 0) for row in ok_rows)
    predicted_pairs = []
    for row in ok_rows:
        predicted = row.get("predicted_output_num_tokens")
        actual = row.get("output_num_tokens")
        if predicted is None or actual is None:
            continue
        predicted_pairs.append((float(predicted), float(actual)))
    if len(predicted_pairs) >= 2:
        xs = [item[0] for item in predicted_pairs]
        ys = [item[1] for item in predicted_pairs]
        _stderr("predicted_vs_actual pearson=", pearson(xs, ys), "spearman=", spearman(xs, ys))
    lane_counts = {}
    for row in ok_rows:
        lane = row.get("lane") or ""
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
    if lane_counts:
        _stderr("lane_histogram=", json.dumps(lane_counts, sort_keys=True))
    payload = {
        "total_elapsed_time": total_elapsed_s,
        "energy_joules": 0.0,
        "total_output_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_num_tokens": total_input_tokens + total_output_tokens,
        "model_name": args.model,
        "tokens_per_second": (total_output_tokens / total_elapsed_s) if total_elapsed_s and total_output_tokens else None,
        "num_requests": len(ordered),
        "mean_ttft": _mean(ttfts),
        "99.9th_ttft": _percentile(ttfts, 0.999),
        "mean_tpot": _mean(tpot_means),
        "99.9th_tpot": _percentile(tpot_means, 0.999),
        "requests": ordered,
        "energy_num_samples": 0,
        "energy_per_npu_joules": {},
        "power_samples_series": [],
    }
    if args.prompt_source == "synthetic" and not getattr(args, "op_conflict", False):
        payload["request_shape"] = {
            "prompt_tokens": int(args.synthetic_prompt_tokens),
            "output_tokens": int(args.max_tokens),
        }
        payload["shape_valid"] = bool(results) and all(row.get("shape_valid") for row in results)
    if getattr(args, "op_conflict", False):
        payload["shape_valid"] = bool(results) and all(row.get("shape_valid") for row in results)
    return payload


def build_parser():
    parser = argparse.ArgumentParser(prog="src.client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt-source", default="lmsys", choices=("lmsys", "lmsys-seq-pred", "builtin", "azure-mix", "synthetic"))
    parser.add_argument("--synthetic-prompt-tokens", type=int)
    parser.add_argument("--synthetic-token-id", type=int)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-in-flight", default=200, type=int)
    parser.add_argument("--request-timeout-seconds", type=float,
                        help="Override client read and total deadlines for a controlled run")
    parser.add_argument("--no-stream-metrics", dest="stream_metrics", action="store_false")
    parser.add_argument("--stream-metrics", dest="stream_metrics", action="store_true")
    parser.set_defaults(stream_metrics=True)
    parser.add_argument("--result-json", default=_expand("~/Projects/workspace/result.json"))
    parser.add_argument("--result-csv", default="")
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument("--lmsys-data-dir", default=default_lmsys_dataset_dir())
    parser.add_argument("--seq-pred-lmsys-dir", default=default_seq_pred_lmsys_vicuna_dir())
    parser.add_argument("--seq-pred-decode-tokenizer", default="lmsys/vicuna-13b-v1.3")
    parser.add_argument("--max-prompts", default=50, type=int)
    parser.add_argument("--skip-rows", default=0, type=int)
    parser.add_argument("--azure-conv-csv", default="")
    parser.add_argument("--azure-code-csv", default="")
    parser.add_argument("--azure-timestamp-col", default="TIMESTAMP")
    parser.add_argument("--sharegpt52k", default="")
    parser.add_argument("--arxiv-summarization", default="")
    parser.add_argument("--openai-humaneval", default="")
    parser.add_argument("--active-pools", default="all", choices=("all", "conversation", "summarization", "code"))
    parser.add_argument("--pool-limit", default=0, type=int)
    parser.add_argument("--max-arrivals", default=50, type=int)
    parser.add_argument("--azure-mix-schedule-csv", default=DEFAULT_SCHEDULE_CSV)
    parser.add_argument("--rebuild-azure-mix-schedule", action="store_true")
    parser.add_argument("--specialization-gap", action="store_true", help="Experiment-only source-pool metadata and strict stream completion")
    parser.add_argument("--op-conflict", action="store_true", help="Experiment-only synthetic open-loop cohort and strict completion")
    parser.add_argument("--op-conflict-cohort", default="", help="Frozen synthetic cohort JSON")
    parser.add_argument("--dry-run", action="store_true")
    return parser


async def _run(args):
    started_s = time.time()
    if getattr(args, "op_conflict", False):
        events = _load_op_conflict_events(args.op_conflict_cohort)
        if args.dry_run:
            for event in events:
                print("%.6f|%s|%d" % (event["arrival_sec"], event["request_shape"], event["request_index"]))
            return []
        results = await _run_op_conflict(args, events)
    elif args.prompt_source == "azure-mix":
        events = _load_or_build_azure_events(args)
        if args.dry_run:
            for event in events:
                print("%.6f|%s|%s" % (float(event.get("arrival_sec", 0.0)), event.get("type"), event.get("dataset_idx")))
            return []
        results = await _run_azure_mix(args, events)
    else:
        prompts = _iter_prompts(args)
        if args.dry_run:
            for index, _ in enumerate(prompts, start=1):
                print(index)
            return []
        results = await _run_prompt_list(args, prompts)
    _annotate_synthetic_results(args, results)
    total_elapsed_s = time.time() - started_s
    payload = _summarize_results(args, results, total_elapsed_s)
    if args.result_json:
        _json_dump_atomic(args.result_json, payload)
    if args.result_csv:
        _write_result_csv(args.result_csv, results)
    return results


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _request_timeout(args)
    if args.specialization_gap and (args.prompt_source != "azure-mix" or not args.stream_metrics):
        parser.error("--specialization-gap requires azure-mix and streaming metrics")
    if args.op_conflict and (args.specialization_gap or args.prompt_source != "synthetic"
                             or not args.stream_metrics or not args.op_conflict_cohort):
        parser.error("--op-conflict requires a synthetic cohort, streaming metrics, and no --specialization-gap")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
