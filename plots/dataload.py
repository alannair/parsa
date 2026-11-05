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
import math
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

batch_sizes = [1, 2, 5, 10, 50, 100, 200]
ilens = [1, 10, 100, 1000, 10000]
olens = [1, 10, 100, 300, 1000]
freqs = list(range(800, 1801, 100))
tps = [1, 2, 4, 8]
models = ["qwen3-4b", "qwen3-8b"]
experiments = ["batch-size", "tp"]
base_path = Path("../../experiments/results-all")


def load_huawei_prompts(path=None, **read_csv_kwargs):
    if path is None:
        path = Path(__file__).resolve().parent / "../data/seq_pred/huawei.csv"
    df = pd.read_csv(Path(path), **read_csv_kwargs)
    df["type"] = 2
    return df


def load_seq_data(conv_path=None, code_path=None, huawei_path=None, **read_csv_kwargs):
    azure = load_azure_conv_and_code(conv_path, code_path)
    huawei = load_huawei_prompts(huawei_path, **read_csv_kwargs)
    cols = ["ContextTokens", "GeneratedTokens", "type"]
    missing_a = set(cols) - set(azure.columns)
    missing_h = set(cols) - set(huawei.columns)
    if missing_a:
        raise KeyError(f"Azure data missing columns: {missing_a}")
    if missing_h:
        raise KeyError(f"Huawei data missing columns: {missing_h}")
    return pd.concat([azure[cols], huawei[cols]], ignore_index=True)


def load_azure_conv_and_code(conv_path=None, code_path=None):
    base = Path(__file__).resolve().parent / "../data/seq_pred"
    conv_path = Path(conv_path) if conv_path is not None else base / "azure_conv.csv"
    code_path = Path(code_path) if code_path is not None else base / "azure_code.csv"

    df_conv = pd.read_csv(conv_path)
    df_conv = df_conv.assign(type=0)

    df_code = pd.read_csv(code_path)
    df_code = df_code.assign(type=1)

    return pd.concat([df_conv, df_code], ignore_index=True)


def normalize_timestamp_to_first_entry(df, timestamp_col = None):
    if timestamp_col is None:
        if "timestamp" in df.columns:
            timestamp_col = "timestamp"
        elif "TIMESTAMP" in df.columns:
            timestamp_col = "TIMESTAMP"
        else:
            raise KeyError(
                "Could not infer timestamp column. Pass timestamp_col=... "
                f"(available columns: {list(df.columns)})"
            )

    out = df.copy()
    if len(out) == 0:
        return out

    s = out[timestamp_col]
    if pd.api.types.is_numeric_dtype(s):
        t0 = s.iloc[0]
        out[timestamp_col] = s - t0
        return out

    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        raise ValueError(
            f"Column {timestamp_col!r} is not numeric and could not be parsed as datetimes."
        )
    t0 = dt.iloc[0]
    out[timestamp_col] = (dt - t0).dt.total_seconds()
    return out


def queries_over_time_2s_window(
    df,
    *,
    timestamp_col = None,
    type_col = "type",
    expected_types = None,
    window_s = 2,
    include_empty_seconds = True,
):
    if window_s < 1:
        raise ValueError(f"window_s must be >= 1, got {window_s}")

    if timestamp_col is None:
        if "timestamp" in df.columns:
            timestamp_col = "timestamp"
        elif "TIMESTAMP" in df.columns:
            timestamp_col = "TIMESTAMP"
        else:
            raise KeyError(
                "Could not infer timestamp column. Pass timestamp_col=... "
                f"(available columns: {list(df.columns)})"
            )

    if len(df) == 0:
        return pd.DataFrame(
            {
                "timestamp_sec": pd.Series(dtype="int64"),
                type_col: pd.Series(dtype="int64"),
                "queries_in_window": pd.Series(dtype="int64"),
                "qps": pd.Series(dtype="float64"),
            }
        )

    ts = pd.to_numeric(df[timestamp_col], errors="coerce")
    if ts.isna().all():
        raise ValueError(f"Timestamp column {timestamp_col!r} could not be converted to numeric seconds.")

    if type_col not in df.columns:
        raise KeyError(
            f"type_col {type_col!r} not in DataFrame (available columns: {list(df.columns)})"
        )

    base = pd.DataFrame({type_col: df[type_col].values, "timestamp_sec": np.floor(ts).astype("int64")})
    per_sec = (
        base.groupby([type_col, "timestamp_sec"], observed=True)
        .size()
        .rename("queries_in_sec")
        .reset_index()
    )

    if expected_types is None:
        types = sorted(pd.unique(per_sec[type_col]).tolist())
    else:
        types = [int(t) for t in expected_types]


    if len(per_sec) == 0:
        sec_idx = pd.RangeIndex(0, 0)
    else:
        sec_min = int(per_sec["timestamp_sec"].min())
        sec_max = int(per_sec["timestamp_sec"].max())
        sec_idx = pd.RangeIndex(sec_min, sec_max + 1) if include_empty_seconds else None

    out_parts = []
    for tval in types:
        grp = per_sec[per_sec[type_col] == tval]
        counts = grp.set_index("timestamp_sec")["queries_in_sec"].sort_index()
        if include_empty_seconds:
            counts = counts.reindex(sec_idx, fill_value=0)
        window_sum = counts.rolling(window=window_s, min_periods=1).sum().astype("int64")
        window_len = counts.rolling(window=window_s, min_periods=1).count().astype("int64")
        qps = (window_sum / window_len).astype("float64")
        out_parts.append(
            pd.DataFrame(
                {
                    "timestamp_sec": window_sum.index.astype("int64"),
                    type_col: int(tval),
                    "queries_in_window": window_sum.values,
                    "qps": qps.values,
                }
            )
        )

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(
        {
            "timestamp_sec": pd.Series(dtype="int64"),
            type_col: pd.Series(dtype="int64"),
            "queries_in_window": pd.Series(dtype="int64"),
            "qps": pd.Series(dtype="float64"),
        }
    )


def load_lmsys(
    dataset_dir=None,
    split=None,
    max_rows=None,
    columns=None,
):
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as e:
        raise ImportError(
            "load_lmsys requires the datasets package. pip install datasets"
        ) from e

    if dataset_dir is None:
        dataset_dir = (
            Path(__file__).resolve().parent
            / "../data/seq_pred/lmsys/data/lmsys_first_round_data_vicuna-13b_1000K"
        )
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    ds = load_from_disk(str(dataset_dir))
    if isinstance(ds, DatasetDict):
        keys = list(ds.keys())
        if split is not None:
            if split not in keys:
                raise KeyError(f"Split {split!r} not in dataset; available: {keys}")
            ds = ds[split]
        elif "train" in keys:
            ds = ds["train"]
        elif len(keys) == 1:
            ds = ds[keys[0]]
        else:
            raise ValueError(f"Multiple splits {keys}; pass split=... explicitly.")

    if max_rows is not None:
        n = min(int(max_rows), len(ds))
        ds = ds.select(range(n))

    df = ds.to_pandas()
    if "input_ids" in df.columns:
        df["ContextTokens"] = df["input_ids"].map(len)
    if "num_tokens" in df.columns:
        df = df.rename(columns={"num_tokens": "GeneratedTokens"})
    df["type"] = 0
    if columns is not None:
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"columns not in DataFrame: {missing}")
        df = df[list(columns)]
    return df


load_lmsys_chat_dataframe = load_lmsys


def preprocess_lmsys(
    df,
    *,
    max_input_tokens=512,
    head_tail=False,
    head_tokens=128,
    tail_tokens=384,
    bert_pretrained_name="bert-base-uncased",
    tokenizer=None,
    drop_prompt=True,
):
    out = df.copy()
    if head_tail and head_tokens + tail_tokens != max_input_tokens:
        raise ValueError(
            f"head_tail requires head_tokens + tail_tokens == max_input_tokens; "
            f"got {head_tokens}+{tail_tokens} != {max_input_tokens}"
        )

    def _truncate_lists(ids, tt, am):
        if ids is None:
            return ids, tt, am
        ids = list(ids)
        n = len(ids)
        if n <= max_input_tokens:
            return ids, list(tt) if tt is not None else [0] * n, list(am) if am is not None else [1] * n
        if head_tail:
            a, b = head_tokens, tail_tokens
            tt = list(tt) if tt is not None else [0] * n
            am = list(am) if am is not None else [1] * n
            new_ids = ids[:a] + ids[-b:]
            return new_ids, tt[:a] + tt[-b:], am[:a] + am[-b:]
        tt = list(tt) if tt is not None else [0] * n
        am = list(am) if am is not None else [1] * n
        return ids[-max_input_tokens:], tt[-max_input_tokens:], am[-max_input_tokens:]

    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "preprocess_lmsys requires transformers when tokenizer is not provided."
            ) from e
        tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_name, use_fast=True)

    if "prompt" in out.columns:
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        for text in out["prompt"].astype(str):
            enc = tokenizer(text, truncation=False, add_special_tokens=True)
            ids = enc["input_ids"]
            tt = enc.get("token_type_ids")
            am = enc.get("attention_mask")
            ids, tt, am = _truncate_lists(ids, tt, am)
            if tt is None:
                tt = [0] * len(ids)
            if am is None:
                am = [1] * len(ids)
            input_ids_list.append(ids)
            token_type_ids_list.append(tt)
            attention_mask_list.append(am)
        out["input_ids"] = input_ids_list
        out["token_type_ids"] = token_type_ids_list
        out["attention_mask"] = attention_mask_list
        if drop_prompt:
            out = out.drop(columns=["prompt"])
    elif "input_ids" in out.columns:
        new_ids = []
        new_tt = []
        new_am = []
        has_tt = "token_type_ids" in out.columns
        has_am = "attention_mask" in out.columns
        for i in range(len(out)):
            row = out.iloc[i]
            ids, tt, am = _truncate_lists(
                row["input_ids"],
                row["token_type_ids"] if has_tt else None,
                row["attention_mask"] if has_am else None,
            )
            new_ids.append(ids)
            new_tt.append(tt if tt is not None else [0] * len(ids))
            new_am.append(am if am is not None else [1] * len(ids))
        out["input_ids"] = new_ids
        out["token_type_ids"] = new_tt
        out["attention_mask"] = new_am
    else:
        raise KeyError(
            "preprocess_lmsys needs a 'prompt' column or 'input_ids' column."
        )

    out["ContextTokens"] = out["input_ids"].map(len)
    return out




LMSYS_CHAT_1M_MODEL_NAMES = (
    "vicuna-13b",
    "wizardlm-13b",
    "palm-2",
    "llama-2-13b-chat",
    "koala-13b",
    "claude-instant-1",
    "oasst-pythia-12b",
    "alpaca-13b",
    "mpt-7b-chat",
    "vicuna-7b",
    "dolly-v2-12b",
    "mpt-30b-chat",
    "fastchat-t5-3b",
    "chatglm-6b",
    "claude-1",
    "gpt-4",
    "vicuna-33b",
    "guanaco-33b",
    "RWKV-4-Raven-14B",
    "stablelm-tuned-alpha-7b",
    "llama-13b",
    "gpt-3.5-turbo",
    "llama-2-7b-chat",
    "claude-2",
    "gpt4all-13b-snoozy",
)


def default_lmsys_preprocessed_root():
    return (Path(__file__).resolve().parent / "../data/seq_pred/lmsys/data").resolve()


def lmsys_preprocessed_disk_name(
    *,
    model_name="vicuna-13b",
    task_type=2,
    first_round_only=True,
    head_tail=False,
    data_size_k=1000,
    full_split=False,
    no_cap=False,
    vicuna_only=True,
):
    path = model_name.lower() + "_" if vicuna_only else ""
    if task_type == 0:
        pass
    elif task_type == 1:
        path = path + "cls_"
    else:
        path = path + "multi_cls_"
    if first_round_only:
        path = "first_round_data_" + path
    elif head_tail:
        path = "headtail_" + path
    else:
        path = "tail_" + path
    if full_split:
        path = "lmsys_" + path + "full"
    else:
        path = "lmsys_" + path + f"{int(data_size_k)}K"
    if no_cap:
        path = path.replace("lmsys_", "lmsys_nocap_", 1)
    return path


def _lmsys_exact_multi_round(dataset, vicuna_tokenizer, task_type, multi_cls_thresholds, no_cap):
    from datasets import Dataset

    df = dataset.to_pandas()
    ans_df = pd.DataFrame(
        columns=["prompt", "labels", "num_tokens", "conversation_id", "turn_id", "model"]
    )
    for conversation_id in range(len(df)):
        if conversation_id % 10000 == 0:
            print("Processing conversation " + str(conversation_id))
        sample = df.iloc[conversation_id]
        conversation = sample["conversation"]
        dialogue_so_far = ""
        new_samples = {
            "prompt": [],
            "labels": [],
            "num_tokens": [],
            "conversation_id": [],
            "turn_id": [],
            "model": [],
        }
        for turn_idx, sentence in enumerate(conversation):
            if sentence["role"] == "user":
                dialogue_so_far += "[USER]: " + sentence["content"] + "\n"
            else:
                assistant_content = sentence["content"]
                encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
                ntok = len(encoded_response["input_ids"])
                if ntok <= 1:
                    break
                if not no_cap and ntok >= 512:
                    break
                new_samples["prompt"].append(dialogue_so_far)
                new_samples["conversation_id"].append(conversation_id)
                new_samples["model"].append(sample["model"])
                new_samples["turn_id"].append(turn_idx // 2)
                new_samples["num_tokens"].append(ntok)
                if task_type == 0:
                    new_samples["labels"].append(ntok)
                else:
                    for ti, thresh in enumerate(multi_cls_thresholds):
                        if ntok < thresh:
                            new_samples["labels"].append(ti)
                            break
                dialogue_so_far += "[ASSISTANT]: " + sentence["content"] + "\n"
        new_samples = pd.DataFrame(new_samples)
        ans_df = pd.concat([ans_df, new_samples], ignore_index=True)
    return Dataset.from_pandas(ans_df)


def _lmsys_calc_percentile(dataset, vicuna_only, num_models, percentiles):
    if vicuna_only:
        output_token_lengths = []
        for sample in dataset:
            output_token_lengths.append(sample["num_tokens"])
        s = pd.Series(output_token_lengths)
        print(s.describe(percentiles=[0.25, 0.5, 0.75, 0.99]))
        return dataset
    output_token_lengths = [[] for _ in range(num_models)]
    for sample in dataset:
        output_token_lengths[sample["model"]].append(sample["num_tokens"])
    for model_id in range(num_models):
        s = pd.Series(output_token_lengths[model_id])
        desc = s.describe(percentiles=[0.25, 0.5, 0.75, 0.99])
        percentiles[model_id].extend(
            [desc["25%"], desc["50%"], desc["75%"], desc["99%"], 1000000]
        )

    def recalc_labels_and_one_hot_model_name(example):
        for ti, thresh in enumerate(percentiles[example["model"]]):
            if example["num_tokens"] < thresh:
                example["labels"] = ti
                break
        arr = [0 for _ in range(num_models)]
        arr[example["model"]] = 1
        example["model"] = arr
        return example

    return dataset.map(recalc_labels_and_one_hot_model_name)


def download_lmsys_chat_1m(
    output_dir=None,
    *,
    task_type=2,
    all_models=False,
    multi_round=False,
    head_tail=False,
    data_size_k=1000,
    model_name="vicuna-13b",
    no_cap=False,
    full_split=False,
    dataset_name="lmsys/lmsys-chat-1m",
    vicuna_tokenizer_name="lmsys/vicuna-13b-v1.3",
    bert_pretrained_name="bert-base-uncased",
    shuffle_seed=1,
    save=True,
    mkdir=True,
):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("download_lmsys_chat_1m requires datasets. pip install datasets") from e
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "download_lmsys_chat_1m requires transformers. pip install transformers"
        ) from e

    if model_name not in LMSYS_CHAT_1M_MODEL_NAMES and not all_models:
        raise ValueError(
            f"model_name {model_name!r} not in LMSYS_CHAT_1M_MODEL_NAMES; pass all_models=True "
            "or use a known model id."
        )

    vicuna_only = not all_models
    first_round_only = not multi_round
    selected_n = 1000 * int(data_size_k)

    if task_type == 1:
        multi_cls_thresholds = [141, 503, 1000000]
    else:
        multi_cls_thresholds = (
            [42, 141, 294, 503, 1000000]
            if first_round_only
            else [58, 147, 280, 499, 100000]
        )

    model_name_to_idx = {m: i for i, m in enumerate(LMSYS_CHAT_1M_MODEL_NAMES)}
    num_models = len(LMSYS_CHAT_1M_MODEL_NAMES)

    vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_tokenizer_name, legacy=False)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_name, use_fast=True)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    def extract_first_round_prompt(example):
        conversation = example["conversation"]
        user_content = ""
        for i, sentence in enumerate(conversation):
            if sentence["role"] == "user":
                if i > 0:
                    user_content += "\n"
                user_content += sentence["content"]
            else:
                break
        assistant_content = ""
        for j in range(i, len(conversation)):
            sentence = conversation[j]
            if sentence["role"] == "assistant":
                if j > i:
                    assistant_content += "\n"
                assistant_content += conversation[j]["content"]
            else:
                break
        example["prompt"] = user_content
        encoded_response = vicuna_tokenizer(assistant_content, truncation=False)
        example["num_tokens"] = len(encoded_response["input_ids"])
        ntok = example["num_tokens"]
        if task_type == 0:
            example["labels"] = ntok
        else:
            for ti, thresh in enumerate(multi_cls_thresholds):
                if ntok < thresh:
                    example["labels"] = ti
                    break
        return example

    def tokenize_function(example):
        enc = bert_tokenizer(example["prompt"], truncation=False)
        if no_cap:
            return enc
        if len(enc["input_ids"]) >= 512:
            if head_tail:
                enc["input_ids"] = enc["input_ids"][:128] + enc["input_ids"][-384:]
                enc["token_type_ids"] = enc["token_type_ids"][:128] + enc["token_type_ids"][-384:]
                enc["attention_mask"] = enc["attention_mask"][:128] + enc["attention_mask"][-384:]
            else:
                enc["input_ids"] = enc["input_ids"][-512:]
                enc["token_type_ids"] = enc["token_type_ids"][-512:]
                enc["attention_mask"] = enc["attention_mask"][-512:]
        return enc

    def replace_model_name_by_idx(example):
        example["model"] = model_name_to_idx[example["model"]]
        if task_type == 0:
            arr = [0 for _ in range(num_models)]
            arr[example["model"]] = 1
            example["model"] = arr
        return example

    def preprocess_hf_dataset(dataset):
        dataset = dataset.remove_columns(
            ["openai_moderation", "redacted", "language", "conversation_id", "turn"]
        )
        dataset = dataset.add_column("prompt", [""] * len(dataset))
        dataset = dataset.add_column("labels", [0] * len(dataset))
        if task_type != 0:
            dataset = dataset.add_column("num_tokens", [0] * len(dataset))

        if first_round_only:
            dataset = dataset.map(extract_first_round_prompt, remove_columns=["conversation"])
            print("Num samples before filtering: ", len(dataset))
            if no_cap:
                if task_type == 0:
                    dataset = dataset.filter(lambda ex: ex["labels"] > 1)
                else:
                    dataset = dataset.filter(lambda ex: ex["num_tokens"] > 1)
            else:
                if task_type == 0:
                    dataset = dataset.filter(
                        lambda ex: ex["labels"] > 1 and ex["labels"] < 512
                    )
                else:
                    dataset = dataset.filter(
                        lambda ex: ex["num_tokens"] > 1 and ex["num_tokens"] < 512
                    )
            print("Num samples after filtering: ", len(dataset))
        else:
            dataset = _lmsys_exact_multi_round(
                dataset, vicuna_tokenizer, task_type, multi_cls_thresholds, no_cap
            )
        if vicuna_only:
            dataset = dataset.remove_columns(["model"])
        else:
            dataset = dataset.map(replace_model_name_by_idx)
        if no_cap:
            pass
        else:
            dataset = dataset.map(tokenize_function, batched=False, remove_columns=["prompt"])
        return dataset

    raw = load_dataset(dataset_name, split="train")
    if not full_split:
        n = min(selected_n, len(raw))
        raw = raw.select(range(n))
    if vicuna_only:
        raw = raw.filter(lambda ex: ex["model"] == model_name)
    raw = raw.shuffle(seed=shuffle_seed)
    dataset = preprocess_hf_dataset(raw)

    percentiles = [[] for _ in range(num_models)]
    if task_type != 0:
        dataset = _lmsys_calc_percentile(dataset, vicuna_only, num_models, percentiles)
    if not no_cap:
        dataset.set_format("torch")

    disk_name = lmsys_preprocessed_disk_name(
        model_name=model_name,
        task_type=task_type,
        first_round_only=first_round_only,
        head_tail=head_tail,
        data_size_k=data_size_k,
        full_split=full_split,
        no_cap=no_cap,
        vicuna_only=vicuna_only,
    )
    if output_dir is None:
        out_path = default_lmsys_preprocessed_root() / disk_name
    else:
        out_path = Path(output_dir).expanduser().resolve() / disk_name

    if save:
        if mkdir:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(out_path))
        print("Saved dataset to " + str(out_path))
        return out_path, dataset
    return None, dataset


def generate_filename(experiment, model, batch_size, input_size, output_size, frequency, tp):
    return f"freq_{experiment}_{model}_ilen{input_size}_olen{output_size}_bs{batch_size}_f{frequency}_tp{tp}.json"


def extract_parameters_from_filename(filename):
    pattern = r"freq_(?P<experiment>[\w-]+)_(?P<model>[\w-]+)_ilen(?P<input_size>\d+)_olen(?P<output_size>\d+)_bs(?P<batch_size>\d+)_f(?P<frequency>\d+)_tp(?P<tp>\d+)\.json"
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    raise ValueError("Filename does not match expected pattern")


def load_json_with_parameters(file_path, model="qwen3-4b"):
    filename = os.path.basename(file_path)
    params = extract_parameters_from_filename(filename)
    with open(file_path, "r") as f:
        json_data = json.load(f)
    json_data.update(params)
    return json_data


def load_all_jsons(param_combinations, model="qwen3-4b"):
    all_data = []
    for params in param_combinations:
        file_name = generate_filename(*params)
        if (base_path / file_name).exists():
            all_data.append(load_json_with_parameters(base_path / file_name, model))
        else:
            pass
    return pd.DataFrame(all_data)



_TTFT_JSON_ALIASES = (
    "ttft_s",
    "ttft",
    "TTFT_s",
    "time_to_first_token_s",
    "time_to_first_token",
)
_TPOT_JSON_ALIASES = (
    "tpot_mean_s",
    "tpot_mean",
    "tpot_s",
    "tpot",
    "TPOT",
    "inter_chunk_latency_s",
)


def _dataframe_from_latency_json_payload(payload):
    if isinstance(payload, list):
        rows = payload
        summary = {}
    elif isinstance(payload, dict):
        rows = None
        for key in ("requests", "data", "results", "per_request"):
            if key in payload and isinstance(payload[key], list):
                rows = payload[key]
                break
        if rows is None:
            has_request_metric = any(
                k in payload and payload[k] is not None
                for k in _TTFT_JSON_ALIASES + _TPOT_JSON_ALIASES
            )
            if has_request_metric:
                rows = [payload]
            else:
                raise ValueError(
                    "JSON dict has no list under 'requests', 'data', 'results', or "
                    "'per_request', and no TTFT/TPOT fields at the top level. "
                    f"Keys (first 40): {list(payload.keys())[:40]}"
                )
        skip = {"requests", "data", "results", "per_request", "power_samples_series"}
        summary = {
            k: v
            for k, v in payload.items()
            if k not in skip and not isinstance(v, (list, dict))
        }
    else:
        raise TypeError(f"Expected list or dict at JSON root, got {type(payload).__name__}")

    df = pd.json_normalize(rows)
    if df.empty:
        return df, {}

    def _first_alias(candidates):
        for name in candidates:
            if name in df.columns:
                return name
        return None

    ttft_src = _first_alias(_TTFT_JSON_ALIASES)
    if ttft_src is not None and "ttft_s" not in df.columns:
        df = df.rename(columns={ttft_src: "ttft_s"})
    tpot_src = _first_alias(_TPOT_JSON_ALIASES)
    if tpot_src is not None and "tpot_mean_s" not in df.columns:
        df = df.rename(columns={tpot_src: "tpot_mean_s"})

    for col in ("ttft_s", "tpot_mean_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, summary


class _ServingConditionError(ValueError):
    pass


def _latency_serving_condition(payload, path):
    parsed = parse_profile_latency_dir_name(path.parent.name)
    suffix = parsed.get("serving_condition") if parsed else None
    condition = payload.get("serving_condition") if isinstance(payload, dict) else None
    if condition and (not isinstance(condition, str) or not condition.strip()):
        raise _ServingConditionError(f"invalid serving condition metadata: {path}")
    if suffix and condition and suffix != condition:
        raise _ServingConditionError(f"serving condition directory/metadata mismatch: {path}")
    return condition or suffix or "legacy"


def _select_latency_serving_condition(df, serving_condition):
    if df.empty:
        return df
    conditions = sorted(df["serving_condition"].unique())
    if serving_condition is not None:
        return df.loc[df["serving_condition"] == serving_condition].copy()
    if len(conditions) > 1:
        raise ValueError("multiple serving conditions; pass serving_condition from: " + ", ".join(conditions))
    return df


def load_latency_json_result(path, *, merge_summary = True):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    condition = _latency_serving_condition(payload, path)
    df, summary = _dataframe_from_latency_json_payload(payload)
    if "serving_condition" in df.columns and any(
        value and value != condition for value in df["serving_condition"].dropna()
    ):
        raise _ServingConditionError(f"serving condition request/result metadata mismatch: {path}")
    df["serving_condition"] = condition
    df.attrs["serving_condition"] = condition
    if merge_summary and summary and len(df) > 0:
        for k, v in summary.items():
            if k not in df.columns:
                df[k] = v
    if len(df) > 0 and "_result_json_path" not in df.columns:
        df = df.copy()
        df["_result_json_path"] = str(path.resolve())
    return df


def load_latency_json_results(paths, *, merge_summary = True, serving_condition = None):
    frames = []
    for p in paths:
        frames.append(load_latency_json_result(p, merge_summary=merge_summary))
    if not frames:
        return pd.DataFrame()
    return _select_latency_serving_condition(pd.concat(frames, ignore_index=True), serving_condition)


def load_result_json_rows(root, pattern="*/result.json"):
    rows = []
    for path in sorted(Path(root).glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)
        if not isinstance(row, dict):
            raise ValueError("result JSON must contain an object: %s" % path)
        row["_result_json_path"] = str(path.resolve())
        rows.append(row)
    if not rows:
        raise FileNotFoundError("no result JSON files matching %s under %s" % (pattern, root))
    return pd.DataFrame(rows)


_PROFILE_LATENCY_DIR_RE = re.compile(
    r"^profile-(?P<experiment>.+)-(?P<workload>[^-]+)-mbt(?P<max_num_batched_tokens>\d+)-ms(?P<max_num_seqs>\d+)-f(?P<frequency_mhz>\d+)(?:-sp-(?P<serving_hash>[0-9a-f]{24}))?$",
    re.IGNORECASE,
)


def parse_profile_latency_dir_name(folder_name):
    match = _PROFILE_LATENCY_DIR_RE.match(str(folder_name).strip())
    if not match:
        return None
    groups = match.groupdict()
    parsed = {
        "experiment": groups["experiment"],
        "workload": groups["workload"],
        "max_num_batched_tokens": int(groups["max_num_batched_tokens"]),
        "max_num_seqs": int(groups["max_num_seqs"]),
        "frequency_mhz": int(groups["frequency_mhz"]),
    }
    if groups["serving_hash"] is not None:
        parsed["serving_condition"] = "sp-" + groups["serving_hash"].lower()
    return parsed


def load_latency_experiments_dataframe(
    experiments_root,
    *,
    result_filename = "result.json",
    merge_summary = True,
    skip_non_matching_dirs = True,
    serving_condition = None,
):
    root = Path(experiments_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiments root is not a directory: {root}")

    frames = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        parsed = parse_profile_latency_dir_name(child.name)
        if parsed is None:
            if skip_non_matching_dirs:
                continue
            parsed = {
                "experiment": child.name,
                "workload": None,
                "max_num_batched_tokens": np.nan,
                "max_num_seqs": np.nan,
                "frequency_mhz": np.nan,
            }

        result_path = child / result_filename
        if not result_path.is_file():
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "run_folder": child.name,
                            "run_dir": str(child.resolve()),
                            "load_error": f"missing {result_filename}",
                            "_result_json_path": str(result_path.resolve()),
                            "serving_condition": parsed.get("serving_condition", "legacy"),
                            **parsed,
                        }
                    ]
                )
            )
            continue

        try:
            df = load_latency_json_result(result_path, merge_summary=merge_summary)
        except _ServingConditionError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "run_folder": child.name,
                            "run_dir": str(child.resolve()),
                            "load_error": str(exc),
                            "_result_json_path": str(result_path.resolve()),
                            "serving_condition": parsed.get("serving_condition", "legacy"),
                            **parsed,
                        }
                    ]
                )
            )
            continue

        if df.empty:
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "run_folder": child.name,
                            "run_dir": str(child.resolve()),
                            "load_error": "no latency rows",
                            "_result_json_path": str(result_path.resolve()),
                            "serving_condition": df.attrs.get("serving_condition", "legacy"),
                            **parsed,
                        }
                    ]
                )
            )
            continue

        df = df.copy()
        df["run_folder"] = child.name
        df["run_dir"] = str(child.resolve())
        for key, value in parsed.items():
            df[key] = value
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return _select_latency_serving_condition(pd.concat(frames, ignore_index=True), serving_condition)


def trim_idle_edges(series):
    """Keep the active window plus one adjacent idle sample at either edge."""
    if not isinstance(series, list) or not series:
        raise ValueError("power_samples_series must be a nonempty list")
    devices = None
    previous_timestamp = None
    first, stop = None, 0
    for index, sample in enumerate(series):
        try:
            timestamp = float(sample["timestamp"])
            per_npu = sample["per_npu"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("power_samples_series contains an invalid sample") from exc
        if not math.isfinite(timestamp):
            raise ValueError("power_samples_series timestamps must be finite")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("power_samples_series timestamps must be strictly increasing")
        previous_timestamp = timestamp
        if not isinstance(per_npu, dict) or not per_npu:
            raise ValueError("power_samples_series requires readings for every monitored NPU")
        if devices is None:
            devices = set(per_npu)
        elif set(per_npu) != devices:
            raise ValueError("power_samples_series changes monitored NPUs")
        idle = True
        for values in per_npu.values():
            try:
                power = float(values["power_sample"])
                frequency = float(values["frequency"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("power_samples_series requires power and frequency for every NPU") from exc
            if not math.isfinite(power) or not math.isfinite(frequency):
                raise ValueError("power_samples_series power and frequency must be finite")
            if frequency != 800:
                idle = False
        if not idle:
            if first is None:
                first = index
            stop = index + 1
    if first is None:
        raise ValueError("power_samples_series has no non-800 MHz samples")
    first = max(0, first - 1)
    stop = min(len(series), stop + 1)
    if stop - first < 2:
        raise ValueError("power_samples_series needs at least two retained samples after trimming")
    return series[first:stop]


def _measure_base_power(samples, window=4):
    if not samples:
        return 0.0
    window = max(1, min(window, len(samples)))
    front_avg = np.mean([row[1] for row in samples[:window]])
    back_avg = np.mean([row[1] for row in samples[-window:]])
    base_power = min(front_avg, back_avg)
    return float(base_power)


def trim_series(rows, base_window=4):
    if not rows:
        return []
    freq_index = 3
    power_index = 1
    left, right = 0, len(rows) - 1
    has_non_800 = any(row[freq_index] != 800 for row in rows)
    if has_non_800:
        while left <= right and rows[left][freq_index] == 800:
            left += 1
        while right >= left and rows[right][freq_index] == 800:
            right -= 1
        left -= 1
    else:
        threshold = _measure_base_power(rows, base_window)
        while left <= right and rows[left][power_index] < threshold:
            left += 1
        while right >= left and rows[right][power_index] < threshold:
            right -= 1
    return rows[left : right + 2]


def trim_energy_samples_series(df):
    def trim_cell(cell):
        if not isinstance(cell, dict) or "0" not in cell:
            return cell
        cell_copy = cell.copy()
        cell_copy["0"] = trim_series(cell_copy["0"])
        return cell_copy

    df_copy = df.copy()
    df_copy["energy_samples_series"] = df_copy["energy_samples_series"].map(trim_cell)
    return df_copy


def energy_from_samples(samples):
    if len(samples) < 2:
        return 0.0
    energy_joules = 0.0
    threshold = _measure_base_power(samples, window=4)
    for prev, curr in zip(samples, samples[1:]):
        dt = curr[0] - prev[0]
        if dt > 0:
            energy_joules += (prev[1] * dt) - threshold
    return energy_joules


def recompute_energy(df):
    def row_energy(cell):
        if isinstance(cell, dict) and "0" in cell and isinstance(cell["0"], list):
            return energy_from_samples(cell["0"])
        return np.nan

    df = df.copy()
    df["energy_joules"] = df["energy_samples_series"].map(row_energy)
    return df







_RUNS_BF_SUFFIX_RE = re.compile(
    r"^(.+)_b(?P<batch_size>\d+)_f(?P<frequency>\d+)$",
    re.IGNORECASE,
)


def default_parsa_tests_runs_root():
    return (Path(__file__).resolve().parent.parent / "tests" / "_runs").resolve()


def parse_tests_run_folder_name(folder_name):
    m = _RUNS_BF_SUFFIX_RE.match(str(folder_name).strip())
    if not m:
        return None
    stem = m.group(1)
    batch = int(m.group("batch_size"))
    freq = int(m.group("frequency"))
    if "__" in stem:
        experiment, model_slug = stem.split("__", 1)
    else:
        experiment, model_slug = stem, None
    return {
        "run_stem": stem,
        "experiment": experiment,
        "model_slug": model_slug,
        "batch_size": batch,
        "frequency_mhz": freq,
    }


def load_parsa_tests_run_result_row(
    run_dir,
    *,
    result_filename = "result.json",
):
    path = Path(run_dir) / result_filename
    out = {"result_json_path": str(path.resolve())}
    if not path.is_file():
        out["load_error"] = f"missing {result_filename}"
        return out
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as e:
        out["load_error"] = str(e)
        return out
    if not isinstance(data, dict):
        out["load_error"] = f"expected object at JSON root, got {type(data).__name__}"
        return out

    def _f(key):
        v = data.get(key)
        if v is None:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if np.isnan(x) or np.isinf(x):
            return None
        return x

    out["energy_joules"] = _f("energy_joules")
    out["mean_tpot"] = _f("mean_tpot")
    out["99.9th_tpot"] = _f("99.9th_tpot")
    out["tpot_mean_s"] = out["mean_tpot"]
    out["tpot_p99_9_s"] = out["99.9th_tpot"]
    out["mean_ttft"] = _f("mean_ttft")
    out["99.9th_ttft"] = _f("99.9th_ttft")
    out["ttft_mean_s"] = out["mean_ttft"]
    out["ttft_p99_9_s"] = out["99.9th_ttft"]
    out["tokens_per_second"] = _f("tokens_per_second")
    out["total_elapsed_time"] = _f("total_elapsed_time")
    out["total_output_tokens"] = _f("total_output_tokens")
    out["total_input_tokens"] = _f("total_input_tokens")
    out["total_num_tokens"] = _f("total_num_tokens")
    out["num_requests"] = data.get("num_requests")
    out["model_name"] = data.get("model_name")
    return out


def load_parsa_tests_runs_dataframe(
    runs_root = None,
    *,
    result_filename = "result.json",
    skip_non_matching_dirs = True,
):
    root = Path(runs_root).expanduser().resolve() if runs_root is not None else default_parsa_tests_runs_root()
    if not root.is_dir():
        raise FileNotFoundError(f"runs root is not a directory: {root}")

    rows = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        parsed = parse_tests_run_folder_name(child.name)
        if parsed is None:
            if skip_non_matching_dirs:
                continue
            parsed = {
                "run_stem": child.name,
                "experiment": child.name,
                "model_slug": None,
                "batch_size": np.nan,
                "frequency_mhz": np.nan,
            }
        payload = load_parsa_tests_run_result_row(child, result_filename=result_filename)
        row = {
            "run_folder": child.name,
            **parsed,
            **payload,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_mean_metric_and_adjust_p999(
    df,
    *,
    mean_col,
    p999_col,
    mean_out,
    p999_out,
    ref_col,
    reference = None,
    reference_frequency_mhz = 1650,
    reference_batch_size = 5,
    freq_col = "frequency_mhz",
    batch_col = "batch_size",
    reference_how = "median",
    inplace = False,
):
    how = str(reference_how or "median").strip().lower()
    allowed = frozenset({"median", "mean", "min", "max"})
    if how not in allowed:
        raise ValueError(f"reference_how must be one of {sorted(allowed)}, got {reference_how!r}")

    out = df if inplace else df.copy()
    m = pd.to_numeric(out[mean_col], errors="coerce")
    p = pd.to_numeric(out[p999_col], errors="coerce")

    use_setup = (
        reference_frequency_mhz is not None and reference_batch_size is not None
    )
    one_setup_kw = (reference_frequency_mhz is None) != (reference_batch_size is None)
    if one_setup_kw:
        raise ValueError(
            "Pass both reference_frequency_mhz and reference_batch_size, or set both to None "
            "to use reference_how.",
        )

    if reference is not None:
        ref = float(reference)
        if not np.isfinite(ref) or ref <= 0:
            raise ValueError(f"reference must be finite and > 0, got {reference!r}")
    elif use_setup:
        fq = float(reference_frequency_mhz)
        bq = float(reference_batch_size)
        if freq_col not in out.columns or batch_col not in out.columns:
            raise KeyError(
                f"Baseline match requires columns {freq_col!r} and {batch_col!r}; "
                f"have {list(out.columns)}",
            )
        fc = pd.to_numeric(out[freq_col], errors="coerce")
        bc = pd.to_numeric(out[batch_col], errors="coerce")
        mask = (fc == fq) & (bc == bq)
        sel = m[mask].dropna()
        sel = sel[np.isfinite(sel) & (sel > 0)]
        if sel.empty:
            raise ValueError(
                f"No rows with {freq_col}={fq} and {batch_col}={bq} with finite {mean_col} > 0 "
                f"to build normalization reference.",
            )
        ref = float(sel.mean())
    else:
        valid = m[np.isfinite(m) & (m > 0)]
        if valid.empty:
            ref = float("nan")
        elif how == "median":
            ref = float(valid.median())
        elif how == "mean":
            ref = float(valid.mean())
        elif how == "min":
            ref = float(valid.min())
        else:
            ref = float(valid.max())
        if not np.isfinite(ref) or ref <= 0:
            ref = float("nan")

    out[ref_col] = ref
    out[mean_out] = m / ref
    m_pos = m.where(m > 0)
    out[p999_out] = (p / m_pos) * out[mean_out]
    return out


def normalize_tpot_mean_and_adjust_p999(
    df,
    *,
    mean_col = "tpot_mean_s",
    p999_col = "tpot_p99_9_s",
    reference = None,
    reference_frequency_mhz = 1650,
    reference_batch_size = 5,
    freq_col = "frequency_mhz",
    batch_col = "batch_size",
    reference_how = "median",
    mean_out = "tpot_mean_s_normalized",
    p999_out = "tpot_p99_9_s_normalized",
    ref_col = "tpot_normalization_ref_s",
    inplace = False,
):
    return normalize_mean_metric_and_adjust_p999(
        df,
        mean_col=mean_col,
        p999_col=p999_col,
        mean_out=mean_out,
        p999_out=p999_out,
        ref_col=ref_col,
        reference=reference,
        reference_frequency_mhz=reference_frequency_mhz,
        reference_batch_size=reference_batch_size,
        freq_col=freq_col,
        batch_col=batch_col,
        reference_how=reference_how,
        inplace=inplace,
    )


def normalize_ttft_mean_and_adjust_p999(
    df,
    *,
    mean_col = "ttft_mean_s",
    p999_col = "ttft_p99_9_s",
    reference = None,
    reference_frequency_mhz = 1650,
    reference_batch_size = 5,
    freq_col = "frequency_mhz",
    batch_col = "batch_size",
    reference_how = "median",
    mean_out = "ttft_mean_s_normalized",
    p999_out = "ttft_p99_9_s_normalized",
    ref_col = "ttft_normalization_ref_s",
    inplace = False,
):
    return normalize_mean_metric_and_adjust_p999(
        df,
        mean_col=mean_col,
        p999_col=p999_col,
        mean_out=mean_out,
        p999_out=p999_out,
        ref_col=ref_col,
        reference=reference,
        reference_frequency_mhz=reference_frequency_mhz,
        reference_batch_size=reference_batch_size,
        freq_col=freq_col,
        batch_col=batch_col,
        reference_how=reference_how,
        inplace=inplace,
    )





OUTPUT_LENGTH_TYPE_LABELS = {
    0: "Conversation (type 0)",
    1: "Code (type 1)",
    2: "Summarization (type 2)",
}


def load_hf_tokenizer(tokenizer_name, prefer_fast=True):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "load_hf_tokenizer requires `transformers` (pip install transformers)."
        ) from exc

    name = str(tokenizer_name)
    if prefer_fast:
        try:
            return AutoTokenizer.from_pretrained(name, use_fast=True)
        except Exception as exc:
            print(
                "tokenizer %s: fast load failed (%s); using slow SentencePiece tokenizer"
                % (name, exc),
                flush=True,
            )
    return AutoTokenizer.from_pretrained(name, use_fast=False)


def load_output_length_dataset(
    csv_path,
    tokenizer=None,
    tokenizer_name="microsoft/deberta-v3-base",
    keep_prompt=True,
    show_progress=True,
    max_length=None,
):
    import csv as _csv
    import sys as _sys

    try:
        _csv.field_size_limit(_sys.maxsize)
    except OverflowError:
        _csv.field_size_limit(2**31 - 1)

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"output-length CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    required = {"prompt", "type", "output_length"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"output-length CSV missing columns: {missing}")
    if df.empty:
        raise ValueError(f"output-length CSV is empty: {path}")

    df["type"] = pd.to_numeric(df["type"], errors="coerce").astype("Int64")
    df["output_length"] = pd.to_numeric(df["output_length"], errors="coerce")
    df = df.dropna(subset=["prompt", "type", "output_length"]).copy()
    df["type"] = df["type"].astype(int)
    df["output_length"] = df["output_length"].astype(int)
    df["prompt"] = df["prompt"].astype(str)

    if tokenizer is None:
        tokenizer = load_hf_tokenizer(tokenizer_name)

    prompts = df["prompt"].tolist()
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(prompts, desc="tokenizing prompts", dynamic_ncols=True, leave=False)
        except ImportError:
            iterator = prompts
    else:
        iterator = prompts

    context_tokens = []
    cap = int(max_length) if max_length is not None else None
    for text in iterator:
        if cap is not None and cap > 0:
            ids = tokenizer(
                text,
                truncation=True,
                max_length=cap,
                add_special_tokens=True,
            )["input_ids"]
        else:
            ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        context_tokens.append(len(ids))

    df["ContextTokens"] = context_tokens
    df["GeneratedTokens"] = df["output_length"].astype(int)
    if not keep_prompt:
        df = df.drop(columns=["prompt"])
    return df.reset_index(drop=True)


PREDICTOR_RESULTS_COLUMNS = ("type", "input_length", "output_length", "prediction")


def _raise_csv_field_limit():
    import csv as _csv
    import sys as _sys

    try:
        _csv.field_size_limit(_sys.maxsize)
    except OverflowError:
        _csv.field_size_limit(2**31 - 1)


def save_predictor_results_csv(df, path, include_prompt=False):
    import csv as _csv

    _raise_csv_field_limit()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    required = set(PREDICTOR_RESULTS_COLUMNS)
    if include_prompt:
        required.add("prompt")


    if "ContextTokens" in work.columns and "input_length" not in work.columns:
        work["input_length"] = work["ContextTokens"]
    if "GeneratedTokens" in work.columns and "output_length" not in work.columns:
        work["output_length"] = work["GeneratedTokens"]
    if "PredictedTokens" in work.columns and "prediction" not in work.columns:
        work["prediction"] = work["PredictedTokens"]

    missing = required - set(work.columns)
    if missing:
        raise KeyError("predictor results missing columns: %s" % sorted(missing))

    columns = list(PREDICTOR_RESULTS_COLUMNS)
    if include_prompt and "prompt" in work.columns:
        columns = ["prompt"] + columns

    work = work[columns].copy()
    work["type"] = pd.to_numeric(work["type"], errors="coerce").astype(int)
    work["input_length"] = pd.to_numeric(work["input_length"], errors="coerce").astype(int)
    work["output_length"] = pd.to_numeric(work["output_length"], errors="coerce").astype(int)
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")

    with open(out, "w", encoding="utf-8", newline="") as handle:
        work.to_csv(handle, index=False, quoting=_csv.QUOTE_MINIMAL)
    return out


def load_predictor_results_csv(path, keep_prompt=False):
    _raise_csv_field_limit()
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError("predictor results CSV not found: %s" % csv_path)

    df = pd.read_csv(csv_path, encoding="utf-8")
    required = set(PREDICTOR_RESULTS_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise KeyError("predictor results CSV missing columns: %s" % sorted(missing))
    if df.empty:
        raise ValueError("predictor results CSV is empty: %s" % csv_path)

    df["type"] = pd.to_numeric(df["type"], errors="coerce").astype(int)
    df["input_length"] = pd.to_numeric(df["input_length"], errors="coerce")
    df["output_length"] = pd.to_numeric(df["output_length"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["type", "input_length", "output_length", "prediction"]).copy()
    df["input_length"] = df["input_length"].astype(int)
    df["output_length"] = df["output_length"].astype(int)

    df["ContextTokens"] = df["input_length"]
    df["GeneratedTokens"] = df["output_length"]
    df["PredictedTokens"] = df["prediction"]
    for bucket_col in ("true_bucket", "pred_bucket"):
        if bucket_col in df.columns:
            df[bucket_col] = pd.to_numeric(df[bucket_col], errors="coerce").astype("Int64")

    if not keep_prompt and "prompt" in df.columns:
        df = df.drop(columns=["prompt"])
    return df.reset_index(drop=True)


def _ensure_predictor_bundle_importable():
    import sys as _sys

    root = str(Path(__file__).resolve().parents[1])
    if root not in _sys.path:
        _sys.path.insert(0, root)


def _load_predictor_bundle(bundle_path):
    import pickle

    _ensure_predictor_bundle_importable()
    path = Path(bundle_path)
    if path.is_dir():
        candidate = path / "bert.pkl"
    else:
        candidate = path
    if not candidate.is_file():
        raise FileNotFoundError(f"predictor bundle not found: {candidate}")
    with open(candidate, "rb") as handle:
        return candidate, pickle.load(handle)


def _resolve_predictor_object(bundle):
    if isinstance(bundle, dict):
        predictor = bundle.get("predictor")
        if predictor is not None and hasattr(predictor, "predict_generated_tokens"):
            return predictor
    if hasattr(bundle, "predict_generated_tokens"):
        return bundle
    raise TypeError(
        "loaded bundle does not expose a predictor with predict_generated_tokens"
    )


def predict_output_length_from_predictor(
    df,
    predictor,
    *,
    prompt_col="prompt",
    type_col="type",
    pred_col="PredictedTokens",
    batch_size=16,
    show_progress=True,
):
    import torch

    from predictors.output_length.common import build_dataloader, get_progress_iter

    if prompt_col not in df.columns:
        raise KeyError(f"DataFrame missing required column: {prompt_col!r}")
    if type_col not in df.columns:
        raise KeyError(f"DataFrame missing required column: {type_col!r}")

    infer_device = predictor.device
    model = predictor.model
    max_length = int(getattr(predictor, "max_length", 512))
    prompts = df[prompt_col].astype(str).tolist()
    types = (
        pd.to_numeric(df[type_col], errors="coerce")
        .fillna(predictor.default_type)
        .astype(int)
        .tolist()
    )
    rows = [
        {"prompt": prompt, "type": prompt_type, "output_length": 0.0}
        for prompt, prompt_type in zip(prompts, types)
    ]
    loader = build_dataloader(
        rows,
        predictor.tokenizer,
        max(1, int(batch_size)),
        max_length,
        shuffle=False,
    )

    preds = []
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    iterator = loader
    if show_progress:
        iterator = get_progress_iter(loader, total=total_batches, desc="predicting")

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(infer_device)
            attention_mask = batch["attention_mask"].to(infer_device)
            type_tensor = batch["type"].to(infer_device).long()
            raw = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_type=type_tensor,
            )
            decoded = predictor._decode_predictions(raw.detach().float()).cpu().tolist()
            preds.extend(float(v) for v in decoded)

    if len(preds) != len(prompts):
        raise RuntimeError(
            "predictor returned %d values for %d prompts" % (len(preds), len(prompts))
        )

    out = df.copy()
    out[pred_col] = preds
    return out


def predict_output_length_from_bundle(
    df,
    bundle_path,
    *,
    prompt_col="prompt",
    type_col="type",
    pred_col="PredictedTokens",
    batch_size=16,
    device=None,
    ascend_devices=None,
    show_progress=True,
    predictor=None,
):
    from predictors.output_length.common import load_inference_predictor

    if predictor is None:
        predictor = load_inference_predictor(
            bundle_path,
            ascend_devices=ascend_devices,
            device=device,
        )
    return predict_output_length_from_predictor(
        df,
        predictor,
        prompt_col=prompt_col,
        type_col=type_col,
        pred_col=pred_col,
        batch_size=batch_size,
        show_progress=show_progress,
    )
