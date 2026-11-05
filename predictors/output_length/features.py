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


import re

from predictors.output_length.input_length_encoder import normalize_input_length

STRUCTURAL_FEATURE_NAMES = (
    "newline_count",
    "code_block_count",
    "message_count",
)
NUM_STRUCTURAL_FEATURES = len(STRUCTURAL_FEATURE_NAMES)


def truncate_head_tail(token_ids, max_length, head_tokens=None):
    ids = list(token_ids)
    cap = max(1, int(max_length))
    if len(ids) <= cap:
        return ids
    head = int(head_tokens) if head_tokens is not None else cap // 2
    head = max(1, min(head, cap - 1))
    tail = cap - head
    return ids[:head] + ids[-tail:]


def count_tokens(tokenizer, text, *, add_special_tokens=True):
    return len(tokenizer.encode(str(text), add_special_tokens=bool(add_special_tokens)))


def tokenize_prompt_with_metadata(
    tokenizer,
    text,
    max_length,
    *,
    head_tail_truncation=False,
    add_special_tokens=True,
):
    original_count = count_tokens(tokenizer, text, add_special_tokens=add_special_tokens)
    full_ids = tokenizer.encode(str(text), add_special_tokens=add_special_tokens)
    if head_tail_truncation:
        input_ids = truncate_head_tail(full_ids, int(max_length))
    elif len(full_ids) > int(max_length):
        input_ids = full_ids[: int(max_length)]
    else:
        input_ids = full_ids
    truncated_count = len(input_ids)
    return {
        "input_ids": input_ids,
        "original_input_token_count": int(original_count),
        "truncated_input_token_count": int(truncated_count),
        "normalized_input_length": float(normalize_input_length(original_count)),
    }


def encode_prompt_head_tail(tokenizer, text, max_length, *, add_special_tokens=True):
    return {
        "input_ids": tokenize_prompt_with_metadata(
            tokenizer,
            text,
            max_length,
            head_tail_truncation=True,
            add_special_tokens=add_special_tokens,
        )["input_ids"]
    }


def structural_features_from_text(text):
    raw = str(text or "")
    newlines = raw.count("\n")
    code_blocks = len(re.findall(r"```", raw))
    messages = raw.count('"role"') + raw.count("'role'")
    if messages <= 0:
        messages = max(1, raw.count("\n\n") + 1)
    return {
        "newline_count": float(newlines),
        "code_block_count": float(code_blocks),
        "message_count": float(messages),
    }


def structural_feature_vector(text):
    feats = structural_features_from_text(text)
    return [feats[name] for name in STRUCTURAL_FEATURE_NAMES]
