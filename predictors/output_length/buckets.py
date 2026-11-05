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


import math

BUCKETS = [
    (0, 64),
    (64, 256),
    (256, 1024),
    (1024, 4096),
    (4096, float("inf")),
]

NUM_BUCKETS = len(BUCKETS)


def num_buckets():
    return NUM_BUCKETS


def token_length_to_bucket_id(length):
    try:
        value = float(length)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value < 0:
        value = 0.0
    for index, (low, high) in enumerate(BUCKETS):
        if value >= low and value < high:
            return index
    return NUM_BUCKETS - 1


def bucket_id_to_range(bucket_id):
    index = max(0, min(int(bucket_id), NUM_BUCKETS - 1))
    return BUCKETS[index]


def bucket_id_to_midpoint(bucket_id):
    low, high = bucket_id_to_range(bucket_id)
    if math.isinf(high):
        return float(low) * 2.0
    return 0.5 * (float(low) + float(high))


def bucket_label(bucket_id):
    low, high = bucket_id_to_range(bucket_id)
    if math.isinf(high):
        return "%g+" % low
    return "%g-%g" % (low, high)


def decode_token_prediction(pred_log_length, bucket_id=None):
    import torch

    if hasattr(pred_log_length, "detach"):
        value = pred_log_length.detach().float()
        if value.numel() == 1:
            log_len = float(value.reshape(-1)[0].item())
        else:
            log_len = value.reshape(-1)
            return torch.expm1(log_len.clamp(min=0.0))
    else:
        log_len = float(pred_log_length)
    tokens = math.expm1(max(0.0, log_len))
    if bucket_id is not None:

        low, high = bucket_id_to_range(int(bucket_id))
        if not math.isinf(high):
            tokens = min(max(tokens, low), high * 1.25)
    return tokens
