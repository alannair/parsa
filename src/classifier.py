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

from .output_len_predictor import default_output_len_predictor, extract_request_text


class RequestFeatures:

    __slots__ = (
        "text",
        "l_in",
        "l_out_raw",
        "l_out",
        "prefill_share",
        "decode_share",
        "stream",
        "priority",
    )

    def __init__(self, text, l_in, l_out_raw, l_out, prefill_share, decode_share, stream=False, priority=0):
        self.text = text
        self.l_in = float(l_in or 0.0)
        self.l_out_raw = float(l_out_raw or 0.0)
        self.l_out = float(l_out or 0.0)
        self.prefill_share = float(prefill_share or 0.0)
        self.decode_share = float(decode_share or 0.0)
        self.stream = bool(stream)
        self.priority = int(priority or 0)

    def to_dict(self):
        return {
            "text": self.text,
            "l_in": self.l_in,
            "l_out_raw": self.l_out_raw,
            "l_out": self.l_out,
            "prefill_share": self.prefill_share,
            "decode_share": self.decode_share,
            "stream": self.stream,
            "priority": self.priority,
        }


class RequestClassifier:

    __slots__ = ("token_counter", "output_len_predictor", "out_len_safety_factor")

    def __init__(self, token_counter=None, output_len_predictor=None, out_len_safety_factor=1.2):
        factor = float(out_len_safety_factor)
        if factor <= 0:
            raise ValueError("out_len_safety_factor must be > 0")
        self.token_counter = token_counter
        self.output_len_predictor = output_len_predictor
        self.out_len_safety_factor = factor

    def extract_text(self, request_json):
        return extract_request_text(request_json)

    def estimate_l_in(self, request_json):
        text = self.extract_text(request_json)
        if self.token_counter is not None:
            try:
                return float(self.token_counter(text))
            except Exception:
                return float(len(str(text or "").split()))
        return float(len(str(text or "").split())) if str(text or "").strip() else 0.0

    def estimate_l_out_raw(self, request_json, force_body_only=False, quantile=None):
        predictor = None if force_body_only else self.output_len_predictor
        if predictor is None:
            raw = default_output_len_predictor(request_json)
        elif quantile is not None and hasattr(predictor, "predict_quantile"):
            raw = predictor.predict_quantile(request_json, quantile)
        else:
            raw = predictor(request_json)
        try:
            raw = float(raw)
        except Exception:
            raw = 128.0
        if math.isnan(raw) or math.isinf(raw):
            raw = 128.0
        return raw

    def estimate_l_out(self, request_json, raw=None):
        if raw is None:
            raw = self.estimate_l_out_raw(request_json)
        try:
            effective = max(0.0, float(raw)) * self.out_len_safety_factor
        except Exception:
            effective = 128.0 * self.out_len_safety_factor
        if math.isnan(effective) or math.isinf(effective):
            effective = 128.0 * self.out_len_safety_factor
        return effective

    def classify(self, request_json, force_body_only=False, quantile=None):
        text = self.extract_text(request_json)
        l_in = self.estimate_l_in(request_json)
        l_out_raw = self.estimate_l_out_raw(request_json, force_body_only=force_body_only, quantile=quantile)
        l_out = self.estimate_l_out(request_json, raw=l_out_raw)
        total = max(1.0, l_in + max(0.0, l_out_raw))
        prefill_share = l_in / total
        decode_share = max(0.0, l_out_raw) / total
        priority = request_json.get("priority", 0)
        try:
            priority = int(priority)
        except Exception:
            priority = 0
        return RequestFeatures(
            text=text,
            l_in=l_in,
            l_out_raw=l_out_raw,
            l_out=l_out,
            prefill_share=prefill_share,
            decode_share=decode_share,
            stream=bool(request_json.get("stream")),
            priority=priority,
        )
