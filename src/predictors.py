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

from .output_len_predictor import (
    build_output_len_predictor,
    default_output_len_predictor,
    output_len_predictor_from_env_or_arg,
)


class TokenCounter:

    __slots__ = ("tokenizer",)

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name):
        if not name:
            return cls()
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers is required for --hf-tokenizer") from exc
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        return cls(tokenizer=tokenizer)

    def count_text_tokens(self, text):
        text = str(text or "")
        if not text.strip():
            return 0
        if self.tokenizer is None:
            return len(text.split())
        try:
            encoded = self.tokenizer(text, add_special_tokens=False)
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                input_ids = self.tokenizer.encode(text, add_special_tokens=False)
            return len(input_ids or [])
        except Exception:
            return len(text.split())

    def __call__(self, text):
        return self.count_text_tokens(text)
