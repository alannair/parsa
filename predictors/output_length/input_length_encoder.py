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

import torch
import torch.nn as nn


def normalize_input_length(token_count, *, device=None, dtype=None):
    if hasattr(token_count, "to"):
        value = token_count.float().clamp(min=0.0)
        return torch.log1p(value)
    count = max(0.0, float(token_count))
    if not math.isfinite(count):
        count = 0.0
    return math.log1p(count)


class InputLengthEncoder(nn.Module):

    def __init__(self, hidden_dim, *, enabled=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.enabled = bool(enabled)
        self.proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(self, original_input_token_count, prompt_type=None):
        if not self.enabled:
            if hasattr(original_input_token_count, "dim"):
                batch = int(original_input_token_count.shape[0])
                device = original_input_token_count.device
                dtype = original_input_token_count.dtype
                return torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
            return None

        if hasattr(original_input_token_count, "dim"):
            counts = original_input_token_count.float().reshape(-1, 1)
            norm = torch.log1p(counts.clamp(min=0.0))
            length_emb = self.proj(norm)
            if prompt_type is not None:

                type_gate = 1.0 + 0.1 * prompt_type.float().reshape(-1, 1)
                length_emb = length_emb * type_gate
            return length_emb

        norm = float(normalize_input_length(original_input_token_count))
        return self.proj(
            torch.tensor([[norm]], device=next(self.parameters()).device, dtype=torch.float32)
        )
