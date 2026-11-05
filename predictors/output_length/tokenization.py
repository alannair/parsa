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


from pathlib import Path


def _stdout(*parts):
    print(*parts, flush=True)


def load_bundle_tokenizer(tokenizer_dir, max_length=512, *, spm_wrapper_cls=None, rust_wrapper_cls=None):
    tokenizer_dir = Path(tokenizer_dir)
    if spm_wrapper_cls is None or rust_wrapper_cls is None:
        from predictors.output_length.common import _RustTokenizerWrapper, _SpmTokenizerWrapper

        if spm_wrapper_cls is None:
            spm_wrapper_cls = _SpmTokenizerWrapper
        if rust_wrapper_cls is None:
            rust_wrapper_cls = _RustTokenizerWrapper

    spm_path = tokenizer_dir / "spm.model"
    if spm_path.is_file():
        _stdout("CPU load: sentencepiece tokenizer from %s" % spm_path.name)
        return spm_wrapper_cls(spm_path, max_length=max_length)

    tok_json = tokenizer_dir / "tokenizer.json"
    if tok_json.is_file():
        try:
            from tokenizers import Tokenizer

            _stdout("CPU load: huggingface tokenizers JSON from %s" % tok_json.name)
            return rust_wrapper_cls(tok_json, max_length=max_length)
        except ImportError:
            pass

    _stdout("CPU load: falling back to transformers.AutoTokenizer (may segfault on Ascend)")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=False, local_files_only=True)


__all__ = ["load_bundle_tokenizer"]
