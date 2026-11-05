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
import importlib
import sys
import time
from pathlib import Path

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    _limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(_limit)
            break
        except OverflowError:
            _limit = _limit // 10
            if _limit < 2**20:
                csv.field_size_limit(2**20)
                break

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CSV = REPO_ROOT / "predictors" / "output_length" / "dataset_output_length.csv"
DEFAULT_BERT_MODEL_DIR = REPO_ROOT / "models" / "output_length"
DEFAULT_DEBERTA_MODEL_DIR = REPO_ROOT / "models" / "output_length_deberta"
NUM_TYPES = 3

PREDICTION_POINT = "point"
PREDICTION_LOGNORMAL = "lognormal"
PREDICTION_HYBRID_BUCKET = "hybrid_bucket"

PKL_NAME = "bert.pkl"
INFERENCE_CONFIG_NAME = "inference_config.json"
BACKBONE_DIR_NAME = "backbone"
WEIGHTS_NAME_LEGACY = "bert_regression_model.pth"
WEIGHTS_NAME_DEBERTA = "deberta_distribution_model.pth"
WEIGHTS_NAME_DEBERTA_HYBRID = "deberta_hybrid_bucket_model.pth"
DEFAULT_MAX_LENGTH = 512


def _stderr(*parts):
    print(*parts, file=sys.stderr)


def _stdout(*parts):
    print(*parts, file=sys.stdout, flush=True)


def get_progress_iter(iterable, total=None, desc=""):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)
    except ImportError:
        return _FallbackProgress(iterable, total=total, desc=desc)


class _FallbackProgress:

    def __init__(self, iterable, total=None, desc=""):
        self._iterable = iterable
        try:
            self._total = int(total if total is not None else len(iterable))
        except TypeError:
            self._total = None
        self._desc = str(desc or "")
        self._count = 0
        self._start = time.time()
        self._last_print = 0.0

    def __iter__(self):
        for item in self._iterable:
            yield item
            self._count += 1
            now = time.time()
            if now - self._last_print >= 1.0 or (self._total and self._count == self._total):
                self._last_print = now
                self._render(now)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, now):
        elapsed = max(1e-6, now - self._start)
        rate = self._count / elapsed
        if self._total:
            pct = 100.0 * self._count / max(1, self._total)
            bar_width = 24
            filled = min(bar_width, int(bar_width * self._count / max(1, self._total)))
            bar = "#" * filled + "." * (bar_width - filled)
            line = "%s [%s] %d/%d %.1f%% %.1fit/s" % (
                self._desc,
                bar,
                self._count,
                self._total,
                pct,
                rate,
            )
        else:
            line = "%s %d %.1fit/s" % (self._desc, self._count, rate)
        sys.stdout.write("\r" + line)
        sys.stdout.flush()


def core_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


class _SpmTokenizerWrapper:

    def __init__(self, spm_path, max_length=512):
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(spm_path))
        self.model_max_length = int(max_length)
        self.max_length = int(max_length)
        self.pad_token_id = 0
        self.padding_side = "right"

    def _pad_batch(self, batch, *, max_length=None, pad_to_multiple_of=None):
        ids_list = [item["input_ids"] for item in batch]
        target_len = max(len(ids) for ids in ids_list) if ids_list else 0
        if max_length is not None:
            target_len = max(target_len, int(max_length))
        if pad_to_multiple_of:
            step = int(pad_to_multiple_of)
            if step > 0 and target_len % step != 0:
                target_len += step - (target_len % step)
        padded = []
        masks = []
        pad_id = int(getattr(self, "pad_token_id", 0) or 0)
        for ids in ids_list:
            pad_len = max(0, target_len - len(ids))
            padded.append(list(ids) + [pad_id] * pad_len)
            masks.append([1] * len(ids) + [0] * pad_len)
        return padded, masks

    def _encode_one(self, text, max_length, add_special_tokens):
        ids = self._sp.encode(str(text), out_type=int)
        if add_special_tokens:
            bos = int(self._sp.bos_id())
            eos = int(self._sp.eos_id())
            if bos >= 0:
                ids = [bos] + list(ids)
            if eos >= 0:
                ids = list(ids) + [eos]
        if max_length and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def encode(self, text, add_special_tokens=True):
        return self._encode_one(str(text), None, bool(add_special_tokens))

    def __call__(
        self,
        text,
        truncation=True,
        max_length=None,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
    ):
        cap = int(max_length or self.max_length)
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        out = []
        for item in texts:
            ids = self._encode_one(item, cap if truncation else None, add_special_tokens)
            out.append({"input_ids": ids})
        if padding and len(out) > 1:
            return self.pad(out, padding=True, return_tensors=return_tensors)
        if return_tensors == "pt":
            import torch

            ids = out[0]["input_ids"]
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return out[0] if len(out) == 1 else out

    def pad(self, batch, padding=True, return_tensors="pt", max_length=None, pad_to_multiple_of=None, **kwargs):
        import torch

        padded, masks = self._pad_batch(
            batch,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return [{"input_ids": ids} for ids in padded]


class _RustTokenizerWrapper:

    def __init__(self, tokenizer_json, max_length=512):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(str(tokenizer_json))
        self.model_max_length = int(max_length)
        self.max_length = int(max_length)

    def encode(self, text, add_special_tokens=True):
        return list(
            self._tok.encode(str(text), add_special_tokens=bool(add_special_tokens)).ids
        )

    def __call__(
        self,
        text,
        truncation=True,
        max_length=None,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
    ):
        import torch

        cap = int(max_length or self.max_length)
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        encodings = self._tok.encode_batch(texts, add_special_tokens=add_special_tokens)
        batch = []
        for enc in encodings:
            ids = enc.ids
            if truncation and len(ids) > cap:
                ids = ids[:cap]
            batch.append({"input_ids": ids})
        if padding and len(batch) > 1:
            return self.pad(batch, padding=True, return_tensors=return_tensors)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([batch[0]["input_ids"]], dtype=torch.long)}
        return batch[0] if len(batch) == 1 else batch

    def pad(self, batch, padding=True, return_tensors="pt", max_length=None, pad_to_multiple_of=None, **kwargs):
        wrapper = _SpmTokenizerWrapper.__new__(_SpmTokenizerWrapper)
        wrapper.pad_token_id = 0
        return _SpmTokenizerWrapper.pad(
            wrapper,
            batch,
            padding=padding,
            return_tensors=return_tensors,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )


class OutputLengthPredictor:

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length,
        default_type=0,
        prediction_mode=PREDICTION_POINT,
        *,
        regression_loss_type="mse",
        head_tail_truncation=False,
        use_structural_features=False,
        use_input_length_feature=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = int(max_length)
        self.default_type = int(default_type)
        self.prediction_mode = str(prediction_mode or PREDICTION_POINT)
        self.regression_loss_type = str(regression_loss_type or "mse")
        self.head_tail_truncation = bool(head_tail_truncation)
        self.use_structural_features = bool(use_structural_features)
        self.use_input_length_feature = bool(use_input_length_feature)

    def _coerce_text_and_type(self, samples):
        if isinstance(samples, dict):
            try:
                from src.output_len_predictor import extract_request_text

                text = extract_request_text(samples)
            except Exception:
                text = str(samples.get("prompt") or "")
            raw_type = samples.get("type")
            if raw_type is None:
                energy = samples.get("energy_aware") or {}
                if isinstance(energy, dict):
                    raw_type = energy.get("type")
            try:
                prompt_type = int(raw_type)
            except (TypeError, ValueError):
                prompt_type = self.default_type
            return text, prompt_type
        if isinstance(samples, str):
            return samples, self.default_type
        if isinstance(samples, list) and samples:
            if isinstance(samples[0], str):
                return samples, [self.default_type] * len(samples)
        raise TypeError("samples must be request dict, string, or list of strings")

    def _encode_texts(self, texts):
        import torch

        from predictors.output_length.features import (
            structural_feature_vector,
            tokenize_prompt_with_metadata,
        )

        if isinstance(texts, str):
            texts = [texts]
        batch_enc = []
        struct_rows = []
        original_lengths = []
        for text in texts:
            if self.use_input_length_feature or self.head_tail_truncation:
                meta = tokenize_prompt_with_metadata(
                    self.tokenizer,
                    text,
                    self.max_length,
                    head_tail_truncation=self.head_tail_truncation,
                )
                batch_enc.append({"input_ids": meta["input_ids"]})
                original_lengths.append(meta["original_input_token_count"])
            else:
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
                batch_enc.append(enc)
                original_lengths.append(len(enc["input_ids"]))
            if self.use_structural_features:
                struct_rows.append(structural_feature_vector(text))
        padded = self.tokenizer.pad(batch_enc, padding=True, return_tensors="pt")
        struct_tensor = None
        if struct_rows:
            struct_tensor = torch.tensor(struct_rows, dtype=torch.float32)
        length_tensor = torch.tensor(original_lengths, dtype=torch.float32)
        return padded, struct_tensor, length_tensor

    def _decode_predictions(self, raw, quantile=None):
        from .deberta_distribution import expected_tokens_from_lognormal, quantile_tokens_from_lognormal
        from .deberta_hybrid import decode_hybrid_outputs

        if self.prediction_mode == PREDICTION_HYBRID_BUCKET:
            if isinstance(raw, dict):
                outputs = raw
            else:
                raise ValueError("hybrid predictor expects dict outputs from the model")
            decoded = decode_hybrid_outputs(outputs, self.regression_loss_type)
            return decoded["predicted_tokens"]
        if self.prediction_mode == PREDICTION_LOGNORMAL:
            if raw.dim() == 1 and raw.numel() == 2:
                raw = raw.unsqueeze(0)
            if raw.dim() != 2 or raw.size(-1) != 2:
                raise ValueError("lognormal predictor expects [batch, 2] distribution parameters")
            mu_log = raw[:, 0]
            log_var = raw[:, 1]
            if quantile is not None:
                return quantile_tokens_from_lognormal(mu_log, log_var, quantile)
            return expected_tokens_from_lognormal(mu_log, log_var)
        return raw.reshape(-1)

    def predict_generated_tokens_detail(self, samples, quantile=None):
        import torch

        from .buckets import bucket_id_to_range, bucket_label

        text, prompt_type = self._coerce_text_and_type(samples)
        if isinstance(text, list):
            texts = text
            types = prompt_type if isinstance(prompt_type, list) else [prompt_type] * len(texts)
        else:
            texts = [text]
            types = [prompt_type]

        enc, struct_tensor, length_tensor = self._encode_texts(texts)
        enc = {key: value.to(self.device) for key, value in enc.items()}
        if struct_tensor is not None:
            struct_tensor = struct_tensor.to(self.device)
        length_tensor = length_tensor.to(self.device)
        type_tensor = torch.tensor(types, dtype=torch.long, device=self.device)

        self.model.eval()
        from predictors.output_length.training_memory import (
            resolve_training_precision,
            training_autocast,
        )

        infer_precision = resolve_training_precision(self.device, auto_on_accelerator=True)
        with torch.no_grad():
            with training_autocast(self.device, infer_precision):
                model_kwargs = {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc.get("attention_mask"),
                    "prompt_type": type_tensor,
                    "collect_metadata": self.prediction_mode == PREDICTION_HYBRID_BUCKET,
                }
                if struct_tensor is not None:
                    model_kwargs["structural_features"] = struct_tensor
                if self.use_input_length_feature:
                    model_kwargs["original_input_token_count"] = length_tensor
                if self.prediction_mode == PREDICTION_HYBRID_BUCKET:
                    model_kwargs["return_dict"] = True
                raw = self.model(**model_kwargs)
        if self.prediction_mode == PREDICTION_HYBRID_BUCKET:
            from .deberta_hybrid import decode_hybrid_outputs
            from .input_length_encoder import normalize_input_length

            decoded = decode_hybrid_outputs(raw, self.regression_loss_type)
            bucket_ids = decoded["predicted_bucket"].cpu().tolist()
            probs = decoded["bucket_probs"].cpu().tolist()
            tokens = decoded["predicted_tokens"].cpu().tolist()
            if length_tensor is None and raw.get("original_input_token_count") is not None:
                length_tensor = raw["original_input_token_count"]
            input_counts = (
                length_tensor.cpu().tolist()
                if length_tensor is not None
                else [None] * len(bucket_ids)
            )
            details = []
            for index, bucket_id in enumerate(bucket_ids):
                low, high = bucket_id_to_range(bucket_id)
                count = input_counts[index]
                detail = {
                    "predicted_tokens": float(tokens[index]),
                    "predicted_bucket": int(bucket_id),
                    "bucket_label": bucket_label(bucket_id),
                    "bucket_range": (float(low), float(high)),
                    "bucket_probs": [float(v) for v in probs[index]],
                    "input_token_count": int(count) if count is not None else None,
                    "normalized_input_length": float(normalize_input_length(count))
                    if count is not None
                    else None,
                    "truncated_input_token_count": int(enc["attention_mask"][index].sum().item()),
                }
                details.append(detail)
            if isinstance(samples, (str, dict)):
                return details[0]
            return details

        values = self._decode_predictions(raw.detach().float(), quantile=quantile).cpu().tolist()
        if isinstance(samples, (str, dict)):
            return {"predicted_tokens": float(values[0])}
        return [{"predicted_tokens": float(v)} for v in values]

    def predict_generated_tokens(self, samples, quantile=None):
        detail = self.predict_generated_tokens_detail(samples, quantile=quantile)
        if isinstance(detail, dict):
            if "predicted_tokens" in detail:
                return detail["predicted_tokens"]
            return detail.get("predicted_tokens")
        return [item["predicted_tokens"] for item in detail]


_LAZY_EXPORTS = {

    "_parse_devices": ("predictors.output_length.ascend", "_parse_devices"),
    "configure_ascend_inference_env": ("predictors.output_length.ascend", "configure_ascend_inference_env"),
    "configure_ascend_training_env": ("predictors.output_length.ascend", "configure_ascend_training_env"),
    "bootstrap_ascend_training_env": ("predictors.output_length.ascend", "bootstrap_ascend_training_env"),
    "early_bootstrap_ascend_from_argv": ("predictors.output_length.ascend", "early_bootstrap_ascend_from_argv"),
    "setup_training_devices": ("predictors.output_length.ascend", "setup_training_devices"),
    "setup_inference_device": ("predictors.output_length.ascend", "setup_inference_device"),
    "wrap_model_for_devices": ("predictors.output_length.ascend", "wrap_model_for_devices"),
    "setup_hybrid_training": ("predictors.output_length.ascend", "setup_hybrid_training"),
    "wrap_model_distributed": ("predictors.output_length.ascend", "wrap_model_distributed"),
    "is_main_training_process": ("predictors.output_length.ascend", "is_main_training_process"),
    "distributed_barrier": ("predictors.output_length.ascend", "distributed_barrier"),

    "resolve_bundle_dir": ("predictors.output_length.bundle_io", "resolve_bundle_dir"),
    "read_bundle_inference_config": ("predictors.output_length.bundle_io", "read_bundle_inference_config"),
    "_build_inference_model": ("predictors.output_length.bundle_io", "_build_inference_model"),
    "_deberta_subset_from_checkpoint": ("predictors.output_length.bundle_io", "_deberta_subset_from_checkpoint"),
    "_head_subset_from_checkpoint": ("predictors.output_length.bundle_io", "_head_subset_from_checkpoint"),
    "_load_checkpoint_cpu": ("predictors.output_length.bundle_io", "load_checkpoint_cpu"),
    "_write_deberta_backbone_weights_raw": (
        "predictors.output_length.bundle_io",
        "write_deberta_backbone_weights_raw",
    ),
    "materialize_bundle_backbone_from_pth_raw": (
        "predictors.output_length.bundle_io",
        "materialize_bundle_backbone_from_pth_raw",
    ),
    "materialize_bundle_backbone_from_pth": (
        "predictors.output_length.bundle_io",
        "materialize_bundle_backbone_from_pth",
    ),
    "materialize_bundle_backbone_weights": (
        "predictors.output_length.bundle_io",
        "materialize_bundle_backbone_weights",
    ),
    "materialize_bundle_backbone_via_training_builder": (
        "predictors.output_length.bundle_io",
        "materialize_bundle_backbone_via_training_builder",
    ),
    "ensure_bundle_backbone": ("predictors.output_length.bundle_io", "ensure_bundle_backbone"),
    "ensure_inference_config": ("predictors.output_length.bundle_io", "ensure_inference_config"),
    "_ensure_repo_root_on_path": ("predictors.output_length.bundle_io", "_ensure_repo_root_on_path"),
    "_load_predictor_from_pickle": ("predictors.output_length.bundle_io", "_load_predictor_from_pickle"),
    "_load_backbone_config_from_json": ("predictors.output_length.bundle_io", "_load_backbone_config_from_json"),
    "save_predictor_bundle": ("predictors.output_length.bundle_io", "save_predictor_bundle"),

    "_load_bundle_tokenizer": ("predictors.output_length.tokenization", "load_bundle_tokenizer"),
    "load_labeled_csv": ("predictors.output_length.dataset_io", "load_labeled_csv"),
    "split_rows": ("predictors.output_length.dataset_io", "split_rows"),
    "build_dataloader": ("predictors.output_length.dataset_io", "build_dataloader"),
    "write_predict_results_csv": ("predictors.output_length.dataset_io", "write_predict_results_csv"),
    "_tokenize_context_lengths": ("predictors.output_length.dataset_io", "tokenize_context_lengths"),

    "_build_deberta_from_state_dict_cpu": (
        "predictors.output_length.inference",
        "_build_deberta_from_state_dict_cpu",
    ),
    "_build_hybrid_from_state_dict_cpu": (
        "predictors.output_length.inference",
        "_build_hybrid_from_state_dict_cpu",
    ),
    "_is_hybrid_bundle_config": ("predictors.output_length.inference", "_is_hybrid_bundle_config"),
    "_checkpoint_has_input_length_encoder": (
        "predictors.output_length.inference",
        "_checkpoint_has_input_length_encoder",
    ),
    "resolve_use_input_length_feature": (
        "predictors.output_length.inference",
        "resolve_use_input_length_feature",
    ),
    "load_predictor_cpu_safe": ("predictors.output_length.inference", "load_predictor_cpu_safe"),
    "npu_warmup_forward": ("predictors.output_length.inference", "npu_warmup_forward"),
    "load_inference_predictor": ("predictors.output_length.inference", "load_inference_predictor"),
    "predict_labeled_bundle_csv": ("predictors.output_length.inference", "predict_labeled_bundle_csv"),
    "default_bundle_dir": ("predictors.output_length.inference", "default_bundle_dir"),
    "predict_output_tokens": ("predictors.output_length.inference", "predict_output_tokens"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "REPO_ROOT",
    "DEFAULT_CSV",
    "DEFAULT_BERT_MODEL_DIR",
    "DEFAULT_DEBERTA_MODEL_DIR",
    "NUM_TYPES",
    "PREDICTION_POINT",
    "PREDICTION_LOGNORMAL",
    "PREDICTION_HYBRID_BUCKET",
    "PKL_NAME",
    "INFERENCE_CONFIG_NAME",
    "BACKBONE_DIR_NAME",
    "WEIGHTS_NAME_LEGACY",
    "WEIGHTS_NAME_DEBERTA",
    "WEIGHTS_NAME_DEBERTA_HYBRID",
    "DEFAULT_MAX_LENGTH",
    "_stderr",
    "_stdout",
    "get_progress_iter",
    "_FallbackProgress",
    "core_model",
    "_SpmTokenizerWrapper",
    "_RustTokenizerWrapper",
    "OutputLengthPredictor",
    *_LAZY_EXPORTS.keys(),
]
