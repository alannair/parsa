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


from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPrecision:

    @property
    def autocast_dtype(self):
        import torch

        if self.name == "bf16":
            return torch.bfloat16
        if self.name == "fp16":
            return torch.float16
        return torch.float32


def _device_type(device):
    if hasattr(device, "type"):
        return str(device.type)
    text = str(device)
    if text.startswith("npu"):
        return "npu"
    if text.startswith("cuda"):
        return "cuda"
    return "cpu"


def _supports_bf16(device):
    import torch

    dtype = torch.bfloat16
    dev_type = _device_type(device)
    if dev_type == "cuda" and torch.cuda.is_available():
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    if dev_type == "npu":
        try:
            return bool(torch.npu.is_bf16_supported())
        except Exception:
            pass
        return True
    return False


def _supports_fp16(device):
    dev_type = _device_type(device)
    return dev_type in ("cuda", "npu")


def resolve_training_micro_batch_size(
    requested_batch_size,
    *,
    device,
    max_length,
    train_head_only = False,
    low_memory = False,
    explicit_micro_batch_size = None,
):

    batch_size = max(1, int(requested_batch_size or 1))
    if explicit_micro_batch_size is not None:
        return max(1, min(batch_size, int(explicit_micro_batch_size)))

    dev_type = _device_type(device)
    if low_memory:
        return 1

    if dev_type == "npu":
        if train_head_only:
            if int(max_length or 0) >= 512:
                return min(batch_size, 2)
            return min(batch_size, 4)
        return 1

    if dev_type == "cuda":
        if train_head_only:
            return min(batch_size, 8)
        if int(max_length or 0) >= 512:
            return min(batch_size, 2)
        return min(batch_size, 4)

    return batch_size


def resolve_training_precision(
    device,
    *,
    bf16 = False,
    fp16 = False,
    fp32 = False,
    low_memory_mode = False,
    auto_on_accelerator = True,
):
    if sum(bool(x) for x in (bf16, fp16, fp32)) > 1:
        raise ValueError("Use at most one of --bf16, --fp16, --fp32")

    if fp32:
        return TrainingPrecision("fp32", use_autocast=False, use_grad_scaler=False)

    if bf16 or low_memory_mode:
        if _supports_bf16(device):
            return TrainingPrecision("bf16", use_autocast=True, use_grad_scaler=False)
        if fp16 or low_memory_mode:
            if _supports_fp16(device):
                from predictors.output_length.common import _stdout

                _stdout("WARN: bf16 unavailable on %s; using fp16" % device)
                return TrainingPrecision("fp16", use_autocast=True, use_grad_scaler=True)
        from predictors.output_length.common import _stdout

        _stdout("WARN: bf16 unavailable on %s; using fp32" % device)
        return TrainingPrecision("fp32", use_autocast=False, use_grad_scaler=False)

    if fp16:
        if _supports_fp16(device):
            return TrainingPrecision("fp16", use_autocast=True, use_grad_scaler=True)
        from predictors.output_length.common import _stdout

        _stdout("WARN: fp16 unavailable on %s; using fp32" % device)
        return TrainingPrecision("fp32", use_autocast=False, use_grad_scaler=False)

    if auto_on_accelerator and _device_type(device) in ("cuda", "npu"):
        if _supports_bf16(device):
            return TrainingPrecision("bf16", use_autocast=True, use_grad_scaler=False)
        if _supports_fp16(device):
            return TrainingPrecision("fp16", use_autocast=True, use_grad_scaler=True)

    return TrainingPrecision("fp32", use_autocast=False, use_grad_scaler=False)


def training_autocast(device, precision):
    import torch

    if not precision.use_autocast:
        return torch.autocast(device_type=_device_type(device), enabled=False)
    return torch.autocast(
        device_type=_device_type(device),
        dtype=precision.autocast_dtype,
        enabled=True,
    )


def create_grad_scaler(device, precision):
    import torch

    if not precision.use_grad_scaler:
        return None
    dev_type = _device_type(device)
    if dev_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    if dev_type == "npu":
        try:
            return torch.npu.amp.GradScaler()
        except Exception:
            try:
                return torch.amp.GradScaler(dev_type)
            except Exception:
                return None
    return None


class MemoryDiagnostics:

    def __init__(self, device, *, enabled = False, main_process = True):
        self.device = device
        self.enabled = bool(enabled) and bool(main_process)
        self._peak_reset = False

    def _read(self):
        import torch

        dev_type = _device_type(self.device)
        alloc = reserved = peak = None
        if dev_type == "cuda" and torch.cuda.is_available():
            idx = (
                int(self.device.index)
                if getattr(self.device, "index", None) is not None
                else torch.cuda.current_device()
            )
            alloc = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
            if self._peak_reset:
                peak = torch.cuda.max_memory_allocated(idx) / (1024**3)
        elif dev_type == "npu":
            if hasattr(torch.npu, "memory_allocated"):
                alloc = torch.npu.memory_allocated(self.device) / (1024**3)
            if hasattr(torch.npu, "memory_reserved"):
                reserved = torch.npu.memory_reserved(self.device) / (1024**3)
            if self._peak_reset and hasattr(torch.npu, "max_memory_allocated"):
                peak = torch.npu.max_memory_allocated(self.device) / (1024**3)
        return alloc, reserved, peak

    def reset_peak(self):
        import torch

        if not self.enabled:
            return
        dev_type = _device_type(self.device)
        try:
            if dev_type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(
                    self.device.index if getattr(self.device, "index", None) is not None else None
                )
            elif dev_type == "npu" and hasattr(torch.npu, "reset_peak_memory_stats"):
                torch.npu.reset_peak_memory_stats(self.device)
        except Exception:
            pass
        self._peak_reset = True

    def log(self, label):
        if not self.enabled:
            return
        from predictors.output_length.common import _stdout

        alloc, reserved, peak = self._read()
        parts = ["memory %s:" % label]
        if alloc is not None:
            parts.append("allocated=%.2f GiB" % alloc)
        if reserved is not None:
            parts.append("reserved=%.2f GiB" % reserved)
        if peak is not None:
            parts.append("peak=%.2f GiB" % peak)
        if len(parts) > 1:
            _stdout(" ".join(parts))


def log_device_memory(device, label = ""):
    MemoryDiagnostics(device, enabled=True, main_process=True).log(label or "snapshot")
