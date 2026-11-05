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


import os


def _stdout(*parts):
    print(*parts, flush=True)


def _parse_devices(devices):
    items = [item.strip() for item in str(devices or "").split(",") if item.strip()]
    if not items:
        return []
    return items


def configure_ascend_inference_env(ascend_devices=None):
    os.environ.setdefault("TASK_QUEUE_ENABLE", "0")
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("COMBINED_ENABLE", "1")

    raw = ascend_devices
    if raw is None or not str(raw).strip():
        raw = os.environ.get("ASCEND_DEVICES", "0")
    device_list = _parse_devices(raw)
    if not device_list:
        device_list = ["0"]
    primary = str(device_list[0])
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = primary
    return int(primary), device_list


def configure_ascend_training_env(npu_list):
    os.environ.setdefault("TASK_QUEUE_ENABLE", "0")
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("COMBINED_ENABLE", "1")
    visible = ",".join(str(int(i)) for i in npu_list)
    os.environ["ASCEND_DEVICES"] = visible
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible
    _stdout(
        "Ascend training: ASCEND_RT_VISIBLE_DEVICES=%s (logical npu:0..%d)"
        % (visible, max(len(npu_list) - 1, 0))
    )


def bootstrap_ascend_training_env(ascend_devices=None):
    from predictors.output_length._ascend_env import early_bootstrap_ascend_from_argv

    if ascend_devices is None:
        return early_bootstrap_ascend_from_argv([])
    return early_bootstrap_ascend_from_argv(["--ascend-devices", str(ascend_devices)])


def early_bootstrap_ascend_from_argv(argv):
    from predictors.output_length._ascend_env import early_bootstrap_ascend_from_argv as _bootstrap

    return _bootstrap(argv)


def setup_training_devices(ascend_devices=None):
    import torch

    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError(
            "Ascend training requires torch_npu (install the CANN-matched wheel for your PyTorch)."
        ) from exc

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required for training but torch.npu.is_available() is False")

    raw = ascend_devices
    if raw is None or not str(raw).strip():
        from predictors.output_length._ascend_env import default_ascend_device_spec

        raw = default_ascend_device_spec("0,1")

    device_list = _parse_devices(raw)
    if not device_list:
        raise ValueError("ascend_devices must list at least one Ascend device id")

    npu_list = []
    seen = set()
    for item in device_list:
        index = int(item)
        if index not in seen:
            seen.add(index)
            npu_list.append(index)

    configure_ascend_training_env(npu_list)

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    use_data_parallel = len(npu_list) > 1
    available = int(torch.npu.device_count())
    _stdout("torch.npu.device_count()=%d" % available)
    if use_data_parallel:
        if available < len(npu_list):
            raise RuntimeError(
                "Requested %d NPUs (--ascend-devices) but torch.npu.device_count()=%d.\n"
                "torch_npu was imported before ASCEND_RT_VISIBLE_DEVICES took effect.\n"
                "Fix (pick one):\n"
                "  export ASCEND_RT_VISIBLE_DEVICES=%s\n"
                "  python -m predictors.output_length.train_output_length_deberta ...\n"
                "Or use fewer devices / a single NPU with --gradient-checkpointing."
                % (len(npu_list), available, ",".join(str(i) for i in npu_list))
            )
        _stdout(
            "DataParallel will use logical npu:0..%d (~batch_size/%d samples per step per device)"
            % (len(npu_list) - 1, len(npu_list))
        )
    return device, npu_list, use_data_parallel


def setup_inference_device(ascend_devices=None):
    import torch

    primary_id, device_list = configure_ascend_inference_env(ascend_devices)

    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError(
            "Ascend inference requires torch_npu (install the CANN-matched wheel for your PyTorch)."
        ) from exc

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required for inference but torch.npu.is_available() is False")

    if len(device_list) > 1:
        _stdout(
            "inference uses NPU %s only (%d devices listed; DataParallel disabled for predict)"
            % (primary_id, len(device_list))
        )

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    return device, primary_id, device_list


def wrap_model_for_devices(model, npu_list, *, distributed=False):
    import torch.nn as nn

    if distributed:
        return model
    if npu_list and len(npu_list) > 1:
        try:
            import torch

            if torch.npu.is_available():
                _stdout(
                    "Ascend: nn.DataParallel is disabled (does not shard memory reliably). "
                    "Use torchrun --nproc_per_node=N for multi-NPU, or pass a single --ascend-devices id."
                )
                return model
        except ImportError:
            pass
        device_ids = list(range(len(npu_list)))
        _stdout(
            "DataParallel on logical NPUs %s (output_device=0)"
            % ",".join(str(i) for i in device_ids)
        )
        return nn.DataParallel(model, device_ids=device_ids, output_device=0)
    return model


def setup_hybrid_training(ascend_devices=None):
    import torch
    import torch.distributed as dist

    try:
        import torch_npu
    except ImportError as exc:
        raise ImportError(
            "Ascend training requires torch_npu (install the CANN-matched wheel for your PyTorch)."
        ) from exc

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is required for training but torch.npu.is_available() is False")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    raw = ascend_devices
    if raw is None or not str(raw).strip():
        from predictors.output_length._ascend_env import default_ascend_device_spec

        raw = default_ascend_device_spec("0,1")
    npu_list = []
    seen = set()
    for item in _parse_devices(raw):
        index = int(item)
        if index not in seen:
            seen.add(index)
            npu_list.append(index)

    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl")
        configure_ascend_training_env(npu_list)
        available = int(torch.npu.device_count())
        if available < world_size:
            raise RuntimeError(
                "torchrun WORLD_SIZE=%d but torch.npu.device_count()=%d"
                % (world_size, available)
            )
        torch.npu.set_device(local_rank)
        device = torch.device("npu:%d" % local_rank)
        _stdout(
            "DDP rank %d/%d local_rank=%d device=%s (per-rank batch_size from --batch-size)"
            % (rank, world_size, local_rank, device)
        )
        return {
            "device": device,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "is_distributed": True,
            "npu_list": npu_list,
            "use_data_parallel": False,
        }

    configure_ascend_training_env(npu_list)
    available = int(torch.npu.device_count())
    _stdout("torch.npu.device_count()=%d" % available)
    if len(npu_list) == 1 and available > 1:
        _stdout(
            "WARN: single-NPU training but device_count=%d — unset ASCEND_RT_VISIBLE_DEVICES "
            "in your shell or pass only one --ascend-devices id."
            % available
        )
    if len(npu_list) > 1:
        raise RuntimeError(
            "Requested %d NPUs via --ascend-devices but this process is not a torchrun worker.\n"
            "Ascend nn.DataParallel is disabled (OOM). Use one of:\n"
            "  Single NPU:\n"
            "    export ASCEND_RT_VISIBLE_DEVICES=%s\n"
            "    python -m predictors.output_length.train_output_length_deberta --bucket ... "
            "--ascend-devices %s --batch-size 4 --gradient-checkpointing\n"
            "  Multi-NPU (DDP):\n"
            "    export ASCEND_RT_VISIBLE_DEVICES=%s\n"
            "    torchrun --nproc_per_node=%d -m predictors.output_length.train_output_length_deberta "
            "--bucket ... --batch-size 2 --gradient-accumulation-steps 2"
            % (
                len(npu_list),
                npu_list[0],
                npu_list[0],
                ",".join(str(i) for i in npu_list),
                len(npu_list),
            )
        )
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    return {
        "device": device,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "is_distributed": False,
        "npu_list": npu_list,
        "use_data_parallel": False,
    }


def unwrap_model_distributed(model, training_ctx):
    if not training_ctx.get("is_distributed"):
        return model
    if hasattr(model, "module"):
        return model.module
    return model


def wrap_model_distributed(model, training_ctx, *, find_unused_parameters=False):
    if not training_ctx.get("is_distributed"):
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP

    local_rank = int(training_ctx["local_rank"])
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=bool(find_unused_parameters),
    )


def is_main_training_process(training_ctx):
    return int(training_ctx.get("rank", 0)) == 0


def distributed_barrier(training_ctx):
    if not training_ctx.get("is_distributed"):
        return
    import torch.distributed as dist

    dist.barrier()


__all__ = [
    "_parse_devices",
    "configure_ascend_inference_env",
    "configure_ascend_training_env",
    "bootstrap_ascend_training_env",
    "early_bootstrap_ascend_from_argv",
    "setup_training_devices",
    "setup_inference_device",
    "wrap_model_for_devices",
    "setup_hybrid_training",
    "unwrap_model_distributed",
    "wrap_model_distributed",
    "is_main_training_process",
    "distributed_barrier",
]
