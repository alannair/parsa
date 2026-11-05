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


def _parse_devices(devices):
    items = [item.strip() for item in str(devices or "").split(",") if item.strip()]
    return items


def default_ascend_device_spec(default="0,1"):
    for name in ("ASCEND_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return str(default)


def configure_ascend_training_env(npu_list):
    os.environ.setdefault("TASK_QUEUE_ENABLE", "0")
    os.environ.setdefault("ASCEND_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("COMBINED_ENABLE", "1")
    visible = ",".join(str(int(i)) for i in npu_list)
    os.environ["ASCEND_DEVICES"] = visible
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible
    print(
        "Ascend training: ASCEND_RT_VISIBLE_DEVICES=%s (logical npu:0..%d)"
        % (visible, max(len(npu_list) - 1, 0)),
        flush=True,
    )


def early_bootstrap_ascend_from_argv(argv):
    argv = list(argv or [])
    if any(flag in argv for flag in ("--predict-only", "--materialize-backbone")):
        return []
    raw = default_ascend_device_spec("0,1")
    for index, tok in enumerate(argv):
        if tok == "--ascend-devices" and index + 1 < len(argv):
            raw = argv[index + 1]
            break
        if tok.startswith("--ascend-devices="):
            raw = tok.split("=", 1)[1]
            break
    npu_list = []
    seen = set()
    for item in _parse_devices(raw):
        index = int(item)
        if index not in seen:
            seen.add(index)
            npu_list.append(index)
    if npu_list:
        configure_ascend_training_env(npu_list)
    return npu_list
