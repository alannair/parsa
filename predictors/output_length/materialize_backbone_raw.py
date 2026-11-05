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


import argparse
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _TorchNpuBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch_npu" or fullname.startswith("torch_npu."):
            return importlib.util.spec_from_loader(
                fullname,
                loader=None,
                is_package=fullname.count(".") > 0,
            )
        return None


def _strip_ascend_env():
    for key in list(os.environ):
        if key.startswith("ASCEND_") or key in ("LD_PRELOAD", "NPU_VISIBLE_DEVICES"):
            os.environ.pop(key, None)


def main(argv=None):
    sys.meta_path.insert(0, _TorchNpuBlocker())
    _strip_ascend_env()

    parser = argparse.ArgumentParser(
        prog="predictors.output_length.materialize_backbone_raw",
        description="Extract DeBERTa weights into bundle/backbone/ (no transformers model)",
    )
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "models" / "output_length_deberta"),
    )
    args = parser.parse_args(argv)

    from predictors.output_length.bundle_io import materialize_bundle_backbone_from_pth_raw

    materialize_bundle_backbone_from_pth_raw(args.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
