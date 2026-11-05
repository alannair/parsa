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


import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path, module_name):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PredictorTrainingConfigTests(unittest.TestCase):
    def test_default_ascend_device_spec_prefers_visible_devices(self):
        module = _load_module("predictors/output_length/_ascend_env.py", "ascend_env_test")
        old_ascend = os.environ.pop("ASCEND_DEVICES", None)
        old_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        try:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
            self.assertEqual(module.default_ascend_device_spec("0,1"), "0,1,2,3,4,5,6,7")
        finally:
            if old_visible is None:
                os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            else:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = old_visible
            if old_ascend is not None:
                os.environ["ASCEND_DEVICES"] = old_ascend

    def test_early_bootstrap_does_not_shrink_visible_devices(self):
        module = _load_module("predictors/output_length/_ascend_env.py", "ascend_env_bootstrap_test")
        old_ascend = os.environ.pop("ASCEND_DEVICES", None)
        old_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        try:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
            npu_list = module.early_bootstrap_ascend_from_argv([])
            self.assertEqual(npu_list, [0, 1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(os.environ.get("ASCEND_DEVICES"), "0,1,2,3,4,5,6,7")
            self.assertEqual(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "0,1,2,3,4,5,6,7")
        finally:
            if old_visible is None:
                os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            else:
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = old_visible
            if old_ascend is None:
                os.environ.pop("ASCEND_DEVICES", None)
            else:
                os.environ["ASCEND_DEVICES"] = old_ascend

    def test_resolve_training_micro_batch_size_is_memory_safe_on_npu(self):
        module = _load_module("predictors/output_length/training_memory.py", "training_memory_test")
        device = types.SimpleNamespace(type="npu")
        self.assertEqual(
            module.resolve_training_micro_batch_size(
                16,
                device=device,
                max_length=512,
                train_head_only=False,
                low_memory=False,
                explicit_micro_batch_size=None,
            ),
            1,
        )
        self.assertEqual(
            module.resolve_training_micro_batch_size(
                16,
                device=device,
                max_length=512,
                train_head_only=True,
                low_memory=False,
                explicit_micro_batch_size=None,
            ),
            2,
        )
        self.assertEqual(
            module.resolve_training_micro_batch_size(
                16,
                device=device,
                max_length=512,
                train_head_only=False,
                low_memory=False,
                explicit_micro_batch_size=3,
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
