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

import sys
import unittest

from experiments.azure_mix import _vllm_command


class RunAzureMixTests(unittest.TestCase):
    def test_vllm_command_uses_all_visible_devices_for_tensor_parallelism(self):
        self.assertEqual(
            _vllm_command("/models/qwen3-8b", 8010, ["0", "2", "3"]),
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                "/models/qwen3-8b",
                "--port",
                "8010",
                "--tensor-parallel-size",
                "3",
            ],
        )


if __name__ == "__main__":
    unittest.main()
