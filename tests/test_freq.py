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
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import freq


class FreqTests(unittest.TestCase):
    def test_ascend_ml_library_path_supports_architecture_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "aarch64-linux" / "lib64" / "libascend_ml.so"
            expected.parent.mkdir(parents=True)
            expected.touch()
            self.assertEqual(freq._ascend_ml_library_path(root, expected.name), expected)


if __name__ == "__main__":
    unittest.main()
