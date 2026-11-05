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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictors.common.train_cli import main_for_module
from predictors.performance.common import FEATURE_COLS, TARGET_COL
from predictors.performance import train as performance_train_module
from predictors.performance.train import DEFAULT_MODEL_DIR


def main(argv=None):
    return main_for_module(
        performance_train_module,
        target_col=TARGET_COL,
        default_model_dir=DEFAULT_MODEL_DIR,
        feature_cols=FEATURE_COLS,
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
