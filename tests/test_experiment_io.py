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

import json
import tempfile
import unittest
from pathlib import Path

from predictors.common.experiment_io import (
    dump_benchmark_csv,
    extract_parameters_from_filename,
    generate_experiment_filename,
    load_experiment_json_row,
)


class ExperimentIoTests(unittest.TestCase):
    def test_extract_parameters_from_filename(self):
        name = generate_experiment_filename(
            "batch-size",
            "qwen3-4b",
            batch_size=50,
            input_size=10,
            output_size=100,
            frequency=1200,
            tp=2,
        )
        params = extract_parameters_from_filename(name)
        self.assertEqual(params["experiment"], "batch-size")
        self.assertEqual(params["model"], "qwen3-4b")
        self.assertEqual(params["input_size"], "10")
        self.assertEqual(params["output_size"], "100")
        self.assertEqual(params["batch_size"], "50")
        self.assertEqual(params["frequency"], "1200")
        self.assertEqual(params["tp"], "2")

    def test_dump_benchmark_csv_from_experiments_root(self):
        payload = {
            "energy_joules": 123.4,
            "tokens_per_second": 56.7,
            "total_elapsed_time": 10.0,
            "total_output_tokens": 567.0,
        }
        name = generate_experiment_filename(
            "batch-size",
            "qwen3-4b",
            batch_size=50,
            input_size=10,
            output_size=100,
            frequency=1200,
            tp=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiments"
            root.mkdir()
            json_path = root / name
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            out_csv = Path(tmp) / "benchmark.csv"
            out_path, df = dump_benchmark_csv(root, out_csv)
            self.assertEqual(out_path, out_csv.resolve())
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(int(row["input_size"]), 10)
            self.assertEqual(int(row["output_size"]), 100)
            self.assertEqual(int(row["batch_size"]), 50)
            self.assertEqual(int(row["frequency"]), 1200)
            self.assertEqual(int(row["tp"]), 2)
            self.assertAlmostEqual(float(row["energy_joules"]), 123.4)
            self.assertAlmostEqual(float(row["tokens_per_second"]), 56.7)

            row_from_file = load_experiment_json_row(json_path)
            self.assertEqual(row_from_file["experiment"], "batch-size")
            self.assertEqual(row_from_file["model"], "qwen3-4b")


if __name__ == "__main__":
    unittest.main()
