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


import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class BenchmarkPredictorBundleTests(unittest.TestCase):
    def _sample_frame(self, seed=7, *, include_tp=False):
        rng = np.random.default_rng(seed)
        rows = 48
        data = {
            "input_size": rng.integers(1, 1000, rows),
            "output_size": rng.integers(1, 1000, rows),
            "batch_size": rng.integers(1, 200, rows),
            "frequency": rng.integers(800, 1800, rows),
            "energy_joules": rng.random(rows) * 1000.0,
            "tokens_per_second": rng.random(rows) * 100.0,
        }
        if include_tp:
            data["tp"] = rng.integers(1, 8, rows)
        return pd.DataFrame(data)

    def test_performance_bundle_roundtrip(self):
        from predictors.performance import load_predictor, train_performance_lr

        sample = {
            "input_size": 10,
            "output_size": 100,
            "batch_size": 50,
            "frequency": 1200,
            "tp": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "performance"
            result = train_performance_lr(self._sample_frame(include_tp=True), model_dir=bundle_dir)
            self.assertTrue((bundle_dir / "predictor.pkl").is_file())
            self.assertTrue((bundle_dir / "inference_config.json").is_file())
            self.assertTrue((bundle_dir / "model.joblib").is_file())
            self.assertIn("bundle_dir", result)

            (bundle_dir / "predictor.pkl").unlink()
            reloaded = load_predictor(bundle_dir)
            expected = result["predict"]([sample])[0]["tokens_per_second"]
            actual = reloaded.predict(sample)[0]["tokens_per_second"]
            self.assertAlmostEqual(expected, actual, places=4)

    def test_energy_bundle_roundtrip(self):
        from predictors.energy import load_predictor, train_energy_lr

        sample = {
            "input_size": 10,
            "output_size": 100,
            "batch_size": 50,
            "frequency": 1200,
            "tp": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "energy"
            result = train_energy_lr(self._sample_frame(seed=11, include_tp=True), model_dir=bundle_dir)
            self.assertTrue((bundle_dir / "predictor.pkl").is_file())
            self.assertTrue((bundle_dir / "bundle_meta.json").is_file())

            (bundle_dir / "predictor.pkl").unlink()
            reloaded = load_predictor(bundle_dir)
            expected = result["predict"]([sample])[0]["energy_joules"]
            actual = reloaded.predict(sample)[0]["energy_joules"]
            self.assertAlmostEqual(expected, actual, places=4)


if __name__ == "__main__":
    unittest.main()
