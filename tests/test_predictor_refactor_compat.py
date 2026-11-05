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


import ast
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


class PredictorRefactorCompatibilityTests(unittest.TestCase):
    def _assert_class_declared_in_file(self, rel_path, class_name):
        file_path = Path(rel_path)
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn(class_name, classes, msg=f"{class_name} missing in {rel_path}")

    def test_pickle_sensitive_class_paths_remain_stable(self):
        from predictors.output_length.common import OutputLengthPredictor

        self.assertEqual(OutputLengthPredictor.__module__, "predictors.output_length.common")
        self._assert_class_declared_in_file(
            "predictors/output_length/deberta_distribution.py",
            "DebertaDistributionModel",
        )
        self._assert_class_declared_in_file(
            "predictors/output_length/deberta_hybrid.py",
            "DebertaHybridBucketModel",
        )
        self._assert_class_declared_in_file(
            "predictors/output_length/train_output_length.py",
            "BertRegressionModel",
        )

    def test_common_facade_delegates_to_new_modules(self):
        from predictors.output_length import common
        from predictors.output_length import ascend, dataset_io, inference

        self.assertIs(common.setup_training_devices, ascend.setup_training_devices)
        self.assertIs(common.load_labeled_csv, dataset_io.load_labeled_csv)
        self.assertIs(common.load_predictor_cpu_safe, inference.load_predictor_cpu_safe)
        self.assertIs(common.predict_output_tokens, inference.predict_output_tokens)

    def test_labeled_csv_loader_accepts_legacy_minimum_schema(self):
        from predictors.output_length.dataset_io import load_labeled_csv

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "labels.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["prompt", "type", "output_length", "ignored"],
                )
                writer.writeheader()
                for i in range(12):
                    writer.writerow(
                        {
                            "prompt": f"hello {i}",
                            "type": i % 3,
                            "output_length": 10 + i,
                            "ignored": "x",
                        }
                    )
            rows = load_labeled_csv(csv_path)
            self.assertEqual(len(rows), 12)
            self.assertEqual(set(rows[0].keys()), {"prompt", "type", "output_length"})

    def test_prediction_csv_layout_is_legacy_by_default(self):
        from predictors.output_length.dataset_io import write_predict_results_csv

        rows = [
            {
                "prompt": "hi",
                "type": 1,
                "input_length": 5,
                "output_length": 12,
            }
        ]
        preds = [12.5]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pred.csv"
            write_predict_results_csv(rows, preds, out, include_prompt=False)
            with open(out, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    ["type", "input_length", "output_length", "prediction"],
                )
                first = next(reader)
                self.assertEqual(first["type"], "1")
                self.assertEqual(first["input_length"], "5")
                self.assertEqual(first["output_length"], "12")

    def test_cli_entrypoints_keep_old_flags(self):
        from predictors.output_length.train_output_length_deberta import build_parser as build_deberta_parser

        if importlib.util.find_spec("torch") is not None:
            from predictors.output_length.train_output_length import build_parser as build_bert_parser

            bert = build_bert_parser().parse_args(
                [
                    "--csv-path",
                    "x.csv",
                    "--model-dir",
                    "m",
                    "--batch-size",
                    "8",
                    "--ascend-devices",
                    "0",
                ]
            )
            self.assertEqual(bert.batch_size, 8)
            self.assertEqual(bert.ascend_devices, "0")

        deberta = build_deberta_parser().parse_args(
            [
                "--csv-path",
                "x.csv",
                "--model-dir",
                "m",
                "--batch-size",
                "8",
                "--gradient-accumulation-steps",
                "2",
                "--bucket",
                "--bf16",
                "--gradient-checkpointing",
                "--low-memory",
                "--low-memory-mode",
                "--train-head-only",
                "--backbone-on-cpu",
                "--ascend-devices",
                "0,1",
            ]
        )
        self.assertTrue(deberta.bucket)
        self.assertEqual(deberta.gradient_accumulation_steps, 2)
        self.assertTrue(deberta.bf16)
        self.assertEqual(deberta.ascend_devices, "0,1")

    def test_internal_api_exports_expected_functions(self):
        from predictors.output_length import api

        self.assertTrue(callable(api.train_bert_point))
        self.assertTrue(callable(api.train_deberta_lognormal))
        self.assertTrue(callable(api.train_deberta_hybrid))
        self.assertTrue(callable(api.predict_output_tokens))
        self.assertTrue(callable(api.load_predictor_cpu_safe))

    def test_benchmark_predictors_share_the_common_implementation(self):
        for rel_path in (
            "predictors/energy/train.py",
            "predictors/performance/train.py",
            "predictors/energy/inference.py",
            "predictors/performance/inference.py",
        ):
            source = Path(rel_path).read_text(encoding="utf-8")
            self.assertIn(
                "from predictors.common.benchmark import BenchmarkPredictor",
                source,
            )


if __name__ == "__main__":
    unittest.main()
