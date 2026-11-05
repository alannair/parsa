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

from plots import dataload


CONDITION = "sp-" + "a" * 24
FOLDER = "profile-decode-heavy-conv-mbt8192-ms5-f1300"


def result(root, folder, condition=None):
    path = root / folder / "result.json"
    path.parent.mkdir(parents=True)
    payload = {"requests": [{"ttft_s": 1.5, "tpot_mean_s": 0.25}]}
    if condition is not None:
        payload["serving_condition"] = condition
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class LatencyServingConditionsTests(unittest.TestCase):
    def test_legacy_parse_shape_is_preserved_and_suffix_is_loaded(self):
        expected = {"experiment": "decode-heavy", "workload": "conv", "max_num_batched_tokens": 8192,
                    "max_num_seqs": 5, "frequency_mhz": 1300}
        self.assertEqual(dataload.parse_profile_latency_dir_name(FOLDER), expected)
        self.assertEqual(dataload.parse_profile_latency_dir_name(FOLDER + "-" + CONDITION),
                         {**expected, "serving_condition": CONDITION})
        self.assertIsNone(dataload.parse_profile_latency_dir_name(FOLDER + "-sp-short"))
        for suffix, metadata, expected_condition in (("", None, "legacy"), ("", CONDITION, CONDITION),
                                                    ("-" + CONDITION, CONDITION, CONDITION),
                                                    ("-" + CONDITION, None, CONDITION)):
            with self.subTest(suffix=suffix, metadata=metadata), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = result(root, FOLDER + suffix, metadata)
                frame = dataload.load_latency_experiments_dataframe(root, merge_summary=False)
                self.assertEqual(frame["ttft_s"].tolist(), [1.5])
                self.assertEqual(frame["serving_condition"].tolist(), [expected_condition])
                self.assertEqual(dataload.load_latency_json_result(path, merge_summary=False)["serving_condition"].tolist(),
                                 [expected_condition])

    def test_suffix_metadata_mismatch_is_not_silently_loaded_or_filtered_away(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = result(root, FOLDER + "-" + CONDITION, "sp-" + "b" * 24)
            with self.assertRaisesRegex(ValueError, "serving condition.*mismatch"):
                dataload.load_latency_json_result(path)
            with self.assertRaisesRegex(ValueError, "serving condition.*mismatch"):
                dataload.load_latency_experiments_dataframe(root, serving_condition="legacy")

    def test_mixed_conditions_require_filter_before_downstream_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [result(root, FOLDER), result(root, FOLDER + "-" + CONDITION, CONDITION)]
            with self.assertRaisesRegex(ValueError, "multiple serving conditions.*serving_condition"):
                dataload.load_latency_experiments_dataframe(root)
            with self.assertRaisesRegex(ValueError, "multiple serving conditions"):
                dataload.load_latency_json_results(paths)
            for condition in ("legacy", CONDITION):
                frame = dataload.load_latency_experiments_dataframe(root, serving_condition=condition)
                self.assertEqual(frame["serving_condition"].tolist(), [condition])
                self.assertEqual(frame["ttft_s"].mean(), 1.5)
                self.assertEqual(dataload.load_latency_json_results(paths, serving_condition=condition)["serving_condition"].tolist(),
                                 [condition])


if __name__ == "__main__":
    unittest.main()
