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

import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.azure_mix import (
    coerce_prompt,
    events_with_prompts,
    filter_events_for_active_pools,
    load_schedule_events,
    normalize_schedule_events,
    read_csv_dict_rows,
)
from src.client import _request_timeout, _write_result_csv, _write_schedule_cache


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCHEDULE = REPO_ROOT / "tests" / "fixtures" / "scheduled_prompts.csv"


class ClientAzureMixTests(unittest.TestCase):
    def test_request_timeout_override_applies_to_queue_and_total_deadlines(self):
        timeout = _request_timeout(SimpleNamespace(request_timeout_seconds=25200))
        self.assertEqual(timeout.connect, 120)
        self.assertEqual(timeout.sock_read, 25200)
        self.assertEqual(timeout.total, 25200)

    def test_request_timeout_uses_existing_default_when_unspecified(self):
        timeout = _request_timeout(SimpleNamespace(request_timeout_seconds=None))
        self.assertEqual(timeout.sock_read, 600)
        self.assertEqual(timeout.total, 720)

    def test_coerce_prompt_rejects_nan_and_empty(self):
        self.assertIsNone(coerce_prompt(None))
        self.assertIsNone(coerce_prompt(float("nan")))
        self.assertIsNone(coerce_prompt(""))
        self.assertEqual(coerce_prompt(" hello "), "hello")

    def test_coerce_prompt_flattens_text_parts(self):
        self.assertEqual(
            coerce_prompt([{"type": "text", "text": "line one"}, {"text": "line two"}]),
            "line one\nline two",
        )

    def test_filter_events_for_active_pools(self):
        events = [
            {"pool": "conversation", "prompt": "a"},
            {"pool": "code", "prompt": "b"},
        ]
        filtered = filter_events_for_active_pools(events, "code")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["prompt"], "b")

    def test_events_with_prompts_drops_invalid(self):
        events = [
            {"prompt": "ok"},
            {"prompt": None},
            {"prompt": math.nan},
        ]
        usable = events_with_prompts(events)
        self.assertEqual(len(usable), 1)
        self.assertEqual(usable[0]["prompt"], "ok")

    def test_normalize_schedule_events_from_fixture(self):
        with open(FIXTURE_SCHEDULE, "r", encoding="utf-8") as handle:
            import csv

            rows = list(csv.DictReader(handle))
        events = normalize_schedule_events(rows)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["pool"], "conversation")

    def test_load_schedule_only_without_build_inputs(self):
        events = load_schedule_events(
            FIXTURE_SCHEDULE,
            active_pools="summarization",
        )
        self.assertEqual(len(events), 1)
        self.assertIn("Summarize", events[0]["prompt"])

    def test_read_csv_dict_rows_accepts_large_prompt_fields(self):
        long_prompt = "x" * 300_000
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".csv", delete=False) as handle:
            path = Path(handle.name)
            handle.write("arrival_sec,type,pool,dataset_idx,prompt\n")
            handle.write("0.0,0,summarization,0,%s\n" % long_prompt)
        try:
            rows = read_csv_dict_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["prompt"]), 300_000)
        finally:
            path.unlink(missing_ok=True)

    def test_write_schedule_cache_ignores_event_metadata_not_in_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.csv"
            _write_schedule_cache(
                path,
                [
                    {
                        "arrival_sec": 0.0,
                        "type": 0,
                        "pool": "conversation",
                        "dataset_idx": 0,
                        "prompt": "hello",
                        "input_size": 42,
                    }
                ],
            )
            with open(path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(
            rows,
            [
                {
                    "arrival_sec": "0.0",
                    "type": "0",
                    "pool": "conversation",
                    "dataset_idx": "0",
                    "prompt": "hello",
                }
            ],
        )

    def test_write_result_csv_preserves_schedule_order_and_failed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            _write_result_csv(
                path,
                [
                    {"prompt_idx": 3, "ok": False, "input_num_tokens": None, "output_num_tokens": 12},
                    {"prompt_idx": 1, "ok": True, "input_num_tokens": 10, "output_num_tokens": 0},
                    {"prompt_idx": 2, "ok": True, "input_num_tokens": 20, "output_num_tokens": 30},
                ],
            )
            with open(path, "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(
            rows,
            [
                {"index": "0", "input_length": "10", "output_length": "0"},
                {"index": "1", "input_length": "20", "output_length": "30"},
                {"index": "2", "input_length": "", "output_length": ""},
            ],
        )


if __name__ == "__main__":
    unittest.main()
