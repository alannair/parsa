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
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import kv_meter
from src.kv_projection import SCHEMA_VERSION, KVProjectionSnapshot
from tests.test_kv_project import make_projection_payload


class KvMeterTests(unittest.TestCase):
    def test_targets_from_arg_parses_preset_urls(self):
        targets = kv_meter._targets_from_arg("lane_d=http://127.0.0.1:8200,lane_p=http://127.0.0.1:8100")
        self.assertEqual(
            targets,
            {
                "lane_d": "http://127.0.0.1:8200",
                "lane_p": "http://127.0.0.1:8100",
            },
        )

    def test_snapshot_records_lane_metrics(self):
        payload = make_projection_payload("lane_d", 0.72, used_blocks=72, capacity_blocks=100)
        snapshot = KVProjectionSnapshot.from_payload(payload, expected_engine_id="lane_d")
        metrics = snapshot.to_metrics_dict()
        self.assertEqual(metrics["current_utilization"], 0.72)
        self.assertEqual(metrics["peak_util_p95"], 0.72)
        self.assertEqual(metrics["npu_used_blocks"], 72)

    def test_snapshot_writes_per_lane_metrics(self):
        payload = make_projection_payload("lane_d", 0.62, used_blocks=62, capacity_blocks=100)
        seen = {}

        class FakeKvModule:
            @staticmethod
            def admin_kv_projection_url(backend_url):
                return backend_url.rstrip("/") + "/admin/parsa/kv-projection"

            @staticmethod
            def fetch_kv_projection_snapshot_sync(url, timeout_s, expected_engine_id=None):
                seen["expected_engine_id"] = expected_engine_id
                return KVProjectionSnapshot.from_payload(payload, expected_engine_id="lane_d")

        fake_module = FakeKvModule()
        with patch.object(kv_meter, "_load_kv_projection_module", return_value=fake_module):
            result = kv_meter.snapshot({"lane_d": "http://127.0.0.1:8200"})
        self.assertEqual(seen["expected_engine_id"], "lane_d")
        sample = result["per_lane"]["lane_d"]
        self.assertEqual(sample["current_utilization"], 0.62)
        self.assertNotIn("error", sample)

    def test_summarize_log_computes_utilization_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "kv.jsonl"
            lines = []
            for used in (20, 40, 60):
                payload = make_projection_payload("lane_d", used / 100.0, used_blocks=used, capacity_blocks=100)
                snapshot = KVProjectionSnapshot.from_payload(payload, expected_engine_id="lane_d")
                lines.append(
                    {
                        "timestamp": time.time() + used,
                        "per_lane": {"lane_d": snapshot.to_metrics_dict()},
                    }
                )
            with open(log_path, "w", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            summary = kv_meter.summarize_log(str(log_path))
        self.assertEqual(summary["samples"], 3)
        self.assertAlmostEqual(summary["mean_utilization"], 0.4)
        self.assertAlmostEqual(summary["peak_utilization"], 0.6)
        self.assertEqual(summary["host_swap_samples"], 0)
        self.assertIn("lane_d", summary["by_lane"])
        self.assertEqual(len(summary["kv_samples_series"]), 3)


if __name__ == "__main__":
    unittest.main()
