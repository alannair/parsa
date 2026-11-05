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
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run


class RunKvMergeTests(unittest.TestCase):
    def test_merge_kv_into_result_adds_kv_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result_json = state_dir / "result.json"
            kv_summary = state_dir / "kv_summary.json"
            result_json.write_text(
                json.dumps({"total_output_tokens": 1000, "energy_joules": 0.0}),
                encoding="utf-8",
            )
            kv_summary.write_text(
                json.dumps(
                    {
                        "samples": 3,
                        "duration_s": 2.5,
                        "host_swap_samples": 0,
                        "mean_utilization": 0.55,
                        "p95_utilization": 0.72,
                        "peak_utilization": 0.8,
                        "mean_peak_util_p95": 0.76,
                        "by_lane": {"lane_d": {"mean_utilization": 0.55}},
                        "kv_samples_series": [{"timestamp": 1.0, "per_lane": {}}],
                    }
                ),
                encoding="utf-8",
            )
            cfg = run.RunConfig(
                online_serving=REPO_ROOT,
                vllm_root=REPO_ROOT,
                model="demo",
                tp_size=2,
                ascend_devices="0,1",
                wait_secs=0,
                wait_bar_width=10,
                router_port=8010,
                lane_p_port="8100",
                lane_d_port="8200",
                slo_max_latency_s=120.0,
                presets_json=REPO_ROOT / "src/presets.example.json",
                profile_json=REPO_ROOT / "src/profile.example.json",
                router_extra_args=[],
                routing_policy="",
                routing_kwargs="",
                client_max_arrivals=10,
                client_extra_args=[],
                kill_after_client=True,
                vllm_serve_extra_args=[],
                power_monitor_enabled=True,
                power_interval_s=1.0,
                kv_monitor_enabled=True,
                kv_interval_s=1.0,
                kv_timeout_s=1.0,
                result_json=result_json,
                demo_state_dir=state_dir,
                rescue_preset_id="rescue",
                lane_devices_json="",
                skip_vllm=False,
                skip_vllm_lanes=set(),
                skip_vllm_p=False,
                skip_vllm_d=False,
                skip_router=False,
                skip_set_freq_from_presets=False,
            )
            run.merge_kv_into_result(cfg, result_fresh=True)
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        self.assertEqual(payload["kv_samples"], 3)
        self.assertEqual(payload["kv_mean_utilization"], 0.55)
        self.assertEqual(payload["kv_p95_utilization"], 0.72)
        self.assertEqual(len(payload["kv_samples_series"]), 1)


if __name__ == "__main__":
    unittest.main()
