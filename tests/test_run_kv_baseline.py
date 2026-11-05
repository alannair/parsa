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
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run
from experiments import kv_baseline
from experiments.experiment_config import load_experiments
from experiments.kv_baseline import _build_baseline_env, _result_row, write_summary_csv
from tests.test_serving_profiles import ORIGINS, VERSIONS


class RunKvBaselineTests(unittest.TestCase):
    def test_kv_baseline_config_expands_to_eight_runs(self):
        config_path = REPO_ROOT / "experiments" / "kv-baseline.json"
        experiments = load_experiments(config_path)
        self.assertEqual(len(experiments), 8)
        conv = [item for item in experiments if item.get("workload") == "conv"]
        self.assertEqual(len(conv), 5)
        seq_lens = [item for item in experiments if item.get("workload") in ("short", "medium", "long")]
        self.assertEqual(len(seq_lens), 3)

    def test_build_baseline_env_skips_router_and_sets_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "sample-run"
            run_dir.mkdir()
            presets_path = run_dir / "presets.json"
            presets_path.write_text("[]", encoding="utf-8")
            exp = {
                "name": "profile-kv-baseline-conv-mbt8192-ms50-f1300",
                "model": "demo-model",
                "tp": 2,
                "total_num_reqs": 100,
                "tuning": {"max_num_seqs": 50, "max_num_batched_tokens": 8192, "frequency_mhz": 1300},
                "client": {
                    "prompt_source": "azure-mix",
                    "azure_mix_schedule_csv": "/datasets/azure-mix/scheduled_prompts.csv",
                    "max_tokens": 4096,
                },
            }
            env = _build_baseline_env(exp, presets_path, run_dir / "result.json", run_dir)
        self.assertEqual(env["SKIP_ROUTER"], "1")
        self.assertEqual(env["POWER_MONITOR_ENABLED"], "1")
        self.assertEqual(env["KV_MONITOR_ENABLED"], "1")
        self.assertNotIn("ROUTING_POLICY", env)
        self.assertNotIn("ROUTING_KWARGS", env)
        self.assertEqual(env["DEMO_STATE_DIR"], str(run_dir / "state"))
        self.assertIn("--max-in-flight 50", env["CLIENT_EXTRA_ARGS"])

    def test_baseline_env_transports_experiment_serving_without_mutating_ambient_env(self):
        serving = {"args": {"gpu_memory_utilization": 0.85}, "env": {"OMP_NUM_THREADS": "2"}}
        exp = {"model": "demo-model", "tp": 2, "total_num_reqs": 10, "serving": serving,
               "tuning": {"max_num_seqs": 5}, "client": {"max_in_flight": 100}}
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "8", "ROUTING_POLICY": "ambient",
                                    "ROUTING_KWARGS": "{}", "CLIENT_EXTRA_ARGS": "ambient"}, clear=True):
            original = dict(os.environ)
            root = Path("/runs/example")
            env = _build_baseline_env(exp, root / "presets.json", root / "result.json", root)
            self.assertEqual(json.loads(env["VLLM_SERVING_OVERRIDES"]), {
                "args": {"gpu-memory-utilization": 0.85}, "env": {"OMP_NUM_THREADS": "2"},
            })
            self.assertEqual(env["OMP_NUM_THREADS"], "8")
            self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", env)
            self.assertIn("--max-in-flight 5", env["CLIENT_EXTRA_ARGS"])
            self.assertEqual(exp["client"]["max_in_flight"], 100)
            self.assertEqual(dict(os.environ), original)

    def test_baseline_plans_all_profiles_before_directories_or_launch(self):
        for model, versions in (("Other/model", VERSIONS),
                                ("Qwen/Qwen3-8B", {**VERSIONS, "vllm": "0.11.1"})):
            experiments = [{"name": "valid", "model": "Qwen/Qwen3-8B", "tp": 1,
                            "total_num_reqs": 2, "lanes": [{}]},
                           {"name": "invalid", "model": model, "tp": 1,
                            "total_num_reqs": 2, "lanes": [{}]}]
            with self.subTest(model=model, versions=versions), tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, {"VLLM_SERVING_PROFILES_ENABLED": "0"}, clear=True,
            ), patch.object(kv_baseline, "load_experiments", return_value=experiments), patch.object(
                kv_baseline, "RUNS_ROOT", Path(directory) / "runs",
            ), patch.object(kv_baseline, "runtime_package_context", return_value={"versions": versions, "origins": ORIGINS}), patch.object(
                kv_baseline.subprocess, "run", side_effect=AssertionError("launch before all profiles planned"),
            ):
                with self.assertRaises(ValueError):
                    kv_baseline.main(["--summary-csv", str(Path(directory) / "summary.csv")])
                self.assertFalse((Path(directory) / "runs").exists())
                self.assertFalse((Path(directory) / "summary.csv").exists())

    def test_baseline_parent_child_namespace_and_summary_use_actual_result_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "VLLM_SERVE_EXTRA_ARGS": "--dtype=auto --gpu-memory-utilization 0.7",
            "VLLM_SERVING_PROFILES_ENABLED": "0",
        }, clear=True), patch.object(kv_baseline, "RUNS_ROOT", Path(directory) / "runs"):
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps([{
                "name": "baseline", "model": "Qwen/Qwen3-8B", "tp": 1, "total_num_reqs": 2,
                "lanes": [{"preset_id": "lane", "backend_url": "http://localhost:8200", "max_num_batched_tokens": 8192}],
                "tuning": {"max_num_seqs": 5}, "client": {"max_tokens": 4},
                "serving": {"args": {"gpu-memory-utilization": 0.85}, "env": {"OMP_NUM_THREADS": "2"}},
            }]), encoding="utf-8")
            launched = []

            def launch(command, **kwargs):
                env = dict(kwargs["env"])
                launched.append(env)
                self.assertEqual(command, [sys.executable, str(REPO_ROOT / "run.py")])
                self.assertEqual(env["VLLM_SERVE_EXTRA_ARGS"], "--dtype=auto --gpu-memory-utilization 0.7")
                self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", env)
                self.assertNotIn("OMP_NUM_THREADS", env)
                args, updates, metadata, child_condition = run.plan_serving_profile(
                    env["MODEL"], int(env["TP_SIZE"]), run._split_extra_args(env["VLLM_SERVE_EXTRA_ARGS"]),
                    env, versions={**VERSIONS, "vllm": "0.23.0"}, origins=ORIGINS,
                )
                self.assertEqual(metadata["status"], "applied")
                self.assertEqual(args[args.index("--gpu-memory-utilization") + 1], "0.85")
                self.assertEqual(updates, {"OMP_NUM_THREADS": "2", "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"})
                self.assertEqual(child_condition, env["VLLM_EXPECTED_SERVING_CONDITION"])
                result = Path(env["RESULT_JSON"])
                self.assertTrue(result.parent.name.endswith("-" + child_condition))
                self.assertEqual(Path(env["DEMO_STATE_DIR"]), result.parent / "state")
                self.assertEqual(Path(env["PRESETS_JSON"]).parent, result.parent)
                result.write_text('{"kv_mean_utilization":0.55,"energy_joules":120.0}', encoding="utf-8")
                return SimpleNamespace(returncode=0)

            summary = root / "summary.csv"
            with patch.object(kv_baseline, "runtime_package_context", return_value={
                "versions": {**VERSIONS, "vllm": "0.23.0"}, "origins": ORIGINS,
            }), patch.object(kv_baseline.subprocess, "run", side_effect=launch), redirect_stdout(io.StringIO()):
                self.assertEqual(kv_baseline.main([str(config), "--summary-csv", str(summary)]), 0)
                self.assertEqual(kv_baseline.main([str(config), "--summary-csv", str(summary)]), 0)
                with summary.open(encoding="utf-8", newline="") as handle:
                    measured = next(csv.DictReader(handle))
                self.assertEqual(measured["result_json"], launched[0]["RESULT_JSON"])
                self.assertEqual(measured["kv_mean_utilization"], "0.55")
                self.assertEqual(measured["energy_joules"], "120.0")
                with patch.dict(os.environ, {"VLLM_SERVE_EXTRA_ARGS": "--dtype=bfloat16 --gpu-memory-utilization 0.7"}):
                    self.assertEqual(kv_baseline.main([str(config), "--summary-csv", str(summary), "--dry-run"]), 0)
            self.assertEqual(launched[0]["RESULT_JSON"], launched[1]["RESULT_JSON"])
            with summary.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertNotEqual(row["result_json"], launched[0]["RESULT_JSON"])
            self.assertRegex(Path(row["result_json"]).parent.name, r"^baseline-sp-[0-9a-f]{24}$")
            self.assertEqual(row["kv_mean_utilization"], "")

    def test_baseline_rejects_profile_opt_out(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            kv_baseline.build_parser().parse_args(["--no-serving-profile"])
        self.assertEqual(raised.exception.code, 2)

    def test_client_base_url_uses_lane_port_when_router_skipped(self):
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
            result_json=REPO_ROOT / "result.json",
            demo_state_dir=REPO_ROOT / "tmp",
            rescue_preset_id="rescue",
            lane_devices_json="",
            skip_vllm=False,
            skip_vllm_lanes=set(),
            skip_vllm_p=False,
            skip_vllm_d=False,
            skip_router=True,
            skip_set_freq_from_presets=False,
        )
        rows = [
            run.LanePlanRow(
                preset_id="lane_d",
                port="8200",
                devices="0,1",
                max_tokens=8192,
                max_seqs=50,
                frequency_mhz="1300",
                start=True,
            )
        ]
        self.assertEqual(run.client_base_url(cfg, rows), "http://127.0.0.1:8200/v1")

    def test_write_summary_csv_includes_kv_and_energy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            result_json = run_dir / "result.json"
            result_json.write_text(
                json.dumps(
                    {
                        "kv_mean_utilization": 0.55,
                        "kv_p95_utilization": 0.72,
                        "energy_joules": 120.0,
                        "energy_joules_per_output_token": 0.12,
                        "tokens_per_second": 42.0,
                        "mean_ttft": 0.4,
                        "99.9th_ttft": 1.2,
                        "mean_tpot": 0.03,
                        "99.9th_tpot": 0.08,
                        "total_output_tokens": 1000,
                        "num_requests": 100,
                    }
                ),
                encoding="utf-8",
            )
            exp = {
                "name": "profile-kv-baseline-conv-mbt8192-ms50-f1300",
                "workload": "conv",
                "total_num_reqs": 100,
                "tuning": {"max_num_batched_tokens": 8192, "max_num_seqs": 50, "frequency_mhz": 1300},
                "client": {"max_tokens": 4096},
            }
            row = _result_row(exp, result_json)
            summary_path = run_dir / "summary.csv"
            write_summary_csv([row], summary_path)
            text = summary_path.read_text(encoding="utf-8")
        self.assertIn("kv_mean_utilization", text)
        self.assertIn("0.55", text)
        self.assertIn("energy_joules_per_output_token", text)


if __name__ == "__main__":
    unittest.main()
