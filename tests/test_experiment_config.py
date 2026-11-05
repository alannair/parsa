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

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import benchmarks
from experiments.experiment_config import _client_extra_args, expand_sweep_config, load_experiments
from experiments.benchmarks import _progress_status
import run
from src import serving_profiles as profiles
from tests.test_serving_profiles import ORIGINS, VERSIONS


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTests(unittest.TestCase):
    def test_multi_experiment_progress_status_includes_completion_and_exit_code(self):
        self.assertEqual(
            _progress_status(17, 400, "profile-request-shape-r1", 0),
            "[17/400] finished profile-request-shape-r1 (exit 0)",
        )
        self.assertIsNone(_progress_status(1, 1, "only-run", 0))

    def test_experiments_json_expands_to_expected_grid(self):
        config_path = REPO_ROOT / "experiments" / "benchmarks.json"
        experiments = load_experiments(config_path)
        self.assertEqual(len(experiments), 135)

        decode = [item for item in experiments if item.get("role") == "decode-heavy"]
        prefill = [item for item in experiments if item.get("role") == "prefill-heavy"]
        self.assertEqual(len(decode), 90)
        self.assertEqual(len(prefill), 45)

        conv = [item for item in decode if item.get("workload") == "conv"]
        code = [item for item in decode if item.get("workload") == "code"]
        self.assertEqual(len(conv), 45)
        self.assertEqual(len(code), 45)

        sample = next(item for item in conv if item["name"].endswith("mbt8192-ms100-f1300-r1"))
        self.assertEqual(sample["lanes"][0]["max_num_batched_tokens"], 8192)
        self.assertEqual(sample["lanes"][0]["max_num_seqs"], 100)
        self.assertEqual(sample["lanes"][0]["frequency_mhz"], 1300)
        self.assertEqual(sample["lanes"][0]["capabilities"], {"kv_projection": True})
        self.assertEqual(sample["client"]["azure_mix_schedule_csv"], "/datasets/azure-mix/scheduled_prompts.csv")
        self.assertEqual(sample["client"]["active_pools"], "conversation")
        self.assertEqual(sample["client"]["max_tokens"], 8192)

        sum_sample = next(item for item in prefill if item.get("workload") == "sum")
        self.assertEqual(sum_sample["client"]["active_pools"], "summarization")
        self.assertEqual(sum_sample["client"]["max_tokens"], 1024)
        self.assertEqual(len(sum_sample["lanes"]), 1)

    def test_build_env_uses_shared_schedule_without_rebuild(self):
        from experiments.benchmarks import _build_env

        run_dir = REPO_ROOT / "experiments" / "_runs" / "sample-run"
        exp = {
            "model": "demo-model",
            "tp": 2,
            "total_num_reqs": 10,
            "workload": "conv",
            "tuning": {"frequency_mhz": 1300, "max_num_seqs": 50, "max_num_batched_tokens": 8192},
            "client": {
                "prompt_source": "azure-mix",
                "azure_mix_schedule_csv": "/datasets/azure-mix/scheduled_prompts.csv",
                "active_pools": "conversation",
            },
        }
        env = _build_env(exp, run_dir / "presets.json", run_dir / "result.json", run_dir)
        extra = env.get("CLIENT_EXTRA_ARGS", "")
        self.assertNotIn("--rebuild-azure-mix-schedule", extra)
        self.assertIn("/datasets/azure-mix/scheduled_prompts.csv", extra)
        self.assertIn("--active-pools conversation", extra)

    def test_build_env_defaults_power_interval_but_preserves_override(self):
        exp = {
            "model": "demo-model", "tp": 1, "total_num_reqs": 1,
            "tuning": {"frequency_mhz": 1300, "max_num_seqs": 1, "max_num_batched_tokens": 1},
        }
        with patch.dict(os.environ, {}, clear=True):
            env = benchmarks._build_env(exp, "presets.json", "result.json", Path("run"))
        self.assertEqual(env["POWER_INTERVAL_S"], "0.05")
        with patch.dict(os.environ, {"POWER_INTERVAL_S": "1.0"}, clear=True):
            env = benchmarks._build_env(exp, "presets.json", "result.json", Path("run"))
        self.assertEqual(env["POWER_INTERVAL_S"], "1.0")

    def test_legacy_array_format_still_loads(self):
        experiments = load_experiments(REPO_ROOT / "experiments" / "example.json")
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["name"], "n-lane-latency")

    def test_expand_sweep_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            expand_sweep_config({"format": "unknown"})

    def test_kv_energy_config_expands_concurrency_ladder(self):
        config_path = REPO_ROOT / "experiments" / "kv-energy.json"
        experiments = load_experiments(config_path)
        self.assertEqual(len(experiments), 5)
        seq_values = sorted(item["lanes"][0]["max_num_seqs"] for item in experiments)
        self.assertEqual(seq_values, [10, 25, 50, 100, 200])
        sample = experiments[0]
        self.assertEqual(sample["lanes"][0]["frequency_mhz"], 1300)
        self.assertEqual(sample["lanes"][0]["capabilities"], {"kv_projection": True})

    def test_request_shape_characterization_expands_all_operating_points_and_repetitions(self):
        experiments = load_experiments(REPO_ROOT / "experiments" / "request-shape-characterization.json")
        self.assertEqual(len(experiments), 2 * 5 * 8 * 5)
        sample = next(item for item in experiments if item["name"].endswith("prefill-heavy-mbt8192-ms50-f1300-r3"))
        self.assertEqual(sample["tp"], 2)
        self.assertEqual(sample["repetition"], 3)
        self.assertEqual(sample["metadata"]["request_shape"], "prefill-heavy")
        self.assertEqual(sample["client"]["synthetic_prompt_tokens"], 1000)
        self.assertEqual(sample["client"]["max_tokens"], 1)
        self.assertEqual(sample["client"]["request_timeout_seconds"], 25200)
        self.assertIn("--request-timeout-seconds 25200", " ".join(_client_extra_args(sample["client"])))

    def test_benchmarks_condition_directories_are_stable_and_expected_key_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), patch.object(
            benchmarks, "RUNS_ROOT", Path(directory) / "runs",
        ):
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({
                "format": "parsa-benchmark-sweep-v1", "model": "/models/qwen3-8b", "tp": 2,
                "serving": {"args": {"gpu-memory-utilization": 0.85}, "env": {"OMP_NUM_THREADS": "2"}},
                "repetitions": 2, "lanes": [{
                    "role": "request-shape", "preset_id": "lane_d", "lane_type": "generic",
                    "backend_url": "http://127.0.0.1:8200", "workloads": [{"suffix": "decode-heavy", "total_num_reqs": 2}],
                    "sweep": {"max_num_batched_tokens": [8192], "max_num_seqs": [5], "frequency_mhz": [1300]},
                }],
            }), encoding="utf-8")
            launched = []

            def launch(command, **kwargs):
                launched.append((command, dict(kwargs["env"])))
                return SimpleNamespace(returncode=0)

            with patch.object(profiles.importlib.util, "find_spec", side_effect=lambda name: SimpleNamespace(
                origin=ORIGINS[name]["origin"], submodule_search_locations=[ORIGINS[name]["root"]],
            )), patch.object(profiles.distribution_metadata, "version", return_value="0.11.0"), patch.object(
                benchmarks.subprocess, "run", side_effect=launch,
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(benchmarks.main([str(config)]), 0)
                self.assertEqual(benchmarks.main([str(config)]), 0)
                with patch.dict(os.environ, {"VLLM_SERVE_EXTRA_ARGS": "--dtype=bfloat16"}):
                    self.assertEqual(benchmarks.main([str(config)]), 0)
            self.assertEqual(len(launched), 6)
            first, repeated, changed = launched[0][1], launched[2][1], launched[4][1]
            self.assertEqual(first["RESULT_JSON"], repeated["RESULT_JSON"])
            self.assertNotEqual(first["RESULT_JSON"], changed["RESULT_JSON"])
            self.assertEqual(first["VLLM_EXPECTED_SERVING_CONDITION"], launched[1][1]["VLLM_EXPECTED_SERVING_CONDITION"])
            for command, env in launched:
                condition = env["VLLM_EXPECTED_SERVING_CONDITION"]
                self.assertRegex(condition, r"^sp-[0-9a-f]{24}$")
                result = Path(env["RESULT_JSON"])
                self.assertEqual(result.name, "result.json")
                self.assertTrue(result.parent.name.startswith("profile-request-shape-decode-heavy-"))
                self.assertTrue(result.parent.name.endswith("-" + condition))
                self.assertEqual(Path(env["PRESETS_JSON"]).parent, result.parent)
                self.assertEqual(json.loads(Path(env["PRESETS_JSON"]).read_text())[0]["max_num_seqs"], 5)
                self.assertEqual(command, [benchmarks.sys.executable, str(REPO_ROOT / "run.py")])
                _, _, metadata, child_condition = run.plan_serving_profile(
                    env["MODEL"], int(env["TP_SIZE"]), run._split_extra_args(env.get("VLLM_SERVE_EXTRA_ARGS", "")),
                    env, versions=VERSIONS, origins=ORIGINS,
                )
                self.assertEqual(metadata["package_origins"], ORIGINS)
                self.assertEqual(child_condition, condition)
                self.assertEqual(metadata["effective_extra_args"][:2], ["--gpu-memory-utilization", "0.85"])
                self.assertEqual(metadata["relevant_env"]["OMP_NUM_THREADS"], "2")
                self.assertNotIn("OMP_NUM_THREADS", env)
                self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", env)
                self.assertNotIn("--gpu-memory-utilization", env.get("VLLM_SERVE_EXTRA_ARGS", ""))
                shadow = {**ORIGINS, "vllm": {"origin": "/shadow/vllm/__init__.py", "root": "/shadow/vllm"}}
                _, _, _, shadow_condition = run.plan_serving_profile(
                    env["MODEL"], int(env["TP_SIZE"]), run._split_extra_args(env.get("VLLM_SERVE_EXTRA_ARGS", "")),
                    env, versions=VERSIONS, origins=shadow,
                )
                self.assertNotEqual(shadow_condition, condition)

            output = io.StringIO()
            with patch.object(benchmarks, "runtime_package_context", return_value={"versions": VERSIONS, "origins": ORIGINS}), patch.dict(
                os.environ, {"VLLM_SERVING_PROFILES_ENABLED": "0"},
            ), redirect_stdout(output):
                self.assertEqual(benchmarks.main([str(config), "--dry-run"]), 0)
            self.assertIn("status=applied", output.getvalue())
            self.assertIn("VLLM_EXPECTED_SERVING_CONDITION=sp-", output.getvalue())
            self.assertIn("serving args:", output.getvalue())

    def test_benchmarks_reject_reserved_metadata_before_any_subprocess(self):
        for key in ("serving_profile", "serving_condition"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "config.json"
                config.write_text(json.dumps([
                    {"name": "valid", "model": "Qwen/Qwen3-8B", "tp": 2, "total_num_reqs": 2,
                     "lanes": [{"preset_id": "lane", "backend_url": "http://localhost:8200", "max_num_batched_tokens": 8192}],
                     "tuning": {"max_num_batched_tokens": 8192, "max_num_seqs": 5, "frequency_mhz": 1300}},
                    {"name": "invalid", "model": "Qwen/Qwen3-8B", "tp": 2, "total_num_reqs": 2,
                     "lanes": [{}], "metadata": {key: "spoofed"},
                     "tuning": {"max_num_batched_tokens": 8192, "max_num_seqs": 5, "frequency_mhz": 1300}},
                ]), encoding="utf-8")
                with patch.dict(os.environ, {}, clear=True), patch.object(
                    benchmarks, "RUNS_ROOT", Path(directory) / "runs",
                ), patch.object(benchmarks, "runtime_package_context", return_value={"versions": VERSIONS, "origins": ORIGINS}), patch.object(
                    benchmarks.subprocess, "run", side_effect=AssertionError("subprocess before validation"),
                ):
                    with self.assertRaisesRegex(ValueError, "reserved"):
                        benchmarks.main([str(config)])
                    self.assertFalse((Path(directory) / "runs").exists())

    def test_sweep_serving_layers_root_lane_workload_without_client_leakage(self):
        config = {"format": "parsa-benchmark-sweep-v1", "model": "Qwen/Qwen3-8B", "tp": 1,
                  "serving": {"args": {"gpu_memory_utilization": 0.8, "additional-config": {"root": 1},
                                       "compilation-config": {"nested": {"root": 1, "replace": 0}}, "enforce-eager": True},
                              "env": {"OMP_NUM_THREADS": "8"}},
                  "lanes": [{"preset_id": "lane", "backend_url": "http://localhost:8200",
                             "serving": {"args": {"additional-config": {"lane": 2},
                                                  "compilation-config": {"nested": {"lane": 2}}}, "env": {"OMP_NUM_THREADS": "4"}},
                             "sweep": {"max_num_batched_tokens": [8192], "max_num_seqs": [5], "frequency_mhz": [1300]},
                             "workloads": [{"total_num_reqs": 2, "serving": {
                                 "args": {"gpu-memory-utilization": 0.85, "additional-config": {"root": 3},
                                          "compilation-config": {"nested": {"replace": 3}}},
                                 "env": {"OMP_NUM_THREADS": "2"}}}]}]}
        exp = expand_sweep_config(config)[0]
        self.assertEqual(exp["serving"], {"args": {"gpu-memory-utilization": 0.85,
                         "additional-config": {"root": 3, "lane": 2}, "enforce-eager": True,
                         "compilation-config": {"nested": {"root": 1, "lane": 2, "replace": 3}}},
                         "env": {"OMP_NUM_THREADS": "2"}})
        self.assertNotIn("serving", exp["client"])
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / "flat.json"
            path.write_text(json.dumps([exp]), encoding="utf-8")
            self.assertEqual(load_experiments(path)[0]["serving"], exp["serving"])
            env = benchmarks._build_env(exp, path, path, Path(directory))
            self.assertNotIn("OMP_NUM_THREADS", env)
            args, updates, _, _ = run.plan_serving_profile(exp["model"], 1, [], env,
                                                          versions={**VERSIONS, "vllm": "0.23.0"})
            self.assertEqual(args[args.index("--gpu-memory-utilization") + 1], "0.85")
            self.assertEqual(updates["OMP_NUM_THREADS"], "2")
            self.assertEqual(updates["PYTORCH_NPU_ALLOC_CONF"], "expandable_segments:True")

    def test_flat_serving_intent_survives_normalization_and_env_builder_without_global_mutation(self):
        serving = {"args": {"gpu_memory_utilization": 0.85, "enforce_eager": False,
                            "served-model-name": ["first", "second"],
                            "additional_config": {"nested": {"array": [1, 2]}},
                            "compilation_config": {"nested": {"keep": 1}}},
                   "env": {"OMP_NUM_THREADS": "2", "PYTORCH_NPU_ALLOC_CONF": "experiment"}}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "OMP_NUM_THREADS": "8", "PYTORCH_NPU_ALLOC_CONF": "ambient",
            "VLLM_SERVE_EXTRA_ARGS": "--gpu-memory-utilization 0.7 --enforce-eager",
        }, clear=True):
            original = dict(os.environ)
            path = Path(directory) / "flat.json"
            path.write_text(json.dumps([{
                "name": "flat", "model": "Qwen/Qwen3-8B", "tp": 1, "total_num_reqs": 2,
                "lanes": [{}], "tuning": {"frequency_mhz": 1300, "max_num_seqs": 5, "max_num_batched_tokens": 8192},
                "serving": serving,
            }]), encoding="utf-8")
            exp = load_experiments(path)[0]
            self.assertEqual(exp["serving"], {
                "args": {"gpu-memory-utilization": 0.85, "enforce-eager": False,
                         "served-model-name": ["first", "second"],
                         "additional-config": {"nested": {"array": [1, 2]}},
                         "compilation-config": {"nested": {"keep": 1}}}, "env": serving["env"],
            })
            env = benchmarks._build_env(exp, path, path, Path(directory))
            self.assertEqual(json.loads(env["VLLM_SERVING_OVERRIDES"]), exp["serving"])
            self.assertEqual(env["OMP_NUM_THREADS"], "8")
            self.assertEqual(env["PYTORCH_NPU_ALLOC_CONF"], "ambient")
            args, updates, _, _ = run.plan_serving_profile(
                exp["model"], exp["tp"], run._split_extra_args(env["VLLM_SERVE_EXTRA_ARGS"]), env,
                versions={**VERSIONS, "vllm": "0.23.0"},
            )
            self.assertEqual(args[args.index("--gpu-memory-utilization") + 1], "0.85")
            self.assertIn("--no-enforce-eager", args)
            self.assertEqual(updates, {"OMP_NUM_THREADS": "2", "PYTORCH_NPU_ALLOC_CONF": "experiment"})
            self.assertEqual(dict(os.environ), original)

    def test_root_only_serving_applies_to_every_workload(self):
        config = {"format": "parsa-benchmark-sweep-v1", "model": "Qwen/Qwen3-8B", "tp": 1,
                  "repetitions": 2, "serving": {"args": {"dtype": "auto"}, "env": {"OMP_NUM_THREADS": "2"}},
                  "lanes": [{"preset_id": "lane", "backend_url": "http://localhost:8200",
                             "sweep": {"max_num_batched_tokens": [8192], "max_num_seqs": [5], "frequency_mhz": [1300]},
                             "workloads": [{"suffix": "one"}, {"suffix": "two"}]}]}
        experiments = expand_sweep_config(config)
        self.assertEqual(len(experiments), 4)
        for exp in experiments:
            self.assertEqual(exp["serving"], {"args": {"dtype": "auto"}, "env": {"OMP_NUM_THREADS": "2"}})
            self.assertNotIn("serving", exp["client"])
        experiments[0]["serving"]["env"]["OMP_NUM_THREADS"] = "changed"
        self.assertEqual(experiments[1]["serving"]["env"]["OMP_NUM_THREADS"], "2")
        self.assertEqual(config["serving"]["env"]["OMP_NUM_THREADS"], "2")

    def test_benchmarks_reject_profile_opt_out(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            benchmarks.build_parser().parse_args(["--no-serving-profile"])
        self.assertEqual(raised.exception.code, 2)

    def test_benchmarks_plan_all_models_before_writes_or_subprocesses(self):
        experiments = [{"name": model, "model": model, "tp": 1, "total_num_reqs": 2, "lanes": [{}],
                        "tuning": {"frequency_mhz": 1300, "max_num_seqs": 5, "max_num_batched_tokens": 8192}}
                       for model in ("Qwen/Qwen3-8B", "Other/model")]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"VLLM_SERVING_PROFILES_ENABLED": "0"}, clear=True), patch.object(
            benchmarks, "load_experiments", return_value=experiments,
        ), patch.object(benchmarks, "RUNS_ROOT", Path(directory) / "runs"), patch.object(
            benchmarks, "runtime_package_context", return_value={"versions": VERSIONS, "origins": ORIGINS},
        ), patch.object(benchmarks.subprocess, "run", side_effect=AssertionError("early launch")):
            with self.assertRaisesRegex(ValueError, "Unknown serving model"):
                benchmarks.main([])
            self.assertFalse((Path(directory) / "runs").exists())


if __name__ == "__main__":
    unittest.main()
