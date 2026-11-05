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
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run
from src import serving_profiles as profiles
from tests.test_serving_profiles import ORIGINS


VERSIONS = {"vllm": "0.11.0", "vllm_ascend": "0.11.0"}


class RunServingProfileTests(unittest.TestCase):
    def test_runtime_mismatch_fails_before_cleanup_frequency_or_state_writes(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            presets = root / "presets.json"
            presets.write_text(json.dumps([{"preset_id": "local", "backend_url": "http://localhost:8200",
                                            "max_num_batched_tokens": 8192}]), encoding="utf-8")
            stack.enter_context(patch.dict(os.environ, {"MODEL": "Qwen/Qwen3-8B", "PRESETS_JSON": str(presets),
                "DEMO_STATE_DIR": str(root / "state"), "VLLM_ROOT": directory}, clear=True))
            stack.enter_context(patch.object(profiles.importlib.util, "find_spec", side_effect=lambda name: SimpleNamespace(
                origin=ORIGINS[name]["origin"], submodule_search_locations=[ORIGINS[name]["root"]])))
            stack.enter_context(patch.object(profiles.distribution_metadata, "version", return_value="0.11.0"))
            mismatched = {"versions": {"vllm": "0.23.0", "vllm_ascend": "0.11.0"}, "origins": ORIGINS}
            stack.enter_context(patch.object(run.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(mismatched))))
            for name in ("prepare_services", "set_freq_from_presets", "start_vllm_lanes", "teardown_services"):
                stack.enter_context(patch.object(run, name, side_effect=AssertionError("lifecycle before verification")))
            with self.assertRaisesRegex(ValueError, "runtime package"):
                run.main()
            self.assertFalse((root / "state").exists())

    def test_plan_is_mandatory_without_precision_or_tp_prerequisites(self):
        for model, tp, environment in (
            ("/models/qwen3-8b", 1, {}),
            ("Qwen/Qwen3-4B", 3, {"VLLM_SERVING_PROFILES_ENABLED": "0"}),
            ("Qwen/Qwen3-14B", 2, {"VLLM_SERVING_PROFILE_QUALIFIERS": '{"precision":"int8"}'}),
        ):
            environment = {**environment, "TASK_QUEUE_ENABLE": "0"}
            with self.subTest(model=model, tp=tp):
                args, updates, metadata, condition = run.plan_serving_profile(
                    model, tp, ["--dtype=auto"], environment, versions=VERSIONS,
                )
                self.assertEqual(args, ["--dtype=auto"])
                self.assertEqual(updates, {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256"})
                self.assertEqual(metadata["status"], "applied")
                self.assertEqual(metadata["relevant_env"]["TASK_QUEUE_ENABLE"], "0")
                self.assertEqual(metadata["qualifiers"], {
                    **json.loads(environment.get("VLLM_SERVING_PROFILE_QUALIFIERS", "{}")), "tp": tp,
                })
                self.assertEqual(metadata["versions_source"], "launcher-interpreter-distribution-metadata")
                self.assertEqual(metadata["launcher_interpreter"], sys.executable)
                self.assertEqual(metadata["scope"], "local-launch-plan")
                self.assertRegex(condition, r"^sp-[0-9a-f]{24}$")

    def test_missing_model_or_release_fails_before_any_lifecycle_or_probe(self):
        for model, versions in (("Other/model", VERSIONS), ("Qwen/Qwen3-8B", {}),
                                ("Qwen/Qwen3-8B", {**VERSIONS, "vllm": "0.11.1"})):
            with self.subTest(model=model, versions=versions), tempfile.TemporaryDirectory() as directory:
                state = Path(directory) / "state"
                with patch.dict(os.environ, {
                    "MODEL": model, "DEMO_STATE_DIR": str(state), "VLLM_SERVING_PROFILES_ENABLED": "0",
                }, clear=True), patch.object(
                    run, "runtime_package_context", return_value={"versions": versions, "origins": ORIGINS},
                ), patch.object(run.subprocess, "run", side_effect=AssertionError("probe before model validation")), patch.object(
                    run, "prepare_services", side_effect=AssertionError("cleanup before model validation"),
                ), self.assertRaises(ValueError):
                    run.main()
                self.assertFalse(state.exists())

    def test_transported_json_boolean_flag_and_env_override_precedence(self):
        environment = profiles.apply_serving_config({
            "VLLM_SERVING_OVERRIDES": '{"args":{"additional-config":{"ambient":1},"enforce-eager":true}}',
            "PYTORCH_NPU_ALLOC_CONF": "ambient", "OMP_NUM_THREADS": "8",
        }, {"args": {"gpu-memory-utilization": 0.85, "enforce-eager": False,
                     "additional-config": {"experiment": 2}},
            "env": {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:128", "OMP_NUM_THREADS": "2"}})
        args, updates, metadata, _ = run.plan_serving_profile(
            "Qwen/Qwen3-8B", 1, ["--gpu-memory-utilization", "0.7", "--enforce-eager",
                                    "--additional-config", '{"cli":3}'], environment,
            versions={**VERSIONS, "vllm": "0.23.0"}, origins=ORIGINS,
        )
        self.assertEqual(args[args.index("--gpu-memory-utilization") + 1], "0.85")
        self.assertIn("--no-enforce-eager", args)
        self.assertNotIn("--enforce-eager", args)
        self.assertEqual(json.loads(args[args.index("--additional-config") + 1]),
                         {"ambient": 1, "cli": 3, "experiment": 2})
        self.assertEqual(updates, {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:128", "OMP_NUM_THREADS": "2"})
        self.assertEqual(metadata["package_origins"], ORIGINS)
        self.assertEqual(environment["PYTORCH_NPU_ALLOC_CONF"], "ambient")

    def test_plan_uses_explicit_qualifiers_and_actual_integer_tp(self):
        environment = {"VLLM_SERVING_PROFILE_QUALIFIERS": '{"precision":"bf16","hardware":"Ascend-910B"}'}
        _, _, metadata, condition = run.plan_serving_profile(
            "/models/qwen3-8b", 2, [], environment, versions=VERSIONS,
        )
        self.assertEqual(metadata["qualifiers"], {"precision": "bf16", "hardware": "Ascend-910B", "tp": 2})
        _, _, _, other_version = run.plan_serving_profile(
            "/models/qwen3-8b", 2, [], environment,
            versions={"vllm": "0.23.0", "vllm_ascend": "0.23.0"},
        )
        self.assertNotEqual(condition, other_version)
        self.assertEqual(environment["VLLM_SERVING_PROFILE_QUALIFIERS"], '{"precision":"bf16","hardware":"Ascend-910B"}')

    def test_invalid_profile_inputs_and_reserved_metadata_fail_before_lifecycle(self):
        cases = [
            ({"VLLM_SERVE_EXTRA_ARGS": "--max_num_seqs=8"}, "Protected"),
            ({"VLLM_SERVE_EXTRA_ARGS": "-tp4", "VLLM_SERVING_PROFILES_ENABLED": "0"}, "Protected"),
            ({"VLLM_SERVING_OVERRIDES": '{"args":{"max-num-batched-tokens":4096}}'}, "max_num_batched_tokens"),
            ({"VLLM_SERVING_OVERRIDES": '{"args":{"model":"Other/model"}}'}, "experiment.model"),
            ({"VLLM_SERVING_OVERRIDES": '{"env":{"ASCEND_RT_VISIBLE_DEVICES":"3"}}'}, "authoritative"),
            ({"VLLM_SERVING_PROFILE_QUALIFIERS": "{"}, "JSON object"),
            ({"VLLM_SERVING_PROFILE_QUALIFIERS": "[]"}, "JSON object"),
            ({"VLLM_SERVING_PROFILE_QUALIFIERS": '{"tp":4}'}, "TP"),
            ({"VLLM_SERVING_PROFILE_QUALIFIERS": '{"tp":2}'}, "TP"),
            ({"VLLM_EXPECTED_SERVING_CONDITION": "parent-condition"}, "VLLM_EXPECTED_SERVING_CONDITION"),
        ]
        for key in ("serving_profile", "serving_condition"):
            cases.append(({"EXPERIMENT_METADATA": json.dumps({key: "spoofed"})}, "reserved"))
        for extra_env, message in cases:
            with self.subTest(environment=extra_env), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                state = Path(directory) / "state"
                stack.enter_context(patch.dict(os.environ, {
                    "MODEL": "Qwen/Qwen3-8B", "DEMO_STATE_DIR": str(state), **extra_env,
                }, clear=True))
                stack.enter_context(patch.object(run, "runtime_package_context", return_value={"versions": VERSIONS, "origins": ORIGINS}))
                for name in ("setup_process_env", "prepare_services", "set_freq_from_presets", "start_vllm_lanes", "teardown_services"):
                    stack.enter_context(patch.object(run, name, side_effect=AssertionError("lifecycle reached before preflight")))
                with self.assertRaisesRegex(ValueError, message):
                    run.main()
                self.assertFalse(state.exists())

    def test_launch_requires_active_plan_before_state_writes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "DEMO_STATE_DIR": str(Path(directory) / "state"),
        }, clear=True):
            cfg = run.load_config()
            cfg.serving_launch_env = {}
            rows = [run.LanePlanRow("local", "8200", "0,1", 8192, 50, "1300", True)]
            with self.assertRaisesRegex(ValueError, "serving profile"):
                run.start_vllm_lanes(cfg, rows)
            self.assertFalse(cfg.demo_state_dir.exists())

    def test_launcher_uses_stored_plan_and_lane_only_environment(self):
        versions = {"vllm": "0.23.0", "vllm_ascend": "0.23.0"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "DEMO_STATE_DIR": directory, "VLLM_ROOT": directory,
            "VLLM_SERVE_EXTRA_ARGS": "--dtype=auto", "TASK_QUEUE_ENABLE": "0",
            "PYTHONPATH": "/registered",
        }, clear=True):
            cfg = run.load_config()
            cfg.serving_extra_args, cfg.serving_env_updates, cfg.serving_profile, cfg.serving_condition = run.plan_serving_profile(
                cfg.model, cfg.tp_size, cfg.vllm_serve_extra_args, os.environ, versions=versions, origins=ORIGINS,
            )
            rows = [
                run.LanePlanRow("local", "8200", "2,3", 8192, 50, "1300", True),
                run.LanePlanRow("second", "8300", "4,5", 4096, 25, "900", True),
            ]
            launched = []

            anchor = {"versions": versions, "origins": ORIGINS}
            with patch.object(run.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(anchor))):
                run.preflight_serving_runtime(cfg, rows, anchor)
            projected = dict(cfg.serving_launch_env)
            # Later parent mutations must not change the environment already verified.
            os.environ["PYTHONPATH"] = "/late-shadow"
            os.environ["TASK_QUEUE_ENABLE"] = "late-change"

            def popen(command, **kwargs):
                launched.append((command, kwargs))
                return SimpleNamespace(pid=123)

            with patch.object(run.subprocess, "Popen", side_effect=popen):
                run.start_vllm_lanes(cfg, rows)
            self.assertEqual(len(launched), 2)
            command, kwargs = launched[0]
            self.assertEqual(command, [
                sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", cfg.model,
                "--port", "8200", "--tensor-parallel-size", "2", "--enable-chunked-prefill",
                "--scheduling-policy", "priority", "--max-num-batched-tokens", "8192",
                "--max-num-seqs", "50", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--dtype=auto",
            ])
            self.assertEqual(kwargs["cwd"], directory)
            self.assertEqual(kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "2,3")
            self.assertEqual(kwargs["env"]["TASK_QUEUE_ENABLE"], "0")
            self.assertEqual(kwargs["env"]["PYTORCH_NPU_ALLOC_CONF"], "expandable_segments:True")
            self.assertEqual(kwargs["env"], {**projected, "ASCEND_RT_VISIBLE_DEVICES": "2,3"})
            self.assertEqual(kwargs["env"]["PYTHONPATH"], str(cfg.online_serving) + ":/registered")
            self.assertEqual(cfg.serving_profile["verified_package_origins"], ORIGINS)
            self.assertEqual(os.environ["TASK_QUEUE_ENABLE"], "late-change")
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", os.environ)
            local, second = cfg.serving_profile["lanes"]
            self.assertEqual(local["command"], command)
            self.assertEqual(local["cwd"], directory)
            self.assertEqual(local["env"], {"ASCEND_RT_VISIBLE_DEVICES": "2,3", "TASK_QUEUE_ENABLE": "0",
                                           "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"})
            self.assertTrue(local["profile_applied"])
            self.assertEqual(second["status"], "launched")
            self.assertTrue(second["profile_applied"])
            self.assertEqual(second["command"], launched[1][0])
            self.assertEqual(second["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4,5")
            self.assertTrue(second["runtime_verified"])
            cfg.result_json = Path(directory) / "result.json"
            cfg.result_json.write_text('{"num_requests":2}', encoding="utf-8")
            run.merge_experiment_metadata(cfg, result_fresh=True)
            payload = json.loads(cfg.result_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["serving_profile"]["lanes"][0]["command"], command)
            self.assertEqual(payload["serving_condition"], cfg.serving_condition)
            self.assertEqual(payload["num_requests"], 2)

    def test_main_persists_authoritative_provenance_after_experiment_merge(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            presets = root / "presets.json"
            presets.write_text(json.dumps([{
                "preset_id": "lane_d", "backend_url": "http://127.0.0.1:8200",
                "max_num_batched_tokens": 8192, "max_num_seqs": 50,
            }]), encoding="utf-8")
            result = root / "result.json"
            stack.enter_context(patch.dict(os.environ, {
                "MODEL": "Qwen/Qwen3-8B", "TP_SIZE": "2", "ASCEND_DEVICES": "0,1",
                "PRESETS_JSON": str(presets), "RESULT_JSON": str(result),
                "DEMO_STATE_DIR": str(root / "state"), "VLLM_ROOT": directory,
                "WAIT_SECS": "0", "SKIP_ROUTER": "1",
                "POWER_MONITOR_ENABLED": "0", "KV_MONITOR_ENABLED": "0",
                "EXPERIMENT_METADATA": '{"request_shape":"decode-heavy"}',
            }, clear=True))
            stack.enter_context(patch.object(run.kill_module, "kill_all"))
            stack.enter_context(patch.object(run, "runtime_package_context", return_value={"versions": VERSIONS, "origins": ORIGINS}))

            def client(command, **kwargs):
                if command[:2] == [sys.executable, "-c"]:
                    return SimpleNamespace(stdout=json.dumps({"versions": VERSIONS, "origins": ORIGINS}))
                self.assertEqual(command[:3], [sys.executable, "-m", "src.client"])
                result.write_text(json.dumps({
                    "serving_profile": {"spoofed": True}, "serving_condition": "spoofed",
                    "energy_joules": 123, "power_samples_series": [{"timestamp": 1}],
                    "op_conflict": {"recorded": True},
                }), encoding="utf-8")
                return SimpleNamespace(returncode=7)

            stack.enter_context(patch.object(run.subprocess, "run", side_effect=client))
            stack.enter_context(patch.object(run.subprocess, "Popen", return_value=SimpleNamespace(pid=123)))
            stderr = stack.enter_context(redirect_stderr(io.StringIO()))
            self.assertEqual(run.main(), 7)
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["request_shape"], "decode-heavy")
            self.assertEqual(payload["energy_joules"], 123)
            self.assertEqual(payload["power_samples_series"], [{"timestamp": 1}])
            self.assertEqual(payload["op_conflict"], {"recorded": True})
            self.assertNotIn("spoofed", payload["serving_profile"])
            self.assertRegex(payload["serving_condition"], r"^sp-[0-9a-f]{24}$")
            metadata = payload["serving_profile"]
            self.assertEqual(metadata["versions_source"], "launcher-interpreter-distribution-metadata")
            self.assertEqual(metadata["qualifiers"], {"tp": 2})
            self.assertEqual(metadata["lanes"][0]["status"], "launched")
            self.assertTrue(metadata["lanes"][0]["profile_applied"])
            self.assertTrue(metadata["runtime_verified"])
            self.assertEqual(metadata["verified_package_origins"], ORIGINS)
            self.assertIn("status=applied", stderr.getvalue())
            self.assertIn("reason=", stderr.getvalue())


class RunResultFreshnessTests(unittest.TestCase):
    """Written before the bounded fixes; deliberately unrun in static-only work."""

    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.result = self.root / "result.json"
        self.presets = self.root / "presets.json"
        self.presets.write_text(json.dumps([{
            "preset_id": "local", "backend_url": "http://localhost:8200",
            "max_num_batched_tokens": 8192, "capabilities": {"kv_projection": True},
        }]), encoding="utf-8")
        self.enterContext(patch.dict(os.environ, {
            "MODEL": "Qwen/Qwen3-8B", "PRESETS_JSON": str(self.presets),
            "RESULT_JSON": str(self.result), "DEMO_STATE_DIR": str(self.root / "state"),
            "VLLM_ROOT": directory, "WAIT_SECS": "0", "SKIP_ROUTER": "1",
            "ROUTING_KWARGS": '{"specialization_gap":{}}',
            "EXPERIMENT_METADATA": '{"experiment":"current"}',
        }, clear=True))
        self.anchor = {"versions": VERSIONS, "origins": ORIGINS}
        self.enterContext(patch.object(run, "runtime_package_context", return_value=self.anchor))
        self.client_payload = None
        self.client_code = 7
        self.stopped = []
        self.summaries = []
        self.cleanup = []
        self.enterContext(patch.object(run.kill_module, "kill_all", side_effect=lambda *a, **kw: self.cleanup.append("cleanup")))
        self.enterContext(patch.object(run.subprocess, "Popen", side_effect=self.background))
        self.enterContext(patch.object(run.subprocess, "run", side_effect=self.subprocess))
        self.stderr = self.enterContext(redirect_stderr(io.StringIO()))

    def background(self, command, **kwargs):
        name = "server" if "serve" in command else Path(command[1]).stem
        if "monitor" in command:
            output = Path(command[command.index("--output") + 1])
            output.write_text('{"timestamp":1}\n', encoding="utf-8")
        return SimpleNamespace(pid=123, poll=lambda: None,
                               terminate=lambda: self.stopped.append(name), wait=lambda **kw: None)

    def subprocess(self, command, **kwargs):
        if command[:2] == [sys.executable, "-c"]:
            return SimpleNamespace(stdout=json.dumps(self.anchor), returncode=0)
        if "snapshot" in command:
            Path(command[command.index("--output") + 1]).write_text('{"timestamp":1}', encoding="utf-8")
        elif "summarize" in command:
            self.summaries.append(Path(command[1]).stem)
            kwargs["stdout"].write('{"total_joules":23,"samples":4,"mean_utilization":0.5}')
        else:
            self.assertEqual(command[:3], [sys.executable, "-m", "src.client"])
            if self.client_payload is not None:
                sibling = self.result.with_suffix(".client.tmp")
                sibling.write_text(self.client_payload, encoding="utf-8")
                os.replace(sibling, self.result)
            return SimpleNamespace(returncode=self.client_code)
        return SimpleNamespace(returncode=0)

    def test_unchanged_json_is_not_timestamped_or_merged_and_monitors_stop(self):
        old = '{"experiment":"old","energy_joules":9,"serving_condition":"old"}'
        self.result.write_text(old, encoding="utf-8")
        identity = run.output_identity(self.result)
        self.assertEqual(run.main(), 7)
        self.assertEqual(self.result.read_text(encoding="utf-8"), old)
        self.assertEqual(run.output_identity(self.result), identity)
        self.assertEqual(self.summaries, [])
        self.assertEqual(set(self.stopped), {"energy_meter", "kv_meter"})
        self.assertEqual(len(self.cleanup), 2)
        self.assertIn("no current result", self.stderr.getvalue().lower())

    def test_zero_exit_without_fresh_json_is_an_orchestration_failure(self):
        for old in (None, '{"num_requests":99}'):
            with self.subTest(old=old):
                if old is not None:
                    self.result.write_text(old, encoding="utf-8")
                self.client_code = 0
                self.assertNotEqual(run.main(), 0)
                if old is None:
                    self.assertFalse(self.result.exists())
                else:
                    self.assertEqual(self.result.read_text(encoding="utf-8"), old)

    def test_atomic_replacement_is_fresh_even_with_identical_bytes_and_nonzero_exit(self):
        self.client_payload = '{"num_requests":1,"requests":[{"ok":false}],"total_output_tokens":0}'
        self.result.write_text(self.client_payload, encoding="utf-8")
        identity = run.output_identity(self.result)
        self.assertEqual(run.main(), 7)
        payload = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertNotEqual(run.output_identity(self.result), identity)
        self.assertEqual(payload["requests"], [{"ok": False}])
        self.assertIn("benchmark_start_s", payload)
        self.assertIn("benchmark_end_s", payload)
        self.assertEqual(payload["energy_joules"], 23)
        self.assertEqual(payload["kv_samples"], 4)
        self.assertEqual(payload["experiment"], "current")
        self.assertEqual(payload["serving_profile"]["lanes"][0]["status"], "launched")
        self.assertEqual(set(self.summaries), {"energy_meter", "kv_meter"})

    def test_malformed_fresh_json_stays_unmodified_and_cleanup_runs(self):
        for text in ("{", "[]", '"not an object"', '{"requests":"invalid"}'):
            with self.subTest(text=text):
                self.client_payload = text
                with self.assertRaises(ValueError):
                    run.main()
                self.assertEqual(self.result.read_text(encoding="utf-8"), text)
        self.assertEqual(self.summaries, [])
        self.assertEqual(len(self.cleanup), 8)

    def test_all_mergers_and_summarizers_fail_closed_without_freshness(self):
        cfg = run.load_config()
        cfg.demo_state_dir.mkdir()
        old = '{"num_requests":99}'
        self.result.write_text(old, encoding="utf-8")
        identity = run.output_identity(self.result)
        for path in (cfg.power_summary_path, cfg.kv_summary_path):
            path.write_text('{"total_joules":23,"samples":4}', encoding="utf-8")
        for path in (cfg.power_log_path, cfg.kv_log_path):
            path.write_text('{}\n', encoding="utf-8")
        run.merge_power_into_result(cfg)
        run.merge_kv_into_result(cfg)
        run.merge_experiment_metadata(cfg)
        run.summarize_power(cfg, None)
        run.summarize_kv(cfg, None)
        self.assertEqual(run.output_identity(self.result), identity)
        self.assertEqual(self.result.read_text(encoding="utf-8"), old)
        self.assertEqual(self.summaries, [])

    def test_client_output_path_aliases_and_abbreviations_are_rejected(self):
        cfg = run.load_config()
        for extra in (["--result-json", "/other"], ["--result-json=/other"],
                      ["--result_json=/other"], ["--result-j", "/other"], ["--res=/other"]):
            with self.subTest(extra=extra):
                cfg.client_extra_args = extra
                with self.assertRaisesRegex(ValueError, "RESULT_JSON"):
                    run.run_client(cfg, [])
                self.assertFalse(self.result.exists())

    def test_lane_skips_reject_before_preflight_state_or_lifecycle(self):
        for environment in ({"SKIP_VLLM": "1"}, {"SKIP_VLLM_LANES": "local"},
                            {"SKIP_VLLM_P": "1"}, {"SKIP_VLLM_D": "1"},
                            {"SKIP_VLLM_LANES": "unused"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment), patch.object(
                run, "preflight_serving_runtime", side_effect=AssertionError("preflight before ownership check"),
            ):
                with self.assertRaisesRegex(ValueError, "owned.*external.*unsupported"):
                    run.main()
                self.assertFalse((self.root / "state").exists())
                self.assertEqual(self.cleanup, [])
        cfg = run.load_config()
        rows = [run.LanePlanRow("external", "8200", "0,1", 8192, 50, "", False)]
        cfg.serving_profile = {"status": "applied"}
        cfg.serving_launch_env = {}
        for action in (lambda: run.preflight_serving_runtime(cfg, rows, self.anchor),
                       lambda: run.start_vllm_lanes(cfg, rows)):
            with self.assertRaisesRegex(ValueError, "owned.*external.*unsupported"):
                action()
            self.assertFalse(cfg.demo_state_dir.exists())


if __name__ == "__main__":
    unittest.main()
