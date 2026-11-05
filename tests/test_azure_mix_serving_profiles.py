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

"""Direct-launch profile checks, written first and left unrun for static-only work."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import azure_mix as azure
from src import serving_profiles


RUNTIME_ANCHOR = {
    "versions": {"vllm": "0.23.0", "vllm_ascend": "0.23.0"},
    "origins": {name: {"origin": "/runtime/" + name + "/__init__.py", "root": "/runtime/" + name}
                for name in ("vllm", "vllm_ascend")},
}


class AzureMixServingChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": "0,2,3"}, clear=True))
        self.discovery = self.enterContext(patch.object(azure, "runtime_package_context", return_value=copy.deepcopy(RUNTIME_ANCHOR)))
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.directory)
        self.schedule = self.root / "schedule.csv"
        self.schedule.write_text("arrival_sec,type,dataset_idx,prompt,pool\n0,0,1,hello,conversation\n", encoding="utf-8")
        self.csv = self.root / "output" / "tokens.csv"
        self.log = self.root / "logs" / "vllm.log"
        self.arguments = ["--schedule", str(self.schedule), "--result-csv", str(self.csv), "--vllm-log", str(self.log)]
        self.preflight_env = None
        self.server_env = None
        self.server_command = None
        self.client_command = None
        self.client_code = 0
        self.client_payload = "index,input_length,output_length\n1,7,11\n"
        self.stopped = False

    def runtime_process(self, command, **kwargs):
        # Keep the real bounded metadata preflight; replace only its discovery process.
        if command[:3] == [sys.executable, "-c", serving_profiles._RUNTIME_PACKAGE_PROBE]:
            if self.client_command is None:
                self.assertFalse(self.csv.parent.exists())
                self.assertFalse(self.log.parent.exists())
            self.assertEqual(kwargs["cwd"], str(azure.REPO_ROOT))
            self.assertEqual(kwargs["timeout"], 10)
            self.preflight_env = dict(kwargs["env"])
            return SimpleNamespace(stdout=json.dumps(self.discovery.return_value), returncode=0)
        self.client_command = command
        self.assertNotIn("env", kwargs)
        self.assertNotIn("--gpu-memory-utilization", command)
        if self.client_payload is not None:
            sibling = self.csv.with_suffix(".client.tmp")
            sibling.write_text(self.client_payload, encoding="utf-8")
            os.replace(sibling, self.csv)
        return SimpleNamespace(returncode=self.client_code)

    def server(self, command, **kwargs):
        self.server_command = command
        self.server_env = dict(kwargs["env"])
        self.assertEqual(self.server_env, self.preflight_env)
        self.assertEqual(kwargs["cwd"], azure.REPO_ROOT)
        self.assertTrue(kwargs["start_new_session"])
        return SimpleNamespace(pid=4321, poll=lambda: None, wait=lambda **options: None)

    def launch(self, arguments=None, readiness_error=None):
        def stopped(pid, sig):
            self.assertEqual(pid, 4321)
            self.stopped = True

        with patch.object(azure.subprocess, "run", side_effect=self.runtime_process), \
                patch.object(azure.subprocess, "Popen", side_effect=self.server), \
                patch.object(azure, "_wait_for_vllm", side_effect=readiness_error), patch.object(azure.os, "killpg", side_effect=stopped):
            return azure.main(self.arguments + (arguments or []))

    def test_direct_launcher_applies_both_release_profiles_and_records_only_provenance(self):
        os.environ["VLLM_SERVING_PROFILES_ENABLED"] = "0"
        for release, allocator, extra in (("0.11.0", "max_split_size_mb:256", []),
                                          ("0.23.0", "expandable_segments:True", ["--gpu-memory-utilization", "0.9"])):
            with self.subTest(release=release):
                self.discovery.return_value["versions"] = {"vllm": release, "vllm_ascend": release}
                self.assertEqual(self.launch(), 0)
                self.assertEqual(self.discovery.call_count, 1 if release == "0.11.0" else 2)
                self.assertEqual(self.server_command, [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
                    "/models/qwen3-8b", "--port", "8010", "--tensor-parallel-size", "3", *extra])
                self.assertEqual(self.server_env["PYTORCH_NPU_ALLOC_CONF"], allocator)
                sidecar = self.csv.with_suffix(".serving.json")
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                self.assertEqual(metadata["serving_profile"]["profile_id"], "qwen3-8b-vllm-" + release)
                self.assertEqual(metadata["serving_profile"]["status"], "applied")
                self.assertTrue(metadata["serving_profile"]["runtime_verified"])
                self.assertEqual(metadata["serving_profile"]["verified_package_origins"], RUNTIME_ANCHOR["origins"])
                self.assertTrue(metadata["serving_condition"].startswith("sp-"))
                self.assertEqual(metadata["argv"], self.server_command)
                self.assertEqual(metadata["effective_env"]["PYTORCH_NPU_ALLOC_CONF"], allocator)
                self.assertEqual(metadata["client_returncode"], 0)
                self.assertEqual(metadata["result_csv_bytes"], len(self.csv.read_bytes()))
                self.assertEqual(metadata["result_csv_sha256"], hashlib.sha256(self.csv.read_bytes()).hexdigest())
                self.assertEqual(azure.verified_serving_sidecar(self.csv), metadata)
                self.assertNotIn("energy_joules", metadata)
                self.assertEqual(self.csv.read_text(encoding="utf-8"), "index,input_length,output_length\n1,7,11\n")
                self.assertIn(str(self.schedule), self.client_command)
                self.assertTrue(self.stopped)
                self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", os.environ)
                self.csv.unlink()
                sidecar.unlink()
                self.log.unlink()
                self.csv.parent.rmdir()
                self.log.parent.rmdir()

    def test_cli_overrides_merge_ambient_intent_without_mutating_global_env_or_leaking_credentials(self):
        os.environ.update(PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:64", HF_TOKEN="credential-not-to-record",
                          VLLM_SERVE_EXTRA_ARGS="--gpu-memory-utilization 0.7",
                          HTTPS_PROXY="http://alice:example-password@proxy.example:8080",
                          VLLM_PROXY="alice:example-password@proxy.example:8080")
        os.environ["VLLM_SERVING_OVERRIDES"] = json.dumps({"args": {"compilation-config": {"a": 1}, "gpu-memory-utilization": 0.8}})
        before = dict(os.environ)
        self.launch(["--serving-overrides", json.dumps({"args": {"gpu-memory-utilization": 0.85,
                     "compilation-config": {"b": 2}}, "env": {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:128"}})])
        self.assertEqual(self.server_env["PYTORCH_NPU_ALLOC_CONF"], "max_split_size_mb:128")
        self.assertEqual(self.server_command[self.server_command.index("--gpu-memory-utilization") + 1], "0.85")
        self.assertEqual(json.loads(self.server_command[self.server_command.index("--compilation-config") + 1]), {"a": 1, "b": 2})
        record = self.csv.with_suffix(".serving.json").read_text(encoding="utf-8")
        self.assertNotIn("credential-not-to-record", record)
        self.assertNotIn("HF_TOKEN", record)
        self.assertNotIn("example-password", record)
        self.assertNotIn("VLLM_PROXY", record)
        self.assertEqual(self.server_env["HTTPS_PROXY"], before["HTTPS_PROXY"])
        self.assertEqual(self.server_env["VLLM_PROXY"], before["VLLM_PROXY"])
        self.assertEqual(dict(os.environ), before)

    def test_readiness_timeout_preserves_previous_csv_and_bound_sidecar(self):
        self.launch()
        sidecar = self.csv.with_suffix(".serving.json")
        old = (self.csv.read_bytes(), sidecar.read_bytes())
        identities = (azure.output_identity(self.csv), azure.output_identity(sidecar))
        self.stopped = False
        with self.assertRaises(TimeoutError):
            self.launch(["--serving-overrides", '{"args":{"dtype":"float16"}}'],
                        readiness_error=TimeoutError("not ready"))
        self.assertEqual((self.csv.read_bytes(), sidecar.read_bytes()), old)
        self.assertEqual((azure.output_identity(self.csv), azure.output_identity(sidecar)), identities)
        self.assertTrue(self.stopped)

    def test_no_current_csv_preserves_previous_pair_for_zero_and_nonzero_exit(self):
        self.launch()
        sidecar = self.csv.with_suffix(".serving.json")
        old = (self.csv.read_bytes(), sidecar.read_bytes())
        identities = (azure.output_identity(self.csv), azure.output_identity(sidecar))
        self.client_payload = None
        for code in (0, 7):
            with self.subTest(code=code):
                self.client_code = code
                self.stopped = False
                with self.assertRaises(RuntimeError if code == 0 else subprocess.CalledProcessError):
                    self.launch(["--serving-overrides", '{"args":{"dtype":"float16"}}'])
                self.assertEqual((self.csv.read_bytes(), sidecar.read_bytes()), old)
                self.assertEqual((azure.output_identity(self.csv), azure.output_identity(sidecar)), identities)
                self.assertTrue(self.stopped)

    def test_fresh_failed_request_csv_gets_bound_provenance_before_failure_propagates(self):
        self.launch()
        old_identity = azure.output_identity(self.csv)
        self.client_payload = "index,input_length,output_length\n1,7,\n"
        self.client_code = 7
        self.stopped = False
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            self.launch(["--serving-overrides", '{"args":{"dtype":"float16"}}'])
        self.assertEqual(caught.exception.returncode, 7)
        self.assertNotEqual(azure.output_identity(self.csv), old_identity)
        self.assertEqual(self.csv.read_text(encoding="utf-8"), self.client_payload)
        metadata = azure.verified_serving_sidecar(self.csv)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["client_returncode"], 7)
        self.assertEqual(metadata["result_csv_sha256"], hashlib.sha256(self.csv.read_bytes()).hexdigest())
        self.assertEqual(metadata["result_csv_bytes"], len(self.csv.read_bytes()))
        self.assertIn("float16", metadata["argv"])
        self.assertTrue(self.stopped)
        self.assertEqual(list(self.csv.parent.glob("*.tmp")), [])

    def test_identical_bytes_atomic_csv_replacement_still_gets_current_binding(self):
        self.launch()
        identity = azure.output_identity(self.csv)
        self.assertEqual(self.launch(["--serving-overrides", '{"args":{"dtype":"float16"}}']), 0)
        self.assertNotEqual(azure.output_identity(self.csv), identity)
        self.assertIn("float16", azure.verified_serving_sidecar(self.csv)["argv"])

    def test_missing_legacy_malformed_and_mismatched_sidecars_are_unverified(self):
        self.assertIsNone(azure.verified_serving_sidecar(self.csv))
        self.launch()
        sidecar = self.csv.with_suffix(".serving.json")
        bound = json.loads(sidecar.read_text(encoding="utf-8"))
        for record in ({"serving_profile": bound["serving_profile"], "serving_condition": bound["serving_condition"]},
                       [], {**bound, "result_csv_sha256": "0" * 64},
                       {**bound, "result_csv_bytes": bound["result_csv_bytes"] + 1},
                       {**bound, "client_returncode": True}, {**bound, "serving_profile": {}},
                       {**bound, "serving_profile": {**bound["serving_profile"], "runtime_verified": False}}):
            with self.subTest(record=record):
                sidecar.write_text(json.dumps(record), encoding="utf-8")
                self.assertIsNone(azure.verified_serving_sidecar(self.csv))
        sidecar.write_text(json.dumps(bound), encoding="utf-8")
        self.csv.write_text("index,input_length,output_length\n1,7,12\n", encoding="utf-8")
        self.assertIsNone(azure.verified_serving_sidecar(self.csv))

    def test_missing_model_profile_or_protected_overrides_fail_before_directories_or_processes(self):
        for arguments in (["--model", "demo"], ["--serving-overrides", '{"args":{"port":9999}}'],
                          ["--serving-overrides", '{"env":{"ASCEND_RT_VISIBLE_DEVICES":"7"}}'],
                          ["--serving-overrides", "bad-json"], []):
            with self.subTest(arguments=arguments):
                self.discovery.return_value["versions"]["vllm"] = "0.99.0" if not arguments else "0.23.0"
                with patch.object(azure.subprocess, "run", side_effect=AssertionError("unexpected process")), \
                        patch.object(azure.subprocess, "Popen", side_effect=AssertionError("unexpected server")), \
                        self.assertRaises(ValueError):
                    azure.main(self.arguments + arguments)
                self.assertFalse(self.csv.parent.exists())
                self.assertFalse(self.log.parent.exists())

    def test_origin_mismatch_rejects_before_filesystem_or_server(self):
        discovered = copy.deepcopy(RUNTIME_ANCHOR)
        discovered["origins"]["vllm"] = {"origin": "/other/vllm/__init__.py", "root": "/other/vllm"}
        with patch.object(azure.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(discovered))), \
                patch.object(azure.subprocess, "Popen", side_effect=AssertionError("unexpected server")), \
                self.assertRaisesRegex(ValueError, "origins/versions mismatch"):
            azure.main(self.arguments)
        self.assertFalse(self.csv.parent.exists())
        self.assertFalse(self.log.parent.exists())


if __name__ == "__main__":
    unittest.main()
