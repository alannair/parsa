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

import copy
import json
import subprocess
import sys
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from src import serving_profiles as profiles


VERSIONS = {"vllm": "0.11.0", "vllm_ascend": "0.11.0"}
ORIGINS = {
    "vllm": {"origin": "/registered/vllm/__init__.py", "root": "/registered/vllm"},
    "vllm_ascend": {"origin": "/registered/vllm_ascend/__init__.py", "root": "/registered/vllm_ascend"},
}


def ready_profile():
    """Synthetic merge fixture; not a tuning recommendation."""
    return {
        "profile_id": "fixture-qwen3-8b-0.11.0",
        "revision": 2,
        "model": "Qwen/Qwen3-8B",
        "versions": dict(VERSIONS),
        "documentation_version": "0.11.0",
        "validated_version": None,
        "source_provenance": {
            "section": "fixture/serving",
            "official_url": "fixture://docs/v0.11.0/serving",
        },
        "status": "ready",
        "defaults": {
            "args": {"gpu-memory-utilization": 0.8, "enforce-eager": True},
            "env": {
                "PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256",
                "TASK_QUEUE_ENABLE": "1",
            },
        },
        "rationale": "Synthetic test fixture only.",
        "source_evidence": [{
            "url": "fixture://local", "commit": "fixture", "context": "test only",
            "section": "fixture/serving", "official_url": "fixture://docs/v0.11.0/serving",
            "documentation_version": "0.11.0", "validated_version": None,
        }],
        "guidance": [],
    }


def registry_text(*entries):
    return json.dumps({"schema_version": 1, "profiles": list(entries)})


class ServingRegistryTests(unittest.TestCase):
    def test_invalid_json_and_duplicate_json_keys_raise(self):
        for text in ("{", '{"schema_version":1,"schema_version":1,"profiles":[]}'):
            with self.subTest(text=text), patch.object(Path, "read_text", return_value=text), self.assertRaises(ValueError):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_invalid_registry_schema_propagates(self):
        for registry in (
            [], {}, {"schema_version": 2, "profiles": []},
            {"schema_version": True, "profiles": []},
            {"schema_version": 1, "profiles": {}},
            {"schema_version": 1, "profiles": [None]},
        ):
            with self.subTest(registry=registry), patch.object(Path, "read_text", return_value=json.dumps(registry)), self.assertRaises(ValueError):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_invalid_entry_identity_versions_and_revision_raise(self):
        for field, value in (
            ("profile_id", ""), ("profile_id", []), ("model", "Qwen/Qwen3-32B"),
            ("versions", None), ("versions", {"vllm": "0.11.0"}),
            ("versions", {"vllm": [], "vllm_ascend": "0.11.0"}),
            ("versions", {"vllm": "0.11.0rc1", "vllm_ascend": "0.11.0"}),
            ("revision", None), ("revision", True), ("revision", 0),
        ):
            entry = ready_profile()
            entry[field] = value
            with self.subTest(field=field, value=value), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, field):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_duplicate_ids_propagate_even_for_different_models(self):
        first, second = ready_profile(), ready_profile()
        second["model"] = "Qwen/Qwen3-4B"
        with patch.object(Path, "read_text", return_value=registry_text(first, second)), self.assertRaisesRegex(ValueError, "Duplicate profile_id"):
            profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_duplicate_model_versions_propagate_even_for_different_ids(self):
        first, second = ready_profile(), ready_profile()
        second["profile_id"] = "different-id"
        second["versions"] = {"vllm_ascend": "0.21.0", "vllm": "0.11.0"}
        with patch.object(Path, "read_text", return_value=registry_text(first, second)), self.assertRaisesRegex(ValueError, "Duplicate model/versions"):
            profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_malformed_status_propagates(self):
        for status in (None, True, [], {}, "unrecognized", "tuning-unavailable", "disabled"):
            entry = ready_profile()
            entry["status"] = status
            with self.subTest(status=status), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, "status"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_malformed_defaults_raise(self):
        for defaults in (None, [], {}, {"args": {}, "env": []}, {"args": [], "env": {}}):
            entry = ready_profile()
            entry["defaults"] = defaults
            with self.subTest(defaults=defaults), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, "defaults"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_invalid_default_args_raise(self):
        for args, message in (
            ({"port": 8000}, "Protected"),
            ({"compilation-config": "{"}, "Malformed JSON"),
            ({"enforce-eager": True, "no-enforce-eager": True}, "Duplicate"),
            ({"gpu-memory-utilization": float("nan")}, "JSON"),
        ):
            entry = ready_profile()
            entry["defaults"] = {"args": args, "env": {}}
            with self.subTest(args=args), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, message):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_profile_env_remains_limited_to_reviewed_keys(self):
        for environment in ({"PATH": "/bad"}, {"HCCL_BUFFSIZE": "200"}, {"TASK_QUEUE_ENABLE": 1}, {"TASK_QUEUE_ENABLE": None}):
            entry = ready_profile()
            entry["defaults"] = {"args": {}, "env": environment}
            with self.subTest(environment=environment), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, "environment"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_invalid_unselected_entry_is_always_validated(self):
        selected, invalid = ready_profile(), ready_profile()
        invalid.update(profile_id="other-model", model="Qwen/Qwen3-4B", status="bad-status")
        for versions in (VERSIONS, {}, {"vllm": "0.11.1", "vllm_ascend": "0.11.1"}):
            with self.subTest(versions=versions), patch.object(Path, "read_text", return_value=registry_text(selected, invalid)), self.assertRaisesRegex(ValueError, "status"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=versions)

    def test_missing_or_unreadable_registry_is_actionable_value_error(self):
        for error in (FileNotFoundError("registry"), PermissionError("registry")):
            with self.subTest(error=error), patch.object(Path, "read_text", side_effect=error), self.assertRaisesRegex(ValueError, "registry.*serving_profiles.json"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_missing_model_release_profile_raises(self):
        for entries in ([], [ready_profile()]):
            with self.subTest(entries=entries), patch.object(Path, "read_text", return_value=registry_text(*entries)), self.assertRaisesRegex(ValueError, "profile.*Qwen/Qwen3-4B.*0.11.0"):
                profiles.resolve_serving_profile("Qwen/Qwen3-4B", [], {}, versions=VERSIONS)

    def test_registry_sources_record_documentation_validation_and_section(self):
        entries = json.loads((Path(__file__).resolve().parents[1] / "src/serving_profiles.json").read_text())["profiles"]
        expected = {
            "0.11.0": (None, {
                "section": "online-serving/single-npu",
                "official_url": "https://docs.vllm.ai/projects/ascend/en/v0.11.0/tutorials/single_npu.html",
            }),
            "0.23.0": ("0.21.0", {
                "section": "models/Qwen3-Dense",
                "official_url": "https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/Qwen3-Dense.html",
            }),
        }
        for entry in entries:
            version = entry["versions"]["vllm"]
            validated, provenance = expected[version]
            with self.subTest(profile=entry["profile_id"]):
                self.assertEqual(entry["revision"], 2)
                self.assertEqual(entry["documentation_version"], version)
                self.assertEqual(entry["validated_version"], validated)
                self.assertEqual(entry["source_provenance"], provenance)
                for evidence in entry["source_evidence"]:
                    self.assertEqual(evidence["section"], provenance["section"])
                    self.assertEqual(evidence["official_url"], provenance["official_url"])
                    self.assertEqual(evidence["documentation_version"], version)
                    self.assertEqual(evidence["validated_version"], validated)
                _, _, metadata = profiles.resolve_serving_profile(entry["model"], [], {}, versions=entry["versions"])
                self.assertEqual(metadata["profile_revision"], 2)
                self.assertEqual(metadata["source_provenance"], provenance)
                self.assertEqual(metadata["documentation_version"], version)
                self.assertEqual(metadata["validated_version"], validated)

    def test_invalid_provenance_fields_and_evidence_versions_propagate(self):
        for field, value, message in (
            ("documentation_version", None, "documentation_version"),
            ("validated_version", "unknown", "validated_version"),
            ("source_provenance", {}, "source provenance"),
            ("source_evidence", [], "source provenance"),
        ):
            entry = ready_profile()
            entry[field] = value
            with self.subTest(field=field), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, message):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)
        for field in ("documentation_version", "validated_version"):
            entry = ready_profile()
            del entry["source_evidence"][0][field]
            with self.subTest(missing=field), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, "Inconsistent source"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)
        entry = ready_profile()
        entry["source_evidence"][0]["official_url"] = "fixture://docs/v0.21.0/serving"
        with patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, "official_url.*documentation_version"):
            profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)


class ServingArgsTests(unittest.TestCase):
    def test_scalar_lists_and_unrelated_tokens(self):
        defaults = {
            "gpu_memory_utilization": 0.8,
            "served-model-name": ["default", "alias"],
            "hf-overrides": {"example": 1},
        }
        explicit = [
            "--gpu-memory-utilization", "0.85",
            "--dtype", "bfloat16",
            "--gpu_memory_utilization=0.9",
            "--served_model_name", "local", "other",
        ]
        self.assertEqual(
            profiles.merge_serving_args(defaults, explicit),
            [
                "--gpu-memory-utilization", "0.9",
                "--served-model-name", "local", "other",
                "--hf-overrides", '{"example":1}',
                "--dtype", "bfloat16",
            ],
        )
        self.assertEqual(explicit[0:2], ["--gpu-memory-utilization", "0.85"])

    def test_boolean_opposites_and_equals_deduplicate(self):
        self.assertEqual(
            profiles.merge_serving_args(
                {"enforce-eager": True, "enable-prefix-caching": False},
                ["--enforce_eager", "--no-enforce-eager", "--enable_prefix_caching=true"],
            ),
            ["--no-enforce-eager", "--enable-prefix-caching"],
        )
        self.assertEqual(
            profiles.merge_serving_args({"no-enforce-eager": True}, ["--enforce-eager"]),
            ["--enforce-eager"],
        )
        self.assertEqual(
            profiles.merge_serving_args({"enforce-eager": True}, ["--no-enforce_eager=false"]),
            ["--enforce-eager"],
        )
        self.assertEqual(
            profiles.merge_serving_args({}, ["--enforce-eager=false", "--no-enforce_eager", "--enforce-eager", "0", "--dtype=auto"], {"no-enforce-eager": False}),
            ["--enforce-eager", "--dtype=auto"],
        )

    def test_nested_json_merges_leaves_and_replaces_arrays_and_scalars(self):
        defaults = {
            "compilation-config": {
                "nested": {"keep": 1, "replace": 2},
                "sizes": [1, 2],
                "object": {"old": True},
            },
            "additional-config": '{"nested":{"keep":true,"replace":0}}',
        }
        result = profiles.merge_serving_args(defaults, [
            '--compilation_config={"nested":{"replace":9},"sizes":[4],"object":null}',
            "--additional-config", '{"nested":{"replace":3}}',
        ])
        self.assertEqual(json.loads(result[1]), {
            "nested": {"keep": 1, "replace": 9}, "sizes": [4], "object": None,
        })
        self.assertEqual(json.loads(result[3]), {"nested": {"keep": True, "replace": 3}})
        self.assertEqual(defaults["compilation-config"]["nested"]["replace"], 2)
        self.assertEqual(
            profiles.merge_serving_args({"compilation-config": {"old": 1}}, ["-O", "3"]),
            ["--compilation-config", "3"],
        )

    def test_json_last_explicit_occurrence_wins_and_other_json_is_not_merged(self):
        result = profiles.merge_serving_args({"additional-config": {"keep": 1}}, [
            "--additional-config", '{"discard":2}',
            '--additional_config={"winner":3}',
        ])
        self.assertEqual(json.loads(result[1]), {"keep": 1, "winner": 3})
        raw = ["--unknown", "value", "--dtype=auto"]
        self.assertEqual(profiles.merge_serving_args({}, raw), raw)
        self.assertEqual(
            profiles.merge_serving_args({"hf-overrides": {"keep": 1}}, ["--hf-overrides", '{"new":2}']),
            ["--hf-overrides", '{"new":2}'],
        )

    def test_attached_compilation_aliases_share_precedence_and_json_leaf_merge(self):
        for argv in (["-O3"], ["-O=3"], ["-O", "3"], ["--compilation-config=3"]):
            with self.subTest(argv=argv):
                self.assertEqual(profiles.merge_serving_args({}, argv, {"compilation-config": 0}),
                                 ["--compilation-config", "0"])
        self.assertEqual(profiles.merge_serving_args({}, ["-O3", "-O=2", "-O", "1"]),
                         ["--compilation-config", "1"])
        self.assertEqual(profiles.merge_serving_args(
            {"compilation-config": {"nested": {"profile": 1}}},
            ['-O{"nested":{"ambient":2}}'], {"compilation-config": {"nested": {"experiment": 3}}}),
            ["--compilation-config", '{"nested":{"ambient":2,"experiment":3,"profile":1}}'])
        self.assertEqual(profiles.merge_serving_args({}, ['-O{"sizes":[1,2]}']),
                         ["--compilation-config", '{"sizes":[1,2]}'])
        with self.assertRaisesRegex(ValueError, "credentials"):
            profiles.merge_serving_args({}, ['-O{"password":"do-not-record"}'])

    def test_override_keys_not_in_profile_are_deduplicated(self):
        explicit = ["--dtype", "auto", "--max-model-len=1000", "--dtype=bfloat16", "--unknown", "keep"]
        self.assertEqual(
            profiles.merge_serving_args({}, explicit, {"dtype": "float16", "max_model_len": 2048}),
            ["--dtype", "float16", "--max-model-len", "2048", "--unknown", "keep"],
        )
        self.assertEqual(explicit[0:2], ["--dtype", "auto"])

    def test_list_to_scalar_override_consumes_all_ambient_values(self):
        for defaults in ({"served-model-name": ["default"]}, {}):
            with self.subTest(defaults=defaults):
                self.assertEqual(
                    profiles.merge_serving_args(
                        defaults,
                        ["--served-model-name", "old", "alias", "--dtype=auto"],
                        {"served-model-name": "experiment"},
                    ),
                    ["--served-model-name", "experiment", "--dtype=auto"],
                )

    def test_three_layers_merge_json_without_losing_profile_or_ambient_leaves(self):
        defaults = {"compilation-config": {"nested": {"profile": 1, "replace": 1}, "sizes": [1, 2]}}
        explicit = ['--compilation_config={"nested":{"ambient":2,"replace":2},"sizes":[3]}', "--dtype=auto"]
        overrides = {"compilation_config": {"nested": {"experiment": 3, "replace": 3}, "sizes": [4]}}
        originals = copy.deepcopy((defaults, explicit, overrides))
        args = profiles.merge_serving_args(defaults, explicit, overrides)
        self.assertEqual(args[0], "--compilation-config")
        self.assertEqual(json.loads(args[1]), {"nested": {"profile": 1, "ambient": 2, "experiment": 3, "replace": 3}, "sizes": [4]})
        self.assertEqual(args[2:], ["--dtype=auto"])
        self.assertEqual((defaults, explicit, overrides), originals)
        args = profiles.merge_serving_args({}, ["-O", '{"ambient":1}'], {"compilation-config": {"experiment": 2}})
        self.assertEqual(args, ["--compilation-config", '{"ambient":1,"experiment":2}'])

    def test_malformed_profiled_json_or_missing_values_raise(self):
        for name in ("compilation-config", "additional-config"):
            for value in ("{", "", "NaN", '{"x":Infinity}'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    profiles.merge_serving_args({name: {}}, ["--" + name, value])
            with self.assertRaises(ValueError):
                profiles.merge_serving_args({name: "{"}, [])
        with self.assertRaises(ValueError):
            profiles.merge_serving_args({"max-model-len": 4096}, ["--max-model-len", "--dtype=auto"])

    def test_every_protected_name_alias_and_abbreviation_rejected(self):
        protected = (
            "model", "port", "device", "tensor-parallel-size", "data-parallel-size",
            "pipeline-parallel-size", "decode-context-parallel-size",
            "prefill-context-parallel-size", "max-num-batched-tokens", "max-num-seqs",
            "scheduling-policy", "enable-chunked-prefill", "no-enable-chunked-prefill",
            "trust-remote-code", "no-trust-remote-code", "config",
        )
        for name in protected:
            for flag in ("--" + name, "--" + name.replace("-", "_") + "=1"):
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    profiles.merge_serving_args({}, [flag, "value"])
            with self.subTest(default=name), self.assertRaises(ValueError):
                profiles.merge_serving_args({name: True}, [])
        for flag in (
            "-tp", "-tp=2", "-tp2", "-dp", "-dp2", "-pp", "-pp=2", "-m",
            "--tensor_parallel_s=2", "--max-num-s", "--no-enable-chunk",
            "--no", "--no-", "--n",
            "--trust-remote", "--conf=/tmp/profile.yaml", "-c", "-c/tmp/profile.yaml",
            "--devices=0,1", "--data-parallel-rank=1", "--", "-O.tensor_parallel_size=4",
        ):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                profiles.merge_serving_args({}, [flag])
        self.assertEqual(
            profiles.merge_serving_args({}, ["--model-impl", "auto", "--config-format", "hf"]),
            ["--model-impl", "auto", "--config-format", "hf"],
        )

    def test_protected_options_explain_authoritative_owner(self):
        for name, value, owner in (
            ("model", "other", "model"), ("port", 8000, "backend_url"),
            ("tensor-parallel-size", 4, "tp"), ("devices", "0,1", "devices"),
            ("max-num-batched-tokens", 4096, "max_num_batched_tokens"),
            ("max-num-seqs", 8, "max_num_seqs"),
            ("scheduling-policy", "fcfs", "priority"),
            ("enable-chunked-prefill", False, "chunked"), ("config", "file", "experiment"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Protected.*" + owner):
                profiles.merge_serving_args({}, ["--" + name, str(value)])
            with self.subTest(override=name), self.assertRaisesRegex(ValueError, "Protected.*" + owner):
                profiles.normalize_serving_config({"args": {name: value}})


class ServingConfigTests(unittest.TestCase):
    def test_optional_config_canonicalizes_options_booleans_and_json(self):
        self.assertEqual(profiles.normalize_serving_config(None), {"args": {}, "env": {}})
        config = {"args": {"--gpu_memory_utilization": 0.8, "no-enforce_eager": "false", "enable-prefix-caching": "0", "-O": '{"nested":{"keep":1}}'}, "env": {"HCCL_BUFFSIZE": "0", "OMP_NUM_THREADS": ""}}
        self.assertEqual(profiles.normalize_serving_config(config), {
            "args": {"gpu-memory-utilization": 0.8, "enforce-eager": True, "enable-prefix-caching": False, "compilation-config": {"nested": {"keep": 1}}},
            "env": {"HCCL_BUFFSIZE": "0", "OMP_NUM_THREADS": ""},
        })
        normalized = profiles.normalize_serving_config(config)
        normalized["args"]["compilation-config"]["nested"]["keep"] = 9
        self.assertEqual(config["args"]["-O"], '{"nested":{"keep":1}}')

    def test_layering_merges_only_two_json_options_and_returns_independent_objects(self):
        base = {"args": {"additional_config": {"nested": {"base": 1, "replace": 1}, "sizes": [1]}, "hf-overrides": {"old": 1}, "served-model-name": ["old"]}, "env": {"HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "2"}}
        override = {"args": {"--additional-config": {"nested": {"workload": 2, "replace": 2}, "sizes": [2]}, "hf-overrides": {"new": 2}, "served-model-name": ["new"]}, "env": {"OMP_NUM_THREADS": "0"}}
        originals = copy.deepcopy((base, override))
        merged = profiles.merge_serving_configs(base, override)
        self.assertEqual(merged, {
            "args": {"additional-config": {"nested": {"base": 1, "workload": 2, "replace": 2}, "sizes": [2]}, "hf-overrides": {"new": 2}, "served-model-name": ["new"]},
            "env": {"HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "0"},
        })
        merged["args"]["additional-config"]["nested"]["base"] = 99
        self.assertEqual((base, override), originals)
        copied = profiles.merge_serving_configs(base, None)
        copied["args"]["served-model-name"].append("independent")
        self.assertEqual((base, override), originals)

    def test_malformed_config_and_nonfinite_or_duplicate_options_raise(self):
        for config in (False, [], "{}", {"extra": {}}, {"args": None}, {"env": []},
                       {"env": {"HCCL_BUFFSIZE": 0}}, {"env": {"BAD=KEY": "x"}},
                       {"env": {"OMP_NUM_THREADS": "bad\x00value"}},
                       {"args": {"dtype=auto": "float16"}}, {"args": {"dtype": None}},
                       {"args": {"served-model-name": "bad\x00value"}},
                       {"args": {"enforce-eager": "maybe"}},
                       {"args": {"gpu_memory_utilization": 0.8, "--gpu-memory-utilization": 0.9}},
                       {"args": {"enforce-eager": True, "no-enforce-eager": False}},
                       {"args": {"additional-config": '{"x":1,"x":2}'}},
                       {"args": {"additional-config": {"x": float("inf")}}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                profiles.normalize_serving_config(config)

    def test_configuration_env_cannot_replace_experiment_or_transport_controls(self):
        for name in ("MODEL", "TP_SIZE", "DP_SIZE", "PP_SIZE", "PRESETS_JSON", "RESULT_JSON",
                     "ASCEND_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "LANE_DEVICES_JSON",
                     "SKIP_VLLM", "SKIP_ROUTER", "KILL_AFTER_CLIENT", "DEMO_STATE_DIR", "WAIT_SECS",
                     "ROUTER_PORT", "CLIENT_EXTRA_ARGS", "EXPERIMENT_METADATA", "PYTHONPATH", "VLLM_ROOT",
                     "VLLM_SERVE_EXTRA_ARGS", "VLLM_SERVING_OVERRIDES", "VLLM_SERVING_PROFILES_ENABLED",
                     "VLLM_EXPECTED_SERVING_CONDITION", "VLLM_SERVING_PROFILE_QUALIFIERS"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Protected.*environment.*experiment"):
                profiles.normalize_serving_config({"env": {name: "0"}})

    def test_credentials_in_explicit_intent_raise_without_echoing_values(self):
        secret = "do-not-record-this-value"
        for config in ({"env": {"HF_TOKEN": secret}}, {"args": {"api-key": secret}},
                       {"args": {"additional-config": {"nested": {"password": secret}}}},
                       {"args": {"hf-overrides": '{"access_token":"' + secret + '"}'}}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "credentials.*inherit") as caught:
                profiles.normalize_serving_config(config)
            self.assertNotIn(secret, str(caught.exception))
        for explicit in (["--api-key", secret], ["--api_key=" + secret],
                         ["--additional-config", '{"authorization":"' + secret + '"}']):
            with self.subTest(explicit=explicit), self.assertRaisesRegex(ValueError, "credentials.*inherit") as caught:
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", explicit, {}, versions=VERSIONS)
            self.assertNotIn(secret, str(caught.exception))
        _, _, metadata = profiles.resolve_serving_profile(
            "Qwen/Qwen3-8B", [], {"VLLM_ENGINE_CONFIG": '{"api_key":"' + secret + '","tuning":1}'}, versions=VERSIONS)
        self.assertNotIn(secret, json.dumps(metadata))
        self.assertNotIn("VLLM_ENGINE_CONFIG", metadata["relevant_env"])

    def test_explicit_url_userinfo_is_rejected_before_transport_without_echoing(self):
        for value in ("http://alice:example-password@proxy.example:8080",
                      "alice:example-password@proxy.example:8080", "//alice:example-password@proxy.example:8080",
                      "https://alice@proxy.example", "http://alice:example=password@proxy.example"):
            for config in ({"env": {"HTTPS_PROXY": value}}, {"args": {"proxy": value}},
                           {"args": {"additional-config": {"endpoint": value}}}):
                with self.subTest(config=config), self.assertRaisesRegex(ValueError, "credentials.*inherit") as caught:
                    profiles.apply_serving_config({}, config)
                self.assertNotIn("alice", str(caught.exception))
                self.assertNotIn("example-password", str(caught.exception))
            for argv in (["--proxy", value], ["--proxy=" + value]):
                with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, "credentials.*inherit") as caught:
                    profiles.resolve_serving_profile("Qwen/Qwen3-8B", argv, {}, versions=VERSIONS)
                self.assertNotIn("alice", str(caught.exception))

    def test_inherited_credentials_remain_in_child_env_but_not_records_or_condition(self):
        environment = {"HTTPS_PROXY": "http://alice:example-password@proxy.example:8080",
                       "VLLM_PROXY": "alice:example-password@proxy.example:8080",
                       "VLLM_ENGINE_CONFIG": '{"endpoint":"http://alice:example-password@proxy.example"}',
                       "HF_TOKEN": "inherited-secret", "PYTORCH_NPU_ALLOC_CONF": "http://alice:example-password@proxy.example",
                       "OMP_NUM_THREADS": "2"}
        transported = profiles.apply_serving_config(environment, None)
        _, updates, metadata = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], transported, versions=VERSIONS)
        child = {**transported, **updates}
        for key, value in environment.items():
            self.assertEqual(child[key], value)
        recorded = json.dumps(metadata)
        for secret in ("alice", "example-password", "inherited-secret"):
            self.assertNotIn(secret, recorded)
        self.assertEqual(metadata["relevant_env"], {"OMP_NUM_THREADS": "2"})
        changed = {key: value.replace("example-password", "other-secret") for key, value in environment.items()}
        _, _, other = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], changed, versions=VERSIONS)
        self.assertEqual(profiles.serving_condition_key(metadata), profiles.serving_condition_key(other))

    def test_url_path_at_sign_is_not_userinfo(self):
        value = "http://proxy.example:8080/path@routing"
        self.assertEqual(profiles.normalize_serving_config({"env": {"HTTPS_PROXY": value}}),
                         {"args": {}, "env": {"HTTPS_PROXY": value}})

    def test_apply_transports_merged_intent_without_resolving_or_applying_defaults(self):
        environment = {"PATH": "/keep", "OMP_NUM_THREADS": "4", "VLLM_SERVING_OVERRIDES": '{"args":{"additional-config":{"root":1}},"env":{"HCCL_BUFFSIZE":"200"}}'}
        original = dict(environment)
        serving = {"args": {"additional_config": {"workload": 2}}, "env": {"OMP_NUM_THREADS": "0"}}
        with patch.object(profiles, "_load_profiles", side_effect=AssertionError("premature resolution")), patch.object(profiles, "runtime_versions", side_effect=AssertionError("premature discovery")):
            result = profiles.apply_serving_config(environment, serving)
        self.assertEqual(environment, original)
        self.assertIsNot(result, environment)
        self.assertEqual(result["OMP_NUM_THREADS"], "4")
        self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", result)
        self.assertEqual(json.loads(result["VLLM_SERVING_OVERRIDES"]), {"args": {"additional-config": {"root": 1, "workload": 2}}, "env": {"HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "0"}})
        self.assertEqual(profiles.apply_serving_config(result, serving), result)

    def test_invalid_transport_intent_raises_in_parent_and_resolver(self):
        for value in ("", "{", "[]", "null", '{"env":{"MODEL":"other"}}', '{"args":{},"args":{}}'):
            environment = {"VLLM_SERVING_OVERRIDES": value}
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "VLLM_SERVING_OVERRIDES"):
                profiles.apply_serving_config(environment, None)
            with self.subTest(resolve=value), self.assertRaisesRegex(ValueError, "VLLM_SERVING_OVERRIDES"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], environment, versions=VERSIONS)


class ServingProfileTests(unittest.TestCase):
    def test_runtime_versions_use_distributions_not_environment(self):
        def version(name):
            if name == "vllm":
                return "0.11.0+parsa"
            if name == "vllm-ascend":
                raise PackageNotFoundError(name)
            self.fail("Unexpected distribution: " + name)

        with patch.dict("os.environ", {"VLLM_VERSION": "0.23.0"}), patch.object(
            profiles.distribution_metadata, "version", side_effect=version,
        ):
            self.assertEqual(profiles.runtime_versions(), {"vllm": "0.11.0+parsa", "vllm_ascend": None})

    def test_all_six_seeds_apply_documented_defaults_and_keep_source_pins(self):
        registry = json.loads((Path(__file__).resolve().parents[1] / "src/serving_profiles.json").read_text())
        entries = registry["profiles"]
        expected = {
            ("Qwen/Qwen3-4B", "0.11.0"), ("Qwen/Qwen3-8B", "0.11.0"),
            ("Qwen/Qwen3-14B", "0.11.0"), ("Qwen/Qwen3-4B", "0.23.0"),
            ("Qwen/Qwen3-8B", "0.23.0"), ("Qwen/Qwen3-14B", "0.23.0"),
        }
        self.assertEqual(len(entries), 6)
        self.assertEqual({(entry["model"], entry["versions"]["vllm"]) for entry in entries}, expected)
        self.assertEqual(len({entry["profile_id"] for entry in entries}), 6)
        for entry in entries:
            with self.subTest(profile=entry["profile_id"]):
                version = entry["versions"]["vllm"]
                expected_env = {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256" if version == "0.11.0" else "expandable_segments:True"}
                expected_args = ["--dtype=auto"] if version == "0.11.0" else ["--gpu-memory-utilization", "0.9", "--dtype=auto"]
                args, env, metadata = profiles.resolve_serving_profile(
                    entry["model"], ["--dtype=auto"], {}, versions=entry["versions"],
                )
                self.assertEqual((args, env), (expected_args, expected_env))
                self.assertEqual(metadata["status"], "applied")
                self.assertEqual(metadata["profile_revision"], 2)
                self.assertTrue(metadata["rationale"])
                self.assertTrue(metadata["source_evidence"])
                self.assertRegex(metadata["profile_hash"], r"^[0-9a-f]{64}$")
                commit = "2f1aed98ccdb0fcbe1ff4fd0abab225bfd8d0367" if version == "0.11.0" else "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd"
                self.assertEqual(metadata["source_evidence"][0]["commit"], commit)
                self.assertIn(commit, metadata["source_evidence"][0]["url"])

    def test_exact_canonical_and_local_alias_matching(self):
        for model in ("Qwen/Qwen3-8B", "/models/qwen3-8b", "/models/Qwen3-8B", "./models/QWEN3-8B", "qwen3-8b", "Qwen3-8B", Path("/models/Qwen3-8B")):
            with self.subTest(model=model):
                _, _, metadata = profiles.resolve_serving_profile(model, [], {}, versions=VERSIONS)
                self.assertEqual(metadata["model"]["canonical"], "Qwen/Qwen3-8B")
                self.assertIsNotNone(metadata["profile_id"])
        for model in (
            "qwen/Qwen3-8B", "Other/Qwen3-8B", "Other/qwen3-8b",
            "/models/qwen3-8b-awq", "/models/qwen3-8b-fp8", "Qwen/Qwen3-32B", "", None,
        ):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "Unknown.*model.*Qwen3"):
                profiles.resolve_serving_profile(model, [], {}, versions=VERSIONS)

    def test_local_suffix_matches_but_rc_dev_and_exact_mismatches_do_not(self):
        _, _, metadata = profiles.resolve_serving_profile(
            "Qwen/Qwen3-8B", [], {}, versions={"vllm": "0.11.0+parsa.1", "vllm_ascend": "0.11.0+local"},
        )
        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["versions"]["vllm"], "0.11.0+parsa.1")
        _, _, metadata = profiles.resolve_serving_profile(
            "Qwen/Qwen3-8B", [], {}, versions={"vllm": "v0.11.0+local", "vllm_ascend": "0.11.0"},
        )
        self.assertEqual(metadata["status"], "applied")
        for value in ("0.11.0rc1", "0.11.0.dev1", "0.11.0.post1", "0.11.1", "0.11", "0.24.0", None, ""):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "vLLM.*0.11.0.*0.23.0"):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions={**VERSIONS, "vllm": value})

    def test_ascend_version_is_recorded_without_a_recipe_match_constraint(self):
        for value in ("0.21.0", "0.23.0rc1", "0.11.0+custom", None):
            with self.subTest(value=value):
                _, _, metadata = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions={**VERSIONS, "vllm_ascend": value})
                self.assertEqual(metadata["status"], "applied")
                self.assertEqual(metadata["versions"]["vllm_ascend"], value)

    def test_ready_profile_merges_and_inherited_env_wins(self):
        environment = {"PYTORCH_NPU_ALLOC_CONF": "", "TASK_QUEUE_ENABLE": "0", "PATH": "/keep", "ASCEND_RT_VISIBLE_DEVICES": "2,3"}
        explicit = ["--gpu_memory_utilization=0.9", "--dtype", "bfloat16"]
        with patch.object(profiles, "_load_profiles", return_value=[ready_profile()]):
            args, env, metadata = profiles.resolve_serving_profile(
                "/models/qwen3-8b", explicit, environment, versions=VERSIONS,
            )
        self.assertEqual(args, ["--gpu-memory-utilization", "0.9", "--enforce-eager", "--dtype", "bfloat16"])
        self.assertEqual(env, {"PYTORCH_NPU_ALLOC_CONF": "", "TASK_QUEUE_ENABLE": "0"})
        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["effective_extra_args"], args)
        self.assertEqual(metadata["relevant_env"], env)
        self.assertEqual(metadata["explicit_overrides"], {"args": explicit, "env": env, "serving": {"args": {}, "env": {}}})
        self.assertEqual(metadata["profile_versions"], VERSIONS)
        self.assertEqual(environment["PATH"], "/keep")
        with patch.object(profiles, "_load_profiles", return_value=[ready_profile()]):
            _, env, _ = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)
        self.assertEqual(env, {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256", "TASK_QUEUE_ENABLE": "1"})

    def test_context_is_provenance_only_and_never_gates_defaults(self):
        for context in (None, {}, {"precision": "awq", "tp": 4, "hardware": "other"}):
            with self.subTest(context=context):
                _, env, metadata = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS, qualifiers=context)
                self.assertEqual(env, {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256"})
                self.assertEqual(metadata["status"], "applied")
                self.assertEqual(metadata["qualifiers"], context or {})

    def test_experiment_env_and_argv_override_ambient_without_mutation(self):
        environment = {"PYTORCH_NPU_ALLOC_CONF": "ambient", "TASK_QUEUE_ENABLE": "0", "HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "4", "PATH": "/keep", "HF_TOKEN": "inherited-secret"}
        explicit = ["--gpu-memory-utilization=0.85", "--dtype=auto"]
        serving = {"args": {"gpu_memory_utilization": 0.7}, "env": {"PYTORCH_NPU_ALLOC_CONF": "", "HCCL_BUFFSIZE": "0", "OMP_NUM_THREADS": ""}}
        original = copy.deepcopy((environment, explicit, serving))
        args, updates, metadata = profiles.resolve_serving_profile("Qwen/Qwen3-8B", explicit, environment, versions={**VERSIONS, "vllm": "0.23.0"}, overrides=serving)
        self.assertEqual(args, ["--gpu-memory-utilization", "0.7", "--dtype=auto"])
        self.assertEqual(updates, {"PYTORCH_NPU_ALLOC_CONF": "", "HCCL_BUFFSIZE": "0", "OMP_NUM_THREADS": ""})
        self.assertEqual(metadata["relevant_env"], {**updates, "TASK_QUEUE_ENABLE": "0"})
        self.assertEqual(metadata["explicit_overrides"], {"args": explicit, "env": {"PYTORCH_NPU_ALLOC_CONF": "ambient", "TASK_QUEUE_ENABLE": "0", "HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "4"}, "serving": {"args": {"gpu-memory-utilization": 0.7}, "env": serving["env"]}})
        self.assertNotIn("inherited-secret", json.dumps(metadata))
        self.assertEqual((environment, explicit, serving), original)

    def test_parent_transport_and_direct_child_intent_have_identical_conditions(self):
        serving = {"args": {"enforce-eager": True, "additional-config": {"root": 1}}, "env": {"OMP_NUM_THREADS": "0"}}
        environment = {"HCCL_BUFFSIZE": "200"}
        transported = profiles.apply_serving_config(environment, serving)
        parent = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], transported, versions=VERSIONS)
        child = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], environment, versions=VERSIONS, overrides=serving)
        self.assertEqual(parent, child)
        self.assertEqual(profiles.serving_condition_key(parent[2]), profiles.serving_condition_key(child[2]))
        args, updates, metadata = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], transported, versions=VERSIONS, overrides={"args": {"additional-config": {"child": 2}}, "env": {"OMP_NUM_THREADS": "1"}})
        self.assertEqual(args, ["--additional-config", '{"child":2,"root":1}', "--enforce-eager"])
        self.assertEqual(updates["OMP_NUM_THREADS"], "1")
        self.assertEqual(metadata["explicit_overrides"]["serving"]["args"], {"additional-config": {"root": 1, "child": 2}, "enforce-eager": True})

    def test_profile_cannot_set_unreviewed_environment(self):
        for name in ("PATH", "VLLM_VERSION", "ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ARBITRARY"):
            entry = ready_profile()
            entry["defaults"]["env"][name] = "bad"
            with self.subTest(name=name), patch.object(Path, "read_text", return_value=registry_text(entry)), self.assertRaisesRegex(ValueError, "Unreviewed environment: " + name):
                profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)

    def test_explicit_protected_keys_fail_before_resolution(self):
        with self.assertRaisesRegex(ValueError, "Protected"):
            profiles.resolve_serving_profile("Qwen/Qwen3-8B", ["--max_num_seqs=8"], {}, versions=VERSIONS)
        entry = ready_profile()
        entry["defaults"]["args"]["tensor-parallel-size"] = 2
        with patch.object(profiles, "_load_profiles", return_value=[entry]), self.assertRaises(ValueError):
            profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)


class ServingConditionTests(unittest.TestCase):
    def test_package_origins_distinguish_otherwise_identical_conditions(self):
        _, _, metadata = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)
        metadata["package_origins"] = copy.deepcopy(ORIGINS)
        key = profiles.serving_condition_key(metadata)
        metadata["package_origins"]["vllm"]["root"] = "/shadow/vllm"
        self.assertNotEqual(profiles.serving_condition_key(metadata), key)

    def test_key_is_stable_ignores_sweep_fields_and_changes_with_serving_condition(self):
        with patch.object(profiles, "_load_profiles", return_value=[ready_profile()]):
            _, _, metadata = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)
        key = profiles.serving_condition_key(metadata)
        self.assertRegex(key, r"^sp-[0-9a-f]{24}$")
        reordered = dict(reversed(list(metadata.items())))
        reordered.update({"timestamp": 999, "port": 8100, "frequency_mhz": 1000, "max_num_seqs": 8})
        reordered["qualifiers"] = {"frequency_mhz": 1000, "max_num_seqs": 8, "port": 8100}
        reordered["effective_extra_args"] = metadata["effective_extra_args"] + ["--port=8100", "--max_num_seqs", "8", "--frequency-mhz", "1000"]
        self.assertEqual(profiles.serving_condition_key(reordered), key)
        # A sweep name used as an option VALUE is still part of the condition.
        port_alias = {**metadata, "effective_extra_args": ["--served-model-name", "port"]}
        frequency_alias = {**metadata, "effective_extra_args": ["--served-model-name", "frequency"]}
        self.assertNotEqual(profiles.serving_condition_key(port_alias), profiles.serving_condition_key(frequency_alias))
        for field, value in (
            ("versions", {**VERSIONS, "vllm": "0.11.0+other"}),
            ("profile_id", "different"), ("profile_hash", "0" * 64),
            ("profile_revision", 3),
            ("effective_extra_args", ["--gpu-memory-utilization", "0.9"]),
            ("relevant_env", {"TASK_QUEUE_ENABLE": "0"}),
            ("model", {"requested": "Qwen/Qwen3-4B", "canonical": "Qwen/Qwen3-4B", "match": "canonical"}),
            ("qualifiers", {"precision": "fp16"}),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(profiles.serving_condition_key({**metadata, field: value}), key)
        changed = copy.deepcopy(ready_profile())
        changed["defaults"]["args"]["gpu-memory-utilization"] = 0.75
        with patch.object(profiles, "_load_profiles", return_value=[changed]):
            _, _, updated = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {}, versions=VERSIONS)
        self.assertNotEqual(updated["profile_hash"], metadata["profile_hash"])
        self.assertNotEqual(profiles.serving_condition_key(updated), key)

    def test_tuning_env_outside_profile_defaults_distinguishes_variants(self):
        _, _, base = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {"HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "4"}, versions=VERSIONS)
        key = profiles.serving_condition_key(base)
        for environment in ({"HCCL_BUFFSIZE": "201", "OMP_NUM_THREADS": "4"}, {"HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "0"}):
            with self.subTest(environment=environment):
                _, _, changed = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], environment, versions=VERSIONS)
                self.assertNotEqual(profiles.serving_condition_key(changed), key)
        for name in ("TOKENIZERS_PARALLELISM", "MALLOC_ARENA_MAX", "INF_NAN_MODE_ENABLE", "ENABLE_FLASHCOMM1"):
            with self.subTest(name=name):
                _, _, first = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {name: "0"}, versions=VERSIONS)
                _, _, second = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {name: "1"}, versions=VERSIONS)
                self.assertEqual(first["relevant_env"][name], "0")
                self.assertNotEqual(profiles.serving_condition_key(first), profiles.serving_condition_key(second))
        for serving in ({"env": {"CUSTOM_TUNING": "0"}}, {"env": {"OMP_NUM_THREADS": ""}}, {"args": {"enforce-eager": True}}, {"args": {"additional-config": {"nested": {"different": True}}}}):
            with self.subTest(serving=serving):
                _, _, changed = profiles.resolve_serving_profile("Qwen/Qwen3-8B", [], {"HCCL_BUFFSIZE": "200", "OMP_NUM_THREADS": "4"}, versions=VERSIONS, overrides=serving)
                self.assertNotEqual(profiles.serving_condition_key(changed), key)


class RuntimeContextTests(unittest.TestCase):
    def test_anchor_discovers_top_level_packages_without_importing_them(self):
        def spec(name):
            self.assertIn(name, ORIGINS)
            return SimpleNamespace(origin=ORIGINS[name]["origin"],
                                   submodule_search_locations=[ORIGINS[name]["root"]])

        with patch.object(profiles.importlib.util, "find_spec", side_effect=spec), patch.object(
            profiles.distribution_metadata, "version", return_value="0.11.0",
        ):
            self.assertEqual(profiles.runtime_package_context(), {"versions": VERSIONS, "origins": ORIGINS})

    def test_probe_uses_exact_context_and_rejects_discovery_failures(self):
        anchor = {"versions": VERSIONS, "origins": ORIGINS}
        env = {"PYTHONPATH": "/project:/registered", "TASK_QUEUE_ENABLE": "0"}

        def discover(command, **kwargs):
            self.assertEqual(command[:2], [sys.executable, "-c"])
            self.assertEqual(kwargs, {"cwd": "/server-checkout", "env": env,
                                     "check": True, "capture_output": True, "text": True, "timeout": 10})
            return SimpleNamespace(stdout=json.dumps(anchor))

        with patch.object(profiles.subprocess, "run", side_effect=discover):
            self.assertEqual(profiles.verify_runtime_package_context(anchor, "/server-checkout", env), anchor)
        failures = ["{", "[]", json.dumps({}), json.dumps({"versions": VERSIONS, "origins": {}})]
        for field, key, value in (
            ("versions", "vllm", "0.23.0"), ("versions", "vllm_ascend", None),
            ("origins", "vllm", {"origin": "/shadow/vllm/__init__.py", "root": "/shadow/vllm"}),
            ("origins", "vllm_ascend", None),
        ):
            context = copy.deepcopy(anchor)
            context[field][key] = value
            failures.append(json.dumps(context))
        for stdout in failures:
            with self.subTest(stdout=stdout), patch.object(
                profiles.subprocess, "run", return_value=SimpleNamespace(stdout=stdout),
            ), self.assertRaisesRegex(ValueError, "runtime package"):
                profiles.verify_runtime_package_context(anchor, "/server-checkout", env)
        for error in (subprocess.TimeoutExpired("probe", 10), subprocess.CalledProcessError(1, "probe"), OSError("cwd")):
            with self.subTest(error=error), patch.object(profiles.subprocess, "run", side_effect=error), self.assertRaisesRegex(
                ValueError, "runtime package",
            ):
                profiles.verify_runtime_package_context(anchor, "/server-checkout", env)
        missing = copy.deepcopy(anchor)
        missing["origins"]["vllm"] = None
        with patch.object(profiles.subprocess, "run", side_effect=AssertionError("invalid anchor probed")), self.assertRaisesRegex(
            ValueError, "runtime package",
        ):
            profiles.verify_runtime_package_context(missing, "/server-checkout", env)


if __name__ == "__main__":
    unittest.main()
