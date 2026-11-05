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

"""Focused offline checks; not executed during the static-only implementation."""

import asyncio
import contextlib
import copy
import csv
import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import op_conflict as op
from experiments.specialization_gap import _dump, _json, _persist
from src.client import _load_op_conflict_events, _request_payload, _run_op_conflict
from src.controller import EnergyAwareController, SLO
from src.presets import Preset
from src.router_app import EnergyAwareRouter
import run


CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "op-conflict.json"
RUNTIME_ANCHOR = {
    "versions": {"vllm": "0.23.0", "vllm_ascend": "0.23.0"},
    "origins": {name: {"origin": "/runtime/" + name + "/__init__.py", "root": "/runtime/" + name}
                for name in ("vllm", "vllm_ascend")},
}


def characterization(root, config):
    # Hand-derived winners: P=(900,2), D=(1500,5).  Raw sums prefer (1100,3),
    # but normalized energy correctly selects (900,2) as the shared compromise.
    for frequency, seqs, energies in ((900, 2, (2, 10)), (1100, 3, (7, 4)), (1500, 5, (9, 3))):
        for shape, energy in zip(op.SHAPES, energies):
            lengths = config["shapes"][shape]
            for repetition in range(1, 6):
                result = {
                    "model_name": config["model"], "tp": config["tp"],
                    "frequency_mhz": frequency, "max_num_seqs": seqs,
                    "max_num_batched_tokens": config["fixed_max_num_batched_tokens"],
                    "repetition": repetition, "request_shape": shape,
                    "prompt_tokens": lengths["prompt_tokens"], "output_tokens": lengths["output_tokens"],
                    "shape_valid": True, "num_requests": 2, "energy_joules": energy * 2,
                    "energy_num_samples": 2, "energy_per_npu_joules": {"0": energy, "1": energy},
                    "99.9th_ttft": 1, "99.9th_tpot": None if lengths["output_tokens"] == 1 else 0.1,
                    "requests": [{"prompt_idx": index, "ok": True, "shape_valid": True,
                                  "input_num_tokens": lengths["prompt_tokens"],
                                  "output_num_tokens": lengths["output_tokens"]} for index in (1, 2)],
                }
                _dump(root / ("%s-%d-r%d" % (shape, frequency, repetition)) / "result.json", result)


class OPConflictServingChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.discovery = self.enterContext(patch.object(op.gap, "runtime_package_context", return_value=copy.deepcopy(RUNTIME_ANCHOR)))
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.directory) / "output"
        self.config = op.load_config(CONFIG)
        self.config["total_requests"] = 2
        source = Path(self.directory) / "source"
        characterization(source, self.config)
        self.selection = op.select(self.config, source)
        self.launched = []

    def launch(self, command, *, cwd, env, stdout, stderr, check):
        anchor = self.discovery.return_value
        argv, updates, metadata, condition = run.plan_serving_profile(
            env["MODEL"], int(env["TP_SIZE"]), shlex.split(env.get("VLLM_SERVE_EXTRA_ARGS", "")), env,
            versions=anchor["versions"], origins=anchor["origins"],
        )
        self.assertEqual(env["VLLM_EXPECTED_SERVING_CONDITION"], condition)
        self.launched.append((dict(env), argv, updates, metadata, condition))
        _dump(Path(env["RESULT_JSON"]), {"serving_profile": metadata, "serving_condition": condition,
                                        "energy_joules": 40, "total_output_tokens": 17})
        return SimpleNamespace(returncode=0)

    def run_candidates(self, repetition=1, dry_run=False):
        resolved = dict(self.config, candidate_ops=self.selection["candidate_ops"])
        with contextlib.redirect_stdout(io.StringIO()), patch.object(op.subprocess, "run", side_effect=self.launch):
            return op.run_mode(self.config, self.selection, self.root, [self.config["mixtures"][1]],
                               op.banks(resolved, op_indices="0,1"), 2, dry_run, repetition=repetition)

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def test_all_candidates_use_active_profile_and_expected_child_condition(self):
        os.environ["VLLM_SERVING_PROFILES_ENABLED"] = "0"
        before = dict(os.environ)
        self.assertEqual(self.run_candidates(), 0)
        frozen = _json(self.root / "experiment.json")
        self.assertEqual(frozen["serving_profile"]["status"], "applied")
        self.assertEqual(frozen["serving_profile"]["profile_id"], "qwen3-8b-vllm-0.23.0")
        self.assertEqual(frozen["serving_profile"]["package_origins"], RUNTIME_ANCHOR["origins"])
        self.assertEqual(frozen["selection_hash"], op.gap._cohort_hash([self.selection]))
        self.assertEqual(self.selection["serving_condition"], "legacy")
        self.assertEqual(self.discovery.call_count, 1)
        self.assertEqual(len(self.launched), 4)
        self.assertEqual({item[-1] for item in self.launched}, {frozen["serving_condition"]})
        for env, argv, updates, _, _ in self.launched:
            self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", env)
            self.assertEqual(json.loads(env["VLLM_SERVING_OVERRIDES"]), {"args": {}, "env": {}})
            self.assertEqual(argv, ["--gpu-memory-utilization", "0.9"])
            self.assertEqual(updates["PYTORCH_NPU_ALLOC_CONF"], "expandable_segments:True")
            result = _json(Path(env["RESULT_JSON"]))
            self.assertEqual((result["energy_joules"], result["total_output_tokens"]), (40, 17))
            self.assertEqual(result["op_conflict"]["energy_scope"], "sampled_boundary_inclusive_client_subprocess")
            self.assertEqual(env["KV_MONITOR_ENABLED"], "0")
        self.assertEqual(dict(os.environ), before)

    def test_loader_accepts_only_serving_extension_and_config_overrides_ambient(self):
        self.config["serving"] = {"args": {"gpu_memory_utilization": 0.85},
                                  "env": {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:128"}}
        path = Path(self.directory) / "config.json"
        _dump(path, self.config)
        self.config = op.load_config(path)
        self.assertEqual(self.config["serving"]["args"], {"gpu-memory-utilization": 0.85})
        self.selection["config"] = self.config
        os.environ.update(PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:64", VLLM_SERVE_EXTRA_ARGS="--gpu-memory-utilization 0.8")
        before = dict(os.environ)
        self.run_candidates()
        frozen = _json(self.root / "experiment.json")
        self.assertEqual(frozen["config"]["serving"], self.config["serving"])
        self.assertEqual(frozen["serving_profile"]["effective_extra_args"], ["--gpu-memory-utilization", "0.85"])
        self.assertEqual(frozen["serving_profile"]["relevant_env"]["PYTORCH_NPU_ALLOC_CONF"], "max_split_size_mb:128")
        self.assertEqual(dict(os.environ), before)
        for extra in ({"unknown": 1}, {"serving": {"args": {"max-num-seqs": 99}}},
                      {"serving": {"env": {"MODEL": "other"}}}):
            _dump(path, dict(self.config, **extra))
            with self.assertRaises(ValueError):
                op.load_config(path)

    def test_legacy_release_profile_is_active_and_same_condition_can_resume_repetitions(self):
        self.discovery.return_value["versions"] = {"vllm": "0.11.0", "vllm_ascend": "0.11.0"}
        self.assertEqual(self.run_candidates(), 0)
        before = (self.root / "experiment.json").read_bytes()
        self.assertEqual(self.run_candidates(repetition=2), 0)
        self.assertEqual((self.root / "experiment.json").read_bytes(), before)
        self.assertEqual(len(self.launched), 8)
        self.assertEqual(len({item[-1] for item in self.launched}), 1)
        for env, argv, updates, metadata, _ in self.launched:
            self.assertEqual(argv, [])
            self.assertEqual(metadata["profile_id"], "qwen3-8b-vllm-0.11.0")
            self.assertEqual(updates["PYTORCH_NPU_ALLOC_CONF"], "max_split_size_mb:256")
            self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", env)

    def test_unknown_model_or_version_and_fixed_knobs_stop_before_any_output_or_child(self):
        for invalid in ("model", "version", "fixed"):
            with self.subTest(invalid=invalid):
                self.config["model"] = "demo" if invalid == "model" else "/models/qwen3-8b"
                self.discovery.return_value["versions"]["vllm"] = "0.99.0" if invalid == "version" else "0.23.0"
                os.environ["VLLM_SERVE_EXTRA_ARGS"] = "--max-num-seqs 999" if invalid == "fixed" else ""
                with self.assertRaises(ValueError):
                    self.run_candidates()
                self.assertFalse(self.root.exists())
                self.assertEqual(self.launched, [])

    def test_changed_or_missing_profile_identity_rejects_root_without_rewriting(self):
        self.run_candidates()
        original = _json(self.root / "experiment.json")
        for change in ("environment", "version", "origin", "old-root", "config", "no-manifest"):
            with self.subTest(change=change):
                self.discovery.return_value = copy.deepcopy(RUNTIME_ANCHOR)
                os.environ.pop("PYTORCH_NPU_ALLOC_CONF", None)
                self.config.pop("serving", None)
                _dump(self.root / "experiment.json", original)
                if change == "environment":
                    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:128"
                elif change == "version":
                    self.discovery.return_value["versions"] = {"vllm": "0.11.0", "vllm_ascend": "0.11.0"}
                elif change == "origin":
                    self.discovery.return_value["origins"]["vllm"] = {"origin": "/other/vllm/__init__.py", "root": "/other/vllm"}
                elif change == "config":
                    self.config["serving"] = {"args": {"gpu-memory-utilization": 0.85}}
                elif change == "old-root":
                    old = dict(original)
                    del old["serving_profile"]
                    del old["serving_condition"]
                    _dump(self.root / "experiment.json", old)
                else:
                    (self.root / "experiment.json").unlink()
                before, launches = self.snapshot(), len(self.launched)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    self.run_candidates(repetition=2)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(len(self.launched), launches)

    def test_dry_run_plans_profile_without_writes_or_child_launch(self):
        self.assertEqual(self.run_candidates(dry_run=True), 0)
        self.assertEqual(self.discovery.call_count, 1)
        self.assertFalse(self.root.exists())
        self.assertEqual(self.launched, [])


class OPConflictChecks(unittest.TestCase):
    def test_conditions_cannot_complete_each_others_repetition_sets(self):
        config = op.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            characterization(root, config)
            for path in root.glob("*/result.json"):
                result = _json(path)
                result["serving_condition"] = "sp-" + ("a" if result["repetition"] <= 2 else "b") * 24
                _dump(path, result)
            with self.assertRaisesRegex(ValueError, "multiple serving conditions.*--serving-condition") as error:
                op.select(config, root)
            self.assertIn("sp-" + "a" * 24, str(error.exception))
            self.assertIn("sp-" + "b" * 24, str(error.exception))
            for condition in ("sp-" + "a" * 24, "sp-" + "b" * 24):
                with self.assertRaisesRegex(ValueError, "no fully measured"):
                    op.select(config, root, serving_condition=condition)

    def test_condition_filter_precedes_duplicates_counts_and_energy_selection(self):
        config = op.load_config(CONFIG)
        condition = "sp-" + "a" * 24
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            characterization(root, config)
            legacy = op.select(config, root)
            self.assertEqual(legacy["serving_condition"], "legacy")
            for path in list(root.glob("*/result.json")):
                result = _json(path)
                result["serving_condition"] = condition
                _dump(root / (path.parent.name + "-profile") / "result.json", result)
            selected = op.select(config, root, serving_condition=condition)
            self.assertEqual(selected["serving_condition"], condition)
            self.assertEqual(selected["roles"], {"prefill": 0, "decode": 2, "shared": 0})
            self.assertEqual(selected["candidate_ops"], legacy["candidate_ops"])
            self.assertEqual(selected["characterization"][0]["mean_energy_per_request_j"], 10)
            # Other conditions' request counts do not disqualify this complete set.
            path = root / "prefill-heavy-900-r1" / "result.json"
            result = _json(path)
            result["num_requests"] = 3
            _dump(path, result)
            self.assertEqual(op.select(config, root, serving_condition=condition)["roles"], selected["roles"])
            # Irrelevant model/frequency results must not create an ambiguity.
            for path in root.glob("*/result.json"):
                result = _json(path)
                if not result.get("serving_condition"):
                    result["model_name"] = "Other/model"
                    _dump(path, result)
            irrelevant = _json(root / "prefill-heavy-900-r1-profile" / "result.json")
            irrelevant.update(serving_condition="other", frequency_mhz=config["maximum_frequency_mhz"] + 1)
            _dump(root / "irrelevant-frequency" / "result.json", irrelevant)
            self.assertEqual(op.select(config, root)["serving_condition"], condition)
            with self.assertRaisesRegex(ValueError, "available.*sp-"):
                op.select(config, root, serving_condition="absent")

    def test_condition_is_frozen_and_legacy_selection_hash_is_preserved(self):
        config = op.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            characterization(source, config)
            common = ["--characterization-root", str(source), "--output-root", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(op.main(["--select", "--serving-condition", "legacy", *common]), 0)
                frozen = _json(output / "selection.json")
                self.assertEqual(frozen["serving_condition"], "legacy")
                with self.assertRaisesRegex(ValueError, "frozen.*serving condition"):
                    op.main(["--inspect", "--serving-condition", "other", *common])
            del frozen["serving_condition"]
            _dump(output / "selection.json", frozen)
            loaded = op.load_selection(output, config)
            self.assertEqual(loaded.get("serving_condition") or "legacy", "legacy")
            self.assertEqual(loaded, frozen)  # Do not alter hashes of old frozen manifests.
            for mode in ([], ["--aggregate"], ["--dry-run"], ["--list"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    op.main([*mode, "--serving-condition", "legacy", "--output-root", str(output)])

    def test_selection_is_offline_and_requires_every_characterization_repetition(self):
        config = op.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            characterization(root, config)
            selected = op.select(config, root)
            self.assertEqual(selected["roles"], {"prefill": 0, "decode": 2, "shared": 0})
            self.assertEqual(selected["candidate_ops"], [
                {"frequency_mhz": 900, "max_num_seqs": 2},
                {"frequency_mhz": 1100, "max_num_seqs": 3},
                {"frequency_mhz": 1500, "max_num_seqs": 5},
            ])
            path = root / "prefill-heavy-900-r5" / "result.json"
            result = _json(path)
            result["shape_valid"] = False
            _dump(path, result)
            self.assertEqual(op.select(config, root)["candidate_ops"][0]["frequency_mhz"], 1100)
            path.unlink()
            self.assertEqual(op.select(config, root)["candidate_ops"][0]["frequency_mhz"], 1100)
            duplicate = _json(root / "decode-heavy-1500-r1" / "result.json")
            _dump(root / "duplicate" / "result.json", duplicate)
            with self.assertRaises(ValueError):
                op.select(config, root)

    def test_frozen_selection_refuses_changed_inputs_and_run_does_not_reselect(self):
        config = op.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            characterization(source, config)
            selection = op.select(config, source)
            _persist(root / "selection.json", selection)
            _persist(root / "selection.json", selection)
            changed = copy.deepcopy(selection)
            changed["roles"]["prefill"] = 1
            with self.assertRaises(ValueError):
                _persist(root / "selection.json", changed)
            self.assertEqual(op.load_selection(root, config), selection)
            source_result = source / "prefill-heavy-900-r1" / "result.json"
            source_result.unlink()
            self.assertEqual(op.load_selection(root, config), selection)
            config["arrival_interval_s"] *= 2
            with self.assertRaises(ValueError):
                op.load_selection(root, config)

    def test_banks_keep_diagonals_both_ordered_mappings_and_neutral_controls(self):
        config = op.load_config(CONFIG)
        config["candidate_ops"] = [{"frequency_mhz": 900, "max_num_seqs": 2},
                                   {"frequency_mhz": 1100, "max_num_seqs": 3}]
        variants = op.banks(config)
        self.assertEqual([bank["name"] for bank in variants],
                         ["shared-0-0", "specialized-0-1-pd", "specialized-0-1-dp", "shared-1-1"])
        self.assertEqual(len(op.realizations(config, config["mixtures"][0], variants)), 3)
        mixed = config["mixtures"][1]
        pair = op.banks(config, "specialized-0-1")
        self.assertEqual(op.routing(config, mixed, pair[0])["pool_to_lane"],
                         {"prefill-heavy": "lane_a", "decode-heavy": "lane_b"})
        self.assertEqual(op.routing(config, mixed, pair[1])["pool_to_lane"],
                         {"prefill-heavy": "lane_b", "decode-heavy": "lane_a"})

    def test_cohort_has_fixed_shapes_counts_timing_and_identity(self):
        config = op.load_config(CONFIG)
        config["total_requests"] = 4
        config["arrival_interval_s"] = 0.25
        rows = op.cohort(config, config["mixtures"][1])
        self.assertEqual([row["request_index"] for row in rows], [1, 2, 3, 4])
        self.assertEqual([row["arrival_sec"] for row in rows], [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(sum(row["request_shape"] == "prefill-heavy" for row in rows), 2)
        self.assertEqual(rows, op.cohort(config, config["mixtures"][1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cohort.json"
            _dump(path, rows)
            self.assertEqual(_load_op_conflict_events(path), rows)
            rows[0]["arrival_sec"] = float("nan")
            _dump(path, rows)
            with self.assertRaises(ValueError):
                _load_op_conflict_events(path)

    def test_inspect_is_read_only_and_debug_selectors_use_frozen_selection(self):
        config = op.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            characterization(source, config)
            common = ["--characterization-root", str(source), "--output-root", str(output)]
            with contextlib.redirect_stdout(io.StringIO()), patch("subprocess.run", side_effect=AssertionError("offline selector started a process")):
                self.assertEqual(op.main(["--inspect", *common]), 0)
                self.assertFalse(output.exists())
                self.assertEqual(op.main(["--select", *common]), 0)
                self.assertTrue((output / "selection.json").is_file())
                listing = io.StringIO()
                with contextlib.redirect_stdout(listing):
                    self.assertEqual(op.main(["--list", "--output-root", str(output), "--mixture", "50-50",
                                             "--bank", "specialized-0-1", "--mapping", "dp", "--repetition", "3"]), 0)
                self.assertEqual(len(listing.getvalue().splitlines()), 1)
                self.assertIn("specialized-0-1-dp/rep-3", listing.getvalue())
                self.assertFalse((output / "experiment.json").exists())

    def test_open_loop_offers_entire_cohort_before_waiting_for_completion(self):
        async def check():
            all_offered = asyncio.Event()
            offered = []

            class Session:
                def __init__(self, **kwargs):
                    self.connector = kwargs["connector"]

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    await self.connector.close()

            async def request(session, semaphore, args, prompt, index, event):
                async with semaphore:
                    offered.append(index)
                    if len(offered) == 3:
                        all_offered.set()
                    await all_offered.wait()
                    return {"ok": True, "stream_complete": True, "input_num_tokens": 1, "output_num_tokens": 1000}

            events = [{"request_index": index, "arrival_sec": 0, "request_shape": "decode-heavy",
                       "token_id": 64, "prompt_tokens": 1, "output_tokens": 1000} for index in (1, 2, 3)]
            with patch("src.client.aiohttp.ClientSession", Session), patch("src.client.run_one", request):
                results = await asyncio.wait_for(_run_op_conflict(SimpleNamespace(max_in_flight=1), events), 1)
            self.assertEqual(offered, [1, 2, 3])
            self.assertTrue(all(row["shape_valid"] for row in results))

        asyncio.run(check())

    def test_opt_in_synthetic_metadata_routes_without_prediction_and_is_not_forwarded(self):
        args = SimpleNamespace(prompt_source="synthetic", model="model", temperature=0,
                               max_tokens=None, min_tokens=None, ignore_eos=False, op_conflict=True)
        event = {"request_shape": "decode-heavy", "prompt_tokens": 1, "output_tokens": 1000}
        payload = _request_payload(args, [42], True, event=event, prompt_idx=2)
        self.assertEqual(payload["max_tokens"], 1000)
        self.assertEqual(payload["min_tokens"], 1000)
        self.assertTrue(payload["ignore_eos"])
        controller = EnergyAwareController(
            [Preset("lane_a", "http://a", 8192), Preset("lane_b", "http://b", 8192)],
            None, SLO(120), routing_policy="oracle_class_affinity",
            routing_kwargs={"op_conflict": {"lane_ids": ["lane_a", "lane_b"],
                "active_pools": list(op.SHAPES),
                "pool_to_lane": {"prefill-heavy": "lane_a", "decode-heavy": "lane_b"}}},
        )
        proxy = EnergyAwareRouter(controller)
        _, choice, forwarded = asyncio.run(proxy._apply_routing(payload))
        self.assertEqual(choice.preset.preset_id, "lane_b")
        self.assertNotIn("energy_aware", forwarded)
        self.assertNotIn("X-Energy-Aware-L-Out-Predicted-Raw", proxy._routing_headers(choice))
        with self.assertRaises(ValueError):
            controller.choose({"prompt": [42]})
        args.op_conflict = False
        self.assertNotIn("energy_aware", _request_payload(args, [42], True))

    def test_specialization_gap_labels_and_neutral_order_are_preserved(self):
        args = SimpleNamespace(prompt_source="azure-mix", model="model", temperature=0,
                               max_tokens=None, max_layers=None, specialization_gap=True)
        payload = _request_payload(args, "hello", True, event={"pool": "summarization"}, prompt_idx=2)
        original = copy.deepcopy(payload)
        controller = EnergyAwareController(
            [Preset("lane_a", "http://a", 8192), Preset("lane_b", "http://b", 8192)],
            None, SLO(120), routing_policy="round_robin",
            routing_kwargs={"specialization_gap": {"lane_ids": ["lane_a", "lane_b"],
                "active_pools": ["summarization"], "pool_to_lane": {}}},
        )
        _, choice, forwarded = asyncio.run(EnergyAwareRouter(controller)._apply_routing(payload))
        self.assertEqual(choice.preset.preset_id, "lane_b")
        self.assertEqual(payload, original)
        self.assertNotIn("energy_aware", forwarded)

    def test_strict_power_boundary_rejects_missing_devices_power_and_frequency(self):
        expected = {"0": 900, "1": 1100}
        sample = {"timestamp": 1.0, "per_npu": {
            "0": {"power_sample": 2.0, "frequency": 900},
            "1": {"power_sample": 3.0, "frequency": 1100}}}
        run.validate_op_conflict_boundary(sample, expected)
        for key, value in (("power_sample", None), ("power_sample", float("inf")), ("frequency", 900)):
            bad = copy.deepcopy(sample)
            bad["per_npu"]["1"][key] = value
            with self.assertRaises(ValueError):
                run.validate_op_conflict_boundary(bad, expected)
        with self.assertRaises(ValueError):
            run.validate_op_conflict_boundary(sample, {"0": 900})

    def test_aggregation_retains_missing_runs_and_reuses_diagonal_winner(self):
        config = op.load_config(CONFIG)
        config["total_requests"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            characterization(root / "source", config)
            selection = op.select(config, root / "source")
            _persist(root / "selection.json", selection)
            _, _, profile, condition = run.plan_serving_profile(
                config["model"], config["tp"], [], {},
                versions=RUNTIME_ANCHOR["versions"], origins=RUNTIME_ANCHOR["origins"])
            _persist(root / "experiment.json", op._experiment(config, selection, 5, profile, condition))
            mixture = config["mixtures"][1]
            events = op.cohort(config, mixture)
            schedule = root / "50-50" / "cohort.json"
            _persist(schedule, events)
            bank = op.banks(dict(config, candidate_ops=selection["candidate_ops"]), op_indices="0")[0]
            candidate_dir = root / "50-50" / "shared-0-0"
            candidate = op.candidate_manifest(config, selection, mixture, bank, 5, events, schedule, condition)
            _persist(candidate_dir / "candidate.json", candidate)
            for repetition, energy in enumerate((40, 44, 48, 52, 56), 1):
                run_dir = candidate_dir / ("rep-%d" % repetition)
                manifest = dict(candidate, repetition=repetition, result=str(run_dir / "result.json"))
                _dump(run_dir / "manifest.json", manifest)
                readings = {str(device): {"power_sample": energy / 40, "frequency": 900}
                            for lane in config["lanes"] for device in lane["device_map"]}
                log = run_dir / "demo-state" / "power.jsonl"
                log.parent.mkdir()
                log.write_text("".join(json.dumps({"timestamp": stamp, "per_npu": readings}) + "\n"
                                       for stamp in (0, 10)), encoding="utf-8")
                result = {"op_conflict": manifest, "energy_joules": energy, "energy_num_samples": 2,
                          "serving_profile": profile, "serving_condition": condition,
                          "energy_per_npu_joules": {device: energy / 4 for device in readings},
                          "benchmark_start_s": 1, "benchmark_end_s": 9,
                          "requests": [{"prompt_idx": event["request_index"], "request_shape": event["request_shape"],
                              "arrival_sec": event["arrival_sec"],
                              "ok": True, "stream_complete": True, "shape_valid": True,
                              "input_num_tokens": event["prompt_tokens"], "output_num_tokens": event["output_tokens"],
                              "lane": "lane_a" if event["request_index"] == 1 else "lane_b",
                              "ttft_s": 1, "tpot_mean_s": None if event["output_tokens"] == 1 else 0.1}
                              for event in events]}
                _dump(run_dir / "result.json", result)
                _dump(run_dir / "execution.json", {"returncode": 0})
            op.aggregate(root, config, selection)
            with open(root / "banks.csv", newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["mean_energy_j"]), 48)
            self.assertAlmostEqual(float(row["std_energy_j"]), 40**0.5)
            with open(root / "summary.csv", newline="", encoding="utf-8") as handle:
                summary = next(row for row in csv.DictReader(handle) if row["mixture"] == "50-50")
            self.assertEqual(summary["specialization_gap_j"], "0.0")
            self.assertEqual(summary["search_complete"], "False")
            first = candidate_dir / "rep-1"
            valid_result = _json(first / "result.json")
            wrong_energy = copy.deepcopy(valid_result)
            wrong_energy["energy_joules"] = 4
            wrong_energy["energy_per_npu_joules"] = {device: 1 for device in readings}
            self.assertFalse(op.measurement_status(first, wrong_energy, candidate)["measurement_valid"])
            wrong_latency = copy.deepcopy(valid_result)
            wrong_latency["requests"][0]["ttft_s"] = 100
            _dump(first / "result.json", wrong_latency)
            op.aggregate(root, config, selection)
            with open(root / "banks.csv", newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["feasible"], "False")
            self.assertEqual(row["valid_repetitions"], "4")
            _dump(first / "result.json", valid_result)
            (candidate_dir / "rep-5" / "result.json").unlink()
            op.aggregate(root, config, selection)
            with open(root / "banks.csv", newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["valid_repetitions"], "4")
            self.assertEqual(row["feasible"], "False")
            (first / "result.json").write_text("{", encoding="utf-8")
            op.aggregate(root, config, selection)
            with open(root / "runs.csv", newline="", encoding="utf-8") as handle:
                failed = next(csv.DictReader(handle))
            self.assertEqual(failed["eligible"], "False")
            self.assertTrue(failed["result_error"])
            wrong_condition = dict(valid_result, serving_condition="sp-" + "0" * 24)
            _dump(first / "result.json", wrong_condition)
            before = (root / "runs.csv").read_bytes()
            with self.assertRaisesRegex(ValueError, "serving condition"):
                op.aggregate(root, config, selection)
            self.assertEqual((root / "runs.csv").read_bytes(), before)

    def test_plot_uses_sample_sd_and_marks_incomplete_search(self):
        from plots.op_conflict import plot_op_conflict
        import matplotlib.pyplot as plt

        rows = [{"mixture": "50-50", "search_complete": False,
                 "shared_mean_energy_j": 48, "shared_std_energy_j": 40**0.5,
                 "specialized_mean_energy_j": 32, "specialized_std_energy_j": 2}]
        figure = plot_op_conflict(rows)
        axes = figure.axes[0]
        self.assertEqual([patch.get_height() for patch in axes.patches], [48, 32])
        self.assertEqual(axes.get_xticklabels()[0].get_text(), "50-50*")
        segments = axes.containers[0].lines[2][0].get_segments()
        self.assertAlmostEqual(segments[0][1][1] - segments[0][0][1], 2 * 40**0.5)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
