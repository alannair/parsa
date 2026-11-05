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

"""Offline regression checks. NOT executed during the static-only implementation."""

import asyncio
import copy
import contextlib
import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run
from experiments import specialization_gap as gap
from src.client import _consume_sse, _request_payload
from src.controller import EnergyAwareController, SLO
from src.presets import Preset
from src.router_app import EnergyAwareRouter
from experiments.specialization_gap import _candidate_manifest, _cohort_hash, _dump, _measurement_status, _planned, _realizations, _routing, _write_cohort, aggregate, banks, load_config


CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "specialization-gap.json"
RUNTIME_ANCHOR = {
    "versions": {"vllm": "0.23.0", "vllm_ascend": "0.23.0"},
    "origins": {name: {"origin": "/runtime/" + name + "/__init__.py", "root": "/runtime/" + name}
                for name in ("vllm", "vllm_ascend")},
}


class SpecializationGapServingChecks(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.discovery = self.enterContext(patch.object(gap, "runtime_package_context", return_value=copy.deepcopy(RUNTIME_ANCHOR)))
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.directory) / "output"
        self.config = gap.load_config(CONFIG)
        self.config["total_requests"] = 2
        self.config["source_schedule"] = str(Path(self.directory) / "source.csv")
        gap._write_cohort(Path(self.config["source_schedule"]), [
            {"arrival_sec": index, "type": 0, "dataset_idx": index, "prompt": "hello",
             "pool": pool} for index, pool in enumerate(("summarization", "conversation"))
        ])
        self.launched = []

    def launch(self, command, *, cwd, env, check):
        # Only the hardware-bound child is replaced; planning and persistence are real.
        anchor = self.discovery.return_value
        argv, updates, metadata, condition = run.plan_serving_profile(
            env["MODEL"], int(env["TP_SIZE"]), shlex.split(env.get("VLLM_SERVE_EXTRA_ARGS", "")), env,
            versions=anchor["versions"], origins=anchor["origins"],
        )
        self.assertEqual(env["VLLM_EXPECTED_SERVING_CONDITION"], condition)
        self.launched.append((dict(env), argv, updates, metadata, condition))
        gap._dump(Path(env["RESULT_JSON"]), {"serving_profile": metadata, "serving_condition": condition,
                                             "energy_joules": 40, "total_output_tokens": 17})
        return SimpleNamespace(returncode=0)

    def run_candidates(self, repetition=1, dry_run=False):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(gap.subprocess, "run", side_effect=self.launch):
            gap.run_mode(self.config, self.root, [self.config["mixtures"][1]],
                         gap.banks(self.config, op_indices="0,1"), 2, dry_run, repetition=repetition)

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

    def test_active_profiles_preplan_all_banks_and_preserve_raw_child_intent(self):
        os.environ["VLLM_SERVING_PROFILES_ENABLED"] = "0"  # Former opt-out cannot disable mandatory planning.
        before = dict(os.environ)
        self.run_candidates()
        frozen = gap._json(self.root / "experiment.json")
        self.assertEqual(frozen["serving_profile"]["status"], "applied")
        self.assertEqual(frozen["serving_profile"]["profile_id"], "qwen3-8b-vllm-0.23.0")
        self.assertEqual(frozen["serving_profile"]["package_origins"], RUNTIME_ANCHOR["origins"])
        self.assertEqual(self.discovery.call_count, 1)
        self.assertEqual(len(self.launched), 4)
        self.assertEqual({item[-1] for item in self.launched}, {frozen["serving_condition"]})
        for env, argv, updates, _, _ in self.launched:
            self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", env)
            self.assertEqual(json.loads(env["VLLM_SERVING_OVERRIDES"]), {"args": {}, "env": {}})
            self.assertEqual(argv, ["--gpu-memory-utilization", "0.9"])
            self.assertEqual(updates["PYTORCH_NPU_ALLOC_CONF"], "expandable_segments:True")
            result = gap._json(Path(env["RESULT_JSON"]))
            self.assertEqual((result["energy_joules"], result["total_output_tokens"]), (40, 17))
            self.assertEqual(result["specialization_gap"]["energy_scope"], "sampled_boundary_inclusive_client_subprocess")
        self.assertEqual(dict(os.environ), before)

    def test_loader_and_individual_overrides_use_config_over_ambient(self):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:64"
        os.environ["VLLM_SERVE_EXTRA_ARGS"] = "--gpu-memory-utilization 0.8"
        self.config["serving"] = {"args": {"gpu_memory_utilization": 0.85},
                                  "env": {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:128"}}
        path = Path(self.directory) / "config.json"
        gap._dump(path, self.config)
        self.config = gap.load_config(path)
        self.assertEqual(self.config["serving"]["args"], {"gpu-memory-utilization": 0.85})
        before = dict(os.environ)
        self.run_candidates()
        frozen = gap._json(self.root / "experiment.json")
        self.assertEqual(frozen["config"]["serving"], self.config["serving"])
        self.assertEqual(frozen["serving_profile"]["effective_extra_args"], ["--gpu-memory-utilization", "0.85"])
        self.assertEqual(frozen["serving_profile"]["relevant_env"]["PYTORCH_NPU_ALLOC_CONF"], "max_split_size_mb:128")
        self.assertEqual(dict(os.environ), before)

    def test_legacy_release_profile_is_active_and_same_condition_can_resume_repetitions(self):
        self.discovery.return_value["versions"] = {"vllm": "0.11.0", "vllm_ascend": "0.11.0"}
        self.run_candidates()
        before = (self.root / "experiment.json").read_bytes()
        self.run_candidates(repetition=2)
        self.assertEqual((self.root / "experiment.json").read_bytes(), before)
        self.assertEqual(len(self.launched), 8)
        self.assertEqual(len({item[-1] for item in self.launched}), 1)
        for env, argv, updates, metadata, _ in self.launched:
            self.assertEqual(argv, [])
            self.assertEqual(metadata["profile_id"], "qwen3-8b-vllm-0.11.0")
            self.assertEqual(updates["PYTORCH_NPU_ALLOC_CONF"], "max_split_size_mb:256")
            self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", env)

    def test_unknown_model_or_version_and_fixed_knobs_fail_before_writes(self):
        for invalid in ("model", "version", "fixed"):
            with self.subTest(invalid=invalid):
                self.config["model"] = "demo" if invalid == "model" else "/models/qwen3-8b"
                self.discovery.return_value["versions"]["vllm"] = "0.99.0" if invalid == "version" else "0.23.0"
                os.environ["VLLM_SERVE_EXTRA_ARGS"] = "--max-num-seqs 999" if invalid == "fixed" else ""
                with self.assertRaises(ValueError):
                    self.run_candidates()
                self.assertFalse(self.root.exists())
                self.assertEqual(self.launched, [])

    def test_changed_or_missing_frozen_profile_requires_fresh_root_before_writes(self):
        self.run_candidates()
        original = gap._json(self.root / "experiment.json")
        for change in ("environment", "version", "origin", "old-root", "no-manifest"):
            with self.subTest(change=change):
                self.discovery.return_value = copy.deepcopy(RUNTIME_ANCHOR)
                os.environ.pop("PYTORCH_NPU_ALLOC_CONF", None)
                gap._dump(self.root / "experiment.json", original)
                if change == "environment":
                    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:128"
                elif change == "version":
                    self.discovery.return_value["versions"] = {"vllm": "0.11.0", "vllm_ascend": "0.11.0"}
                elif change == "origin":
                    self.discovery.return_value["origins"]["vllm"] = {"origin": "/other/vllm/__init__.py", "root": "/other/vllm"}
                elif change == "old-root":
                    old = dict(original)
                    del old["serving_profile"]
                    del old["serving_condition"]
                    gap._dump(self.root / "experiment.json", old)
                else:
                    (self.root / "experiment.json").unlink()
                before, launches = self.snapshot(), len(self.launched)
                with self.assertRaisesRegex(ValueError, "fresh"):
                    self.run_candidates(repetition=2)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(len(self.launched), launches)

    def test_profile_protection_and_dry_run_do_not_create_output(self):
        for serving in ({"args": {"max-num-seqs": 99}}, {"env": {"TP_SIZE": "4"}}):
            self.config["serving"] = serving
            path = Path(self.directory) / "bad.json"
            gap._dump(path, self.config)
            with self.assertRaises(ValueError):
                gap.load_config(path)
        self.config.pop("serving")
        self.run_candidates(dry_run=True)
        self.assertEqual(self.discovery.call_count, 1)
        self.assertFalse(self.root.exists())
        self.assertEqual(self.launched, [])


def controller(mode, mapping):
    return EnergyAwareController(
        [Preset("lane_a", "http://a", 8192, max_num_seqs=50, frequency_mhz=1100),
         Preset("lane_b", "http://b", 8192, max_num_seqs=100, frequency_mhz=1300)],
        profile=None, slo=SLO(120), routing_policy=mode,
        routing_kwargs={"specialization_gap": {
            "lane_ids": ["lane_a", "lane_b"], "active_pools": ["summarization", "conversation"],
            "pool_to_lane": mapping,
        }},
    )


def body(pool, index):
    return {"messages": [{"role": "user", "content": "hello"}],
            "energy_aware": {"specialization_gap": {"pool": pool, "request_index": index}}}


class SpecializationGapChecks(unittest.TestCase):
    def test_oracle_uses_provenance_without_profile_or_predictor(self):
        for mapping in ({"summarization": "lane_a", "conversation": "lane_b"},
                        {"summarization": "lane_b", "conversation": "lane_a"}):
            router = controller("oracle_class_affinity", mapping)
            for index, pool in enumerate(("conversation", "summarization"), 1):
                self.assertEqual(router.choose(body(pool, index)).preset.preset_id, mapping[pool])
        with self.assertRaises(ValueError):
            EnergyAwareController([], None, SLO(120), routing_policy="oracle_class_affinity")

    def test_neutral_routing_is_cohort_index_round_robin(self):
        router = controller("round_robin", {})
        for index in (4, 1, 3, 2):
            self.assertEqual(router.choose(body("summarization", index)).preset.preset_id,
                             "lane_a" if index % 2 else "lane_b")
        with self.assertRaises(ValueError):
            router.choose(body("unknown", 1))

    def test_metadata_round_trip_and_backend_removal(self):
        args = SimpleNamespace(model="model", temperature=0, max_tokens=None, max_layers=None,
                               specialization_gap=True)
        payload = _request_payload(args, "hello", True, event={"pool": "summarization"}, prompt_idx=1)
        self.assertEqual(payload["energy_aware"]["specialization_gap"]["pool"], "summarization")
        proxy = EnergyAwareRouter(controller("oracle_class_affinity", {
            "summarization": "lane_a", "conversation": "lane_b"}))
        _, choice, forwarded = asyncio.run(proxy._apply_routing(payload))
        self.assertEqual(choice.preset.preset_id, "lane_a")
        self.assertNotIn("energy_aware", forwarded)
        self.assertIn("specialization_gap", payload["energy_aware"])
        args.specialization_gap = False
        self.assertNotIn("energy_aware", _request_payload(args, "hello", True))

    def test_mapping_enumeration_and_controls(self):
        config = load_config(CONFIG)
        variants = banks(config, "specialized-0-1")
        self.assertEqual([item["mapping"] for item in variants], ["pd", "dp"])
        for mixture in config["mixtures"]:
            for bank in banks(config):
                routing = _routing(config, mixture, bank)
                expected = "oracle_class_affinity" if bank["ops"][0] != bank["ops"][1] and mixture["name"] == "50-50" else "round_robin"
                self.assertEqual(routing["routing_mode"], expected)
        self.assertEqual(sum(item["ops"][0] == item["ops"][1] for item in banks(config)), 3)
        self.assertEqual(len(_realizations(config, config["mixtures"][0], banks(config))), 6)
        neutral_pair = _realizations(config, config["mixtures"][0], variants)
        self.assertEqual(len(neutral_pair), 1)
        self.assertEqual(neutral_pair[0]["mapping_aliases"], ["pd", "dp"])
        planned_neutral = list(_planned(config, [config["mixtures"][0]], variants, 1, Path("unused"), mapping="neutral"))
        self.assertEqual(planned_neutral[0][1]["name"], "specialized-0-1-neutral")
        bad = copy.deepcopy(config)
        bad["candidate_ops"][0]["max_num_batched_tokens"] = 4096
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            _dump(path, bad)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_strict_sse_rejects_truncated_work(self):
        class Content:
            def __init__(self, chunks):
                self.chunks = chunks

            async def iter_chunked(self, size):
                for chunk in self.chunks:
                    yield chunk

        message = b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
        response = SimpleNamespace(content=Content([message]))
        with self.assertRaises(ValueError):
            asyncio.run(_consume_sse(response, strict=True))
        response.content = Content([message, b'data: [DONE]\n\n'])
        self.assertTrue(asyncio.run(_consume_sse(response, strict=True))["complete"])
        for reason in (None, "error"):
            import json
            event = {"choices": [{"delta": {"content": "hi"}, "finish_reason": reason}]}
            response.content = Content([("data: " + json.dumps(event) + "\n\n").encode(), b'data: [DONE]\n\n'])
            with self.assertRaises(ValueError):
                asyncio.run(_consume_sse(response, strict=True))
        response.content = Content([message])
        self.assertEqual(asyncio.run(_consume_sse(response))["text"], "hi")

    def test_aggregation_keeps_diagonal_and_excludes_missing_or_unmeasured_runs(self):
        config = load_config(CONFIG)
        config["total_requests"] = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, profile, condition = run.plan_serving_profile(
                config["model"], config["tp"], [], {},
                versions=RUNTIME_ANCHOR["versions"], origins=RUNTIME_ANCHOR["origins"])
            _dump(root / "experiment.json", {"config": config, "repetitions": 2,
                                              "serving_profile": profile, "serving_condition": condition})
            mixture = config["mixtures"][1]
            cohort = [{"arrival_sec": float(index - 1), "type": 0, "dataset_idx": index, "prompt": "hello",
                       "pool": "summarization" if index == 1 else "conversation"} for index in (1, 2)]
            schedule = root / "50-50" / "cohort.csv"
            _write_cohort(schedule, cohort)
            for bank in banks(config, op_indices="0,1"):
                candidate_dir = root / "50-50" / bank["name"]
                metadata = _candidate_manifest(config, mixture, bank, 2, _cohort_hash(cohort), cohort, schedule, condition)
                _dump(candidate_dir / "candidate.json", metadata)
                if bank["name"].endswith("-dp"):
                    continue
                result = {
                    "serving_profile": profile, "serving_condition": condition,
                    "energy_joules": 40 if bank["type"] == "shared" else 1,
                    "energy_num_samples": 2,
                    "99.9th_ttft": 1, "99.9th_tpot": 0.1,
                    "requests": [{"prompt_idx": index, "ok": True, "stream_complete": True,
                                  "pool": "summarization" if index == 1 else "conversation", "lane": "lane_a" if index == 1 else "lane_b"}
                                 for index in (1, 2)],
                    "benchmark_start_s": 1, "benchmark_end_s": 9,
                    "energy_per_npu_joules": {str(device): 10 if bank["type"] == "shared" else 0.25 for lane in config["lanes"] for device in lane["device_map"]},
                }
                for repetition in ((1, 2) if bank["type"] == "shared" else (1,)):
                    run_dir = candidate_dir / ("rep-%d" % repetition)
                    run_manifest = dict(metadata, repetition=repetition)
                    _dump(run_dir / "manifest.json", run_manifest)
                    _dump(run_dir / "result.json", dict(result, specialization_gap=run_manifest))
                    _dump(run_dir / "execution.json", {"returncode": 0})
                    readings = {str(device): {"power_sample": 1 if bank["type"] == "shared" else 0.025, "frequency": metadata["lane%d" % lane]["op"]["frequency_mhz"]}
                                for lane in (0, 1) for device in config["lanes"][lane]["device_map"]}
                    import json
                    raw_path = run_dir / "demo-state" / "power.jsonl"
                    raw_path.parent.mkdir()
                    raw_path.write_text("".join(json.dumps({"timestamp": timestamp, "per_npu": readings}) + "\n" for timestamp in (0, 10)), encoding="utf-8")
                    if bank["type"] == "shared":
                        self.assertTrue(_measurement_status(run_dir, result, metadata)["measurement_valid"])
                        self.assertFalse(_measurement_status(run_dir, dict(result, benchmark_end_s=11), metadata)["measurement_valid"])
                        self.assertFalse(_measurement_status(run_dir, dict(result, energy_num_samples=1), metadata)["measurement_valid"])
                        self.assertFalse(_measurement_status(run_dir, dict(result, energy_num_samples=0), metadata)["measurement_valid"])
                        self.assertFalse(_measurement_status(run_dir, dict(result, energy_joules=1), metadata)["measurement_valid"])
            aggregate(root, config)
            import csv
            with open(root / "banks.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertFalse(any(row["feasible"] == "True" for row in rows if row["bank_type"] == "specialized"))
            self.assertTrue(all(row["repetitions"] == "2" for row in rows))
            pd = next(row for row in rows if row["mapping"] == "pd")
            self.assertEqual(pd["valid_repetitions"], "1")
            with open(root / "summary.csv", newline="", encoding="utf-8") as handle:
                summary = next(row for row in csv.DictReader(handle) if row["mixture"] == "50-50")
            self.assertEqual(summary["best_specialized_bank"], summary["best_shared_bank"])
            self.assertEqual(float(summary["specialization_gap_j"]), 0)
            self.assertEqual(summary["best_specialized_mapping"], "neutral")
            self.assertEqual(summary["search_complete"], "False")
            result_path = root / "50-50" / "shared-0-0" / "rep-1" / "result.json"
            changed = gap._json(result_path)
            changed["serving_condition"] = "sp-" + "0" * 24
            _dump(result_path, changed)
            before = (root / "runs.csv").read_bytes()
            with self.assertRaisesRegex(ValueError, "serving condition"):
                aggregate(root, config)
            self.assertEqual((root / "runs.csv").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
