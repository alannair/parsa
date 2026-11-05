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

"""Task 1 catalog and configuration contract tests."""

import json
import asyncio
import importlib.util
import math
import sys
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from argparse import Namespace
from types import ModuleType
from unittest.mock import patch

from src.op_drift import load_catalog, load_drift_config
from src import op_drift as drift
from src.config import ControllerConfig
from src.profile_table import OperatingPointProfileTable
from src.profile_table import ProfileBinding
from src.cli import build_controller
from src.controller import EnergyAwareController, OverloadedError, SLO
from src.metrics import MetricsPipeline
from src.presets import Preset
from src.profile_table import DictProfileTable
from src.router_app import EnergyAwareRouter


def catalog_fixture(context=None):
    context = context or {
        "model": "fixture-model", "model_revision": "r1",
        "versions": {"vllm": "1", "vllm_ascend": "1"}, "tp": 1,
        "precision": "bf16", "max_num_batched_tokens": 1024,
        "device_cohort": "fixture-device", "serving_condition": "fixture-serving",
        "serving_profile_hash": "profile-hash", "predictor_id": "predictor-v1",
        "routing_policy": "legacy_ratio", "slo_definition": {"default": 1.0},
    }
    coeffs = dict(zip(("a_ttft", "b_ttft", "a_tbt", "b_tbt", "a_ep", "b_ep", "a_ed", "b_ed"), [1.0] * 8))
    return {
        "schema_version": "parsa-op-catalog-v1", "compatibility": context,
        "feature_schema": {"revision": "length-bins-v1", "prompt_edges": [64, 256],
                            "output_edges": [64, 256], "slo_classes": ["default"],
                            "slo_targets": {"default": 0.99}},
        "operating_points": {"op_a": {"frequency_mhz": 1300, "max_num_seqs": 10, "profile": coeffs}},
        "initial_bank": {"lane_a": "op_a"},
        "banks": {"bank_a": {"lanes": {"lane_a": "op_a"}, "regimes": []}},
        "transitions": [],
    }


def measured_regime(regime_id="fixture-regime"):
    return {
        "regime_id": regime_id,
        "arrival_rps": {"min": 0.5, "max": 2.0},
        "class_mix": {"p0:o0:default": {"min": 1.0, "max": 1.0}, "unknown": {"min": 0.0, "max": 0.0}},
        "power_w": {"lo": 200.0, "hi": 210.0},
        "capacity_rps_lo": 3.0,
        "slo_success_lo": {"default": 0.995},
        "calibration_revision": "synthetic-fixture-only",
        "source_run_ids": ["synthetic-fixture-run"],
    }


def multi_bank_fixture():
    catalog = catalog_fixture()
    op_a = catalog["operating_points"]["op_a"]
    catalog["operating_points"]["op_b"] = {
        "frequency_mhz": 1500, "max_num_seqs": 12,
        "profile": {key: 2.0 for key in op_a["profile"]},
    }
    catalog["initial_bank"].update(lane_b="op_a")
    catalog["banks"] = {
        "bank_a": {"lanes": {"lane_a": "op_a", "lane_b": "op_a"}, "regimes": [measured_regime()]},
        "bank_b": {"lanes": {"lane_a": "op_b", "lane_b": "op_a"}, "regimes": []},
    }
    catalog["transitions"] = [{
        "source_bank": "bank_a", "target_bank": "bank_b", "lane_id": "lane_a",
        "duration_hi_s": 10.0, "extra_energy_hi_j": 2.0,
        "remaining_capacity_rps_lo": 1.0, "max_backlog_requests": 3,
        "calibration_revision": "synthetic-fixture-only",
        "source_run_ids": ["synthetic-fixture-transition"],
    }]
    return catalog
class CatalogTests(unittest.TestCase):
    def test_off_defaults_and_serialization_compatibility(self):
        cfg = load_drift_config({})
        self.assertEqual(cfg["mode"], "off")
        self.assertEqual(cfg["window_s"], 120.0)
        self.assertNotIn("op_drift", ControllerConfig().to_dict())
        self.assertEqual(load_drift_config({"op_drift": {"catalog_path": "/missing/catalog.json"}})["mode"], "off")

    def test_shadow_config_requires_identity_and_catalog(self):
        with self.assertRaises(ValueError):
            load_drift_config({"op_drift": {"mode": "shadow"}})
        cfg = load_drift_config({"op_drift": {"mode": "shadow", "catalog_path": "catalog.json", "runtime_context": {"x": 1}}})
        self.assertEqual(cfg["mode"], "shadow")

    def test_live_is_explicitly_unavailable(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            load_drift_config({"op_drift": {"mode": "live"}})

    def test_invalid_mode_unknown_key_and_numeric_values(self):
        for value in ({"mode": "future"}, {"unexpected": True}, {"window_s": True}, {"margin_j": -1}, {"min_saving_fraction": 1}, {"evaluate_every_s": math.inf}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_drift_config({"op_drift": value})

    def test_loads_catalog_and_hashes_deterministically(self):
        context = catalog_fixture()["compatibility"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(catalog_fixture()), encoding="utf-8")
            first = load_catalog(path, context)
            second = load_catalog(path, context)
        self.assertEqual(first["identity_source"], "operator_declared")
        self.assertEqual(first["catalog_hash"], second["catalog_hash"])
        self.assertEqual(first["profile_hashes"], second["profile_hashes"])
        self.assertEqual(first["banks"]["bank_a"]["regimes"], [])

    def test_loads_multi_bank_regime_and_directed_transition(self):
        catalog = multi_bank_fixture()
        context = catalog["compatibility"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            loaded = load_catalog(path, context)
        self.assertEqual(set(loaded["banks"]), {"bank_a", "bank_b"})
        self.assertEqual(loaded["banks"]["bank_a"]["regimes"][0]["power_w"], {"lo": 200.0, "hi": 210.0})
        self.assertEqual(loaded["transitions"][0]["lane_id"], "lane_a")

    def test_hash_is_key_order_independent_but_content_sensitive(self):
        catalog = catalog_fixture()
        context = catalog["compatibility"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            first = load_catalog(path, context)["catalog_hash"]
            path.write_text(json.dumps(json.loads(json.dumps(catalog), object_pairs_hook=lambda pairs: dict(reversed(pairs)))), encoding="utf-8")
            reordered = load_catalog(path, context)["catalog_hash"]
            changed = catalog_fixture()
            changed["operating_points"]["op_a"]["frequency_mhz"] += 1
            path.write_text(json.dumps(changed), encoding="utf-8")
            content_changed = load_catalog(path, context)["catalog_hash"]
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, content_changed)

    def test_rejects_duplicate_json_keys_and_context_mismatch(self):
        context = catalog_fixture()["compatibility"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_catalog(path, context)
            path.write_text(json.dumps(catalog_fixture()), encoding="utf-8")
            altered = dict(context, model="other")
            with self.assertRaises(ValueError):
                load_catalog(path, altered)

    def test_rejects_invalid_catalog_units_and_references(self):
        context = catalog_fixture()["compatibility"]
        mutations = (
            lambda value: value["operating_points"]["op_a"].update(frequency_mhz=0),
            lambda value: value["operating_points"]["op_a"]["profile"].pop("b_ep"),
            lambda value: value["operating_points"]["op_a"]["profile"].update(unknown_coefficient=1.0),
            lambda value: value["initial_bank"].update(lane_a="missing-op"),
            lambda value: value["feature_schema"].update(prompt_edges=[256, 64]),
            lambda value: value["banks"]["bank_a"].update(lanes={"wrong_lane": "op_a"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for mutate in mutations:
                candidate = catalog_fixture()
                mutate(candidate)
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                    load_catalog(path, context)

    def test_rejects_type_aliases_and_missing_or_empty_identity(self):
        catalog = catalog_fixture()
        context = catalog["compatibility"]
        cases = []
        bad_context = dict(context, tp=True)
        cases.append((catalog, bad_context))
        bad_context = dict(context, max_num_batched_tokens=1024.0)
        cases.append((catalog, bad_context))
        bad_catalog = catalog_fixture()
        bad_catalog["compatibility"]["slo_definition"] = {}
        cases.append((bad_catalog, bad_catalog["compatibility"]))
        bad_catalog = catalog_fixture()
        bad_catalog["compatibility"]["slo_definition"] = None
        cases.append((bad_catalog, bad_catalog["compatibility"]))
        bad_catalog = catalog_fixture()
        bad_catalog["compatibility"]["slo_definition"] = {"nested": [True]}
        cases.append((bad_catalog, bad_catalog["compatibility"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for candidate, runtime in cases:
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                    load_catalog(path, runtime)

    def test_rejects_nested_slo_definition_mismatch(self):
        catalog = catalog_fixture()
        context = json.loads(json.dumps(catalog["compatibility"]))
        context["slo_definition"]["default"] = 0.95
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_catalog(path, context)

    def test_rejects_empty_lists_in_slo_definition_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for definition in ({"default": []}, {"default": [[]]}):
                catalog = catalog_fixture()
                catalog["compatibility"]["slo_definition"] = definition
                path.write_text(json.dumps(catalog), encoding="utf-8")
                with self.subTest(definition=definition), self.assertRaises(ValueError):
                    load_catalog(path, catalog["compatibility"])

    def test_rejects_wrong_power_shape_and_invalid_slo_class_elements(self):
        context = catalog_fixture()["compatibility"]
        wrong_power = multi_bank_fixture()
        wrong_power["banks"]["bank_a"]["regimes"][0]["power_w"] = {"min": 200, "max": 210}
        bad_classes = catalog_fixture()
        bad_classes["feature_schema"]["slo_classes"] = [{}]
        for candidate in (wrong_power, bad_classes):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "catalog.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_catalog(path, candidate["compatibility"] if candidate is bad_classes else context)

    def test_rejects_infeasible_class_mix_and_duplicate_ids_or_maps(self):
        bad_mix = multi_bank_fixture()
        bad_mix["banks"]["bank_a"]["regimes"][0]["class_mix"] = {}
        low_mix = multi_bank_fixture()
        low_mix["banks"]["bank_a"]["regimes"][0]["class_mix"] = {"p0:o0:default": {"min": 0.0, "max": 0.2}}
        high_mix = multi_bank_fixture()
        high_mix["banks"]["bank_a"]["regimes"][0]["class_mix"] = {"p0:o0:default": {"min": 0.8, "max": 1.0}, "unknown": {"min": 0.8, "max": 1.0}}
        duplicate_regime = multi_bank_fixture()
        duplicate_regime["banks"]["bank_a"]["regimes"].append(measured_regime())
        duplicate_map = multi_bank_fixture()
        duplicate_map["banks"]["bank_c"] = {"lanes": dict(duplicate_map["banks"]["bank_a"]["lanes"]), "regimes": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            for candidate in (bad_mix, low_mix, high_mix, duplicate_regime, duplicate_map):
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    load_catalog(path, candidate["compatibility"])

    def test_prepared_profile_is_immutable_until_generation_activation(self):
        catalog = catalog_fixture()
        catalog["operating_points"]["op_b"] = {
            "frequency_mhz": 1500, "max_num_seqs": 12,
            "profile": {key: 2.0 for key in catalog["operating_points"]["op_a"]["profile"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            validated = load_catalog(path, catalog["compatibility"])
        table = OperatingPointProfileTable(validated)
        before = table.estimate("lane_a", 0, 1).ttft_s
        prepared = table.prepare_binding("lane_a", "op_b", 1)
        self.assertEqual(table.estimate("lane_a", 0, 1).ttft_s, before)
        with self.assertRaises(TypeError):
            prepared.coefficients["a_ttft"] = 4
        table.activate_binding(prepared, expected_generation=0)
        self.assertNotEqual(table.estimate("lane_a", 0, 1).ttft_s, before)
        with self.assertRaises(ValueError):
            table.activate_binding(prepared, expected_generation=0)

    def test_rejects_forged_or_mutable_profile_bindings(self):
        catalog = multi_bank_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            table = OperatingPointProfileTable(load_catalog(path, catalog["compatibility"]))
        prepared = table.prepare_binding("lane_a", "op_b", 1)
        for forged in (
            prepared._replace(op_id="missing"),
            prepared._replace(generation=float("nan")),
            prepared._replace(coefficients={**prepared.coefficients, "a_ttft": 99}),
            prepared._replace(generation=0),
        ):
            with self.subTest(forged=forged), self.assertRaises((TypeError, ValueError, KeyError)):
                table.activate_binding(forged, expected_generation=0)
        owned = dict(prepared.coefficients)
        alias = ProfileBinding(prepared.lane_id, prepared.op_id, 1, owned)
        table.activate_binding(alias, expected_generation=0)
        owned["a_ttft"] = 99
        self.assertEqual(table.estimate("lane_a", 0, 1).ttft_s, 2.0)

    def test_config_positional_compatibility_shadow_roundtrip_and_early_cli_rejection(self):
        original = ControllerConfig({}, {}, {}, {}, {}, {}, {}, {}, 37)
        self.assertEqual(original.random_seed, 37)
        self.assertEqual(original.op_drift["mode"], "off")
        shadow = ControllerConfig.from_routing_kwargs({"op_drift": {"mode": "shadow", "catalog_path": "synthetic.json", "runtime_context": {"model": "fixture"}}})
        self.assertEqual(shadow.to_dict()["op_drift"]["mode"], "shadow")
        for mode in ("bad", "live"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ControllerConfig.from_routing_kwargs({"op_drift": {"mode": mode}})
        args = Namespace(routing_kwargs={"op_drift": {"mode": "live"}})
        with patch("src.cli.load_presets_json") as load_presets, patch("src.cli.build_output_len_predictor") as build_predictor:
            with self.assertRaises(ValueError):
                build_controller(args)
            load_presets.assert_not_called()
            build_predictor.assert_not_called()


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def window_config(**overrides):
    return load_drift_config({"op_drift": {
        "window_s": 20.0, "evaluate_every_s": 5.0, "min_arrivals": 10,
        "max_observations": 3, **overrides,
    }})


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(drift, "DemandWindow"), "Task 2 DemandWindow is missing")
        self.clock = FakeClock()
        self.schema = catalog_fixture()["feature_schema"]
        self.window = drift.DemandWindow(window_config(), self.schema, clock=self.clock)

    def arrival(self, request_id, at=0.0, features=None, slo="default", context=None):
        return drift.observe_arrival(self.window, request_id, at, features, slo, context)

    def test_offered_counts_and_mix_survive_diagnostic_cap(self):
        for i in range(2000):
            self.arrival(str(i), features={"l_in": 2, "l_out_raw": 3})
        for i in range(500):
            context = self.arrival("unknown-%d" % i)
            drift.observe_completion(self.window, context.request_id, 0, "rejected", context)
        snapshot = drift.window_snapshot(self.window, 20)
        self.assertEqual(snapshot["arrival_count"], 2500)
        self.assertEqual(snapshot["arrival_rps"], 125)
        self.assertEqual(snapshot["class_mix"], {"p0:o0:default": 0.8, "unknown": 0.2})
        self.assertEqual(snapshot["unknown_count"], 500)
        self.assertEqual(snapshot["diagnostic_count"], 3)
        self.assertEqual(snapshot["outcome_counts"]["rejected"], 500)
        # A per-request registry would grow here; the collector owns only buckets
        # and capped numeric diagnostics, never the caller-owned contexts/IDs.
        self.assertLessEqual(len(self.window.buckets), 1024)
        self.assertFalse(hasattr(self.window, "__dict__"))
        for slot in self.window.__slots__:
            value = getattr(self.window, slot)
            if isinstance(value, (dict, list, set, tuple)):
                self.assertNotIn("unknown-499", value)

    def test_unknown_is_enriched_once_in_its_original_bucket(self):
        context = self.arrival("request")
        self.assertEqual(drift.window_snapshot(self.window, 1)["unknown_count"], 1)
        self.assertIs(self.arrival("request", 2, {"l_in": 64, "l_out_raw": 256}, context=context), context)
        self.arrival("request", 3, {"l_in": 1, "l_out_raw": 1}, context=context)
        snapshot = drift.window_snapshot(self.window, 4)
        self.assertEqual(snapshot["arrival_count"], 1)
        self.assertEqual(snapshot["class_mix"], {"p1:o2:default": 1.0})
        self.assertEqual(snapshot["diagnostic_count"], 1)
        self.assertEqual(context.arrival_at_s, 0)
        self.assertFalse(hasattr(context, "text"))

    def test_half_open_edges_and_invalid_features_or_slo_remain_unknown(self):
        cases = [
            ({"l_in": 0, "l_out_raw": 0}, "default"),
            ({"l_in": 64, "l_out_raw": 64}, "default"),
            (Namespace(l_in=256, l_out_raw=256, text="not retained"), "default"),
            ({"l_in": 63.9, "l_out_raw": 255.9}, "default"),
            ({"l_in": math.nan, "l_out_raw": 1}, "default"),
            ({"l_in": 1, "l_out_raw": math.inf}, "default"),
            ({"l_in": -1, "l_out_raw": 1}, "default"),
            ({"l_in": True, "l_out_raw": 1}, "default"),
            ({"l_in": 1, "l_out": 1}, "default"),
            ({"l_in": 1, "l_out_raw": 1}, "arbitrary-user-string"),
            ({"l_in": 1, "l_out_raw": 1}, []),
        ]
        for i, (features, slo) in enumerate(cases):
            self.arrival(str(i), features=features, slo=slo)
        snapshot = drift.window_snapshot(self.window, 1)
        self.assertEqual(snapshot["unknown_count"], 7)
        self.assertEqual(snapshot["class_mix"], {
            "p0:o0:default": 1 / 11, "p1:o1:default": 1 / 11,
            "p2:o2:default": 1 / 11, "p0:o1:default": 1 / 11, "unknown": 7 / 11,
        })
        self.assertEqual(snapshot["slo_counts"], {"default": 9})
        self.assertEqual(snapshot["slo_unknown_count"], 2)

    def test_enrichment_can_supply_slo_once_without_recounting(self):
        schema = deepcopy(self.schema)
        schema["slo_classes"].append("urgent")
        schema["slo_targets"]["urgent"] = 0.999
        self.window = drift.DemandWindow(window_config(), schema, self.clock)
        context = self.arrival("request", slo="unclassified")
        self.arrival("request", 1, {"l_in": 1, "l_out_raw": 1}, "urgent", context)
        snapshot = drift.window_snapshot(self.window, 2)
        self.assertEqual(snapshot["arrival_count"], 1)
        self.assertEqual(snapshot["slo_counts"], {"urgent": 1})
        self.assertEqual(snapshot["class_mix"], {"p0:o0:urgent": 1.0})

    def test_catalog_slo_named_unknown_is_distinct_from_an_unclassified_slo(self):
        schema = deepcopy(self.schema)
        schema["slo_classes"].append("unknown")
        schema["slo_targets"]["unknown"] = 0.9
        self.window = drift.DemandWindow(window_config(), schema, self.clock)
        self.arrival("known", features={"l_in": 1, "l_out_raw": 1}, slo="unknown")
        self.arrival("unclassified", slo="not-in-catalog")
        snapshot = drift.window_snapshot(self.window, 1)
        self.assertEqual(snapshot["class_mix"], {"p0:o0:unknown": 0.5, "unknown": 0.5})
        self.assertEqual(snapshot["slo_counts"], {"unknown": 1})
        self.assertEqual(snapshot["slo_unknown_count"], 1)

    def test_expiry_recycled_bucket_and_expired_context_cannot_change_new_arrival(self):
        context = self.arrival("old")
        drift.window_snapshot(self.window, 25)
        self.arrival("new", 25, {"l_in": 1, "l_out_raw": 1})
        self.arrival("old", 25, {"l_in": 256, "l_out_raw": 256}, context=context)
        snapshot = drift.window_snapshot(self.window, 26)
        self.assertEqual(snapshot["arrival_count"], 1)
        self.assertEqual(snapshot["class_mix"], {"p0:o0:default": 1.0})
        self.assertEqual(snapshot["diagnostic_count"], 1)
        self.assertEqual(snapshot["exposure_s"], 21)

    def test_bucket_rounded_exposure_and_cold_start(self):
        for i in range(10):
            self.arrival(str(i))
        self.assertFalse(drift.window_snapshot(self.window, 19.99)["ready"])
        self.assertTrue(drift.window_snapshot(self.window, 20)["ready"])
        snapshot = drift.window_snapshot(self.window, 24)
        self.assertEqual(snapshot["exposure_s"], 24)
        self.assertEqual(snapshot["arrival_rps"], 10 / 24)
        self.assertEqual(snapshot["exposure_start_s"], 0)
        self.assertEqual(snapshot["bucket_width_s"], 5)
        self.assertNotIn("headroom", snapshot)
        self.assertEqual(drift.window_snapshot(self.window, 25)["arrival_count"], 0)
        self.assertFalse(drift.window_snapshot(self.window, 25)["ready"])

    def test_outcomes_are_separate_idempotent_and_never_correct_predictions(self):
        statuses = ["completed", "completed", "failed", "rejected", "cancelled"]
        for i, status in enumerate(statuses):
            context = self.arrival(str(i), features={"l_in": 1, "l_out_raw": 1})
            outcome = {"status": status, "slo_met": True if i == 0 else False if i == 1 else None,
                       "l_out_raw": 9999, "text": "not retained"}
            self.assertTrue(drift.observe_completion(self.window, str(i), 0, outcome, context))
            self.assertFalse(drift.observe_completion(self.window, str(i), 0, outcome, context))
        self.arrival("unfinished")
        snapshot = drift.window_snapshot(self.window, 1)
        self.assertEqual(snapshot["arrival_count"], 6)
        self.assertEqual(snapshot["class_mix"], {"p0:o0:default": 5 / 6, "unknown": 1 / 6})
        self.assertEqual(snapshot["outcome_counts"], {"completed": 2, "failed": 1, "rejected": 1, "cancelled": 1})
        self.assertEqual(snapshot["slo_outcome_counts"], {"known": 2, "unknown": 3, "success": 1, "failure": 1})

    def test_completed_unknown_cannot_be_enriched_from_future_information(self):
        context = self.arrival("request")
        drift.observe_completion(self.window, "request", 1, "completed", context)
        self.arrival("request", 2, {"l_in": 1, "l_out_raw": 1}, context=context)
        self.assertEqual(drift.window_snapshot(self.window, 3)["unknown_count"], 1)

    def test_completion_of_expired_arrival_counts_at_completion_time(self):
        context = self.arrival("request")
        drift.observe_completion(self.window, "request", 30, {"status": "failed", "slo_met": True}, context)
        snapshot = drift.window_snapshot(self.window, 31)
        self.assertEqual(snapshot["arrival_count"], 0)
        self.assertEqual(snapshot["outcome_counts"]["failed"], 1)
        self.assertEqual(snapshot["slo_outcome_counts"]["success"], 0)
        self.assertEqual(snapshot["slo_outcome_counts"]["failure"], 1)

    def test_context_scope_and_clock_validation(self):
        context = self.arrival("request")
        other = drift.DemandWindow(window_config(), self.schema, self.clock)
        with self.assertRaises(ValueError):
            drift.observe_arrival(other, "request", 0, context=context)
        with self.assertRaises(ValueError):
            self.arrival("different", context=context)
        with self.assertRaises(ValueError):
            drift.observe_completion(self.window, "request", 0, "completed")
        with self.assertRaises(ValueError):
            drift.observe_completion(self.window, "request", 0, "unrecognized", context)
        for time in (math.nan, math.inf, True, -1):
            with self.subTest(time=time), self.assertRaises(ValueError):
                drift.window_snapshot(self.window, time)
        self.clock.now = 4
        self.assertEqual(drift.window_snapshot(self.window, None)["at_s"], 4)
        with self.assertRaises(ValueError):
            drift.DemandWindow(window_config(), self.schema, FakeClock(math.nan))

    def test_bucket_count_is_bounded_even_for_tiny_evaluation_interval(self):
        window = drift.DemandWindow(window_config(evaluate_every_s=0.00001), self.schema, self.clock)
        self.assertLessEqual(len(window.buckets), 1024)
        drift.observe_arrival(window, "request", 0)
        self.assertTrue(drift.window_snapshot(window, 20)["exposure_s"] >= 20)
        self.assertEqual(drift.window_snapshot(window, 1000000)["arrival_count"], 0)

    def test_snapshot_identity_is_stable_and_burst_bound_is_not_average(self):
        for i in range(10):
            self.arrival(str(i))
        first = drift.window_snapshot(self.window, 20)
        self.assertEqual(first["demand_bound_rps"], 2)
        self.assertEqual(first["observation_id"], drift.window_snapshot(self.window, 20)["observation_id"])
        second = drift.window_snapshot(self.window, 21)
        self.assertNotEqual(first["observation_id"], second["observation_id"])
        self.assertEqual(first["evaluation_index"], second["evaluation_index"])
        self.arrival("boundary", 25)
        self.assertIsNone(drift.window_snapshot(self.window, 25)["demand_bound_rps"])


def validated_catalog(raw):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catalog.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return load_catalog(path, raw["compatibility"])


def shadow_kv_snapshot(lane, *, ts=None, active=2):
    """Synthetic complete producer-shaped payload; no backend or telemetry fetch."""
    import time
    from src.kv_projection import KVProjectionSnapshot, SCHEMA_VERSION
    quantiles = ("p50", "p90", "p95")
    return KVProjectionSnapshot.from_payload({
        "schema_version": SCHEMA_VERSION, "ts_s": time.time() if ts is None else ts,
        "engine_id": lane, "block_size_tokens": 16,
        "capacity": {"npu_blocks": 100, "host_blocks": 0},
        "current": {"npu_used_blocks": 10, "npu_free_blocks": 90,
                    "host_used_blocks": 0, "active_requests": active},
        "projection_basis": {"mode": "router_hints_plus_runtime", "horizon_iters": 1,
                             "quantiles": [0.5, 0.9, 0.95]},
        "projection": {q: [{"iter_offset": 0, "batch_size": active, "npu_kv_blocks": 10}]
                       for q in quantiles},
        "summary": {"peak_npu_blocks": dict.fromkeys(quantiles, 10),
                    "peak_utilization": dict.fromkeys(quantiles, .1),
                    "peak_batch_size": dict.fromkeys(quantiles, active),
                    "time_to_below_utilization_ms": {"0.80": 0, "0.70": 0, "0.60": 0}},
    }, expected_engine_id=lane)


@contextmanager
def fake_streaming_responses():
    """ASGI hand-off double only, not a FastAPI/Starlette integration test."""
    class Response:
        def __init__(self, content, status_code=200, headers=None, media_type=None):
            self.body_iterator = content
            self.status_code = status_code
            self.headers = headers
            self.media_type = media_type

    class StreamingResponse(Response):
        async def __call__(self, scope, receive, send):
            if (await receive())["type"] == "http.disconnect":
                return  # Deliberately return before the first __anext__.
            await send({"type": "http.response.start", "status": self.status_code, "headers": []})
            async for chunk in self.body_iterator:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    modules = {name: ModuleType(name) for name in
               ("fastapi", "fastapi.responses", "starlette", "starlette.responses")}
    for name in ("fastapi", "starlette"):
        modules[name].__path__ = []
        modules[name].responses = modules[name + ".responses"]
        modules[name + ".responses"].Response = Response
        modules[name + ".responses"].StreamingResponse = StreamingResponse
    with patch.dict(sys.modules, modules):
        yield Response, StreamingResponse


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(drift, "choose_retune"), "Task 2 pure selector is missing")
        self.clock = FakeClock()
        self.config = window_config(min_residence_s=10, cooldown_s=10)
        self.raw = multi_bank_fixture()
        candidate = measured_regime("candidate")
        candidate["power_w"] = {"lo": 170, "hi": 180}
        self.raw["banks"]["bank_b"]["regimes"] = [candidate]
        self.raw["transitions"][0].update(duration_hi_s=60, extra_energy_hi_j=2000,
                                            remaining_capacity_rps_lo=3, max_backlog_requests=100)
        self.catalog = validated_catalog(self.raw)
        self.window = drift.DemandWindow(self.config, self.catalog["feature_schema"], self.clock)
        self.next_arrival = 0
        self.state = drift.make_policy_state(self.config, self.catalog, "bank_a",
                                             {"lane_a": 4, "lane_b": 2}, clock=self.clock)

    def sample(self, at, extra=0):
        while self.next_arrival < at:
            i = self.next_arrival
            drift.observe_arrival(self.window, str(i), i, {"l_in": 10, "l_out_raw": 10})
            self.next_arrival += 1
        for i in range(extra):
            drift.observe_arrival(self.window, "extra-%s-%s" % (at, i), at - 0.1,
                                  {"l_in": 10, "l_out_raw": 10})
        self.clock.now = at
        observation = drift.window_snapshot(self.window, at)
        observation["headroom"] = {"fresh": True, "at_s": at,
                                    "healthy_lane_ids": ["lane_a", "lane_b"], "backlog_requests": 0}
        self.state["observation"] = observation
        return self.estimates()

    def estimates(self):
        return {bank: drift.estimate_bank(self.catalog, bank, self.state["observation"])
                for bank in self.catalog["banks"]}

    def choose(self, estimates=None, at=None):
        return drift.choose_retune("bank_a", self.estimates() if estimates is None else estimates,
                                   self.catalog["transitions"], self.state,
                                   self.clock.now if at is None else at)

    def evaluate(self, at, extra=0):
        result = self.choose(self.sample(at, extra))
        self.state = result["policy_state"]
        return result

    def reload(self):
        self.catalog = validated_catalog(self.raw)
        self.state["catalog_hash"] = self.catalog["catalog_hash"]

    def test_benefit_persistence_and_shadow_leave_actual_binding_unchanged(self):
        table = OperatingPointProfileTable(self.catalog)
        binding_before = table.estimate("lane_a", 1, 1).ttft_s
        for at in (20, 25):
            result = self.evaluate(at)
            self.assertEqual((result["action"], result["reason"]), ("NO_CHANGE", "persistence"))
            self.assertEqual(result["benefit_lo_j"], 16800)
        estimates = self.sample(30)
        before = deepcopy({key: value for key, value in self.state.items() if key != "clock"})
        result = self.choose(estimates)
        self.assertEqual((result["action"], result["reason"]), ("RETUNE", "shadow_recommendation"))
        self.assertEqual((result["target_bank"], result["lane_id"], result["expected_generation"]), ("bank_b", "lane_a", 4))
        self.assertEqual((result["benefit_lo_j"], result["extra_energy_hi_j"], result["useful_horizon_s"]), (16800, 2000, 840))
        self.assertEqual(result["required_saving_j"], 9000)
        self.assertEqual(before, {key: value for key, value in self.state.items() if key != "clock"})
        self.assertEqual(result["policy_state"]["current_bank"], "bank_a")
        self.assertEqual(result["policy_state"]["generations"], {"lane_a": 4, "lane_b": 2})
        self.assertIsNone(result["policy_state"]["last_commit_s"])
        self.assertIsNone(result["policy_state"]["last_attempt_s"])
        self.assertEqual(result["policy_state"]["last_proposal_s"], 30)
        self.assertEqual(table.estimate("lane_a", 1, 1).ttft_s, binding_before)

    def test_199_watts_does_not_cover_switching_cost(self):
        self.raw["banks"]["bank_b"]["regimes"][0]["power_w"] = {"lo": 190, "hi": 199}
        self.reload()
        result = self.evaluate(20)
        self.assertEqual((result["action"], result["reason"]), ("NO_CHANGE", "insufficient_savings"))
        self.assertEqual(result["benefit_lo_j"], 840)

    def test_both_saving_thresholds_are_strict_and_transition_time_is_excluded(self):
        self.sample(20)
        for margin, fraction, horizon in ((14800, 0, 900), (0, 0.1, 900), (0, 0, 60)):
            with self.subTest(margin=margin, fraction=fraction, horizon=horizon):
                self.state["config"].update(margin_j=margin, min_saving_fraction=fraction, benefit_horizon_s=horizon)
                result = self.choose()
                self.assertEqual(result["reason"], "insufficient_savings")
                self.assertEqual(result["action"], "NO_CHANGE")
        self.assertEqual(result["benefit_lo_j"], 0)

    def test_missing_and_out_of_coverage_regimes_return_none(self):
        self.sample(20)
        self.assertIsNone(drift.estimate_bank(self.catalog, "absent", self.state["observation"]))
        self.raw["banks"]["bank_b"]["regimes"] = []
        self.reload()
        self.assertIsNone(self.estimates()["bank_b"])
        self.assertEqual(self.choose()["reason"], "no_calibration")
        self.raw["banks"]["bank_a"]["regimes"][0]["arrival_rps"] = {"min": 0, "max": 0.5}
        self.reload()
        self.assertIsNone(self.estimates()["bank_a"])

    def test_all_overlapping_rows_are_combined_conservatively_with_provenance(self):
        second = measured_regime("overlap")
        second.update(power_w={"lo": 190, "hi": 220}, capacity_rps_lo=0.8,
                      slo_success_lo={"default": 0.98}, calibration_revision="second-revision",
                      source_run_ids=["second-source"])
        self.raw["banks"]["bank_a"]["regimes"].append(second)
        self.reload()
        estimate = self.sample(20)["bank_a"]
        self.assertEqual(estimate["power_w"], {"lo": 190, "hi": 220})
        self.assertEqual(estimate["capacity_rps_lo"], 0.8)
        self.assertEqual(estimate["slo_success_lo"], {"default": 0.98})
        self.assertFalse(estimate["steady_feasible"])
        self.assertEqual(estimate["regime_ids"], ["fixture-regime", "overlap"])
        self.assertEqual(estimate["calibration_revisions"], ["second-revision", "synthetic-fixture-only"])
        self.assertEqual(estimate["source_run_ids"], ["second-source", "synthetic-fixture-run"])
        self.assertEqual(self.choose()["reason"], "baseline_capacity")

    def test_mix_minima_are_checked_even_for_unobserved_classes(self):
        self.sample(20)
        mix = self.raw["banks"]["bank_a"]["regimes"][0]["class_mix"]
        mix["p0:o0:default"]["min"] = 0
        mix["p1:o0:default"] = {"min": 0.1, "max": 1}
        self.reload()
        self.assertIsNone(self.estimates()["bank_a"])

    def test_unknown_traffic_needs_explicit_nonzero_coverage(self):
        self.sample(20)
        # Add offered unknown traffic without using diagnostic retention for counts.
        drift.observe_arrival(self.window, "unknown", 20)
        self.state["observation"] = drift.window_snapshot(self.window, 21)
        self.clock.now = 21
        self.assertIsNone(self.estimates()["bank_a"])
        mix = self.raw["banks"]["bank_a"]["regimes"][0]["class_mix"]
        mix["p0:o0:default"]["min"] = 0
        del mix["unknown"]
        self.reload()
        self.assertIsNone(self.estimates()["bank_a"])
        mix["unknown"] = {"min": 0, "max": 1}
        self.reload()
        self.assertIsNotNone(self.estimates()["bank_a"])
        mix["unknown"]["min"] = 0.1
        self.reload()
        self.assertIsNone(self.estimates()["bank_a"])

    def test_mixed_slo_classes_and_unknown_slo_enforce_all_active_targets(self):
        schema = self.raw["feature_schema"]
        schema["slo_classes"].append("urgent")
        schema["slo_targets"]["urgent"] = 0.999
        for bank in self.raw["banks"].values():
            regime = bank["regimes"][0]
            regime["slo_success_lo"]["urgent"] = 0.9
            regime["class_mix"] = {name: {"min": 0, "max": 1}
                                    for name in ("p0:o0:default", "p0:o0:urgent", "unknown")}
        self.reload()
        self.window = drift.DemandWindow(self.config, schema, self.clock)
        self.sample(20)
        self.assertTrue(self.estimates()["bank_a"]["steady_feasible"])
        drift.observe_arrival(self.window, "urgent", 20, {"l_in": 1, "l_out_raw": 1}, "urgent")
        self.state["observation"] = drift.window_snapshot(self.window, 21)
        self.assertFalse(self.estimates()["bank_a"]["steady_feasible"])
        self.assertEqual(self.estimates()["bank_a"]["active_slo_classes"], ["default", "urgent"])
        other = drift.DemandWindow(self.config, schema, FakeClock())
        for i in range(20):
            drift.observe_arrival(other, str(i), i, slo_class="unsupported")
        estimate = drift.estimate_bank(self.catalog, "bank_a", drift.window_snapshot(other, 20))
        self.assertIsNotNone(estimate)
        self.assertFalse(estimate["steady_feasible"])

    def test_named_unknown_slo_uses_its_target_without_activating_unobserved_classes(self):
        self.raw["feature_schema"]["slo_classes"].append("unknown")
        self.raw["feature_schema"]["slo_targets"]["unknown"] = 0.9
        for bank in self.raw["banks"].values():
            regime = bank["regimes"][0]
            regime["class_mix"] = {"p0:o0:unknown": {"min": 1, "max": 1}}
            regime["slo_success_lo"] = {"default": 0, "unknown": 0.95}
        self.reload()
        window = drift.DemandWindow(self.config, self.catalog["feature_schema"], FakeClock())
        for i in range(20):
            drift.observe_arrival(window, str(i), i, {"l_in": 1, "l_out_raw": 1}, "unknown")
        observation = drift.window_snapshot(window, 20)
        estimate = drift.estimate_bank(self.catalog, "bank_a", observation)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate["active_slo_classes"], ["unknown"])
        self.assertTrue(estimate["steady_feasible"])

    def test_observation_schema_invalid_mix_and_cross_snapshot_comparison_are_rejected(self):
        estimates = self.sample(20)
        for field, value in (("feature_revision", "future-revision"), ("feature_schema_id", "different-bins"),
                             ("class_mix", {"arbitrary": 1}), ("arrival_rps", math.nan),
                             ("slo_counts", {"arbitrary": 20})):
            with self.subTest(field=field):
                observation = deepcopy(self.state["observation"])
                observation[field] = value
                self.assertIsNone(drift.estimate_bank(self.catalog, "bank_a", observation))
        # A finite self-consistent identity reaches the deeper ratio consistency guard.
        malformed = deepcopy(self.state["observation"])
        malformed["arrival_rps"] = 999.0
        malformed["observation_id"] = drift._hash({key: value for key, value in malformed.items()
                                                     if key not in ("observation_id", "headroom")})
        self.assertTrue(drift._snapshot_identity(malformed))
        self.assertIsNone(drift.estimate_bank(self.catalog, "bank_a", malformed))
        for field in ("observation_id", "coverage_id", "catalog_hash"):
            with self.subTest(field=field):
                mismatched = deepcopy(estimates)
                mismatched["bank_b"][field] = "different"
                self.assertEqual(self.choose(mismatched)["reason"], "coverage_mismatch")
        estimates["bank_a"] = None
        self.assertEqual(self.choose(estimates)["reason"], "no_calibration")

    def test_cold_start_and_low_samples_have_distinct_abstentions(self):
        self.assertEqual(self.evaluate(10)["reason"], "cold_start")
        self.state["config"]["min_arrivals"] = 30
        self.assertEqual(self.evaluate(20)["reason"], "low_samples")

    def test_candidate_and_baseline_slo_and_capacity_are_hard_constraints(self):
        self.sample(20)
        for bank, field, value, reason in (
            ("bank_a", "slo_success_lo", {"default": 0.98}, "baseline_slo"),
            ("bank_b", "slo_success_lo", {"default": 0.98}, "candidate_slo"),
            ("bank_b", "capacity_rps_lo", 0.9, "candidate_capacity"),
        ):
            with self.subTest(reason=reason):
                original = deepcopy(self.raw["banks"][bank]["regimes"][0])
                self.raw["banks"][bank]["regimes"][0][field] = value
                self.reload()
                self.assertEqual(self.choose()["reason"], reason)
                self.raw["banks"][bank]["regimes"][0] = original

    def test_headroom_is_explicit_fresh_finite_and_complete(self):
        self.sample(20)
        cases = [
            (None, "headroom_missing"), ({}, "headroom_invalid"),
            ({"fresh": False, "at_s": 20, "healthy_lane_ids": ["lane_a", "lane_b"], "backlog_requests": 0}, "stale_headroom"),
            ({"fresh": True, "at_s": 14, "healthy_lane_ids": ["lane_a", "lane_b"], "backlog_requests": 0}, "stale_headroom"),
            ({"fresh": True, "at_s": 21, "healthy_lane_ids": ["lane_a", "lane_b"], "backlog_requests": 0}, "stale_headroom"),
        ]
        good = deepcopy(self.state["observation"]["headroom"])
        for value in (math.nan, math.inf, -1, True, None):
            cases.append(({**good, "backlog_requests": value}, "headroom_invalid"))
        cases.append(({key: value for key, value in good.items() if key != "backlog_requests"}, "headroom_invalid"))
        for headroom, reason in cases:
            with self.subTest(headroom=headroom):
                self.state["observation"]["headroom"] = headroom
                result = self.choose()
                self.assertEqual((result["action"], result["reason"]), ("NO_CHANGE", reason))

    def test_healthy_residual_lanes_and_backlog_envelope_are_required(self):
        self.sample(20)
        self.state["observation"]["headroom"]["healthy_lane_ids"] = ["lane_a"]
        self.assertEqual(self.choose()["reason"], "unhealthy_residual_lanes")
        self.state["observation"]["headroom"]["healthy_lane_ids"] = ["lane_a", "lane_b"]
        self.state["observation"]["headroom"]["backlog_requests"] = 101
        self.assertEqual(self.choose()["reason"], "transition_backlog")
        self.state["observation"]["headroom"]["backlog_requests"] = 0
        self.raw["transitions"][0]["remaining_capacity_rps_lo"] = 0.5
        self.reload()
        self.assertEqual(self.choose()["reason"], "residual_capacity")

    def test_recent_burst_bound_not_just_mean_drives_backlog_check(self):
        for i in range(20):
            drift.observe_arrival(self.window, str(i), 0, {"l_in": 10, "l_out_raw": 10})
        self.clock.now = 20
        observation = drift.window_snapshot(self.window, 20)
        observation["headroom"] = {"fresh": True, "at_s": 20, "healthy_lane_ids": ["lane_a", "lane_b"], "backlog_requests": 0}
        self.state["observation"] = observation
        self.raw["transitions"][0]["max_backlog_requests"] = 10
        self.reload()
        result = self.choose()
        self.assertEqual(observation["arrival_rps"], 1)
        self.assertEqual(observation["demand_bound_rps"], 4)
        self.assertEqual(result["reason"], "transition_backlog")
        self.assertEqual(result["expected_backlog_requests"], 60)

    def test_duplicate_transition_records_cannot_cherry_pick_cost_or_capacity(self):
        self.sample(20)
        expensive = deepcopy(self.raw["transitions"][0])
        expensive.update(duration_hi_s=120, extra_energy_hi_j=30000, remaining_capacity_rps_lo=1,
                         max_backlog_requests=0, source_run_ids=["expensive-source"])
        self.raw["transitions"].append(expensive)
        self.reload()
        result = self.choose()
        self.assertEqual(result["reason"], "insufficient_savings")
        self.assertEqual((result["duration_hi_s"], result["extra_energy_hi_j"], result["benefit_lo_j"]), (120, 30000, 15600))
        self.assertEqual(result["transition_source_run_ids"], ["expensive-source", "synthetic-fixture-transition"])
        expensive["remaining_capacity_rps_lo"] = 0.5
        self.reload()
        self.assertEqual(self.choose()["reason"], "residual_capacity")

    def test_residence_attempt_commit_and_proposal_cooldown_use_actual_times(self):
        self.sample(20)
        self.state["last_commit_s"] = 19
        self.assertEqual(self.choose()["reason"], "min_residence")
        self.state["config"]["min_residence_s"] = 1
        self.state["last_commit_s"] = 12
        self.assertEqual(self.choose()["reason"], "cooldown")
        self.state["last_commit_s"] = None
        self.state["last_attempt_s"] = 19
        self.assertEqual(self.choose()["reason"], "cooldown")
        self.state["last_attempt_s"] = None
        self.state["last_proposal_s"] = 19
        self.assertEqual(self.choose()["reason"], "proposal_cooldown")
        self.state["last_proposal_s"] = None
        self.state["config"]["persist_windows"] = 1
        self.assertEqual(self.evaluate(20)["action"], "RETUNE")
        self.assertEqual(self.evaluate(25)["reason"], "proposal_cooldown")

    def test_duplicate_snapshot_or_same_evaluation_bucket_never_adds_persistence(self):
        self.assertEqual(self.evaluate(20)["policy_state"]["winner_windows"], 1)
        for at in (20, 21):
            result = self.evaluate(at)
            self.assertEqual(result["reason"], "duplicate_window")
            self.assertEqual(result["policy_state"]["winner_windows"], 1)
        self.assertEqual(self.evaluate(25)["policy_state"]["winner_windows"], 2)
        self.assertEqual(self.evaluate(30)["action"], "RETUNE")

    def test_invalidity_and_skipped_windows_reset_persistence(self):
        self.evaluate(20)
        estimates = self.sample(25)
        self.state["observation"].pop("headroom")
        result = self.choose(estimates)
        self.assertEqual(result["reason"], "headroom_missing")
        self.assertEqual(result["policy_state"]["winner_windows"], 0)
        self.state = result["policy_state"]
        self.assertEqual(self.evaluate(30)["policy_state"]["winner_windows"], 1)
        self.assertEqual(self.evaluate(40)["policy_state"]["winner_windows"], 1)

    def test_different_winner_resets_persistence_and_ties_use_bank_id(self):
        third = deepcopy(self.raw["banks"]["bank_b"])
        third["lanes"] = {"lane_a": "op_a", "lane_b": "op_b"}
        self.raw["banks"]["bank_c"] = third
        edge = deepcopy(self.raw["transitions"][0])
        edge.update(target_bank="bank_c", lane_id="lane_b")
        self.raw["transitions"].append(edge)
        self.reload()
        self.assertEqual(self.evaluate(20)["target_bank"], "bank_b")
        third["regimes"][0].update(power_w={"lo": 150, "hi": 160}, arrival_rps={"min": 0.5, "max": 1})
        self.reload()
        self.assertEqual(self.evaluate(25)["target_bank"], "bank_c")
        self.assertEqual(self.state["winner_windows"], 1)
        result = self.evaluate(30, extra=1)
        self.assertEqual(result["target_bank"], "bank_b")
        self.assertEqual(result["policy_state"]["winner_windows"], 1)

    def test_active_transition_missing_directed_edge_and_two_lane_candidate_abstain(self):
        self.sample(20)
        self.state["active_transition"] = True
        self.assertEqual(self.choose()["reason"], "transition_active")
        self.state["active_transition"] = False
        self.raw["transitions"][0].update(source_bank="bank_b", target_bank="bank_a")
        self.reload()
        self.assertEqual(self.choose()["reason"], "missing_transition")
        self.raw["banks"]["bank_c"] = deepcopy(self.raw["banks"]["bank_b"])
        self.raw["banks"]["bank_c"]["lanes"] = {"lane_a": "op_b", "lane_b": "op_b"}
        self.raw["banks"]["bank_b"]["regimes"] = []
        self.reload()
        self.assertEqual(self.choose()["reason"], "no_neighbor")

    def test_state_and_selector_share_injected_monotonic_clock(self):
        self.sample(20)
        result = drift.choose_retune("bank_a", self.estimates(), self.catalog["transitions"], self.state, None)
        self.assertEqual(result["policy_state"]["last_evaluated_s"], 20)
        self.assertEqual(self.choose(at=26)["reason"], "stale_observation")
        for invalid in (math.nan, math.inf, True, -1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.choose(at=invalid)
        self.state["last_attempt_s"] = 21
        with self.assertRaises(ValueError):
            self.choose()
        with self.assertRaises(ValueError):
            drift.make_policy_state(self.config, self.catalog, "bank_a", {"lane_a": True, "lane_b": 0}, clock=self.clock)


class ShadowTests(unittest.IsolatedAsyncioTestCase):
    """Synthetic catalogs only; exercise real controller/queue/router objects."""

    def setUp(self):
        self.assertTrue(hasattr(drift, "ShadowObserver"), "Task 3 shadow integration is missing")
        self.clock = FakeClock()
        self.raw = multi_bank_fixture()
        candidate = measured_regime("candidate")
        candidate["power_w"] = {"lo": 170, "hi": 180}
        self.raw["banks"]["bank_b"]["regimes"] = [candidate]
        self.raw["transitions"][0].update(duration_hi_s=60, extra_energy_hi_j=2000,
                                          remaining_capacity_rps_lo=3, max_backlog_requests=100)

    def controller(self, *, raw=None, presets=None, **kwargs):
        raw = self.raw if raw is None else raw
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "synthetic-only.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        options = {"op_drift": {
            **window_config(min_residence_s=10, cooldown_s=10),
            "mode": "shadow", "catalog_path": str(path),
            "runtime_context": raw["compatibility"],
        }}
        options.update(kwargs.pop("routing_kwargs", {}))
        if presets is None:
            presets = [Preset(lane, "http://unused", 1024, max_num_seqs=10, frequency_mhz=1300)
                       for lane in ("lane_a", "lane_b")]
        profile = DictProfileTable.from_mapping({lane: {"a_ttft": .1, "a_tbt": .01,
                                                        "a_ep": 2, "a_ed": .05}
                                                for lane in ("lane_a", "lane_b")})
        return EnergyAwareController(presets, profile, SLO(20), routing_kwargs=options,
                                     shadow_clock=self.clock, **kwargs)

    def feed(self, controller, start, stop):
        for i in range(start, stop):
            self.clock.now = i
            choice = controller.choose({"prompt": "PRIVATE PROMPT", "max_tokens": 10}, str(i))
            controller.on_request_finish(str(i), choice, .1, status_code=200)
        self.clock.now = stop

    def headroom(self):
        return {"fresh": True, "at_s": self.clock(), "healthy_lane_ids": ["lane_a", "lane_b"],
                "backlog_requests": 0}

    async def test_off_bypasses_catalog_observer_events_and_task(self):
        with patch.object(drift, "load_catalog", side_effect=AssertionError("catalog read")), \
                patch.object(drift, "DemandWindow", side_effect=AssertionError("observer constructed")):
            controller = self.controller(routing_kwargs={"op_drift": {"mode": "off", "catalog_path": "/missing"}})
            self.assertIsNone(controller.shadow)
            before = controller.config.to_dict()
            choice = controller.choose({"prompt": "hello", "max_tokens": 10})
            self.assertEqual(choice.preset.preset_id, "lane_a")
            self.assertIsNone(choice._observation_context)
            self.assertNotIn("op_drift", before)
            self.assertEqual(before, controller.config.to_dict())
            with patch("src.router_app.asyncio.create_task", side_effect=AssertionError("task started")):
                from src.router_app import shadow_lifespan
                async with shadow_lifespan(controller) as task:
                    self.assertIsNone(task)
            self.assertFalse(any(event["event"] == "op_drift" for event in controller.metrics.snapshot()["events"]))

    def test_direct_live_and_experiment_modes_rejected(self):
        with self.assertRaisesRegex(ValueError, "live.*not implemented"):
            self.controller(routing_kwargs={"op_drift": {"mode": "live"}})
        for key in ("specialization_gap", "op_conflict"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "op_drift.*experiment"):
                self.controller(routing_kwargs={key: {}})

    def test_initial_bank_and_actual_policy_must_match(self):
        for field, value in (("preset_id", "other"), ("frequency_mhz", 1400),
                             ("max_num_seqs", 11), ("max_num_batched_tokens", 2048)):
            presets = [Preset(lane, "http://unused", 1024, max_num_seqs=10, frequency_mhz=1300)
                       for lane in ("lane_a", "lane_b")]
            setattr(presets[0], field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "initial_bank|routing_policy"):
                self.controller(presets=presets)
        with self.assertRaisesRegex(ValueError, "routing_policy"):
            self.controller(routing_policy="round_robin")
        with self.assertRaisesRegex(ValueError, "routing_policy"):
            self.controller(random_lb=True)

    def test_recommendations_retain_state_but_never_mutate_runtime(self):
        controller = self.controller()
        shadow = controller.shadow
        self.assertIs(shadow.window.clock, self.clock)
        self.assertIs(shadow.policy_state["clock"], self.clock)
        before = [preset.to_dict() for preset in controller.registry.snapshot()]
        profile = controller.profile
        estimate = profile.estimate("lane_a", 1, 1).ttft_s
        with patch.object(EnergyAwareController, "replace_presets", side_effect=AssertionError("replace")), \
                patch.object(OperatingPointProfileTable, "activate_binding", side_effect=AssertionError("activate")), \
                patch("subprocess.Popen", side_effect=AssertionError("launch")), \
                patch("subprocess.run", side_effect=AssertionError("setter/process")):
            self.feed(controller, 0, 20)
            self.assertEqual(shadow.tick(self.headroom())["reason"], "persistence")
            self.assertEqual(shadow.tick(self.headroom())["reason"], "duplicate_window")
            self.feed(controller, 20, 25)
            self.assertEqual(shadow.tick(self.headroom())["reason"], "persistence")
            self.feed(controller, 25, 30)
            result = shadow.tick(self.headroom())
        self.assertEqual((result["action"], result["benefit_lo_j"]), ("RETUNE", 16800))
        self.assertEqual(shadow.policy_state["winner_windows"], 3)
        self.assertEqual(shadow.policy_state["current_bank"], "bank_a")
        self.assertEqual(shadow.policy_state["generations"], {"lane_a": 0, "lane_b": 0})
        self.assertIsNone(shadow.policy_state["last_commit_s"])
        self.assertEqual(before, [preset.to_dict() for preset in controller.registry.snapshot()])
        self.assertIs(controller.profile, profile)
        self.assertEqual(profile.estimate("lane_a", 1, 1).ttft_s, estimate)

    def test_missing_calibration_or_headroom_abstains(self):
        controller = self.controller()
        self.feed(controller, 0, 20)
        self.assertEqual(controller.shadow.tick()["reason"], "headroom_missing")
        stale = self.headroom()
        stale["at_s"] = 14
        self.assertEqual(controller.shadow.tick(stale)["reason"], "stale_headroom")
        raw = deepcopy(self.raw)
        for bank in raw["banks"].values():
            bank["regimes"] = []
        self.clock.now = 0
        controller = self.controller(raw=raw)
        self.feed(controller, 0, 20)
        result = controller.shadow.tick()
        self.assertEqual((result["action"], result["reason"]), ("NO_CHANGE", "no_calibration"))

    async def test_direct_router_and_wait_observe_once_without_extra_predictions(self):
        calls = []
        def predictor(body):
            calls.append(body["prompt"])
            return 10
        controller = self.controller(output_len_predictor=predictor)
        direct = controller.choose({"prompt": "private-direct"}, "direct")
        waited = await controller.choose_or_wait({"prompt": "private-wait"}, "wait")
        request_id, routed, payload = await EnergyAwareRouter(controller)._apply_routing({"prompt": "private-router"})
        self.assertEqual(calls, ["private-direct", "private-wait", "private-router"])
        self.assertEqual(controller.shadow.snapshot()["arrival_count"], 3)
        self.assertEqual(controller.shadow.snapshot()["unknown_count"], 0)
        for choice in (direct, waited, routed):
            self.assertIsNotNone(choice._observation_context)
            self.assertFalse(hasattr(choice._observation_context, "generation"))
        controller.on_request_finish(request_id, routed, .1, status_code=200)
        controller.on_request_finish(request_id, routed, .1, status_code=200)
        snap = controller.shadow.snapshot()
        self.assertEqual(snap["outcome_counts"]["completed"], 1)
        self.assertEqual(snap["slo_outcome_counts"], {"known": 0, "unknown": 1, "success": 0, "failure": 0})
        router = EnergyAwareRouter(controller)
        self.assertNotIn("observation", json.dumps(payload))
        self.assertNotIn("observation", json.dumps(router._routing_headers(routed)))
        self.assertNotIn("observation", json.dumps(router._backend_metadata_headers(routed)))
        controller.shadow.tick()
        events = json.dumps(controller.metrics.snapshot())
        self.assertNotIn("private-", events)
        self.assertNotIn("private-", repr(controller.shadow.window.diagnostics))
        self.assertNotIn("policy_state", events)

    async def test_queue_full_counts_unknown_before_classifier_and_rejects_once(self):
        controller = self.controller(routing_kwargs={"queue": {"enabled": True, "max_pending_requests": 1},
                                                    "depon": {"enabled": True}})
        waiting = asyncio.create_task(controller.choose_or_wait(
            {"prompt": "blocked", "max_tokens": 10, "energy_aware": {
                "workflow_id": "fixture-workflow", "parent_request_ids": ["missing"]}}, "blocked"))
        await asyncio.sleep(0)
        from src.classifier import RequestClassifier
        with patch.object(RequestClassifier, "classify", side_effect=AssertionError("classified queue-full")):
            with self.assertRaisesRegex(OverloadedError, "dispatch_queue_full"):
                await EnergyAwareRouter(controller)._apply_routing({"prompt": "rejected"})
        snapshot = controller.shadow.snapshot()
        self.assertEqual((snapshot["arrival_count"], snapshot["unknown_count"]), (2, 1))
        self.assertEqual(snapshot["outcome_counts"]["rejected"], 1)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(controller.shadow.snapshot()["outcome_counts"]["cancelled"], 1)

    async def test_dependency_replan_keeps_original_feature_but_existing_evaluations(self):
        calls = []
        def predictor(body):
            calls.append(None)
            return 10 if len(calls) == 1 else 300
        controller = self.controller(output_len_predictor=predictor,
                                     routing_kwargs={"depon": {"enabled": True}})
        choice = await controller.choose_or_wait({"prompt": "hello"}, "dep")
        self.assertEqual(len(calls), 2)  # Existing dependency-only path reclassifies.
        self.assertEqual(choice.routing_l_out_predicted_raw, 300)
        snap = controller.shadow.snapshot()
        self.assertEqual(snap["arrival_count"], 1)
        self.assertEqual(snap["class_mix"], {"p0:o0:default": 1})

    async def test_queue_releases_same_context_and_timeout_is_terminal(self):
        controller = self.controller(routing_kwargs={"queue": {"enabled": True}})
        choice = await controller.choose_or_wait({"prompt": "hello", "max_tokens": 10}, "queued")
        self.assertEqual(choice._observation_context.request_id, "queued")
        self.assertEqual(controller.shadow.snapshot()["arrival_count"], 1)
        controller = self.controller(routing_kwargs={"queue": {"enabled": True, "request_timeout_s": .001},
                                                    "depon": {"enabled": True}})
        with self.assertRaises(OverloadedError):
            await controller.choose_or_wait({"prompt": "hi", "energy_aware": {
                "workflow_id": "fixture-workflow", "parent_request_ids": ["missing"]}}, "timeout")
        self.assertEqual(controller.shadow.snapshot()["outcome_counts"]["rejected"], 1)

    async def test_cancel_before_admission_and_adapter_failure_have_terminal_outcomes(self):
        controller = self.controller(routing_kwargs={"queue": {"enabled": True}})
        from src.dispatch_queue import DispatchQueue
        entered = asyncio.Event()
        async def suspended_snapshot(queue):
            entered.set()
            await asyncio.Future()
        with patch.object(DispatchQueue, "snapshot", suspended_snapshot):
            task = asyncio.create_task(EnergyAwareRouter(controller)._apply_routing({"prompt": "hi"}))
            await entered.wait()
            self.assertEqual(controller.shadow.snapshot()["arrival_count"], 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(controller.shadow.snapshot()["outcome_counts"]["cancelled"], 1)
        adapter_type = type(controller.registry.get("lane_a").adapter)
        with patch.object(adapter_type, "prepare_body", side_effect=RuntimeError("private failure")):
            with self.assertRaisesRegex(RuntimeError, "private failure"):
                await EnergyAwareRouter(controller)._apply_routing({"prompt": "hi"})
        self.assertEqual(controller.shadow.snapshot()["outcome_counts"]["failed"], 1)

    def test_classification_failure_and_rejection_keep_original_errors(self):
        controller = self.controller()
        from src.classifier import RequestClassifier
        for error, status in ((RuntimeError("private classify"), "failed"), (OverloadedError("full"), "rejected")):
            with patch.object(RequestClassifier, "classify", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    controller.choose({"prompt": "hi"})
                self.assertIs(caught.exception, error)
            self.assertEqual(controller.shadow.snapshot()["outcome_counts"][status], 1)
        self.assertEqual(controller.shadow.snapshot()["unknown_count"], 2)

    def test_observation_and_recording_failures_are_isolated_and_clear_persistence(self):
        controller = self.controller()
        self.feed(controller, 0, 20)
        controller.shadow.tick(self.headroom())
        with patch.object(drift, "observe_arrival", side_effect=RuntimeError("PRIVATE ERROR")):
            self.assertEqual(controller.choose({"prompt": "hi"}).preset.preset_id, "lane_a")
        self.assertEqual(controller.shadow.policy_state["winner_windows"], 0)
        self.assertNotIn("PRIVATE ERROR", json.dumps(controller.metrics.snapshot()))
        with patch.object(MetricsPipeline, "record_shadow_event", side_effect=RuntimeError("metrics")), \
                self.assertLogs("src.router_app", level="WARNING"):
            choice = controller.choose({"prompt": "hi"})
            controller.on_request_finish("id", choice, .1, status_code=200)
            self.assertEqual(controller.shadow.tick(self.headroom())["action"], "NO_CHANGE")
        self.assertEqual(controller.shadow.policy_state["winner_windows"], 0)

    def test_events_are_bounded_json_evidence_and_logged(self):
        controller = self.controller(metrics=MetricsPipeline(max_events=32))
        self.feed(controller, 0, 20)
        with self.assertLogs("src.router_app", level="INFO") as logged:
            for _ in range(40):
                controller.shadow.tick(self.headroom())
        events = controller.metrics.snapshot()["events"]
        self.assertEqual(len(events), 32)
        event = events[-1]
        self.assertEqual((event["event"], event["mode"], event["executed"], event["identity_source"]),
                         ("op_drift", "shadow", False, "operator_declared"))
        self.assertEqual(event["decision"]["benefit_lo_j"], 16800)
        self.assertEqual(event["estimates"]["bank_b"]["power_w"], {"lo": 170, "hi": 180})
        self.assertEqual(len(event["catalog_hash"]), 64)
        self.assertEqual(len(event["config_hash"]), 64)
        self.assertEqual(event["runtime_context"], self.raw["compatibility"])
        text = json.dumps(events, allow_nan=False)
        self.assertNotIn("policy_state", text)
        self.assertNotIn("PRIVATE PROMPT", text)
        self.assertTrue(any('"executed": false' in line for line in logged.output))

    def test_diagnostic_sink_failure_cannot_break_a_valid_request(self):
        controller = self.controller()
        import logging
        with patch.object(drift, "observe_arrival", side_effect=RuntimeError("observation")), \
                patch.object(MetricsPipeline, "record_shadow_event", side_effect=RuntimeError("metrics")), \
                patch.object(logging.getLogger("src.router_app"), "warning", side_effect=RuntimeError("logger")):
            self.assertEqual(controller.choose({"prompt": "hi"}).preset.preset_id, "lane_a")

    def test_private_observation_id_failure_cannot_break_direct_routing(self):
        controller = self.controller()
        with patch("src.op_drift.uuid.uuid4", side_effect=OSError("entropy")):
            self.assertEqual(controller.choose({"prompt": "hi"}).preset.preset_id, "lane_a")
        self.assertEqual(controller.metrics.snapshot()["events"][0]["decision"]["reason"], "shadow_error")

    def test_proxy_failure_or_cancellation_after_200_does_not_become_success(self):
        import time
        controller = self.controller()
        router = EnergyAwareRouter(controller)
        for status in ("failed", "cancelled", "completed"):
            choice = controller.choose({"prompt": "hi", "max_tokens": 10}, status)
            lane = controller.queue_manager.get(choice.preset.preset_id)
            lane.on_start()
            router._finish_proxy(status, lane, choice, time.time(), 200, outcome=status)
            controller.on_request_finish(status, choice, .1, status_code=200)
        snap = controller.shadow.snapshot()
        self.assertEqual(snap["outcome_counts"], {"completed": 1, "failed": 1, "cancelled": 1, "rejected": 0})
        self.assertEqual(snap["slo_outcome_counts"], {"known": 0, "unknown": 3, "success": 0, "failure": 0})

    async def test_late_enrichment_uses_current_clock_and_original_bucket(self):
        controller = self.controller()
        context = controller._begin_observation("delayed")
        self.clock.now = 4
        controller.shadow.snapshot()
        self.clock.now = 6
        choice = await controller.choose_or_wait({"prompt": "hi", "max_tokens": 10}, "delayed",
                                                 _observation_context=context)
        snap = controller.shadow.snapshot()
        self.assertEqual((snap["arrival_count"], snap["unknown_count"]), (1, 0))
        self.assertEqual(choice._observation_context.arrival_at_s, 0)
        self.assertEqual(list(controller.shadow.window.diagnostics), [(0, 1.0, 10.0)])

    async def stream_handoff(self, controller):
        """Run the real handler through response return, without any iterator advance."""
        tracker = Namespace(advances=0, releases=0)
        async def chunks(size):
            self.assertEqual(size, 1024)
            tracker.advances += 1
            yield b"data: synthetic\n\n"
        async def release():
            tracker.releases += 1
        async def post(url, *, json, headers):
            return Namespace(status=200, headers={}, release=release,
                             content=Namespace(iter_chunked=chunks))
        async def body():
            return {"prompt": "private stream", "max_tokens": 10, "stream": True}
        request = Namespace(headers={"content-type": "application/json"}, json=body,
                            app=Namespace(state=Namespace(upstream_session=Namespace(post=post))))
        response = await EnergyAwareRouter(controller).handle_completion(request, "/v1/completions")
        self.assertEqual((tracker.advances, tracker.releases), (0, 0))
        self.assertEqual(controller.queue_manager.total_in_flight(), 1)
        if controller.shadow is not None:
            snap = controller.shadow.snapshot()
            self.assertEqual(snap["arrival_count"], 1)
            self.assertEqual(sum(snap["outcome_counts"].values()), 0)
        return response, tracker

    def assert_early_stream_outcome(self, controller, tracker, status):
        self.assertEqual((tracker.advances, tracker.releases), (0, 0))
        self.assertEqual(controller.queue_manager.total_in_flight(), 1)
        self.assertEqual(controller.metrics.snapshot()["completions"], 0)
        snap = controller.shadow.snapshot()
        self.assertEqual(snap["outcome_counts"], {
            key: int(key == status) for key in ("completed", "failed", "rejected", "cancelled")})
        self.assertEqual(snap["slo_outcome_counts"], {"known": 0, "unknown": 1, "success": 0, "failure": 0})

    async def test_stream_asgi_start_send_failure_observes_without_cleanup(self):
        controller = self.controller()
        error = OSError("response-start send failed")
        with fake_streaming_responses() as (response_base, _):
            response, tracker = await self.stream_handoff(controller)
            self.assertIsInstance(response, response_base)
            async def receive():
                return {"type": "http.request"}
            async def send(message):
                self.assertEqual(message["type"], "http.response.start")
                self.assertEqual(tracker.advances, 0)
                raise error
            with self.assertRaises(OSError) as caught:
                await response({"type": "http"}, receive, send)
            self.assertIs(caught.exception, error)
            # Closing an unstarted generator cannot supply the missing terminal.
            await response.body_iterator.aclose()
            self.assert_early_stream_outcome(controller, tracker, "failed")

    async def test_stream_asgi_early_receive_or_send_cancellation_observes_without_cleanup(self):
        for boundary in ("receive", "send"):
            with self.subTest(boundary=boundary), fake_streaming_responses():
                controller = self.controller()
                response, tracker = await self.stream_handoff(controller)
                entered = asyncio.Event()
                async def receive():
                    if boundary == "receive":
                        entered.set()
                        await asyncio.Future()
                    return {"type": "http.request"}
                async def send(message):
                    self.assertEqual(message["type"], "http.response.start")
                    entered.set()
                    await asyncio.Future()
                task = asyncio.create_task(response({"type": "http"}, receive, send))
                await entered.wait()
                self.assertEqual(tracker.advances, 0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await response.body_iterator.aclose()
                self.assert_early_stream_outcome(controller, tracker, "cancelled")

    async def test_stream_asgi_graceful_preiteration_disconnect_observes_without_cleanup(self):
        controller = self.controller()
        with fake_streaming_responses():
            response, tracker = await self.stream_handoff(controller)
            async def receive():
                return {"type": "http.disconnect"}
            async def send(message):
                self.fail("disconnect must return before response-start or iteration")
            await response({"type": "http"}, receive, send)
            await response.body_iterator.aclose()
            self.assert_early_stream_outcome(controller, tracker, "cancelled")

    async def test_stream_asgi_success_records_one_terminal_with_response_identity(self):
        controller = self.controller()
        with fake_streaming_responses() as (response_base, streaming_base):
            response, tracker = await self.stream_handoff(controller)
            self.assertIsInstance(response, response_base)
            self.assertIsInstance(response, streaming_base)
            sent = []
            async def receive():
                return {"type": "http.request"}
            async def send(message):
                sent.append(message)
            await response({"type": "http"}, receive, send)
            await response.body_iterator.aclose()
            self.assertEqual([message["type"] for message in sent],
                             ["http.response.start", "http.response.body", "http.response.body"])
            self.assertEqual((tracker.advances, tracker.releases), (1, 1))
            self.assertEqual(controller.queue_manager.total_in_flight(), 0)
            self.assertEqual(controller.metrics.snapshot()["completions"], 1)
            snap = controller.shadow.snapshot()
            self.assertEqual(snap["outcome_counts"], {"completed": 1, "failed": 0, "rejected": 0, "cancelled": 0})
            self.assertEqual(snap["slo_outcome_counts"], {"known": 0, "unknown": 1, "success": 0, "failure": 0})

    async def test_off_stream_asgi_uses_original_response_and_early_failure_behavior(self):
        controller = self.controller(routing_kwargs={"op_drift": {"mode": "off"}})
        with fake_streaming_responses() as (_, streaming_base):
            response, tracker = await self.stream_handoff(controller)
            self.assertIs(type(response), streaming_base)
            async def receive():
                return {"type": "http.request"}
            async def send(message):
                raise OSError("off response-start send failed")
            with self.assertRaises(OSError):
                await response({"type": "http"}, receive, send)
            await response.body_iterator.aclose()
            self.assertIsNone(controller.shadow)
            self.assertEqual((tracker.advances, tracker.releases), (0, 0))
            self.assertEqual(controller.queue_manager.total_in_flight(), 1)
            self.assertEqual(controller.metrics.snapshot()["completions"], 0)
            self.assertFalse(any(e["event"] == "op_drift" for e in controller.metrics.snapshot()["events"]))

    async def test_production_headroom_requires_fresh_complete_kv_and_known_accounting(self):
        controller = self.controller()
        self.assertIsNone(await controller.collect_shadow_headroom())
        controller.config.kv_project["enabled"] = True
        self.assertIsNone(await controller.collect_shadow_headroom())
        import time
        for lane in ("lane_a", "lane_b"):
            controller.kv_projection_store.update(lane, shadow_kv_snapshot(lane))
        self.clock.now = 20
        controller.queue_manager.get("lane_a").on_start(3)
        # One tracked running request overlaps gateway/backend running counts.
        features, options = controller._classify_request({"prompt": "hi"})
        for rid, status, lane in (("running", "running", "lane_a"), ("reserved", "planned", "lane_b"),
                                  ("pending", "pending", "")):
            record = controller._register_request_record(rid, {}, features, options)
            record.status, record.selected_preset_id = status, lane
        for free_blocks in (20, 100):
            invalid = shadow_kv_snapshot("lane_b")
            invalid.npu_free_blocks = free_blocks
            controller.kv_projection_store.update("lane_b", invalid)
            with self.subTest(free_blocks=free_blocks):
                self.assertIsNone(await controller.collect_shadow_headroom())
        controller.kv_projection_store.update("lane_b", shadow_kv_snapshot("lane_b"))
        headroom = await controller.collect_shadow_headroom()
        # 1 unbound + 1 planned + max(3 gateway, 1 tracked, 2 backend) + 2 backend = 7.
        self.assertEqual(headroom, {"fresh": True, "at_s": 20, "healthy_lane_ids": ["lane_a", "lane_b"],
                                   "backlog_requests": 7})
        controller.kv_projection_store.update("lane_b", shadow_kv_snapshot("lane_b", ts=time.time() - 5))
        self.assertIsNone(await controller.collect_shadow_headroom())
        controller.kv_projection_store.update("lane_b", shadow_kv_snapshot("wrong-engine"))
        self.assertIsNone(await controller.collect_shadow_headroom())
        controller.kv_projection_store.update("lane_b", shadow_kv_snapshot("lane_b", active=-1))
        self.assertIsNone(await controller.collect_shadow_headroom())

    async def test_parser_to_store_missing_host_accounting_is_not_shadow_fresh(self):
        import time
        from src.kv_projection import KVProjectionSnapshot
        controller = self.controller()
        controller.config.kv_project["enabled"] = True
        payloads = {}
        for lane in ("lane_a", "lane_b"):
            payload = {
                "schema_version": "parsa.vllm.kv_projection.v1", "ts_s": time.time(),
                "engine_id": lane, "block_size_tokens": 16,
                "capacity": {"npu_blocks": 100, "host_blocks": 0},
                "current": {"npu_used_blocks": 10, "npu_free_blocks": 90, "active_requests": 0},
                "projection_basis": {"mode": "router_hints_plus_runtime", "horizon_iters": 1,
                                     "quantiles": [0.5, 0.9, 0.95]},
                "projection": {q: [{"iter_offset": 0, "batch_size": 0, "npu_kv_blocks": 10}]
                               for q in ("p50", "p90", "p95")},
                "summary": {"peak_npu_blocks": dict.fromkeys(("p50", "p90", "p95"), 10),
                            "peak_utilization": dict.fromkeys(("p50", "p90", "p95"), .1),
                            "peak_batch_size": dict.fromkeys(("p50", "p90", "p95"), 0),
                            "time_to_below_utilization_ms": {"0.80": 0, "0.70": 0, "0.60": 0}},
            }
            payloads[lane] = payload
            controller.kv_projection_store.update(lane, KVProjectionSnapshot.from_payload(payload, lane))
        missing = controller.kv_projection_store.get_snapshot("lane_b", allow_stale=True)
        self.assertEqual(missing.host_used_blocks, 0)  # Legacy static numeric fallback is preserved.
        self.assertFalse(missing.host_used_blocks_available)
        self.assertNotIn("host_used_blocks_available", missing.to_metrics_dict())
        self.assertIsNone(await controller.collect_shadow_headroom())
        for raw_value in (None, "0", 0.0):
            with self.subTest(raw_value=raw_value):
                payload = deepcopy(payloads["lane_b"])
                payload["current"]["host_used_blocks"] = raw_value
                controller.kv_projection_store.update(
                    "lane_b", KVProjectionSnapshot.from_payload(payload, "lane_b"))
                self.assertFalse(controller.kv_projection_store.get_snapshot(
                    "lane_b", allow_stale=True).host_used_blocks_available)
                self.assertIsNone(await controller.collect_shadow_headroom())
        invalid_bool = deepcopy(payloads["lane_b"])
        invalid_bool["current"]["host_used_blocks"] = True
        with self.assertRaises(TypeError):
            KVProjectionSnapshot.from_payload(invalid_bool, "lane_b")

    async def test_explicit_host_zero_and_swap_alias_zero_are_available(self):
        import time
        from src.kv_projection import KVProjectionSnapshot
        controller = self.controller()
        controller.config.kv_project["enabled"] = True
        snapshots = []
        for lane, key in (("lane_a", "host_used_blocks"), ("lane_b", "swap_used_blocks")):
            payload = {
                "schema_version": "parsa.vllm.kv_projection.v1", "ts_s": time.time(),
                "engine_id": lane, "block_size_tokens": 16,
                "capacity": {"npu_blocks": 100, "host_blocks": 0},
                "current": {"npu_used_blocks": 10, "npu_free_blocks": 90, key: 0, "active_requests": 0},
                "projection_basis": {"mode": "router_hints_plus_runtime", "horizon_iters": 1,
                                     "quantiles": [0.5, 0.9, 0.95]},
                "projection": {q: [{"iter_offset": 0, "batch_size": 0, "npu_kv_blocks": 10}]
                               for q in ("p50", "p90", "p95")},
                "summary": {"peak_npu_blocks": dict.fromkeys(("p50", "p90", "p95"), 10),
                            "peak_utilization": dict.fromkeys(("p50", "p90", "p95"), .1),
                            "peak_batch_size": dict.fromkeys(("p50", "p90", "p95"), 0),
                            "time_to_below_utilization_ms": {"0.80": 0, "0.70": 0, "0.60": 0}},
            }
            snapshot = KVProjectionSnapshot.from_payload(payload, lane)
            snapshots.append(snapshot)
            controller.kv_projection_store.update(lane, snapshot)
        self.assertTrue(all(snapshot.host_used_blocks_available for snapshot in snapshots))
        self.assertEqual((await controller.collect_shadow_headroom())["healthy_lane_ids"], ["lane_a", "lane_b"])

    async def test_headroom_requires_positive_integer_capacity(self):
        controller = self.controller()
        controller.config.kv_project["enabled"] = True
        controller.kv_projection_store.update("lane_a", shadow_kv_snapshot("lane_a"))
        for capacity in (True, 100.0, 0, -1):
            invalid = shadow_kv_snapshot("lane_b")
            invalid.npu_blocks = capacity
            controller.kv_projection_store.update("lane_b", invalid)
            with self.subTest(capacity=capacity):
                self.assertIsNone(await controller.collect_shadow_headroom())

    async def test_unbound_queue_accounting_is_not_counted_twice(self):
        controller = self.controller(routing_kwargs={"depon": {"enabled": True}})
        waiting = asyncio.create_task(controller.choose_or_wait(
            {"prompt": "hi", "energy_aware": {
                "workflow_id": "fixture-workflow", "parent_request_ids": ["missing"]}}, "blocked"))
        await asyncio.sleep(0)
        item = await controller.dispatch_queue.get("blocked")
        self.assertIsNotNone(item.observation_context)
        self.assertNotIn("observation", json.dumps(item.to_dict()))
        # Full backend/accounting coverage is still required, never default to zero.
        controller.config.kv_project["enabled"] = True
        self.assertIsNone(await controller.collect_shadow_headroom())
        for lane in ("lane_a", "lane_b"):
            controller.kv_projection_store.update(lane, shadow_kv_snapshot(lane, active=0))
        self.assertEqual((await controller.collect_shadow_headroom())["backlog_requests"], 1)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting

    async def test_slow_loop_lifecycle_errors_sleep_reset_and_cancellation(self):
        from src.router_app import shadow_lifespan
        controller = self.controller()
        self.feed(controller, 0, 20)
        controller.shadow.tick(self.headroom())
        slept = asyncio.Event()
        delays = []
        async def sleep(delay):
            delays.append(delay)
            slept.set()
            await asyncio.Future()
        async def bad_headroom():
            raise ValueError("PRIVATE COLLECTION")
        async with shadow_lifespan(controller, headroom_producer=bad_headroom, sleep=sleep) as task:
            await slept.wait()
            self.assertFalse(task.done())
            self.assertEqual(delays, [5])
            self.assertEqual(controller.shadow.policy_state["winner_windows"], 0)
            event = controller.metrics.snapshot()["events"][-1]
            self.assertEqual(event["decision"]["reason"], "shadow_error")
            self.assertEqual(event["error_type"], "ValueError")
            self.assertNotIn("PRIVATE COLLECTION", json.dumps(event))
        self.assertTrue(task.done())
        self.assertTrue(task.cancelled())

    async def test_slow_loop_selector_error_and_success_keep_serving(self):
        controller = self.controller()
        self.feed(controller, 0, 20)
        attempts = []
        async def headroom():
            return self.headroom()
        async def sleep(delay):
            attempts.append(delay)
            if len(attempts) == 2:
                raise asyncio.CancelledError()
        original = drift.choose_retune
        calls = 0
        def selector(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("selector")
            return original(*args)
        with patch.object(drift, "choose_retune", side_effect=selector):
            with self.assertRaises(asyncio.CancelledError):
                await controller.shadow.run(headroom, sleep=sleep)
        self.assertEqual(attempts, [5, 5])
        self.assertEqual(controller.shadow.policy_state["winner_windows"], 1)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                            ("aiohttp", "fastapi", "starlette", "httpx", "uvicorn")),
                         "HTTP integration skipped: optional aiohttp/FastAPI/starlette/httpx/uvicorn absent; no install")
    async def test_http_lifespan_integration(self):
        from fastapi.testclient import TestClient
        from src.router_app import create_app
        controller = self.controller()
        with TestClient(create_app(controller)) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            task = client.app.state.shadow_task
            self.assertFalse(task.done())
        self.assertTrue(task.done())


if __name__ == "__main__":
    unittest.main()
