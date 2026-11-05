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

import argparse
import json
import logging

from .controller import EnergyAwareController, ROUTING_POLICIES, SLO
from .metrics_collector import BaselineCalibration
from .op_drift import load_drift_config
from .output_len_predictor import build_output_len_predictor, output_len_predictor_from_env_or_arg
from .predictors import TokenCounter
from .presets import load_presets_json
from .profile_table import DictProfileTable, LatencyScaleFactors, ScaledProfileTable
from .router_app import create_app


def _json_object(value):
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("routing kwargs must decode to a JSON object")
    return parsed


def _merge_dict(base, override):
    merged = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged.get(key), value)
        else:
            merged[key] = value
    return merged


def build_parser():
    parser = argparse.ArgumentParser(prog="src.cli")
    parser.add_argument("--presets", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--slo-max-latency-s", required=True, type=float)
    parser.add_argument("--rescue-preset-id")
    parser.add_argument("--out-len-safety-factor", default=1.2, type=float)
    parser.add_argument("--output-len-predictor")
    parser.add_argument("--hf-tokenizer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--enable-metrics-calibration", action="store_true")
    parser.add_argument("--metrics-interval-s", default=1.0, type=float)
    parser.add_argument("--baseline-ttft-s", default=0.05, type=float)
    parser.add_argument("--baseline-tbt-s", default=0.02, type=float)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--random-lb", action="store_true")
    parser.add_argument("--routing-policy", default="legacy_ratio", choices=ROUTING_POLICIES)
    parser.add_argument("--routing-kwargs", type=_json_object)
    parser.add_argument("--enable-soft-admission", action="store_true")
    parser.add_argument("--queue", action="store_true", help="Enable router-side queued dispatch and batch planning")
    parser.add_argument("--feedback", action="store_true", help="Enable multi-SLO scoring and online feedback calibration")
    parser.add_argument("--depon", action="store_true", help="Enable dependency/workflow-aware scheduling")
    parser.add_argument("--kv-project", action="store_true", help="Require NPU KV projection from the backend and use it in routing")
    return parser


def build_controller(args):
    drift_config = load_drift_config(args.routing_kwargs or {})
    if drift_config["mode"] != "off" and any(
        (args.routing_kwargs or {}).get(key) is not None for key in ("specialization_gap", "op_conflict")
    ):
        raise ValueError("op_drift cannot be enabled in experiment modes")
    presets = load_presets_json(args.presets)
    profile = DictProfileTable.from_json(args.profile)
    experiment = (args.routing_kwargs or {}).get("specialization_gap") is not None
    if experiment and any((args.hf_tokenizer, args.output_len_predictor, args.enable_metrics_calibration,
                           args.enable_soft_admission, args.queue, args.feedback, args.depon, args.kv_project, args.random_lb)):
        raise ValueError("specialization_gap cannot enable predictor-dependent or adaptive router options")
    token_counter = TokenCounter.from_pretrained(args.hf_tokenizer) if args.hf_tokenizer and not experiment else None
    predictor = None
    predictor_spec = output_len_predictor_from_env_or_arg(args.output_len_predictor)
    if not (experiment or args.random_lb or args.routing_policy == "random"):
        predictor = build_output_len_predictor(predictor_spec)
    metrics_calibration = None
    if args.enable_metrics_calibration:
        scales = LatencyScaleFactors()
        profile = ScaledProfileTable(profile, scales)
        cfg = BaselineCalibration(
            baseline_ttft_s=args.baseline_ttft_s,
            baseline_tbt_s=args.baseline_tbt_s,
        )
        metrics_urls = [preset.adapter.metric_url() for preset in presets]
        metrics_calibration = (metrics_urls, scales, cfg, args.metrics_interval_s)
    routing_kwargs = dict(args.routing_kwargs or {})
    if args.enable_soft_admission:
        routing_kwargs = _merge_dict(routing_kwargs, {"soft_admission": {"enabled": True}})
    if args.queue:
        routing_kwargs = _merge_dict(routing_kwargs, {"queue": {"enabled": True}})
    if args.feedback:
        routing_kwargs = _merge_dict(routing_kwargs, {"feedback": {"enabled": True}})
    if args.depon:
        routing_kwargs = _merge_dict(routing_kwargs, {"depon": {"enabled": True}})
    if args.kv_project:
        routing_kwargs = _merge_dict(routing_kwargs, {"kv_project": {"enabled": True}})
    controller = EnergyAwareController(
        presets=presets,
        profile=profile,
        slo=SLO(args.slo_max_latency_s),
        rescue_preset_id=args.rescue_preset_id,
        token_counter=token_counter,
        output_len_predictor=predictor,
        out_len_safety_factor=args.out_len_safety_factor,
        random_lb=args.random_lb,
        routing_policy=args.routing_policy,
        routing_kwargs=routing_kwargs,
    )
    return controller, metrics_calibration


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    controller, metrics_calibration = build_controller(args)
    app = create_app(controller, metrics_calibration=metrics_calibration)
    import uvicorn

    uvicorn.run(app, host=args.host, port=int(args.port), log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()
