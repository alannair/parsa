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

from predictors.common.dataset_io import load_benchmark_csv
from predictors.common.experiment_io import dump_benchmark_csv
from predictors.common.metrics import print_pearson_spearman_kendall_underpred


def _parse_layer_sizes(value):
    text = str(value or "").strip()
    if not text:
        raise argparse.ArgumentTypeError("layer sizes must be a comma-separated list of positive integers")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("layer sizes must not be empty")
    try:
        sizes = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layer sizes must be integers") from exc
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("layer sizes must be positive integers")
    return sizes


def build_benchmark_train_parser(*, prog, description, default_model_dir):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--dump-benchmark-csv",
        action="store_true",
        help="Read experiment JSON files under --experiments-root and write --csv-path",
    )
    parser.add_argument(
        "--experiments-root",
        help="Directory containing experiment JSON files (required with --dump-benchmark-csv)",
    )
    parser.add_argument(
        "--csv-path",
        help="Output CSV for --dump-benchmark-csv, or input CSV for training",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan JSON files directly under --experiments-root",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip experiment JSON files that fail to load when dumping CSV",
    )
    parser.add_argument("--model-dir", default=str(default_model_dir), help="Directory for saved bundle")
    parser.add_argument(
        "--model-type",
        default="lr",
        choices=("lr", "linear_regression", "gb", "gradient_boosting", "nn"),
        help="Training model type (default: lr)",
    )
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no-save", action="store_true", help="Train without writing a bundle")
    parser.add_argument("--layer-sizes", default="64,32", type=_parse_layer_sizes)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--validation-split", default=0.2, type=float)
    parser.add_argument("--learning-rate", default=None, type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--npu-id",
        default=None,
        type=int,
        help="Use Ascend NPU PyTorch training path for --model-type nn",
    )
    return parser


def run_benchmark_dump_cli(*, default_model_dir, argv=None, prog=None):
    parser = build_benchmark_train_parser(
        prog=prog or "predictors.benchmark.dump",
        description="Build benchmark.csv from experiment JSON files",
        default_model_dir=default_model_dir,
    )
    args = parser.parse_args(argv)
    if not args.dump_benchmark_csv:
        parser.error("--dump-benchmark-csv is required for dump mode")
    if not args.experiments_root:
        parser.error("--experiments-root is required with --dump-benchmark-csv")
    if not args.csv_path:
        parser.error("--csv-path is required with --dump-benchmark-csv")

    out_path, df = dump_benchmark_csv(
        args.experiments_root,
        args.csv_path,
        recursive=not args.no_recursive,
        skip_invalid=args.skip_invalid,
    )
    print("wrote %d rows -> %s" % (len(df), out_path), flush=True)
    load_errors = df.attrs.get("load_errors") or []
    for message in load_errors:
        print("skipped: %s" % message, flush=True)
    return 0


def run_benchmark_train_cli(*, target_col, default_model_dir, train_module, feature_cols=None, argv=None):
    prog = train_module.__name__
    parser = build_benchmark_train_parser(
        prog=prog,
        description="Train a benchmark predictor and save a bundle",
        default_model_dir=default_model_dir,
    )
    args = parser.parse_args(argv)
    if args.dump_benchmark_csv:
        return run_benchmark_dump_cli(
            default_model_dir=default_model_dir,
            argv=argv,
            prog=prog,
        )
    if not args.csv_path:
        parser.error("--csv-path is required for training (or use --dump-benchmark-csv to build one)")

    df = load_benchmark_csv(args.csv_path, target_col, feature_cols=feature_cols)
    model_dir = None if args.no_save else args.model_dir
    model_type = "lr" if args.model_type == "linear_regression" else args.model_type

    common = {
        "test_size": args.test_size,
        "random_state": args.seed,
        "model_dir": model_dir,
    }
    if model_type == "nn":
        npu_ids = None if args.npu_id is None else [int(args.npu_id)]
        result = train_module.train_nn(
            df,
            layer_sizes=args.layer_sizes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            learning_rate=args.learning_rate,
            verbose=1 if args.verbose else 0,
            npu_ids=npu_ids,
            **common,
        )
    else:
        result = train_module.train(df, model_type=model_type, **common)

    print_pearson_spearman_kendall_underpred(result["y_true"], result["y_pred"], label=target_col)
    if result.get("bundle_dir"):
        print("saved bundle -> %s" % result["bundle_dir"], flush=True)
    return 0


def main_for_module(train_module, *, target_col, default_model_dir, feature_cols=None, argv=None):
    return run_benchmark_train_cli(
        target_col=target_col,
        default_model_dir=default_model_dir,
        train_module=train_module,
        feature_cols=feature_cols,
        argv=argv,
    )
