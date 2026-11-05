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
import os
import sys
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLOTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LABELS_CSV = REPO_ROOT / "predictors" / "output_length" / "dataset_output_length.csv"
DEFAULT_BERT_BUNDLE = REPO_ROOT / "models" / "output_length"
DEFAULT_DEBERTA_BUNDLE = REPO_ROOT / "models" / "output_length_deberta"
DEFAULT_RESULTS_DEBERTA = PLOTS_DIR / "predictor_results_deberta.csv"
DEFAULT_RESULTS_BERT = PLOTS_DIR / "predictor_results_bert.csv"
DEFAULT_TOKENIZER = "microsoft/deberta-v3-base"


def _log(message):
    print(message, flush=True)


def _save_or_show(fig, output_dir, name):
    import matplotlib.pyplot as plt

    if output_dir is None:
        plt.show()
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _log("  saved %s" % path)
    return path


def _resolve_bundle(path_arg, default_path):
    if path_arg is not None:
        return Path(path_arg)
    candidate = Path(default_path)
    return candidate if candidate.exists() else None


def run_predictor_dump(
    labels_csv,
    bundle_path,
    output_csv,
    *,
    tokenizer_name=DEFAULT_TOKENIZER,
    batch_size=16,
    device=None,
    ascend_devices=None,
    num_prompts=None,
    show_progress=True,
    include_prompt=False,
):
    del tokenizer_name
    from predictors.output_length.predict_bundle import run_predict_bundle

    return run_predict_bundle(
        labels_csv,
        bundle_path,
        output_csv,
        batch_size=batch_size,
        device=device,
        ascend_devices=ascend_devices,
        num_prompts=num_prompts,
        show_progress=show_progress,
        include_prompt=include_prompt,
    )


def run_predictor_plot(
    results_csv,
    *,
    results_csv_bert=None,
    output_dir=None,
    bins=50,
    log_scale=True,
    pred_col="PredictedTokens",
    truth_vs_pred_suffix="",
    plot_bucket_classification=False,
):
    if str(PLOTS_DIR) not in sys.path:
        sys.path.insert(0, str(PLOTS_DIR))
    from dataload import load_predictor_results_csv
    from plots import (
        plot_output_length_dataset_distributions,
        plot_predictor_bucket_classification,
        plot_predictor_truth_vs_pred,
    )

    _log("loading results from %s" % results_csv)
    df = load_predictor_results_csv(results_csv)

    _log("plotting input / output size distributions")
    fig_dist, _axes = plot_output_length_dataset_distributions(
        df,
        title="Output-length dataset: prompt sizes per type",
        bins=bins,
        show=output_dir is None,
        log_x=log_scale,
    )
    dist_name = "dataset_size_distributions%s.pdf" % truth_vs_pred_suffix
    _save_or_show(fig_dist, output_dir, dist_name)

    title = "Predictor: true vs predicted output tokens"
    fig_name = "truth_vs_pred%s.pdf" % truth_vs_pred_suffix
    fig, _ = plot_predictor_truth_vs_pred(
        df,
        pred_col=pred_col,
        title=title,
        show=output_dir is None,
        log_scale=log_scale,
    )
    _save_or_show(fig, output_dir, fig_name)

    if plot_bucket_classification:
        _log("plotting hybrid bucket classification (confusion matrices)")
        bucket_name = "bucket_classification%s.pdf" % truth_vs_pred_suffix
        fig_bucket, _ = plot_predictor_bucket_classification(
            df,
            title="Hybrid bucket classification: true vs predicted bucket",
            show=output_dir is None,
        )
        _save_or_show(fig_bucket, output_dir, bucket_name)

    if results_csv_bert is not None:
        _log("loading BERT results from %s" % results_csv_bert)
        df_bert = load_predictor_results_csv(results_csv_bert)
        fig_bert, _ = plot_predictor_truth_vs_pred(
            df_bert,
            pred_col=pred_col,
            title="Legacy BERT predictor: true vs predicted output tokens",
            show=output_dir is None,
            log_scale=log_scale,
        )
        _save_or_show(fig_bert, output_dir, "truth_vs_pred_bert.pdf")

    return df


def build_parser():
    parser = argparse.ArgumentParser(
        prog="plots.predictor_plots",
        description="Predict output-length bundles to CSV and plot from saved results",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--batch-size", default=16, type=int)
    common.add_argument(
        "--device",
        default=None,
        help="Optional device override (default: Ascend single-NPU safe bundle inference)",
    )
    common.add_argument(
        "--ascend-devices",
        default=os.environ.get("ASCEND_DEVICES", "0"),
        help="Physical NPU id for predict (single card; no DataParallel)",
    )
    common.add_argument(
        "--num-prompts",
        default=0,
        type=int,
        help="Random subsample for quick runs (0 = all rows)",
    )
    common.add_argument("--no-progress", action="store_true")
    common.add_argument(
        "--tokenizer-name",
        default=DEFAULT_TOKENIZER,
        help="Ignored for predict (bundle tokenizer is used); kept for compatibility",
    )

    p_predict = sub.add_parser(
        "predict",
        parents=[common],
        help="Run bundle inference and write results CSV",
    )
    p_predict.add_argument(
        "--labels-csv",
        default=str(DEFAULT_LABELS_CSV),
        help="Labeled prompts CSV (prompt,type,output_length)",
    )
    p_predict.add_argument(
        "--bundle",
        required=True,
        help="Predictor bundle directory or bert.pkl",
    )
    p_predict.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV (type,input_length,output_length,prediction)",
    )
    p_predict.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include prompt text column in the results CSV",
    )

    p_plot = sub.add_parser("plot", help="Plot figures from saved results CSV(s)")
    p_plot.add_argument(
        "--results-csv",
        required=True,
        help="Results CSV from the predict subcommand",
    )
    p_plot.add_argument(
        "--results-csv-bert",
        default=None,
        help="Optional second results CSV (legacy BERT) for an extra true-vs-pred figure",
    )
    p_plot.add_argument(
        "--output-dir",
        default=None,
        help="Save figures as PDFs (omit to show interactively)",
    )
    p_plot.add_argument("--bins", default=50, type=int)
    p_plot.add_argument(
        "--no-log-scale",
        action="store_true",
        help="Disable log-scaled axes",
    )
    p_plot.add_argument(
        "--plot-buckets",
        action="store_true",
        help="Also plot hybrid length-bucket confusion matrices (uses pred_bucket column when present)",
    )

    p_all = sub.add_parser(
        "all",
        parents=[common],
        help="predict then plot (convenience; same as two separate steps)",
    )
    p_all.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    p_all.add_argument("--bundle", required=True)
    p_all.add_argument("--output-csv", required=True)
    p_all.add_argument(
        "--results-csv-bert",
        default=None,
        help="Optional BERT results CSV for a second true-vs-pred figure in plot step",
    )
    p_all.add_argument("--output-dir", default=None)
    p_all.add_argument("--include-prompt", action="store_true")
    p_all.add_argument("--bins", default=50, type=int)
    p_all.add_argument("--no-log-scale", action="store_true")
    p_all.add_argument(
        "--plot-buckets",
        action="store_true",
        help="Also plot hybrid bucket classification confusion matrices",
    )

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    num_prompts = args.num_prompts if getattr(args, "num_prompts", 0) and args.num_prompts > 0 else None
    show_progress = not getattr(args, "no_progress", False)
    ascend_devices = getattr(args, "ascend_devices", None) or None
    log_scale = not getattr(args, "no_log_scale", False)
    tokenizer_name = getattr(args, "tokenizer_name", DEFAULT_TOKENIZER)

    if args.command == "predict":
        run_predictor_dump(
            labels_csv=args.labels_csv,
            bundle_path=args.bundle,
            output_csv=args.output_csv,
            tokenizer_name=tokenizer_name,
            batch_size=args.batch_size,
            device=args.device,
            ascend_devices=ascend_devices,
            num_prompts=num_prompts,
            show_progress=show_progress,
            include_prompt=args.include_prompt,
        )
        return 0

    if args.command == "plot":
        run_predictor_plot(
            args.results_csv,
            results_csv_bert=args.results_csv_bert,
            output_dir=args.output_dir,
            bins=args.bins,
            log_scale=log_scale,
            plot_bucket_classification=getattr(args, "plot_buckets", False),
        )
        return 0

    if args.command == "all":
        run_predictor_dump(
            labels_csv=args.labels_csv,
            bundle_path=args.bundle,
            output_csv=args.output_csv,
            tokenizer_name=tokenizer_name,
            batch_size=args.batch_size,
            device=args.device,
            ascend_devices=ascend_devices,
            num_prompts=num_prompts,
            show_progress=show_progress,
            include_prompt=args.include_prompt,
        )
        run_predictor_plot(
            args.output_csv,
            results_csv_bert=args.results_csv_bert,
            output_dir=args.output_dir,
            bins=args.bins,
            log_scale=log_scale,
            plot_bucket_classification=getattr(args, "plot_buckets", False),
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
