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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _log(message):
    print(message, flush=True)


def _run_predict_bundle_worker(
    labels_csv,
    bundle_path,
    output_csv,
    *,
    batch_size=16,
    device=None,
    ascend_devices=None,
    num_prompts=None,
    show_progress=True,
    include_prompt=False,
):
    from predictors.output_length.api import predict_labeled_bundle_csv

    return predict_labeled_bundle_csv(
        labels_csv,
        bundle_path,
        output_csv,
        batch_size=batch_size,
        device=device,
        ascend_devices=ascend_devices,
        num_prompts=0 if num_prompts is None else int(num_prompts),
        include_prompt=include_prompt,
        show_progress=show_progress,
    )


def run_predict_bundle(
    labels_csv,
    bundle_path,
    output_csv,
    *,
    batch_size=16,
    device=None,
    ascend_devices=None,
    num_prompts=None,
    show_progress=True,
    include_prompt=False,
):
    ascend = ascend_devices if ascend_devices is not None else os.environ.get("ASCEND_DEVICES", "0")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--labels-csv",
        str(labels_csv),
        "--bundle",
        str(bundle_path),
        "--output-csv",
        str(output_csv),
        "--batch-size",
        str(int(batch_size)),
        "--ascend-devices",
        str(ascend),
    ]
    if device is not None and str(device).strip():
        cmd.extend(["--device", str(device).strip()])
    if num_prompts is not None and int(num_prompts) > 0:
        cmd.extend(["--num-prompts", str(int(num_prompts))])
    if include_prompt:
        cmd.append("--include-prompt")
    if not show_progress:
        cmd.append("--no-progress")

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    _log("predict_bundle: spawning clean subprocess")
    _log("  %s" % " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    _log("predict_bundle: done -> %s" % output_csv)
    return Path(output_csv)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="predictors.output_length.predict_bundle",
        description="Run output-length bundle predict via safe bundle inference (subprocess)",
    )
    parser.add_argument(
        "--labels-csv",
        "--csv-path",
        dest="labels_csv",
        default=str(REPO_ROOT / "predictors" / "output_length" / "dataset_output_length.csv"),
    )
    parser.add_argument("--bundle", "--model-dir", dest="bundle", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ascend-devices",
        default=os.environ.get("ASCEND_DEVICES", "0"),
    )
    parser.add_argument("--num-prompts", default=0, type=int)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    num_prompts = args.num_prompts if args.num_prompts and args.num_prompts > 0 else None
    if args.worker:
        _run_predict_bundle_worker(
            args.labels_csv,
            args.bundle,
            args.output_csv,
            batch_size=args.batch_size,
            device=args.device,
            ascend_devices=args.ascend_devices,
            num_prompts=num_prompts,
            show_progress=not args.no_progress,
            include_prompt=args.include_prompt,
        )
        return 0
    run_predict_bundle(
        args.labels_csv,
        args.bundle,
        args.output_csv,
        batch_size=args.batch_size,
        device=args.device,
        ascend_devices=args.ascend_devices,
        num_prompts=num_prompts,
        show_progress=not args.no_progress,
        include_prompt=args.include_prompt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
