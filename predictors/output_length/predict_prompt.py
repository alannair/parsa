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
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="predictors.output_length.predict_prompt",
        description="Predict output token count for one prompt",
    )
    parser.add_argument(
        "--bundle",
        default=str(_REPO_ROOT / "models" / "output_length_deberta"),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--type", default=0, type=int, choices=[0, 1, 2])
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu (default) or npu:0 / Ascend via --ascend-devices",
    )
    parser.add_argument("--ascend-devices", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from predictors.output_length.api import predict_output_tokens

    prediction = predict_output_tokens(
        args.prompt,
        prompt_type=args.type,
        bundle_path=args.bundle,
        device=args.device,
        ascend_devices=args.ascend_devices,
    )

    if args.json:
        print(
            json.dumps(
                {"prompt": args.prompt, "type": args.type, "prediction": prediction},
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print("%.4f" % prediction, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
