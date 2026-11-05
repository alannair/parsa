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
import shutil

from .lmsys_data import default_lmsys_dataset_dir, saved_dataset_dir_exists


def build_parser():
    parser = argparse.ArgumentParser(prog="src.download_lmsys_dataset")
    parser.add_argument("--dataset", default="lmsys/lmsys-chat-1m")
    parser.add_argument("--output-dir", default=default_lmsys_dataset_dir())
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def download_dataset(dataset_name, output_dir, max_samples=None, force=False):
    target = os.path.expanduser(output_dir)
    if saved_dataset_dir_exists(target):
        if not force:
            return target
        shutil.rmtree(target)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    from datasets import load_dataset

    split = "train"
    if max_samples and int(max_samples) > 0:
        split = "train[:%d]" % int(max_samples)
    dataset = load_dataset(dataset_name, split=split, token=True)
    dataset.save_to_disk(target)
    return target


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    target = download_dataset(args.dataset, args.output_dir, max_samples=args.max_samples, force=args.force)
    print(target)


if __name__ == "__main__":
    main()
