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

"""Download pinned specialization inputs; never launch serving or generate schedules."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import tempfile
import urllib.request
from pathlib import Path


HF_DATASETS = (
    ("sharegpt52k", {
        "repo_id": "RyokoAI/ShareGPT52K",
        "revision": "6f9b78cc1dd15dbb51d3c51ccc219c558962fd77",
        "config": None, "split": None, "format": "json",
    }, ("sg_90k_part1.json", "sg_90k_part2.json")),
    ("arxiv-summarization", {
        "repo_id": "ccdv/arxiv-summarization",
        "revision": "240aaf1a969b3f8cd0ade6986bfad0cd730ee288",
        "config": "document", "split": "train", "format": "jsonl", "num_rows": 203037,
    }, ("train.jsonl",)),
    ("humaneval", {
        "repo_id": "openai/openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "config": "openai_humaneval", "split": "test", "format": "jsonl", "num_rows": 164,
    }, ("test.jsonl",)),
)

# Exact native Parquet paths at the revisions above, in deterministic shard order.
PARQUET_FILES = {
    ("ccdv/arxiv-summarization", "document", "train"): tuple(
        "document/train-%05d-of-00015.parquet" % index for index in range(15)
    ),
    ("openai/openai_humaneval", "openai_humaneval", "test"): (
        "openai_humaneval/test-00000-of-00001.parquet",
    ),
}

# Official Azure/AzurePublicDataset Git tree metadata; these CSVs contain no prompts.
AZURE_SOURCE = {
    "repo_id": "Azure/AzurePublicDataset",
    "revision": "215becdacba1ce682c7368642ce97d5a332de7a6",
    "dataset": "Azure LLM inference trace 2023", "format": "csv",
    "git_blobs": {
        "AzureLLMInferenceTrace_conv.csv": {
            "size_bytes": 719188, "sha1": "5b9219f2caf93bbbf95ec29894e206115658e235",
        },
        "AzureLLMInferenceTrace_code.csv": {
            "size_bytes": 320117, "sha1": "62bd8109d0f2b1bc48206d397ecca48d8de03aab",
        },
    },
}
MANIFEST = "download-manifest.json"
FORMAT_VERSION = 1
CHUNK_SIZE = 1024 * 1024


def fingerprint(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("missing, empty, or non-regular dataset file: %s" % path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify_git_blob(path, size, expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(("blob %d\0" % size).encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def validate_existing(destination, source, files):
    try:
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("destination is not a regular directory")
        with (destination / MANIFEST).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format_version") != FORMAT_VERSION or manifest.get("source") != source or set(manifest.get("files", {})) != set(files):
            raise ValueError("source revision or manifest differs")
        for name in files:
            if fingerprint(destination / name) != manifest["files"][name]:
                raise ValueError("size or SHA-256 mismatch: %s" % name)
        for name, expected in source.get("git_blobs", {}).items():
            if not verify_git_blob(destination / name, expected["size_bytes"], expected["sha1"]):
                raise ValueError("Azure Git blob mismatch: %s" % name)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Cannot reuse %s: %s. Data left untouched; use a fresh --output-dir." % (destination, exc)) from exc


def prepare_dataset(root, name, source, files, download):
    destination = root / name
    if os.path.lexists(destination):
        validate_existing(destination, source, files)
        print("Verified reuse: %s" % destination, flush=True)
        return destination
    # ponytail: interrupted exports restart; add chunk checkpoints only if this becomes costly.
    # One downloader at a time: no concurrent writers to the same output directory.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=str(root)) as temporary:
        stage = Path(temporary) / name
        stage.mkdir()
        download(stage)
        manifest = {"format_version": FORMAT_VERSION, "source": source,
                    "files": {filename: fingerprint(stage / filename) for filename in files}}
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if os.path.lexists(destination):
            raise FileExistsError("destination appeared during download; refusing to replace: %s" % destination)
        stage.rename(destination)
    print("Published: %s" % destination, flush=True)
    return destination


def write_jsonl(path, records, expected_rows):
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if count != expected_rows:
        raise ValueError("incomplete dataset export: expected %d rows, got %d" % (expected_rows, count))
    print("Exported %d records: %s" % (count, path.name), flush=True)
    return count


def download_hf(stage, source, files):
    from huggingface_hub import snapshot_download

    if source["split"] is None:
        snapshot = snapshot_download(repo_id=source["repo_id"], repo_type="dataset",
                                     revision=source["revision"], allow_patterns=list(files))
        for name in files:
            shutil.copyfile(Path(snapshot) / name, stage / name)
    else:
        from pyarrow import parquet

        shards = PARQUET_FILES[(source["repo_id"], source["config"], source["split"])]
        snapshot = snapshot_download(repo_id=source["repo_id"], repo_type="dataset",
                                     revision=source["revision"], allow_patterns=list(shards))

        def records():
            for shard in shards:
                with (Path(snapshot) / shard).open("rb") as handle:
                    for batch in parquet.ParquetFile(handle).iter_batches(batch_size=64):
                        yield from batch.to_pylist()

        write_jsonl(stage / files[0], records(), source["num_rows"])


def download_azure(stage):
    prefix = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/%s/data/" % AZURE_SOURCE["revision"]
    for name, expected in AZURE_SOURCE["git_blobs"].items():
        path = stage / name
        with urllib.request.urlopen(prefix + name, timeout=60) as response, path.open("xb") as handle:
            shutil.copyfileobj(response, handle, length=CHUNK_SIZE)
        if not verify_git_blob(path, expected["size_bytes"], expected["sha1"]):
            raise ValueError("Azure download failed its pinned size/Git blob check: %s" % name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/datasets", help="Dataset root; use /data1/datasets when running on the host")
    args = parser.parse_args(argv)
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / ".hf-cache"))
    print("Full datasets, no sampling. Allow substantial disk space; run only one downloader per output root.", flush=True)
    # Validate every existing destination before starting any network downloads.
    for name, source, files in HF_DATASETS:
        if os.path.lexists(root / name):
            validate_existing(root / name, source, files)
    if os.path.lexists(root / "azure-mix"):
        validate_existing(root / "azure-mix", AZURE_SOURCE, tuple(AZURE_SOURCE["git_blobs"]))
    prepare_dataset(root, "azure-mix", AZURE_SOURCE, tuple(AZURE_SOURCE["git_blobs"]), download_azure)
    for name, source, files in HF_DATASETS:
        prepare_dataset(root, name, source, files, lambda stage: download_hf(stage, source, files))
    print("\nInputs for the existing schedule generator (not executed):")
    for option, path in (
        ("--azure-conv-csv", root / "azure-mix" / "AzureLLMInferenceTrace_conv.csv"),
        ("--azure-code-csv", root / "azure-mix" / "AzureLLMInferenceTrace_code.csv"),
        ("--sharegpt52k", root / "sharegpt52k"),
        ("--arxiv-summarization", root / "arxiv-summarization" / "train.jsonl"),
        ("--openai-humaneval", root / "humaneval" / "test.jsonl"),
    ):
        print("%s %s" % (option, shlex.quote(str(path))))
    print("Azure CSVs provide arrival times and token counts, not prompt text.")
    print("Azure source/license: https://github.com/Azure/AzurePublicDataset/blob/%s/AzureLLMInferenceDataset2023.md" % AZURE_SOURCE["revision"])


if __name__ == "__main__":
    main()
