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

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import download_datasets as downloader


class DatasetDownloadTests(unittest.TestCase):
    def test_git_blob_validation_rejects_wrong_size_and_contents(self):
        content = b"TIMESTAMP,ContextTokens,GeneratedTokens\n0,10,20\n"
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            path.write_bytes(content)
            self.assertTrue(downloader.verify_git_blob(path, len(content), digest))
            self.assertFalse(downloader.verify_git_blob(path, len(content) + 1, digest))
            path.write_bytes(content.replace(b"10,20", b"11,20"))
            self.assertFalse(downloader.verify_git_blob(path, len(content), digest))

    def test_failed_download_is_not_published_and_existing_data_is_preserved(self):
        source = {"repo_id": "test/source", "revision": "pinned"}

        def fail_download(stage):
            (stage / "data.jsonl").write_text("partial", encoding="utf-8")
            raise RuntimeError("download interrupted")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                downloader.prepare_dataset(root, "new", source, ("data.jsonl",), fail_download)
            self.assertFalse((root / "new").exists())
            self.assertEqual(list(root.iterdir()), [])
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                downloader.prepare_dataset(root, "existing", source, ("data.jsonl",), fail_download)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_reuse_requires_matching_manifest_and_file_hashes(self):
        source = {"repo_id": "test/source", "revision": "pinned"}
        calls = []

        def download(stage):
            calls.append(stage)
            (stage / "data.jsonl").write_text('{"article":"example"}\n', encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = downloader.prepare_dataset(root, "dataset", source, ("data.jsonl",), download)
            self.assertEqual(downloader.prepare_dataset(root, "dataset", source, ("data.jsonl",), download), destination)
            self.assertEqual(len(calls), 1)
            with self.assertRaises(ValueError):
                downloader.prepare_dataset(root, "dataset", dict(source, revision="different"), ("data.jsonl",), download)
            (destination / "data.jsonl").write_text("corrupt", encoding="utf-8")
            with self.assertRaises(ValueError):
                downloader.prepare_dataset(root, "dataset", source, ("data.jsonl",), download)
            self.assertEqual((destination / "data.jsonl").read_text(encoding="utf-8"), "corrupt")
            self.assertEqual(len(calls), 1)

    def test_jsonl_export_streams_every_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            records = ({"article": str(index), "abstract": "summary"} for index in range(3))
            self.assertEqual(downloader.write_jsonl(path, records, expected_rows=3), 3)
            with path.open(encoding="utf-8") as handle:
                self.assertEqual([json.loads(line)["article"] for line in handle], ["0", "1", "2"])

    def test_pinned_parquet_exports_do_not_import_datasets(self):
        for _, original_source, output_files in downloader.HF_DATASETS[1:]:
            source = dict(original_source)
            key = (source["repo_id"], source["config"], source["split"])
            if source["repo_id"] == "ccdv/arxiv-summarization":
                shards = tuple("document/train-%05d-of-00015.parquet" % index for index in range(15))
                self.assertEqual(original_source["num_rows"], 203037)
            else:
                shards = ("openai_humaneval/test-00000-of-00001.parquet",)
                self.assertEqual(original_source["num_rows"], 164)
            self.assertEqual(downloader.PARQUET_FILES[key], shards)
            source["num_rows"] = len(shards) * 2
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snapshot = root / "snapshot"
                stage = root / "stage"
                stage.mkdir()
                expected = []
                for shard in shards:
                    path = snapshot / shard
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"mock parquet")
                    expected.extend([
                        {"prompt": shard + ":first", "keep": {"field": "unchanged"}},
                        {"prompt": shard + ":second", "keep": "unicode: α"},
                    ])
                opened = []

                class Batch:
                    def __init__(self, rows):
                        self.rows = rows

                    def to_pylist(self):
                        return self.rows

                class Reader:
                    def __init__(self, handle):
                        self.path = Path(handle.name)
                        opened.append(self.path.relative_to(snapshot).as_posix())

                    def iter_batches(self, batch_size):
                        self_shard = self.path.relative_to(snapshot).as_posix()
                        self_rows = [row for row in expected if row["prompt"].startswith(self_shard + ":")]
                        if batch_size != 64:
                            raise AssertionError("export must use bounded record batches")
                        yield Batch(self_rows)

                hub = types.ModuleType("huggingface_hub")
                hub.snapshot_download = Mock(return_value=str(snapshot))
                arrow = types.ModuleType("pyarrow")
                parquet = types.ModuleType("pyarrow.parquet")
                parquet.ParquetFile = Reader
                arrow.parquet = parquet
                with patch.dict(sys.modules, {"huggingface_hub": hub, "pyarrow": arrow,
                                              "pyarrow.parquet": parquet, "datasets": None}):
                    downloader.download_hf(stage, source, output_files)
                hub.snapshot_download.assert_called_once_with(
                    repo_id=source["repo_id"], repo_type="dataset", revision=source["revision"],
                    allow_patterns=list(shards),
                )
                self.assertEqual(opened, list(shards))
                with (stage / output_files[0]).open(encoding="utf-8") as handle:
                    self.assertEqual([json.loads(line) for line in handle], expected)


if __name__ == "__main__":
    unittest.main()
