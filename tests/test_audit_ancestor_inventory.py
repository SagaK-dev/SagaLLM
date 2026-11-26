import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import audit_ancestor_inventory as a


class AncestorInventoryTests(unittest.TestCase):
    def test_scan_exact_bytes_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "public").mkdir()
            path = root / "public/train.jsonl"
            raw = b'{"text":"one"}\r\n{"text":"two"}'
            pin = hashlib.sha256(raw).hexdigest()
            path.write_bytes(raw)
            self.assertEqual(a.scan_file(root, "public/train.jsonl", "text-jsonl", pin, 2),
                             {"rows": 2, "bytes": len(raw), "sha256": pin})
            for expected, rows, fmt in (("0"*64, 2, "text-jsonl"), (pin, 1, "text-jsonl"),
                                        (pin, 2, "chat-jsonl")):
                with self.assertRaises(ValueError):
                    a.scan_file(root, "public/train.jsonl", fmt, expected, rows)
            path.write_bytes(b'{"text":"one","text":"two"}\n')
            with self.assertRaises(ValueError):
                a.scan_file(root, "public/train.jsonl", "text-jsonl",
                            hashlib.sha256(path.read_bytes()).hexdigest(), 1)
            path.unlink()
            path.symlink_to(root / "missing")
            with self.assertRaises(ValueError):
                a.scan_file(root, "public/train.jsonl", "text-jsonl", pin, 2)

    def test_map_recount_and_tamper(self):
        reports = a.load_reports(a.ARCHIVE)
        mapping = a.map_reports(reports)
        mapping["reconstruction_evidence"] = {
            "run_id": 45790130406, "artifact_id": 15101943417,
            "archive_sha256": a.DEVELOPMENT_ZIP, "member_sha256": "a"*64}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.zip"

            def write(value):
                raw = json.dumps(value).encode()
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("checkpoint-exposure-map.json", raw)
                return hashlib.sha256(path.read_bytes()).hexdigest()

            with patch.object(a, "MAP_ZIP", write(mapping)):
                result, _ = a.read_map(path, reports)
                self.assertEqual(result["checkpoint_count"], 12)
                path.write_bytes(path.read_bytes() + b"x")
                with self.assertRaises(ValueError):
                    a.read_map(path, reports)
            modified = copy.deepcopy(mapping)
            modified["checkpoints"][0]["inventory_ready"] = True
            with patch.object(a, "MAP_ZIP", write(modified)):
                with self.assertRaises(ValueError):
                    a.read_map(path, reports)

    def test_build_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = {"file_hashes": {}, "splits": {}}
            sft = {"files": {}}
            for corpus, data in (("public", {"text": "code"}),
                                 ("sft", {"messages": [{"role": "user", "content": "task"},
                                                        {"role": "assistant", "content": "code"}]})):
                directory = root / corpus
                directory.mkdir()
                for split in a.SPLITS:
                    raw = (json.dumps(dict(data, split=split)) + "\n").encode()
                    (directory / (split + ".jsonl")).write_bytes(raw)
                    pin = hashlib.sha256(raw).hexdigest()
                    if corpus == "public":
                        public["file_hashes"][split] = pin
                        public["splits"][split] = 1
                    else:
                        sft["files"][split] = {"sha256": pin, "count": 1}
            with patch.object(a, "verify"):
                raw, rows = a.build(root, public, "a"*64, sft, a.DATA)
            inv = json.loads(raw)
            self.assertEqual(len(a._inventory(inv)), 6)
            self.assertEqual(len(rows), 6)
            self.assertEqual(inv["files"][0]["group"], "pretraining")
            self.assertEqual({x["group"] for x in inv["files"]},
                             {"pretraining", "train", "validation", "test"})
