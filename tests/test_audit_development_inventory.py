import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import audit_development_inventory as a


def fixture(source=False):
    rows = []
    for i in range(6):
        row = {"id": str(i), "family": "f", "task": "task "+str(i),
               "cases": [{"stdin": "1\n", "stdout": "2\n"}] * 3}
        if source:
            row["source"] = "class Main {}"
        rows.append(row)
    return {"schema": "fixture", "independent_benchmark": False, "rows": rows}


class DevelopmentInventoryTests(unittest.TestCase):
    def test_conversion_is_deterministic_and_complete(self):
        for source, count in ((False, 42), (True, 48)):
            raw = a.canonical(fixture(source))
            rows = a.convert_suite(raw, "fixture", source)
            self.assertEqual(len(rows), count)
            self.assertEqual(rows, a.convert_suite(raw, "fixture", source))
            self.assertEqual(rows[0]["field"], "task")
            self.assertIn("2\n", [row["text"] for row in rows])
            self.assertEqual(any(row["field"] == "source" for row in rows), source)

    def test_suite_rejects_tampering_and_unexpected_fields(self):
        for field, value in (("independent_benchmark", True), ("schema", "other")):
            obj = fixture()
            obj[field] = value
            with self.assertRaises(ValueError):
                a.convert_suite(a.canonical(obj), "fixture", False)
        obj = fixture()
        obj["rows"][1]["id"] = obj["rows"][0]["id"]
        with self.assertRaises(ValueError):
            a.convert_suite(a.canonical(obj), "fixture", False)
        obj = fixture()
        obj["rows"][0]["cases"][0]["stdout"] = 1
        with self.assertRaises(ValueError):
            a.convert_suite(a.canonical(obj), "fixture", False)

    def test_derived_bytes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "derived").mkdir()
            rows = a.convert_suite(a.canonical(fixture()), "fixture", False)
            entry, summary = a.derived(root, "suite", rows, "a"*64, "development")
            raw = (root / entry["path"]).read_bytes()
            self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(summary["rows"], 42)
            self.assertEqual(summary["original_sha256"], "a"*64)
            with self.assertRaises(FileExistsError):
                a.derived(root, "suite", rows, "a"*64, "development")

    def test_oracle_count_and_identity(self):
        row = {"id": "x", "family": "f",
               "cases": [{"stdin": "", "stdout": "\n"}] * 3}
        raw = a.canonical(row) + b"\n"
        self.assertEqual(len(a.convert_oracles(raw, 1)), 6)
        with self.assertRaises(ValueError):
            a.convert_oracles(raw, 2)
        with self.assertRaises(ValueError):
            a.convert_oracles(raw * 2, 2)

    def test_ancestor_archive_scope_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ancestor.zip"
            entries = [{"group": "train", "path": f"f{i}.jsonl",
                        "sha256": "a"*64, "format": "chat-jsonl"} for i in range(6)]
            inv = a.canonical({"schema": a.INVENTORY_SCHEMA, "files": entries})
            evidence = {"source_count": 6, "inventory_sha256": a.sha(inv),
                        "status": "SCOPED_ANCESTOR_SOURCE_BYTES_VERIFIED",
                        "full_exposure_inventory": "NOT_ESTABLISHED",
                        "actual_exposure_verified": False,
                        "inventory_ready_for_independent_evaluation": False,
                        "model_selection_permitted": False, "independent_evaluation": "NOT_RUN"}

            def write():
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("ancestor-inventory.json", inv)
                    z.writestr("ancestor-inventory-evidence.json", a.canonical(evidence))
                return a.sha(path.read_bytes())

            with patch.object(a, "ANCESTOR_ZIP", write()):
                self.assertEqual(a.ancestor(path)[0], entries)
                path.write_bytes(path.read_bytes() + b"x")
                with self.assertRaises(ValueError):
                    a.ancestor(path)
            evidence["actual_exposure_verified"] = True
            with patch.object(a, "ANCESTOR_ZIP", write()):
                with self.assertRaises(ValueError):
                    a.ancestor(path)


if __name__ == "__main__":
    unittest.main()
