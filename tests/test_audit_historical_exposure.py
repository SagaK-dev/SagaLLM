import unittest
from scripts.audit_historical_exposure import collect, audit
from pathlib import Path


class ExposureLedgerTests(unittest.TestCase):
    def test_nested_deduplication(self):
        x = collect({"data_manifest_sha256": "a"*64,
                     "models": [{"rows": [{"id": "task-1"}, {"ids": ["task-1", "task-2"]}]}]})
        self.assertEqual(x["reported_ids"], ["task-1", "task-2"])
        self.assertEqual(len(x["declared_data_hashes"]), 1)
        with self.assertRaises(ValueError):
            collect({"data_manifest_sha256": "bad"})

    def test_real_archive(self):
        result = audit(Path("experiments/archive"))
        self.assertEqual(result["report_count"], 12)
        self.assertFalse(result["model_selection_permitted"])
        self.assertTrue(all(not r["actual_exposure_verified"] for r in result["reports"]))
