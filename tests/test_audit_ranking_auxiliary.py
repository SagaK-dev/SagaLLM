import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import audit_ranking_auxiliary as a


class RankingAuditTests(unittest.TestCase):
    def test_pin_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_bytes(b"{}")
            self.assertEqual(a.pin(path, a.sha(b"{}")), b"{}")
            path.write_bytes(b"{}\n")
            with self.assertRaises(ValueError):
                a.pin(path, a.sha(b"{}"))
        with self.assertRaises(ValueError):
            a.parse(b'{"x":1,"x":2}')

    def test_schedule_comparison_rejects_wrong_hash_step_and_checkpoint(self):
        def arm(coefficient, checkpoint):
            return {"schedule_sha256": "a"*64, "coefficient": coefficient,
                    "steps": 160, "selected_step": 160, "broad_rows_seen": 320,
                    "stdio_rows_seen": 320, "aux_rows_seen": 320,
                    "checkpoint_sha256": checkpoint}
        arms = {"control": arm(0, "b"*64), "ranking": arm(1, "c"*64)}
        report = {"results": {"4201": arms}}
        seen = set()
        a.compare_schedule(report, 4201, "a"*64, seen, 320)
        self.assertEqual(seen, {"b"*64, "c"*64})
        with self.assertRaises(ValueError):
            a.compare_schedule(report, 4201, "d"*64, set(), 320)
        with self.assertRaises(ValueError):
            a.compare_schedule(report, 4201, "a"*64, seen, 320)
        arms["ranking"]["selected_step"] = 80
        with self.assertRaises(ValueError):
            a.compare_schedule(report, 4201, "a"*64, set(), 320)

    def test_prior_archive_and_scope(self):
        inv = {"schema": "sagallm-private-java-exposure-v1", "files":
               [{"path": str(i)} for i in range(11)]}
        raw = a.canonical(inv)
        evidence = {"inventory_sha256": a.sha(raw),
                    "status": "SCOPED_DEVELOPMENT_SOURCE_BYTES_VERIFIED",
                    "ancestor_entries_referenced": 6, "new_entries": 5,
                    "full_exposure_inventory": "NOT_ESTABLISHED",
                    "actual_exposure_verified": False,
                    "inventory_ready_for_independent_evaluation": False,
                    "model_selection_permitted": False, "independent_evaluation": "NOT_RUN"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.zip"
            def write():
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("scoped-development-inventory.json", raw)
                    z.writestr("development-inventory-evidence.json", a.canonical(evidence))
                return a.sha(path.read_bytes())
            with patch.object(a, "PRIOR_ZIP", write()):
                self.assertEqual(a.prior(path)[0], inv)
                path.write_bytes(path.read_bytes() + b"x")
                with self.assertRaises(ValueError):
                    a.prior(path)
            evidence["model_selection_permitted"] = True
            with patch.object(a, "PRIOR_ZIP", write()):
                with self.assertRaises(ValueError):
                    a.prior(path)


if __name__ == "__main__":
    unittest.main()
