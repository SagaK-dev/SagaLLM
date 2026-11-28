import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sagallm.java_algorithm_curriculum import rows, sha, canonical
from sagallm.java_algorithm_compact_tasks import COMPACT_TASKS
from scripts.audit_algorithm_curriculum_admission import sft_candidate
from scripts.audit_compact_algorithm_contract_boundaries import (
    CONTRACTS, bind_records, checked_cases, read_compact,
)


class CompactContractBoundaryTests(unittest.TestCase):
    def report(self):
        observations = []
        for r in rows():
            task = COMPACT_TASKS.get(r["id"], r["task"])
            old = sft_candidate(r, "algorithm-v1-")
            new = sft_candidate({**r, "task": task}, "algorithm-compact-v1-")
            observations.append({"id": new["id"], "baseline_id": old["id"],
                                 **{k:r[k] for k in ("family", "variant", "language")},
                                 "baseline_task_sha256": sha(r["task"].encode()),
                                 "task_sha256": sha(task.encode()),
                                 "source_sha256": sha(r["source"].encode()),
                                 "candidate_row_sha256": sha(canonical(new)),
                                 "task_changed": task != r["task"], "compact_positions": 384})
        return {"context": 384, "training_performed": False,
                "semantic_review": "REVIEW_REQUIRED", "rights_review": "REVIEW_REQUIRED",
                "rows": observations,
                "membership_sha256": sha(canonical(sorted(r["candidate_row_sha256"] for r in observations)))}

    def test_explicit_cases_are_new_and_match_oracle(self):
        for family in CONTRACTS:
            with self.subTest(family=family):
                self.assertEqual(len(checked_cases(family)),10)

    def test_bad_expectation_rejected(self):
        with patch("scripts.audit_compact_algorithm_contract_boundaries.oracle", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "expectation"):
                checked_cases("edit_distance")

    def test_exact_record_binding(self):
        records, sources = bind_records(list(rows()),self.report())
        self.assertEqual((len(records),len(sources)),(16,8))

    def test_record_tampering_rejected(self):
        for key, value in (("task_sha256","0"*64), ("source_sha256","0"*64),
                           ("compact_positions",385), ("task_changed",None),
                           ("language","other")):
            report = self.report()
            report["rows"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bind_records(list(rows()),report)

    def test_wrong_membership_rejected(self):
        report = self.report()
        report["membership_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError,"Membership"):
            bind_records(list(rows()),report)

    def test_wrong_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"compact.zip"
            path.write_bytes(b"0"*3326)
            with self.assertRaisesRegex(ValueError,"pin"):
                read_compact(path)
