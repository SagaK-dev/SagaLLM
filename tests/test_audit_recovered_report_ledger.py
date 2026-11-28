from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import audit_recovered_report_ledger as a


class RecoveredLedgerTests(unittest.TestCase):
    def test_declarations_not_generated_hashes(self):
        cp, data, generated = "a"*64, "b"*64, "c"*64
        models, sources = a.declarations({"best_checkpoint_sha256": cp,
            "generated_source_sha256": generated, "results": [{"data_manifest_sha256": data,
            "checkpoint_sha256": cp, "schedule_sha256": generated}]})
        self.assertEqual({m["sha256"] for m in models}, {cp})
        self.assertEqual(len(models), 2)
        self.assertEqual([s["sha256"] for s in sources], [data])

    def test_read_exact_archive_and_reject_extra(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"recovered.zip"
            def write(extra=False):
                with zipfile.ZipFile(p, "w") as z:
                    z.writestr("recovery-manifest.json", '{"status":"test"}')
                    z.writestr("reports/1-2-0.json", b'{}')
                    if extra:
                        z.writestr("unexpected.json", b'{}')
                return a.sha(p.read_bytes())
            digest = write()
            manifest, blobs = a.read_recovery(p, digest)
            self.assertEqual(manifest["status"], "test")
            self.assertEqual(blobs, {"reports/1-2-0.json": b'{}'})
            with self.assertRaises(ValueError):
                a.read_recovery(p, "0"*64)
            with self.assertRaises(ValueError):
                a.read_recovery(p, write(True))

    def test_validate_and_detect_tamper(self):
        raw = b'{"status":"TRAINED_CPU","best_checkpoint_sha256":"' + b"a"*64 + b'"}'
        candidate = {"artifact_id": 2, "run_id": 1, "commit_sha": "b"*40,
                     "name": "sagallm-v016-pinned-java-cpu", "size_in_bytes": 123,
                     "api_zip_sha256": "c"*64, "run_binding": "MATCH",
                     "status": "METADATA_READY_FOR_BYTE_VERIFICATION",
                     "archive_bytes_verified": False, "expires_at": "2027-01-01T00:00:00Z"}
        gate = {"actual_exposure_verified": False, "training_lineage_verified": False,
                "full_exposure_inventory": "NOT_ESTABLISHED", "independent_evaluation": "NOT_RUN",
                "model_selection_permitted": False}
        triage = {**gate, "status": "SCOPED_ARTIFACT_METADATA_TRIAGE",
            "triage": {"run_count": 1, "artifact_count": 1,
                       "runs": [{"run_id": 1,"commit_sha": "b"*40, "artifacts": [candidate]}],
                       "priority_by_expiry": [candidate]}}
        member = {"member": "java-model/train-report.json", "status": "METADATA_SCREEN_PASSED_NOT_RIGHTS_REVIEW",
                  "preserved_file": "1-2-0.json", "bytes": len(raw), "report_sha256": a.sha(raw)}
        rec = {"run_id": 1, "artifact_id": 2, "commit_sha": "b"*40, "artifact_name": candidate["name"],
               "archive_sha256": candidate["api_zip_sha256"], "archive_bytes_verified": True,
               "status": "ZIP_VERIFIED", "report_members": [member], "preserved_bytes": len(raw)}
        manifest = {**gate, "schema": "sagallm-recovered-historical-reports-v1",
                    "status": "BOUNDED_RECOVERY_ATTEMPT_COMPLETE", "all_reports_recovered": False,
                    "new_training": False, "new_inference": False, "new_java_execution": False,
                    "prior_evidence": {"archive_sha256": a.TRIAGE_ZIP, "run_id": a.TRIAGE_RUN, "artifact_id": a.TRIAGE_ARTIFACT},
                    "candidate_count": 1, "artifacts": [rec], "preserved_report_count": 1,
                    "preserved_bytes": len(raw), "zip_verified_count": 1, "all_candidate_archives_verified": True,
                    "artifact_status_counts": {"ZIP_VERIFIED": 1},
                    "report_member_status_counts": {member["status"]: 1}}
        blobs = {"reports/1-2-0.json": raw}
        rows = a.validate(manifest, blobs, triage)
        self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError):
            a.validate(manifest, {"reports/1-2-0.json": raw+b" "}, triage)
        with self.assertRaises(ValueError):
            a.validate(manifest, {**blobs, "reports/1-2-1.json": b"{}"}, triage)
        bad = deepcopy(manifest)
        bad["artifacts"][0]["archive_sha256"] = "d"*64
        with self.assertRaises(ValueError):
            a.validate(bad, blobs, triage)

    def test_report_dedup_does_not_certify_models(self):
        d = {"candidate_checkpoint_sha256": "e"*64, "tokenizer_sha256": "f"*64}
        rows = [{"run_id": n, "artifact_id": n, "report_sha256": "a"*64, "data": d} for n in (1,2)]
        triage = {"combined_scope": {"actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False, "checkpoint_count": 18,
            "checkpoint_declarations": [{"checkpoint_sha256": f"{i:064x}"} for i in range(18)]}}
        result = a.ledger(rows, triage)
        self.assertEqual(result["distinct_report_sha256_count"], 1)
        self.assertEqual(result["duplicate_report_instances"], 1)
        self.assertEqual(result["unmapped_checkpoint_hash_claims"], ["e"*64])
        self.assertFalse(result["actual_exposure_verified"])


if __name__ == "__main__":
    unittest.main()
