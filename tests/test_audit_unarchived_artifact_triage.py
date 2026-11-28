from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import audit_unarchived_artifact_triage as a


GATE = {"actual_exposure_verified": False, "training_lineage_verified": False,
        "full_exposure_inventory": "NOT_ESTABLISHED", "independent_evaluation": "NOT_RUN",
        "model_selection_permitted": False}
OBSERVED = "2026-11-28T00:00:00Z"


def run_row(n=1):
    return {"id": n, "commit_sha": "a"*40, "path": ".github/workflows/java-context-sft-cpu.yml",
            "status": "completed", "conclusion": "success", "classification": "RESEARCH_OR_TRAINING_REVIEW",
            "evidence": "NO_PINNED_REPORT_IN_THIS_SCOPE", "actual_exposure_verified": False}


def artifact(n=10):
    return {"id": n, "name": "research-evidence", "size_in_bytes": 1234,
            "expired": False, "created_at": "2026-11-25T00:00:00Z",
            "expires_at": "2027-01-01T00:00:00Z", "digest": "sha256:" + "b"*64,
            "workflow_run": {"id": 1, "head_sha": "a"*40}}


class ArtifactTriageTests(unittest.TestCase):
    def test_metadata_ready_not_verified(self):
        result = a.artifact_row(artifact(), run_row(), OBSERVED)
        self.assertEqual(result["status"], "METADATA_READY_FOR_BYTE_VERIFICATION")
        self.assertFalse(result["archive_bytes_verified"])
        self.assertIsNone(result["member_report_sha256"])
        for fields, blocker in [
            ({"digest": None}, "SHA256_NOT_RECORDED"),
            ({"workflow_run": {"id": 2, "head_sha": "a"*40}}, "RUN_COMMIT_BINDING_CONFLICT"),
            ({"workflow_run": None}, "RUN_COMMIT_BINDING_MISSING"),
            ({"expired": True}, "EXPIRED_OR_PAST_EXPIRY"),
            ({"size_in_bytes": a.MAX_ARCHIVE+1}, "SIZE_OUTSIDE_CURRENT_ARCHIVE_LIMIT"),
        ]:
            r = a.artifact_row({**artifact(), **fields}, run_row(), OBSERVED)
            self.assertEqual(r["status"], "BLOCKED_METADATA")
            self.assertIn(blocker, r["blockers"])

    def test_pages_and_queue_reject_drift(self):
        run = run_row()
        d = {"run_discovery": {"runs": [run], "live_count": 1,
                               "unarchived_research_or_unknown_run_ids": [1]}}
        out = a.triage(d, OBSERVED, lambda r, t: a.artifact_pages(
            [{"total_count": 1, "artifacts": [artifact()]}], r, t))
        self.assertEqual(out["artifact_count"], 1)
        self.assertEqual(out["priority_by_expiry"][0]["artifact_id"], 10)
        self.assertFalse(out["archive_bytes_verified"])
        d["run_discovery"]["unarchived_research_or_unknown_run_ids"] = []
        with self.assertRaises(ValueError):
            a.review_queue(d)
        with self.assertRaises(ValueError):
            a.artifact_pages([{"total_count": 2, "artifacts": [artifact()]}], run, OBSERVED)

    def test_join_only_declared_pools(self):
        pool = ["sft/train.jsonl", "stdio/train.jsonl", "stdio/validation.jsonl",
                "derived/stdio_validation_oracles.jsonl"]
        records = [{"model": "contrastive-stdio-ab-candidate" if i == 0 else f"m-{i}",
                    "checkpoint_sha256": f"{i+1:064x}", "parent_checkpoint_sha256_declared": a.BASE,
                    "sources": [{"path": p, "sha256": "a"*64} for p in pool],
                    "actual_exposure_verified": False, "checkpoint_bytes_verified_here": False,
                    "inventory_ready": False} for i in range(17)]
        primary = {**GATE, "status": "ARCHIVED_REPORT_SCOPE_INTEGRATED_NOT_COMPLETE",
                   "archive_report_count": 12, "archive_report_scope": [{"run_id": i} for i in range(12)],
                   "checkpoint_count": 17, "checkpoint_declarations": records,
                   "unresolved_checkpoint_declarations": [{"checkpoint_sha256": None}],
                   "inventory_entries": 15, "inventory_sha256": "c"*64,
                   "inventory_ready_for_independent_evaluation": False}
        discovery = {**GATE, "status": "SCOPED_HOSTED_RUN_DISCOVERY", "archive_report_count": 15,
                     "run_discovery": {"archive_run_ids": list(range(12)) + [r[0] for r in a.SPECS.values()]},
                     "stdio_scope": {"candidate_checkpoint_sha256_declared": a.SEMANTIC_CANDIDATE,
                         "parent_checkpoint_sha256_declared": a.BASE, "selected_step_declared": 160,
                         "source_pool_declarations": pool, "actual_exposure_verified": False,
                         "checkpoint_bytes_verified_here": False},
                     "public_scope": {"suite_sha256_declared": a.PUBLIC_SUITE,
                         "source_bytes_verified": False, "matched_full_suite": False,
                         "independent_evaluation": "NOT_RUN"}}
        out = a.join_scope(primary, discovery)
        self.assertEqual(out["checkpoint_count"], 18)
        self.assertEqual(out["new_inventory_entries"], 0)
        self.assertFalse(out["inventory_ready_for_independent_evaluation"])
        bad = deepcopy(discovery)
        bad["stdio_scope"]["actual_exposure_verified"] = True
        with self.assertRaises(ValueError):
            a.join_scope(primary, bad)

    def test_zip_members_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"prior.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("one.json", '{"x":1}')
            digest = a.sha(path.read_bytes())
            self.assertEqual(a.read_evidence(path, digest, ["one.json"])[0]["one.json"], {"x":1})
            with self.assertRaises(ValueError):
                a.read_evidence(path, digest, ["two.json"])
            with self.assertRaises(ValueError):
                a.read_evidence(path, "0"*64, ["one.json"])


if __name__ == "__main__":
    unittest.main()
