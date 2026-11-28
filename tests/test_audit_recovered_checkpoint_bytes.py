from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import audit_recovered_checkpoint_bytes as a
from scripts.audit_ranking_auxiliary import sha


class CheckpointByteAuditTests(unittest.TestCase):
    def test_exact_members_and_role_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"archive.zip"
            best = b"opaque checkpoint best"
            last = b"opaque checkpoint last"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("java-model/java-best.pt", best)
                z.writestr("java-model/java-last.pt", last)
                z.writestr("unknown.pt", b"not read")
            candidate = {"name": "sagallm-v016-pinned-java-cpu",
                         "run_id": 1, "artifact_id": 2, "size_in_bytes": p.stat().st_size,
                         "api_zip_sha256": sha(p.read_bytes())}
            claims = {sha(best): [{"artifact_id": 2, "role": "BEST"}],
                      sha(last): [{"artifact_id": 9, "role": "LAST"}]}
            result = a.verify_archive(p, candidate, claims)
            self.assertEqual(result["members"][0]["status"], "MATCHED_SAME_ARTIFACT_OUTPUT_CLAIM")
            self.assertEqual(result["members"][1]["status"], "MATCHED_OTHER_DECLARATION")
            self.assertFalse(result["members"][0]["checkpoint_deserialized"])
            self.assertEqual(result["unlisted_pt_members"], 1)
            with self.assertRaises(ValueError):
                a.verify_archive(p, {**candidate, "api_zip_sha256": "0"*64}, claims)

    def test_missing_members_and_wrong_claim_role(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"archive.zip"
            raw = b"opaque"
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("java-model/java-best.pt", raw)
            c = {"name": "sagallm-v016-pinned-java-cpu", "run_id": 1, "artifact_id": 2,
                 "size_in_bytes": p.stat().st_size, "api_zip_sha256": sha(p.read_bytes())}
            result = a.verify_archive(p, c, {sha(raw): [{"artifact_id": 2, "role": "PARENT"}]})
            self.assertEqual(result["members"][0]["status"], "MATCHED_OTHER_DECLARATION")
            self.assertEqual(result["members"][1]["status"], "MEMBER_NOT_VISIBLE")
            self.assertFalse(result["members"][1]["checkpoint_bytes_verified"])

    def test_ab_match_does_not_certify_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"archive.zip"
            raw = b"opaque"
            name = "sagallm-java-identifier-ab-validation"
            member = next(iter(a.MEMBERS[name]))
            with zipfile.ZipFile(p, "w") as z:
                z.writestr(member, raw)
            c = {"name": name, "run_id": 1, "artifact_id": 2,
                 "size_in_bytes": p.stat().st_size, "api_zip_sha256": sha(p.read_bytes())}
            r = a.verify_archive(p, c, {sha(raw): [{"artifact_id": 2, "role": "BEST"}]})
            self.assertEqual(r["members"][0]["status"], "MATCHED_SAME_ARTIFACT_OUTPUT_CLAIM_ARM_UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
