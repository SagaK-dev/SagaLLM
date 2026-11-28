from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from scripts import archive_recovered_originals as a


class PermanentOriginalTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        (root/"reports").mkdir()
        artifacts = []
        total = 0
        for i in range(33):
            name = ("sagallm-java-train-only-identifier-candidates" if i < 2
                    else "sagallm-v016-pinned-java-cpu")
            members = []
            for j, member in enumerate(a.ALLOWED[name]):
                filename = f"{i+1}-{i+100}-{j}.json"
                raw = json.dumps({"status":"test","i":i,"j":j}).encode()
                (root/"reports"/filename).write_bytes(raw)
                total += len(raw)
                members.append({"member":member,"preserved_file":filename,"bytes":len(raw),
                                "report_sha256":a.sha(raw),"status":"METADATA_SCREEN_PASSED_NOT_RIGHTS_REVIEW"})
            artifacts.append({"run_id":i+1,"artifact_id":i+100,"artifact_name":name,
                              "archive_bytes_verified":True,"status":"ZIP_VERIFIED","report_members":members})
        d = {"schema":"sagallm-recovered-historical-reports-v1","status":"BOUNDED_RECOVERY_ATTEMPT_COMPLETE",
             "actual_exposure_verified":False,"training_lineage_verified":False,"full_exposure_inventory":"NOT_ESTABLISHED",
             "independent_evaluation":"NOT_RUN","model_selection_permitted":False,"candidate_count":33,
             "zip_verified_count":33,"all_candidate_archives_verified":True,"all_reports_recovered":False,
             "preserved_report_count":35,"preserved_bytes":total,"new_training":False,"new_inference":False,
             "new_java_execution":False,"prior_evidence":{"archive_sha256":a.TRIAGE_ZIP},"artifacts":artifacts}
        raw = json.dumps(d,sort_keys=True).encode()
        (root/"recovery-manifest.json").write_bytes(raw)
        return raw,total

    def test_offline_exact_bytes_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"archive"
            raw,total=self.fixture(root)
            with patch.multiple(a, MANIFEST_SHA=a.sha(raw), MANIFEST_BYTES=len(raw), EXPECTED_BYTES=total):
                result=a.verify_local(root)
                self.assertEqual(result["original_report_instances"],35)
                self.assertFalse(result["requires_actions_artifact_for_local_verification"])
                p=root/"reports/1-100-0.json"
                old=p.read_bytes()
                p.write_bytes(old+b" ")
                with self.assertRaises(ValueError):
                    a.verify_local(root)
                p.write_bytes(old)
                (root/"reports/extra.json").write_text("{}")
                with self.assertRaises(ValueError):
                    a.verify_local(root)
                (root/"reports/extra.json").unlink()
                (root/"recovery-manifest.json").write_bytes(raw+b" ")
                with self.assertRaises(ValueError):
                    a.verify_local(root)

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                a.stage(root,root/"fake.zip",root/"fake.zip")


if __name__ == "__main__":
    unittest.main()
