import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import audit_recover_unarchived_reports as a


class RecoveryTests(unittest.TestCase):
    def test_original_bytes_and_unlisted_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "reports"
            out.mkdir()
            p = root / "archive.zip"
            raw = b'{ "status": "TRAINED_CPU", "steps": 160 }\n'
            with zipfile.ZipFile(p, "w") as z:
                z.writestr("java-model/train-report.json", raw)
                z.writestr("java-model/java-best.pt", b"not a model")
                z.writestr("unlisted.json", b'{"raw_source":"private"}')
            spec = {"run_id": 1, "artifact_id": 2, "commit_sha": "a"*40,
                    "name": "sagallm-v016-pinned-java-cpu", "size_in_bytes": p.stat().st_size,
                    "api_zip_sha256": a.sha(p.read_bytes()), "expires_at": "2027-01-01T00:00:00Z"}
            result = a.verify_zip(p, spec, out)
            self.assertTrue(result["archive_bytes_verified"])
            self.assertEqual(result["other_json_member_count"], 1)
            self.assertEqual((out / "1-2-0.json").read_bytes(), raw)
            self.assertEqual(result["report_members"][0]["report_sha256"], a.sha(raw))
            self.assertEqual(len(list(out.iterdir())), 1)
            with self.assertRaises(ValueError):
                a.verify_zip(p, {**spec, "api_zip_sha256": "0"*64}, out)

    def test_screen_fails_closed(self):
        self.assertEqual(a.screen(b'{"source":"raw text"}'), "RAW_FIELD_REVIEW_REQUIRED")
        self.assertEqual(a.screen(b'{"nested":{"message":"public class Main {}"}}'), "CONTENT_REVIEW_REQUIRED")
        self.assertEqual(a.screen(b'{"sha256":"abc","source_persisted":false}'),
                         "METADATA_SCREEN_PASSED_NOT_RIGHTS_REVIEW")
        with self.assertRaises(ValueError):
            a.screen(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            a.screen(b'[]')

    def test_unsafe_member_and_unknown_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "a.zip"
            out = root / "reports"
            out.mkdir()
            def make(name):
                with zipfile.ZipFile(p, "w") as z:
                    z.writestr(name, b'{}')
                return {"run_id": 1, "artifact_id": 2, "commit_sha": "a"*40,
                        "name": "unknown", "size_in_bytes": p.stat().st_size,
                        "api_zip_sha256": a.sha(p.read_bytes()), "expires_at": "2027-01-01T00:00:00Z"}
            with self.assertRaises(ValueError):
                a.verify_zip(p, make("../bad.json"), out)
            result = a.verify_zip(p, make("report.json"), out)
            self.assertTrue(result["unknown_artifact_name"])
            self.assertEqual(result["preserved_bytes"], 0)
            self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
