import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import zipfile
from scripts import archive_experiment_reports as a


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / "artifact.zip"
        self.raw = b'{"status":"NOT_RUN"}\n'
        with zipfile.ZipFile(self.archive,"w") as z:
            z.writestr("report.json", self.raw)
            z.writestr("ignored.pt", b"never extract")
        self.row = dict(run_id=1, artifact_id=2, commit_sha="a"*40,
                        archive_sha256=a.sha(self.archive.read_bytes()),
                        report_sha256=a.sha(self.raw), member="report.json", slug="test")
        self.manifest = self.root / "manifest.json"

    def test_exact_bytes_and_no_extraction(self):
        self.assertEqual(a.extract_report(self.row,self.archive),self.raw)
        self.assertFalse((self.root/"ignored.pt").exists())

    def test_bad_archive_and_report_pin(self):
        for k in ("archive_sha256","report_sha256"):
            with self.assertRaises(ValueError):
                a.extract_report({**self.row,k:"0"*64},self.archive)

    def test_no_traversal_and_duplicate_outputs(self):
        for rows in ([{**self.row,"member":"../report.json"}],[self.row,self.row]):
            self.manifest.write_text(json.dumps(dict(schema="sagallm-pinned-report-manifest-v1",
                                                    repository=a.REPO,entries=rows)))
            with self.assertRaises(ValueError):
                a.load_manifest(self.manifest)

    def test_wrong_artifact_run(self):
        meta=dict(id=2, expired=False, digest="sha256:"+self.row["archive_sha256"],
                  size_in_bytes=200,workflow_run=dict(id=99,head_sha="a"*40))
        with self.assertRaises(ValueError):
            a.verify_metadata(self.row,meta)

    def test_verify_local_and_detect_tamper(self):
        self.manifest.write_text(json.dumps(dict(schema="sagallm-pinned-report-manifest-v1",
                                                repository=a.REPO,entries=[self.row])))
        target=self.root/"1-test.json"
        target.write_bytes(self.raw)
        a.verify_local(self.manifest,self.root)
        target.write_bytes(b"{}")
        with self.assertRaises(ValueError):
            a.verify_local(self.manifest,self.root)


if __name__ == "__main__":
    unittest.main()
