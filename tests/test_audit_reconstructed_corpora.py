import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.audit_reconstructed_corpora import verify, SPLITS


class ReconstructionTests(unittest.TestCase):
    def test_exact_bytes_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_bytes(b"manifest")
            pin = hashlib.sha256(b"manifest").hexdigest()
            hashes = {}
            for split in SPLITS:
                raw = (split + "\n").encode()
                (root / (split + ".jsonl")).write_bytes(raw)
                hashes[split] = {"sha256": hashlib.sha256(raw).hexdigest()}
            manifest = {"files": hashes}
            self.assertEqual(len(verify(root, manifest, pin, True)["split_sha256"]), 3)
            (root / "test.jsonl").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify(root, manifest, pin, True)
            with self.assertRaises(ValueError):
                verify(root, manifest, "0"*64, True)
