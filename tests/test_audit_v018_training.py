import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import audit_v018_training as audit


class TrainingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.raw = audit.ARCHIVED.read_bytes()
        self.report = json.loads(self.raw)

    def test_historical_report(self):
        result = audit.audit_report(self.raw)
        self.assertEqual(result["best_step"], 50)
        self.assertEqual(result["last_seen"], 1600)
        self.assertEqual(result["compile_only"]["base"]["compiled"], 2)
        self.assertEqual(result["compile_only"]["best"]["compiled"], 0)
        self.assertFalse(result["model_selection_permitted"])

    def test_byte_tamper(self):
        with self.assertRaises(ValueError):
            audit.audit_report(self.raw + b"\n")

    def test_invariant_failures(self):
        for mutate in (
            lambda p: p.update(best_checkpoint_step=400),
            lambda p: p.update(distinct_training_examples_seen=1601),
            lambda p: p.update(java_code_executed=True),
            lambda p: p["compile_only_best"].update(compiled=1),
            lambda p: p["compile_only_best"]["cases"].reverse(),
            lambda p: p["history"][0].update(validation_response_loss=float("nan")),
            lambda p: p["original_split_counts"]["test"].update(accepted=0),
        ):
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(self.report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    audit.recount(changed)

    def test_zip_pins_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("weights.pt", b"opaque bytes")
            sha = lambda data: hashlib.sha256(data).hexdigest()
            archive = sha(path.read_bytes())
            audit.audit_zip(path, archive, {"weights.pt": sha(b"opaque bytes")})
            for ap, members in ((sha(b"wrong"), {"weights.pt": sha(b"opaque bytes")}),
                                (archive, {"weights.pt": sha(b"wrong")}),
                                (archive, {"missing": sha(b"opaque bytes")})):
                with self.assertRaises(ValueError):
                    audit.audit_zip(path, ap, members)
            with zipfile.ZipFile(path, "a") as z:
                z.writestr("weights.pt", b"other")
            with self.assertRaises(ValueError):
                audit.audit_zip(path, sha(path.read_bytes()), {"weights.pt": sha(b"opaque bytes")})
