import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from scripts import audit_checkpoint_exposure as audit


class CheckpointMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = audit.load_reports(audit.ARCHIVE)

    def test_actual_reports(self):
        result = audit.map_reports(self.reports)
        self.assertEqual(result["checkpoint_count"], 12)
        self.assertEqual(len({r["checkpoint_sha256"] for r in result["checkpoints"]}), 12)
        self.assertFalse(result["model_selection_permitted"])
        for row in result["checkpoints"]:
            self.assertFalse(row["inventory_ready"])
            self.assertFalse(row["actual_exposure_verified"])
            self.assertTrue(all(not x["shard_ready"] for x in row["sources"]))

    def test_mutations(self):
        for mutate in (
            lambda p: p[0].update(base_sha256="0"*64),
            lambda p: p[0].update(fresh_used_in_training_or_selection=True),
            lambda p: p[1].update(pair_manifest_sha256="0"*64),
            lambda p: p[1]["results"].pop("4201"),
            lambda p: p[1]["results"]["4201"]["control"].update(coefficient=1),
            lambda p: p[1]["results"]["4201"]["control"].update(
                checkpoint_sha256=p[0]["results"]["4201"]["control"]["checkpoint_sha256"]),
        ):
            with self.subTest(mutate=mutate):
                data = copy.deepcopy(self.reports)
                mutate(data)
                with self.assertRaises(ValueError):
                    audit.map_reports(data)

    def test_coefficient_numeric_types(self):
        for value in (0, 0.0):
            data = copy.deepcopy(self.reports)
            data[1]["results"]["4201"]["control"]["coefficient"] = value
            self.assertEqual(audit.map_reports(data)["checkpoint_count"], 12)
        for value in (False, True, "0", None, 0.5, 1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                data = copy.deepcopy(self.reports)
                data[1]["results"]["4201"]["control"]["coefficient"] = value
                with self.assertRaises(ValueError):
                    audit.map_reports(data)
        for value in (1, 1.0):
            data = copy.deepcopy(self.reports)
            data[1]["results"]["4201"]["parameter_ce"]["coefficient"] = value
            self.assertEqual(audit.map_reports(data)["checkpoint_count"], 12)

    def test_reconstruction_archive_gate(self):
        evidence = {"status": "DECLARED_DEVELOPMENT_DATA_PINS_MATCH",
                    "pins": audit.PINS, "actual_exposure_verified": False,
                    "full_exposure_inventory": "NOT_ESTABLISHED",
                    "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.zip"

            def write(data, extra=False):
                raw = json.dumps(data).encode()
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("development-data-evidence.json", raw)
                    if extra:
                        z.writestr("extra.json", "{}")
                return hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(raw).hexdigest()

            pin, member = write(evidence)
            with patch.object(audit, "DEVELOPMENT_ZIP", pin):
                self.assertEqual(audit.read_reconstruction(path), member)
                path.write_bytes(path.read_bytes() + b"x")
                with self.assertRaises(ValueError):
                    audit.read_reconstruction(path)
            for changed, extra in ((dict(evidence, model_selection_permitted=True), False),
                                   (evidence, True)):
                pin, _ = write(changed, extra)
                with patch.object(audit, "DEVELOPMENT_ZIP", pin):
                    with self.assertRaises(ValueError):
                        audit.read_reconstruction(path)
