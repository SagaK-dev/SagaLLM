import copy
import unittest
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
