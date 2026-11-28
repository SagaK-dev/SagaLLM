from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import audit_remaining_historical_scope as a
from scripts import audit_integrated_exposure_scope as b
from scripts.audit_historical_exposure import audit as archive_audit
from scripts.audit_replay_activation_scope import SPECS, models


class RemainingScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("experiments/archive")
        cls.reports = a.load_reports(cls.root / "reports")

    def test_original_reports(self):
        result = a.verify_early(*self.reports)
        self.assertEqual(result["historical_behavior_cases_recounted"], 54)
        self.assertEqual(result["m1_generation_hash_declarations_compared"], 24)
        self.assertFalse(result["blocked_run_produced_new_checkpoint"])
        self.assertFalse(result["m1_reports_identical"])

    def test_scope_mutations(self):
        for mutate in (
            lambda ab, m, r, blocked: ab["schedule"].update(validation_or_test_in_optimizer=True),
            lambda ab, m, r, blocked: ab["baseline_behavior"]["cases"][0]["case_categories"].__setitem__(0, "PASS"),
            lambda ab, m, r, blocked: m["models"]["base_v018"]["observations"][0].update(source_sha256="0"*64),
            lambda ab, m, r, blocked: r["prompt_metrics"][0].update(prefix_sha256="0"*64),
            lambda ab, m, r, blocked: blocked.update(optimizer_steps_completed=1),
            lambda ab, m, r, blocked: blocked["alignment"].update(passed=True),
        ):
            reports = deepcopy(self.reports)
            mutate(*reports)
            with self.assertRaises(ValueError):
                a.verify_early(*reports)

    def test_model_mapping(self):
        inv = {"schema": "sagallm-private-java-exposure-v1",
               "files": [{"path": p, "group": g, "sha256": "a"*64, "format": "text-jsonl"}
                         for p, g in b.PATHS.items()]}
        scope = {"inventory_sha256": "b"*64}
        for field, spec, arm, key in (
                ("ranking_models", SPECS[0], "ranking", "checkpoint_sha256_declared_by_ranking"),
                ("ce_models", SPECS[4], "parameter_ce", "checkpoint_sha256_declared_by_ce")):
            records = models(b.archived(self.root, spec), arm)
            scope[field] = [{"model": k, key: v[0]} for k, v in records.items()]
        ledger = archive_audit(self.root)
        out = b.integrate(self.root, scope, inv, ledger, a.verify_early(*self.reports))
        self.assertEqual(out["archive_report_count"], 12)
        self.assertEqual(out["checkpoint_count"], 17)
        self.assertEqual(len(out["unresolved_checkpoint_declarations"]), 1)
        self.assertFalse(out["inventory_ready_for_independent_evaluation"])
        self.assertTrue(all(not r["actual_exposure_verified"] for r in out["checkpoint_declarations"]))
        changed = deepcopy(inv)
        changed["files"][0]["group"] = "test"
        with self.assertRaises(ValueError):
            b.inventory_map(changed)
        changed = deepcopy(inv)
        changed["files"][0]["path"] = changed["files"][1]["path"]
        with self.assertRaises(ValueError):
            b.inventory_map(changed)
        broken = deepcopy(ledger)
        broken["reports"].pop()
        with self.assertRaises(ValueError):
            b.integrate(self.root, scope, inv, broken, {})

    def test_zip_pin_and_exact_member_set(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "p.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("e.json", '{"x":1}')
            digest = a.sha(path.read_bytes())
            self.assertEqual(a.zip_json(path, digest, ["e.json"])[0]["e.json"], {"x":1})
            with self.assertRaises(ValueError):
                a.zip_json(path, digest, ["other.json"])
            with self.assertRaises(ValueError):
                a.zip_json(path, "0"*64, ["e.json"])
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                a.zip_json(path, digest, ["e.json"])


if __name__ == "__main__":
    unittest.main()
