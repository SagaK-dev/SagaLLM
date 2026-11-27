from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import audit_prompt_probe_scope as a
from scripts.audit_ranking_auxiliary import canonical


class PromptProbeScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = a.load_reports(Path("experiments/archive/reports"))

    def test_actual_archived_identity_comparison(self):
        records = a.compare(*self.reports)
        self.assertEqual(len(records), 7)
        self.assertEqual(sum(r["signal_train_pairs"] for r in records), 140)
        self.assertEqual(len({r["checkpoint_sha256_declared_by_ce"] for r in records}), 7)

    def test_rejects_identity_length_mode_and_checkpoint_mutations(self):
        for mutate in (
            lambda ce, s, b: s["models"]["base"]["rows"][0]["ids"].reverse(),
            lambda ce, s, b: s["models"]["base"]["rows"][0].update(common_prefix_tokens=-1),
            lambda ce, s, b: b["models"]["base"].pop("bias"),
            lambda ce, s, b: b["models"]["base"]["control"]["regression_generation"]["rows"][0].update(id="validation-forged"),
            lambda ce, s, b: b.update(production_promotion=True),
            lambda ce, s, b: ce["results"]["4201"]["control"].update(checkpoint_sha256=a.BASE),
        ):
            ce, s, b = deepcopy(self.reports)
            mutate(ce, s, b)
            with self.assertRaises(ValueError):
                a.compare(ce, s, b)
        with self.assertRaises(ValueError):
            a.compare(*self.reports, checkpoint_pins=["a" * 64] * 6)

    def test_prior_archive_and_non_promotion_gate(self):
        inv = {"schema": "sagallm-private-java-exposure-v1",
               "files": [{"path": str(i)} for i in range(12)]}
        derived = []
        for name, group, count in zip(("train", "validation", "fresh"),
                                       ("train", "development", "development"), (640, 12, 12)):
            entry = {"path": "derived/parameter_ce_" + name + ".jsonl", "group": group}
            inv["files"].append(entry)
            derived.append({"entry": entry, "rows": count})
        raw = canonical(inv)
        evidence = {"status": "HISTORICAL_PARAMETER_CE_DECLARATIONS_RECONSTRUCTED",
                    "report_sha256": a.SPECS[0][2], "tokenizer_sha256": a.TOKENIZER,
                    "inventory_sha256": a.sha(raw), "inventory_entries": 15,
                    "checkpoint_count": 6, "diagnostic_scope_checked_models": 7,
                    "derived_sources": derived, "actual_exposure_verified": False,
                    "training_lineage_verified": False, "full_exposure_inventory": "NOT_ESTABLISHED",
                    "inventory_ready_for_independent_evaluation": False,
                    "model_selection_permitted": False, "independent_evaluation": "NOT_RUN"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evidence.zip"
            def write(extra=False):
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("scoped-parameter-ce-inventory.json", raw)
                    z.writestr("parameter-ce-evidence.json", canonical(evidence))
                    if extra:
                        z.writestr("unexpected.json", "{}")
                return a.sha(path.read_bytes())
            with patch.object(a, "PRIOR_ZIP", write()):
                self.assertEqual(a.prior(path)[0]["inventory_entries"], 15)
                path.write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    a.prior(path)
            evidence["model_selection_permitted"] = True
            with patch.object(a, "PRIOR_ZIP", write()):
                with self.assertRaises(ValueError):
                    a.prior(path)
            evidence["model_selection_permitted"] = False
            with patch.object(a, "PRIOR_ZIP", write(extra=True)):
                with self.assertRaises(ValueError):
                    a.prior(path)


if __name__ == "__main__":
    unittest.main()
