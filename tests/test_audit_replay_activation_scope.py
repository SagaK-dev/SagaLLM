from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import audit_replay_activation_scope as a
from scripts.audit_ranking_auxiliary import canonical


class ReplayActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = a.load_reports(Path("experiments/archive/reports"))
        ce = cls.reports[4]
        cls.previous = {"models": [
            {"model": name, "checkpoint_sha256_declared_by_ce": record[0]}
            for name, record in a.models(ce, "parameter_ce").items()]}

    def test_actual_report_scope(self):
        rank, replay, binding, act, ce, signal = self.reports
        ranking = a.replay_and_binding(rank, replay, binding, ce)
        activation = a.activation_scope(act, signal, ce, self.previous)
        self.assertEqual(sum(r["generation_rows_recounted"] for r in ranking), 84)
        self.assertEqual(sum(r["binding_pairs_compared"] for r in ranking), 161)
        self.assertEqual(sum(r["equal_length_pairs_compared"] for r in activation), 133)
        self.assertEqual(sum(r["control_checks_recounted"] for r in activation), 3192)

    def test_replay_and_binding_mutations(self):
        for mutate in (
            lambda r, f, b: r["baseline"]["regression"]["rows"][0].update(source_sha256="0"*64),
            lambda r, f, b: f["deterministic_bounds"]["base"]["regression"].update(observed_passes=1),
            lambda r, f, b: f["summaries"]["base"]["regression"].update(unique=99),
            lambda r, f, b: b["models"]["base"]["train"]["rows"][0]["ids"].reverse(),
            lambda r, f, b: b["models"]["4201-control"]["prior_generation_collisions"].update(regression=99),
        ):
            rank, replay, binding = deepcopy(self.reports[:3])
            mutate(rank, replay, binding)
            with self.assertRaises(ValueError):
                a.replay_and_binding(rank, replay, binding, self.reports[4])

    def test_activation_mutations(self):
        for mutate in (
            lambda d: d["models"]["base"]["rows"][0].update(prompt_tokens=0),
            lambda d: d["excluded_unequal_length"][0].reverse(),
            lambda d: d["models"]["base"]["rows"][0].update(control_checks=23),
            lambda d: d.update(production_promotion=True),
        ):
            act = deepcopy(self.reports[3])
            mutate(act)
            with self.assertRaises(ValueError):
                a.activation_scope(act, self.reports[5], self.reports[4], self.previous)

    def test_prior_pin_and_gate(self):
        d = {"schema":"sagallm-prompt-probe-scope-v1",
             "status":"ARCHIVED_PROMPT_PROBE_IDENTITIES_MATCH",
             "reports":[{"run_id":r,"slug":s,"report_sha256":h} for r,s,h in a.CE_SPECS],
             "inventory_entries":15,"new_inventory_entries":0,"inventory_sha256":"a"*64,
             "models":self.previous["models"],"signal_pair_comparisons":140,
             "bias_pair_comparisons":322,"bias_regression_row_comparisons":84,
             "actual_exposure_verified":False,"training_lineage_verified":False,
             "full_exposure_inventory":"NOT_ESTABLISHED",
             "inventory_ready_for_independent_evaluation":False,
             "independent_evaluation":"NOT_RUN","model_selection_permitted":False}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"prior.zip"
            def write(extra=False):
                with zipfile.ZipFile(p,"w") as z:
                    z.writestr("prompt-probe-scope.json",canonical(d))
                    if extra:
                        z.writestr("other.json","{}")
                return a.sha(p.read_bytes())
            with patch.object(a,"PRIOR_ZIP",write()):
                self.assertEqual(a.prior(p)[0]["inventory_entries"],15)
                p.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    a.prior(p)
            d["model_selection_permitted"] = True
            with patch.object(a,"PRIOR_ZIP",write()):
                with self.assertRaises(ValueError):
                    a.prior(p)
            d["model_selection_permitted"] = False
            with patch.object(a,"PRIOR_ZIP",write(extra=True)):
                with self.assertRaises(ValueError):
                    a.prior(p)


if __name__ == "__main__":
    unittest.main()
