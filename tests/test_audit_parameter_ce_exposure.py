from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import audit_parameter_ce_exposure as a


class TinyTokenizer:
    SPECIAL = {"<bos>": 0}
    def encode(self, text):
        return [ord(c) for c in text]
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class ParameterExposureTests(unittest.TestCase):
    def test_four_combinations_and_first_divergence(self):
        tok = TinyTokenizer()
        lookup = {"a": {"id":"a","family":"f","task":"x","source":"abc1"},
                  "b": {"id":"b","family":"f","task":"y","source":"abc2"}}
        p = {"ids":["a","b"], "family":"f", "prefixes":[[0,120],[0,121]],
             "common_response_prefix":tok.encode("abc"), "position":3,
             "alternatives":[49,50]}
        rows = a.screening_rows([p], lookup, tok, lambda x:x)
        self.assertEqual([r["text"] for r in rows], ["xabc1","xabc2","yabc1","yabc2"])
        self.assertEqual([r["correct"] for r in rows], [True,False,False,True])
        with self.assertRaises(ValueError):
            a.screening_rows([p,p], lookup, tok, lambda x:x)
        p["alternatives"] = [49,51]
        with self.assertRaises(ValueError):
            a.screening_rows([p], lookup, tok, lambda x:x)

    def test_schedule_rejects_mismatch(self):
        def row(coeff, cp):
            return {"schedule_sha256":"a"*64, "coefficient":coeff,
                    "steps":160, "selected_step":160, "broad_rows_seen":320,
                    "stdio_rows_seen":320,"auxiliary_directions":320,
                    "checkpoint_sha256":cp}
        arms = {"control":row(0,"b"*64),"parameter_ce":row(1,"c"*64)}
        report = {"schedule_sha256_by_seed":{"4201":"a"*64},"results":{"4201":arms}}
        seen = set()
        a.compare_schedule(report,4201,"a"*64,seen)
        self.assertEqual(len(seen),2)
        with self.assertRaises(ValueError):
            a.compare_schedule(report,4201,"d"*64,set())
        with self.assertRaises(ValueError):
            a.compare_schedule(report,4201,"a"*64,seen)
        arms["parameter_ce"]["auxiliary_directions"] = 319
        with self.assertRaises(ValueError):
            a.compare_schedule(report,4201,"a"*64,set())

    def test_prior_digest_and_scope(self):
        inv = {"schema":"sagallm-private-java-exposure-v1",
               "files":[{"path":str(i)} for i in range(12)]}
        raw = a.canonical(inv)
        evidence = {"inventory_entries":12,"inventory_sha256":a.sha(raw),
                    "status":"HISTORICAL_RANKING_DECLARATIONS_RECONSTRUCTED",
                    "derived_rows":640,"derived_entry":inv["files"][-1],
                    "actual_exposure_verified":False,"full_exposure_inventory":"NOT_ESTABLISHED",
                    "inventory_ready_for_independent_evaluation":False,
                    "model_selection_permitted":False,"independent_evaluation":"NOT_RUN"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"prior.zip"
            def write():
                with zipfile.ZipFile(path,"w") as z:
                    z.writestr("scoped-ranking-inventory.json",raw)
                    z.writestr("ranking-auxiliary-evidence.json",a.canonical(evidence))
                return a.sha(path.read_bytes())
            with patch.object(a,"PRIOR_ZIP",write()):
                self.assertEqual(a.prior(path)[0],inv)
                path.write_bytes(path.read_bytes()+b"x")
                with self.assertRaises(ValueError):
                    a.prior(path)
            evidence["actual_exposure_verified"] = True
            with patch.object(a,"PRIOR_ZIP",write()):
                with self.assertRaises(ValueError):
                    a.prior(path)


if __name__ == "__main__":
    unittest.main()
