import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from sagallm.java_template_subset import (build_template_subset,verify_template_subset,
    verify_subset_train,MEMBERSHIP,COUNT,MODE)
from sagallm.java_diverse_corpus import build_java_diverse_corpus
from scripts.audit_sft_prompt_collisions import audit

class TemplateSubsetTests(unittest.TestCase):
    def test_exact_subset_and_reject_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/"full";out=root/"subset"
            build_java_diverse_corpus(source)
            plan=audit(source)["candidate_plan"]
            hashes={r["row_sha256"] for r in plan["candidate_rows"]}
            self.assertEqual(plan["membership_sha256"],MEMBERSHIP)
            m=build_template_subset(source,out,hashes)
            self.assertEqual(m["files"]["train"]["count"],COUNT)
            for split in ("validation","test"):
                self.assertEqual((source/(split+".jsonl")).read_bytes(),(out/(split+".jsonl")).read_bytes())
            with self.assertRaises(ValueError):build_template_subset(source,root/"bad",set(list(hashes)[:-1]))
            with self.assertRaises(ValueError):build_template_subset(source,out,hashes)
            raw=(out/"train.jsonl").read_bytes()
            lines=raw.splitlines(keepends=True)
            (out/"train.jsonl").write_bytes(b"".join(reversed(lines)))
            with self.assertRaises(ValueError):verify_subset_train(out)
            (out/"train.jsonl").write_bytes(raw)
            (out/"test.jsonl").write_bytes(b"invalid")
            self.assertEqual(verify_subset_train(out),m)
            with self.assertRaises(ValueError):verify_template_subset(out)
    def test_explicit_mode_required(self):
        from sagallm.java_instruction_sft import train_instruction_sft
        for kwargs in ({"dataset_mode":"unknown"},{"dataset_mode":MODE},
                       {"dataset_mode":MODE,"record_exposure":True,"validation_only":True,"require_pinned":False}):
            with self.assertRaises(ValueError):
                train_instruction_sft("none","none","none","none",compile_cases=0,**kwargs)

if __name__=="__main__":unittest.main()
