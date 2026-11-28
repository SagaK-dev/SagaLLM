import unittest
from copy import deepcopy
from scripts import audit_recovered_semantic_lineage as a


class SemanticLineageTests(unittest.TestCase):
    def test_step_zero_exact_arm_and_parent(self):
        d = {"status":"TRAIN_VALIDATION_AB_COMPLETE", "tokenizer_sha256":a.TOKENIZER,
             "base_checkpoint_sha256":a.BEST, "canonical_manifest_sha256":a.DATA,
             "candidate_sha256":a.IDENTIFIER, "steps_per_arm":60,
             "decision":{"production_adoption":False}, "selection_uses_only_validation":True,
             "test_used_for_selection":False, "test_split_evaluated":False, "generated_java_executed":False,
             "results": {arm:{"arm":arm,"optimizer_steps_completed":60,"selected_step":0,
                 "initial_validation_loss":1.5,"best_validation_loss":1.5,
                 "history":[{"validation_response_loss":1.6}], "checkpoint_sha256":digest*64}
                 for arm,digest in (("canonical","a"),("identifier_augmented","b"))}}
        name = "sagallm-java-identifier-ab-validation"
        member = "identifier-ab-evidence/canonical-validation-best.pt"
        result = a.declared_output(d,name,member)
        self.assertEqual(result["sha256"],"a"*64)
        self.assertEqual(result["selected_step"],0)
        self.assertEqual(result["trial_steps_completed"],60)
        self.assertEqual(result["field_path"],["results","canonical","checkpoint_sha256"])
        for key,value in (("selected_step",30),("arm","identifier_augmented"),("best_validation_loss",1.2)):
            bad = deepcopy(d)
            bad["results"]["canonical"][key] = value
            with self.assertRaises(ValueError):
                a.declared_output(bad,name,member)

    def test_sft_output_is_not_parent_or_last(self):
        d = {"status":"TRAINED_CPU","tokenizer_sha256":a.TOKENIZER,
             "data_manifest_sha256":a.DATA,"optimizer_steps_completed":400,
             "test_used_for_model_selection":False,"checkpoint_sha256_base":a.BASE,
             "best_checkpoint_step":50,"best_checkpoint_sha256":"a"*64,
             "last_checkpoint_sha256":"b"*64}
        name = "sagallm-v018-java-context-sft-cpu"
        result = a.declared_output(d,name,"java-context-model/java-sft-best.pt")
        self.assertEqual(result["parent_sha256"],a.BASE)
        self.assertEqual(result["selected_step"],50)
        self.assertEqual(a.declared_output(d,name,"java-context-model/java-sft-last.pt")["selected_step"],400)
        d["checkpoint_sha256_base"]="c"*64
        with self.assertRaises(ValueError):
            a.declared_output(d,name,"java-context-model/java-sft-best.pt")

    def test_declared_graph(self):
        a.acyclic({"a":None,"b":"a","c":"b"})
        for graph in ({"a":"b","b":"a"},{"a":"missing"},{"a":"a"}):
            with self.assertRaises(ValueError):
                a.acyclic(graph)


if __name__ == "__main__":
    unittest.main()
