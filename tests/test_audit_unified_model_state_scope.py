import unittest
from copy import deepcopy
from scripts import audit_unified_model_state_scope as a

class UnifiedScopeTests(unittest.TestCase):
    def test_alias_scope_and_mutations(self):
        g={"actual_exposure_verified":False,"training_lineage_verified":False,
           "full_exposure_inventory":"NOT_ESTABLISHED","independent_evaluation":"NOT_RUN","model_selection_permitted":False}
        aliases=[f"{i+1:064x}" for i in range(6)]
        nodes=[]; rows=[]
        for i,h in enumerate(aliases):
            instances=[]
            for j in range(2 if i<2 else 1):
                m={"artifact_id":i*2+j+1,"member":f"{i}-{j}.pt","run_id":i*2+j+10}
                instances.append(m)
                rows.append({**m,"checkpoint_sha256":h,"arm":"canonical","embedded_parent_sha256":a.BEST,
                    "embedded_selected_step":0,"config_equal":True,"all_tensor_digests_match":True,
                    "different_tensor_names":[],"parent_tensor_index_sha256":"a"*64,
                    "candidate_tensor_index_sha256":"a"*64,"parent_tensor_count":39,"candidate_tensor_count":39,
                    "parent_tensor_bytes":100,"candidate_tensor_bytes":100})
            nodes.append({"sha256":h,"arm":"canonical","selected_step":0,"instances":instances})
        links=[{"checkpoint_sha256":h,"parent_sha256":a.BEST} for h in aliases]
        links += [{"checkpoint_sha256":a.BEST,"parent_sha256":None}]
        links += [{"checkpoint_sha256":f"{i+100:064x}","parent_sha256":a.BEST} for i in range(27)]
        lineage={**g,"combined_checkpoint_hash_count":34,"step_zero_distinct_hashes":6,
                 "declared_parent_graph_acyclic":True,"recovered_outputs":nodes,"parent_links":links,
                 "inventory_sha256":"b"*64,"unresolved_checkpoint_declarations":[]}
        tensor={**g,"lineage_evidence":{"archive_sha256":a.LINEAGE_ZIP},"parent_checkpoint_sha256":a.BEST,
                "member_count":8,"all_selected_step_zero_tensors_match_parent":True,
                "tensor_and_config_equal_member_count":8,"tensor_and_config_equal_distinct_hashes":6,"results":rows}
        inv={"schema":"sagallm-private-java-exposure-v1","files":[{"path":f"{i}.jsonl"} for i in range(17)]}
        trial={**g,"lineage_evidence":{"archive_sha256":a.LINEAGE_ZIP},"prior_inventory_sha256":"b"*64,
               "inventory_sha256":"c"*64,"inventory_entries":17,"selected_checkpoint_exposure_certified":False,
               "new_sources":inv["files"][-2:],"scope":"TRIAL_INPUT_SCREENING_ONLY"}
        entry={"path":"derived/public_unreviewed_io.jsonl"}
        merged,out=a.unify(lineage,tensor,trial,inv,"c"*64,entry,{})
        self.assertEqual(len(merged["files"]),18)
        self.assertEqual(out["model_state_equivalence_classes_upper_bound"],28)
        self.assertEqual(out["exact_distinct_model_state_count"],"NOT_ESTABLISHED")
        for field,value in (("config_equal",False),("candidate_tensor_index_sha256","d"*64),
                            ("embedded_selected_step",80),("checkpoint_sha256","e"*64)):
            altered=deepcopy(tensor);altered["results"][0][field]=value
            with self.assertRaises(ValueError):
                a.unify(lineage,altered,trial,inv,"c"*64,entry,{})
        with self.assertRaises(ValueError):
            a.unify(lineage,tensor,trial,inv,"d"*64,entry,{})

    def test_public_reject_unpinned(self):
        with self.assertRaises(ValueError):
            a.public_scope(b"{}",b"",{})

if __name__=="__main__":
    unittest.main()
