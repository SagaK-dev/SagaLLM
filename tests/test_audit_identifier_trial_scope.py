import unittest
from copy import deepcopy
from scripts import audit_identifier_trial_scope as a


class IdentifierTrialScopeTests(unittest.TestCase):
    def fixture(self):
        gate={"actual_exposure_verified":False,"training_lineage_verified":False,
              "full_exposure_inventory":"NOT_ESTABLISHED","independent_evaluation":"NOT_RUN",
              "model_selection_permitted":False}
        inv={"schema":"sagallm-private-java-exposure-v1",
             "files":[{"group":"train","path":f"data/{i}.jsonl","sha256":"a"*64,"format":"chat-jsonl"}
                      for i in range(15)]}
        lineage={**gate,"status":"DECLARED_PARENT_AND_OUTPUT_ROLES_RECONCILED",
                 "combined_checkpoint_hash_count":34,"step_zero_distinct_hashes":6,
                 "step_zero_member_instances":8,"inventory_entries":15,"inventory_sha256":"b"*64}
        reports=[]
        for rev,(mode,run,artifact,digest) in a.REVISIONS.items():
            reports.append({"schema":"sagallm-identifier-trial-input-v1","status":"RECONSTRUCTED_TRIAL_INPUT_SCOPE",
                "revision":rev,"mode":mode,"run_id":run,"artifact_id":artifact,"report_sha256":digest,
                "source_archives":{str(k):v for k,v in a.ORIGINALS.items()},
                "candidate_sha256":a.CANDIDATE,"candidate_bytes":102959,"candidate_manifest_sha256":a.MANIFEST,
                "paired_rows":120,"train_families":10,"canonical_manifest_sha256":a.DATA,"tokenizer_sha256":a.TOKENIZER,
                "selected_step":0,"source_bytes_verified_here":True,"selected_checkpoint_exposure_certified":False,
                "control_paired_sha256":"c"*64,"control_paired_bytes":100,"derived_schedule_sha256":("d" if mode=="identifier" else "e")*64,
                "derived_selection_ids_sha256":"f"*64,"trial_counts":{"mode":mode},
                **{k:False for k in ("model_loaded","new_training","new_inference","java_compiled","java_executed",
                      "actual_exposure_verified","training_lineage_verified","historical_schedule_digest_recorded","model_selection_permitted")}})
        return reports,lineage,inv

    def test_integrate_conservative_paths(self):
        reports,lineage,inv=self.fixture()
        raw,result=a.integrate(reports,lineage,inv,"b"*64)
        self.assertEqual(len(a.parse(raw)["files"]),17)
        self.assertEqual(result["new_inventory_entries"],2)
        self.assertFalse(result["selected_checkpoint_exposure_certified"])
        self.assertFalse(result["inventory_ready_for_independent_evaluation"])

    def test_reject_trial_or_selection_drift(self):
        reports,lineage,inv=self.fixture()
        for key,value in (("selected_step",60),("derived_schedule_sha256","0"*64),
                          ("control_paired_sha256","0"*64),("new_training",True),
                          ("selected_checkpoint_exposure_certified",True)):
            altered=deepcopy(reports)
            altered[0][key]=value
            with self.assertRaises(ValueError):
                a.integrate(altered,lineage,inv,"b"*64)

    def test_reject_duplicate_source_or_prior_hash(self):
        reports,lineage,inv=self.fixture()
        inv["files"][0]["path"]="identifier/control-paired.jsonl"
        with self.assertRaises(ValueError):
            a.integrate(reports,lineage,inv,"b"*64)
        reports,lineage,inv=self.fixture()
        with self.assertRaises(ValueError):
            a.integrate(reports,lineage,inv,"c"*64)


if __name__=="__main__":
    unittest.main()
