import unittest
from scripts.audit_sweep_prefix_comparison import canonical,sha,compare,SEEDS,NEW_ARMS

class PrefixComparisonTests(unittest.TestCase):
    def test_exact_inputs_but_not_identical_history(self):
        old={};new={}
        for seed in SEEDS:
            h={"base_checkpoint_sha256":"a","tokenizer_sha256":"b","data_manifest_sha256":"c",
               "train_split_sha256":"d","seed":seed,"batch_size":4}
            base={"history":[{"step":s,"last_train_response_loss":2.,"validation_response_loss":1.8} for s in (4,8,12)],
                  "batches":[{"step":i} for i in range(1,13)],"header":h}
            for arm in ("low","reference"):old[seed,arm]=base
            for arm in NEW_ARMS:
                new[seed,arm]={**base,"history":[{**r,"validation_response_loss":1.80000001} for r in base["history"]],
                    "batches":[{"step":i} for i in range(1,25)],"rates":[1e-5]*24,
                    "row":{"best_step":4,"best_validation_loss":1.8,"last_validation_loss":1.81}}
        d=compare(old,new)
        self.assertEqual(d["matching_prefix_input_pair_count"],4)
        self.assertEqual(d["exact_history_pair_count"],0)
        self.assertGreater(d["maximum_validation_loss_absolute_difference"],0)
        self.assertFalse(d["checkpoint_tensor_comparison_performed"])

if __name__=="__main__":unittest.main()
