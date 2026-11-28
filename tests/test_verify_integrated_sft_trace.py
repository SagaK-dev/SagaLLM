import unittest
from scripts.verify_integrated_sft_trace import declared_selection

class OfflineSelectionTests(unittest.TestCase):
    def fixture(self):
        return {"history":[{"step":1,"validation_response_loss":1.9995,"distinct_train_examples_seen":2},
                    {"step":2,"validation_response_loss":1.95,"distinct_train_examples_seen":4},
                    {"step":3,"validation_response_loss":1.9495,"distinct_train_examples_seen":6}],
                "optimizer_steps_completed":3,"initial_validation_response_loss":2.,
                "best_checkpoint_step":2,"best_validation_response_loss":1.95,
                "best_checkpoint_train_examples_seen":4}

    def test_threshold_and_zero_selection(self):
        d=self.fixture()
        self.assertEqual(declared_selection(d),2)
        d.update(history=[],best_checkpoint_step=0,best_validation_response_loss=2.,
                 best_checkpoint_train_examples_seen=0)
        self.assertEqual(declared_selection(d),0)

    def test_reject_bad_declarations(self):
        for key,value in (("best_checkpoint_step",3),("best_validation_response_loss",1.9495),
                          ("best_checkpoint_train_examples_seen",6)):
            d=self.fixture();d[key]=value
            with self.assertRaises(ValueError):declared_selection(d)
        d=self.fixture();d["history"][1]["validation_response_loss"]=float("nan")
        with self.assertRaises(ValueError):declared_selection(d)

if __name__=="__main__":
    unittest.main()
