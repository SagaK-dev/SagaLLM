import unittest
from copy import deepcopy
from scripts.run_validation_only_lr_sweep import aggregate, SEEDS, ARMS

class MatchedSweepTests(unittest.TestCase):
    def rows(self):
        return [{"seed":s,"arm":a,"lr":lr,"schedule_sha256":str(s),
                 "initial_validation_loss":2.0,"best_validation_loss":1.9,
                 "last_validation_loss":2.1,"best_step":4,"test_status":"NOT_RUN"}
                for s in SEEDS for a,lr in ARMS]
    def test_complete_matched_sweep(self):
        result=aggregate(self.rows())
        self.assertEqual(len(result),3)
        self.assertEqual(result[0]["positive_selected_step_count"],2)
        self.assertAlmostEqual(result[0]["mean_best_minus_baseline"],-.1)
    def test_reject_drift(self):
        for key,value in (("schedule_sha256","drift"),("initial_validation_loss",3.0),
                          ("test_status","SCORED"),("lr",0.001)):
            rows=self.rows();rows[0][key]=value
            with self.assertRaises(ValueError):aggregate(rows)
        with self.assertRaises(ValueError):aggregate(self.rows()[:-1])

if __name__=="__main__":unittest.main()
