import math
import unittest
import torch
from scripts.run_equal_integral_lr_sweep import spec, STEPS
from sagallm.sft_lr_schedule import step_lr
from scripts.compare_equal_integral_tensor_state import tensor_record

class EqualIntegralTests(unittest.TestCase):
    def test_rate_integrals(self):
        for arm,lr in (("constant_matched",2.175e-5),("warmup_cosine",4e-5)):
            self.assertTrue(math.isclose(math.fsum(step_lr(spec(arm,lr),STEPS,s) for s in range(1,STEPS+1)),.000522,rel_tol=0,abs_tol=1e-15))
    def test_tensor_bits(self):
        a=torch.tensor([0.,1.])
        self.assertEqual(tensor_record(a),tensor_record(a.clone()))
        self.assertNotEqual(tensor_record(a),tensor_record(torch.tensor([-0.,1.])))
        self.assertEqual(tensor_record(torch.tensor(1.))["bytes"],4)
        self.assertEqual(tensor_record(a[1:])["bytes"],4)
if __name__=="__main__":unittest.main()
