import unittest
import torch
from scripts.compare_traced_context_reproduction import state_fingerprint

class ReproductionFingerprintTests(unittest.TestCase):
    def test_tensor_dtype_shape_and_bits(self):
        a=torch.tensor([0.,1.])
        self.assertEqual(state_fingerprint(a),state_fingerprint(a.clone()))
        for b in (a.double(),a.reshape(2,1),torch.tensor([-0.,1.]),torch.tensor([0.,2.])):
            self.assertNotEqual(state_fingerprint(a),state_fingerprint(b))
    def test_nested_optimizer_and_key_types(self):
        a={"state":{1:{"step":torch.tensor(4.),"m":torch.ones(2)}},"groups":[{"lr":4e-5}]}
        b={"groups":[{"lr":4e-5}],"state":{1:{"m":torch.ones(2),"step":torch.tensor(4.)}}}
        self.assertEqual(state_fingerprint(a),state_fingerprint(b))
        b["state"][1]["step"]+=1
        self.assertNotEqual(state_fingerprint(a),state_fingerprint(b))
        self.assertNotEqual(state_fingerprint({1:True}),state_fingerprint({"1":True}))
        self.assertNotEqual(state_fingerprint([True]),state_fingerprint([1]))

if __name__=="__main__":unittest.main()
