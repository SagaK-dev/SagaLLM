import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import torch
from scripts import audit_step_zero_tensor_equivalence as a


class StepZeroTensorTests(unittest.TestCase):
    def test_tensor_digest_equal_and_different(self):
        p={"model":{"weight":torch.tensor([1.0,2.0]),"bias":torch.tensor([0.0])},"config":{"size":2}}
        c={"model":{k:v.clone() for k,v in p["model"].items()},"config":{"size":2}}
        self.assertTrue(a.tensor_comparison(p,c)["all_tensor_digests_match"])
        c["model"]["weight"][0]=3.0
        self.assertEqual(a.tensor_comparison(p,c)["different_tensor_names"],["weight"])
        c["model"]["weight"]=p["model"]["weight"].double()
        self.assertFalse(a.tensor_comparison(p,c)["all_tensor_digests_match"])
        c["config"]={"size":3}
        self.assertFalse(a.tensor_comparison(p,c)["config_equal"])

    def test_shape_and_bitwise_scope(self):
        p={"model":{"weight":torch.tensor([0.0])},"config":{}}
        c={"model":{"weight":torch.tensor([-0.0])},"config":{}}
        self.assertFalse(a.tensor_comparison(p,c)["all_tensor_digests_match"])
        with self.assertRaises(ValueError):
            a.tensor_index({"model":{"bad":"not a tensor"}})

    def test_safe_loader_and_archive_pin(self):
        state={"model":{"weight":torch.tensor([1.0])},"config":{"size":1}}
        buffer=io.BytesIO()
        torch.save(state,buffer)
        raw=buffer.getvalue()
        loaded=a.load_state(raw)
        self.assertTrue(a.tensor_comparison(state,loaded)["all_tensor_digests_match"])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"test.zip"
            with zipfile.ZipFile(p,"w") as z:
                z.writestr("known.pt",raw)
            with zipfile.ZipFile(io.BytesIO(a.archive_bytes(p,a.sha(p.read_bytes())))) as z:
                self.assertEqual(a.member_bytes(z,"known.pt",a.sha(raw)),raw)
                with self.assertRaises(ValueError):
                    a.member_bytes(z,"known.pt","0"*64)
            with self.assertRaises(ValueError):
                a.archive_bytes(p,"0"*64)


if __name__=="__main__":
    unittest.main()
