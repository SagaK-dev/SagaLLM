import tempfile
from pathlib import Path
import unittest
import torch
from sagallm.sft_lr_schedule import step_lr, validate_schedule
from sagallm.exposure_trace import ExposureTrace, verify_trace, SCHEMA

class ScheduleTests(unittest.TestCase):
    def test_rates(self):
        s={"kind":"warmup_cosine","peak_lr":4e-5,"warmup_steps":4,"min_lr_ratio":.1}
        rates=[step_lr(s,24,i) for i in range(1,25)]
        self.assertEqual(rates[0],1e-5)
        self.assertEqual(rates[3],4e-5)
        self.assertAlmostEqual(rates[-1],4e-6)
        self.assertTrue(all(a>b for a,b in zip(rates[3:],rates[4:])))
        for k,v in (("warmup_steps",23),("peak_lr",float("nan")),("min_lr_ratio",0),("kind","other")):
            with self.assertRaises(ValueError):validate_schedule({**s,k:v},24)
    def test_trace_reject_wrong_rate(self):
        s={"kind":"constant","peak_lr":4e-5,"warmup_steps":0,"min_lr_ratio":1.}
        h={"schema":SCHEMA,"split":"train","base_checkpoint_sha256":"a"*64,"tokenizer_sha256":"b"*64,
            "data_manifest_sha256":"c"*64,"train_split_sha256":"d"*64,"seed":1,"max_steps":2,"batch_size":1,"lr_schedule":s}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"trace";r={"id":"i","family":"f","x":torch.tensor([1,2]),"y":torch.tensor([-100,3]),"response_tokens":1}
            with ExposureTrace(p,h) as t:
                with self.assertRaises(ValueError):t._begin([r],r["x"][None,:],r["y"][None,:],1e-5)
                intent=t._begin([r],r["x"][None,:],r["y"][None,:],4e-5)
                t._applied(intent)
            self.assertEqual(verify_trace(p)["completed_optimizer_returns"],1)

if __name__=="__main__":unittest.main()
