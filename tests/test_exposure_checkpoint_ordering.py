import tempfile
import unittest
from pathlib import Path
from sagallm.exposure_trace import ExposureTrace, verify_trace, MAX_EVENTS, MAX_STEPS
import torch
from sagallm.exposure_trace import SCHEMA

def header():
    return {'schema':SCHEMA,'split':'train','base_checkpoint_sha256':'a'*64,'tokenizer_sha256':'b'*64,
            'data_manifest_sha256':'c'*64,'train_split_sha256':'d'*64,'seed':1,'max_steps':2,'batch_size':1}

def batch():
    r={'id':'one','family':'fixture','x':torch.tensor([1,2]),'y':torch.tensor([-100,3]),'response_tokens':1}
    return [r],r['x'].reshape(1,-1),r['y'].reshape(1,-1)

class TraceCheckpointOrderingTests(unittest.TestCase):
    def test_step_zero_best_last_and_terminal_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"trace"; cp=Path(d)/"opaque"; cp.write_bytes(b"fixture")
            with ExposureTrace(p,header()) as t:
                t.bind_checkpoint("best",cp,t.checkpoint_metadata())
                with self.assertRaises(ValueError):
                    t.bind_checkpoint("best",cp,t.checkpoint_metadata())
                t.bind_checkpoint("last",cp,t.checkpoint_metadata())
                with self.assertRaises(ValueError):
                    t._begin(*batch())
                final=t.finish()
            self.assertEqual(verify_trace(p,expected_tip=final["tip_sha256"])["completed_optimizer_returns"],0)
        self.assertGreaterEqual(MAX_EVENTS,MAX_STEPS*3+4)

if __name__=="__main__":
    unittest.main()
