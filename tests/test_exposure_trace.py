from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from sagallm import exposure_trace as a
from scripts.smoke_optimizer_exposure_trace import smoke

def header():
    return {"schema":a.SCHEMA,"split":"train","base_checkpoint_sha256":"a"*64,
            "tokenizer_sha256":"b"*64,"data_manifest_sha256":"c"*64,"train_split_sha256":"d"*64,
            "seed":1,"max_steps":2,"batch_size":1}

def batch():
    r={"id":"sample","family":"fixture","x":torch.tensor([1,2]),"y":torch.tensor([-100,3]),"response_tokens":1}
    return [r],r["x"].reshape(1,-1),r["y"].reshape(1,-1)

class ExposureTraceTests(unittest.TestCase):
    def test_real_training_and_embedded_checkpoint_bindings(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/"smoke"
            result=smoke(root)
            self.assertEqual(result["actual_fixture_optimizer_returns"],2)
            self.assertEqual(result["checkpoint_bindings_rechecked"],2)
            trace=result["fixture_trace"]
            self.assertEqual(trace["status"],"FINALIZED_LOCAL_TRACE")
            self.assertFalse(trace["authenticated_attestation"])
            p=root/"optimizer-exposure.jsonl"
            raw=p.read_bytes()
            p.write_bytes(raw.replace(b'"kind":"step_intent"',b'"kind":"step_intenT"',1))
            with self.assertRaises(ValueError):
                a.verify_trace(p)
            p.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                a.verify_trace(p)
            p.write_bytes(b"\n".join(raw.splitlines()[:-1])+b"\n")
            self.assertEqual(a.verify_trace(p)["status"],"INCOMPLETE_LOCAL_TRACE")
            with self.assertRaises(ValueError):
                a.verify_trace(p,expected_tip=trace["tip_sha256"])

    def test_pending_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"trace.jsonl"
            with a.ExposureTrace(p,header()) as trace:
                intent=trace._begin(*batch())
                with self.assertRaises(ValueError):
                    trace._applied("0"*64)
                with self.assertRaises(ValueError):
                    trace.finish()
            r=a.verify_trace(p)
            self.assertEqual(r["pending_step"],1)
            self.assertEqual(r["completed_optimizer_returns"],0)
            self.assertEqual(r["status"],"INCOMPLETE_LOCAL_TRACE")
            with self.assertRaises(FileExistsError):
                a.ExposureTrace(p,header())

    def test_exact_batch_and_label_mask(self):
        rows,x,y=batch()
        r=a.batch_record(rows,x,y)
        self.assertEqual(r["loss_targets"],1)
        with self.assertRaises(ValueError):
            a.batch_record(rows,x,y+1)
        with self.assertRaises(ValueError):
            a.batch_record(rows,x,torch.tensor([[-100,-100]]))
        bad={**header(),"split":"test"}
        with self.assertRaises(ValueError):
            a.validate_header(bad)

    def test_optimizer_error_is_not_completed(self):
        with tempfile.TemporaryDirectory() as t:
            from sagallm.config import ModelConfig
            from sagallm.model import SagaTransformer
            model=SagaTransformer(ModelConfig(vocab_size=260,hidden_size=16,num_layers=1,num_heads=2,
                    num_kv_heads=1,intermediate_size=32,max_position_embeddings=16))
            optim=torch.optim.AdamW(model.parameters())
            p=Path(t)/"trace.jsonl"
            with a.ExposureTrace(p,header()) as trace:
                with patch.object(optim,"step",side_effect=RuntimeError("fixture optimizer failure")):
                    with self.assertRaises(RuntimeError):
                        a.traced_sft_step(model,optim,trace,*batch())
                self.assertTrue(trace.poisoned)
                with self.assertRaises(ValueError):
                    trace.checkpoint_metadata()
            self.assertEqual(a.verify_trace(p)["pending_step"],1)
            self.assertEqual(a.verify_trace(p)["completed_optimizer_returns"],0)

if __name__=="__main__":
    unittest.main()
