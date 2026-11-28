from pathlib import Path
import tempfile
import unittest

from sagallm.exposure_source_binding import replay_source_binding
from sagallm.exposure_trace import ExposureTrace, SCHEMA, canonical, digest, file_sha
from sagallm.java_instruction_sft import encode_row, pack
from sagallm.tokenization import ByteBPETokenizer
from scripts.smoke_optimizer_exposure_trace import smoke


def record(ident="fixture-1", number=1):
    return {"id":ident,"family":"fixture",
            "messages":[{"role":"user","content":f"User request (data):\nPrint {number}\nCurrent source (data):\n// fixture"},
                        {"role":"assistant","content":f"public class Main {{ public static void main(String[] args) {{ System.out.println({number}); }} }}"}]}


class SourceBindingTests(unittest.TestCase):
    def test_actual_smoke_reencoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=smoke(Path(tmp)/"run")["fixture_source_binding"]
            self.assertEqual(d["status"],"RECORDED_TRAIN_INPUTS_MATCH_PINNED_SOURCE")
            self.assertEqual(d["recorded_row_exposures_rechecked"],4)
            self.assertEqual(d["packed_batches_rechecked"],2)
            self.assertFalse(d["validation_or_test_source_read"])
            self.assertFalse(d["actual_training_process_attested"])

    def fixture(self,root,*,recorded=None,source=None):
        # Synthetic journal construction for negative tests; not a training run.
        ds=root/"dataset";ds.mkdir()
        raw=b"".join(canonical(r)+b"\n" for r in (source if source is not None else [record()]))
        (ds/"train.jsonl").write_bytes(raw)
        (ds/"manifest.json").write_bytes(canonical({"files":{"train":{"filename":"train.jsonl","sha256":digest(raw),
                "count":len(raw.splitlines())}}}))
        tokenizer=ByteBPETokenizer()
        tok=root/"tok.json";tokenizer.save(tok)
        base=root/"base.pt";base.write_bytes(b"opaque fixture")
        h={"schema":SCHEMA,"split":"train","base_checkpoint_sha256":file_sha(base),
           "tokenizer_sha256":file_sha(tok),"data_manifest_sha256":file_sha(ds/"manifest.json"),
           "train_split_sha256":digest(raw),"seed":1,"max_steps":1,"batch_size":1}
        p=root/"trace.jsonl"
        rows=[encode_row(recorded if recorded is not None else record(),tokenizer,256)]
        x,y=pack(rows)
        with ExposureTrace(p,h) as trace:
            intent=trace._begin(rows,x,y)
            trace._applied(intent)
            for role in ("best","last"):
                trace.bind_checkpoint(role,base,trace.checkpoint_metadata())
            final=trace.finish()
        args={"expected_trace_sha256":final["trace_sha256"],"expected_tip":final["tip_sha256"],
              "dataset_dir":ds,"tokenizer_path":tok,"base_checkpoint":base,"context":256}
        return p,args

    def test_train_membership_and_token_digest(self):
        for row in (record("unknown-2"),record(number=2)):
            with tempfile.TemporaryDirectory() as tmp:
                p,args=self.fixture(Path(tmp),recorded=row)
                with self.assertRaises(ValueError):
                    replay_source_binding(p,**args)

    def test_pinned_source_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,args=self.fixture(Path(tmp))
            result=replay_source_binding(p,**args)
            self.assertEqual(result["train_source_rows"],1)
            self.assertEqual(result["recorded_row_exposures_rechecked"],1)
            raw=(args["dataset_dir"]/"train.jsonl").read_bytes()
            (args["dataset_dir"]/"train.jsonl").write_bytes(raw+b" ")
            with self.assertRaises(ValueError):
                replay_source_binding(p,**args)
        with tempfile.TemporaryDirectory() as tmp:
            p,args=self.fixture(Path(tmp),source=[record(),record(number=2)])
            with self.assertRaises(ValueError):
                replay_source_binding(p,**args)


if __name__=="__main__":
    unittest.main()
