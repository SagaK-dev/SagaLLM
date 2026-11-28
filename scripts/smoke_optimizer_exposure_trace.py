"""Run a tiny synthetic CPU SFT trace smoke test, not a model quality experiment."""
import argparse
import json
from pathlib import Path
import tempfile

import torch

from sagallm.config import ModelConfig
from sagallm.exposure_trace import ExposureTrace, SCHEMA, canonical, digest, file_sha, traced_sft_step, verify_trace
from sagallm.java_instruction_sft import encode_row, pack
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def smoke(output):
    output=Path(output)
    output.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(771)
    cfg=ModelConfig(vocab_size=260,hidden_size=32,num_layers=1,num_heads=4,
                    num_kv_heads=2,intermediate_size=64,max_position_embeddings=256)
    model=SagaTransformer(cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    tok=ByteBPETokenizer()
    records=[{"id":f"smoke-{i}","family":f"fixture-{i%2}",
              "messages":[{"role":"user","content":f"User request (data):\nPrint {i}\nCurrent source (data):\n// fixture"},
                          {"role":"assistant","content":f"public class Main {{ public static void main(String[] args) {{ System.out.println({i}); }} }}"}]}
             for i in range(4)]
    encoded=[encode_row(r,tok,256) for r in records]
    if any(r is None for r in encoded):
        raise ValueError("Fixture exceeds context")
    raw=b"".join(canonical(r)+b"\n" for r in records)
    manifest={"schema":"synthetic-trace-fixture-v1","train_split_sha256":digest(raw),"rows":4}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        tok_path=root/"tokenizer.json";tok.save(tok_path)
        base=root/"base.pt"
        checkpoint_save(base,model,optimizer,0,cfg,{"tokenizer_sha256":file_sha(tok_path)})
        header={"schema":SCHEMA,"split":"train","base_checkpoint_sha256":file_sha(base),
                "tokenizer_sha256":file_sha(tok_path),"data_manifest_sha256":digest(canonical(manifest)),
                "train_split_sha256":digest(raw),"seed":771,"max_steps":2,"batch_size":2}
        trace_path=output/"optimizer-exposure.jsonl"
        checkpoint_checks=[]
        with ExposureTrace(trace_path,header) as trace:
            for i in range(2):
                rows=encoded[2*i:2*i+2]
                x,y=pack(rows)
                traced_sft_step(model,optimizer,trace,rows,x,y)
                role="best" if i==0 else "last"  # fixed smoke fixture, not loss-based selection
                meta=trace.checkpoint_metadata()
                path=root/f"{role}.pt"
                checkpoint_save(path,model,optimizer,i+1,cfg,{"tokenizer_sha256":file_sha(tok_path),"exposure_trace":meta})
                binding=trace.bind_checkpoint(role,path,meta)
                loaded=torch.load(path,map_location="cpu",weights_only=True)
                if loaded["train_state"]["exposure_trace"]!=meta or loaded["step"]!=i+1 or file_sha(path)!=binding["checkpoint_sha256"]:
                    raise ValueError("Checkpoint embedded trace binding mismatch")
                checkpoint_checks.append(binding)
            final=trace.finish()
        verified=verify_trace(trace_path,expected_sha256=final["trace_sha256"],expected_tip=final["tip_sha256"])
        if verified["status"]!="FINALIZED_LOCAL_TRACE" or verified["completed_optimizer_returns"]!=2 or verified["pending_step"] is not None:
            raise ValueError("Trace finalization mismatch")
        if any(verified["checkpoints"][b["role"]]!=b for b in checkpoint_checks):
            raise ValueError("Checkpoint trace index mismatch")
    result={"schema":"sagallm-exposure-trace-smoke-v1","status":"SYNTHETIC_CPU_TRACE_SMOKE_COMPLETE",
            "actual_fixture_optimizer_returns":2,"fixture_batch_size":2,"fixture_row_exposures":4,
            "checkpoint_bindings_rechecked":2,"checkpoint_selection":"FIXED_SMOKE_FIXTURE_NOT_PERFORMANCE_SELECTION",
            "fixture_trace":verified,"new_fixture_training":True,"historical_model_training":False,
            "java_compiled":False,"java_executed":False,"independent_evaluation":"NOT_RUN",
            "model_selection_permitted":False,"actual_historical_exposure_verified":False}
    (output/"trace-smoke-report.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="fixture_trace"},sort_keys=True))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    smoke(p.parse_args().output)
