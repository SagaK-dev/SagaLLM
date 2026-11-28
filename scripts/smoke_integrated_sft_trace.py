"""Exercise the existing Java SFT entrypoint with an opt-in journal on CPU."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from sagallm.config import ModelConfig
from sagallm.java_diverse_corpus import build_java_diverse_corpus
from sagallm.java_instruction_sft import train_instruction_sft, sha_file
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def run(output):
    output=Path(output);output.mkdir()
    torch.manual_seed(917)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        dataset=root/"data";build_java_diverse_corpus(dataset)
        tok=root/"tokenizer.json";ByteBPETokenizer().save(tok)
        cfg=ModelConfig(vocab_size=260,hidden_size=32,num_layers=1,num_heads=4,
            num_kv_heads=2,intermediate_size=64,max_position_embeddings=512)
        model=SagaTransformer(cfg)
        optim=torch.optim.AdamW(model.parameters())
        base=root/"base.pt"
        checkpoint_save(base,model,optim,0,cfg,{"tokenizer_sha256":sha_file(tok)})
        dest=root/"sft"
        report=train_instruction_sft(base,tok,dataset,dest,steps=2,batch_size=2,
            eval_every=1,eval_count=6,test_count=6,compile_cases=0,max_seconds=60,
            require_pinned=False,record_exposure=True)
        exp=report["optimizer_exposure"]
        if (report["optimizer_steps_completed"]!=2 or
            exp["source_binding"]["recorded_row_exposures_rechecked"]!=4 or
            exp["trace"]["checkpoints"]["best"]["checkpoint_sha256"]!=report["best_checkpoint_sha256"] or
            exp["trace"]["checkpoints"]["last"]["checkpoint_sha256"]!=report["last_checkpoint_sha256"] or
            report["test_used_for_model_selection"] is not False or report["java_code_executed"] is not False):
            raise ValueError("Integrated smoke scope mismatch")
        (output/"optimizer-exposure.jsonl").write_bytes((dest/"optimizer-exposure.jsonl").read_bytes())
        (output/"train-report.json").write_bytes((dest/"train-report.json").read_bytes())
        summary={"status":"INTEGRATED_CPU_SFT_TRACE_SMOKE_COMPLETE",
                 "optimizer_steps":2,"row_exposures":4,
                 "canonical_train_rows":report["original_split_counts"]["train"]["total"],
                 "best_step":report["best_checkpoint_step"],
                 "trace_sha256":exp["trace"]["trace_sha256"],"checkpoint_bindings":2,
                 "fresh_synthetic_fixture_base":True,"historical_base_used":False,
                 "java_compiled":False,"java_executed":False,
                 "independent_evaluation":"NOT_RUN","model_selection_permitted":False}
        (output/"smoke-summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
        print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args().output)
