"""Frozen-weight first-layer prompt attention bias: bounded CPU research pilot.

No Java execution, training, external evaluation, or modification to model.py.
The two arms use the identical uncached forward, differing only in one
attention-logit bias. Only previously exposed TRAIN and regression probes.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import time
import zipfile

import torch
import torch.nn.functional as F

from sagallm.config import ModelConfig
from sagallm.model import rotary_cache, apply_rope
from sagallm.tokenization import load_tokenizer
from sagallm.training import verify_checkpoint_tokenizer
from sagallm.java_instruction_sft import compact_prompt
from scripts import audit_java_m3_v2 as a
from scripts import diagnose_java_m3_parameters as d
from scripts import diagnose_java_m3_prompt_signal as s
from scripts import train_java_m3_parameter_ce as p
from scripts import train_java_m3_ranking as r
from scripts.evaluate_java_m3_v2 import suites

SWAP_ZIP="e428e95300195cfcde474592f721b9be45995789bbe51b31549cc780bd1c878d"
SWAP_REPORT="3d8436e4cfee22aee4918a8718a5eb270d94f7a3af832adb724ff4691e47742c"
BIAS=math.log(2.)
ATOL=1e-4


def verify_swap(path):
    raw=a.read_pin(path,SWAP_ZIP,1000000)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        a.require(z.namelist()==["activation-swap.json"] and z.infolist()[0].file_size<5000000,
                  "Unexpected activation archive")
        data=z.read("activation-swap.json")
    a.require(a.sha(data)==SWAP_REPORT,"Activation report mismatch")
    obj=json.loads(data)
    a.require(obj["status"]=="ACTIVATION_SWAP_DIAGNOSTIC_COMPLETE" and obj["matched_model_pairs"]==133
              and obj["control_checks"]==3192 and obj["archive_sha256"]==s.ARCHIVE
              and obj["training_report_sha256"]==s.REPORT and obj["pair_manifest_sha256"]==p.PAIR_SHA
              and obj["production_promotion"] is False and obj["behavior"]=="NOT_RUN",
              "Activation evidence mismatch")
    a.require(len(obj["models"])==7 and all(v["pairs"]==19 for v in obj["models"].values()),"Coverage mismatch")
    for v in obj["models"].values():
        a.require(v["summary"]["block_1_attention"]["final_response"]["mean_donor_directed_shift"]>0
                  and v["summary"]["block_2_attention"]["final_response"]["mean_donor_directed_shift"]<0,
                  "Layer hypothesis not supported")
    return obj


def attention_mask(length,prompt_length,bias,*,device=None,dtype=torch.float32):
    if not (type(length) is int and type(prompt_length) is int and 2<=prompt_length<=length<=384
            and bias in (0.,BIAS)):
        raise ValueError("Invalid fixed attention-bias configuration")
    q=torch.arange(length,device=device)[:,None]
    k=torch.arange(length,device=device)[None,:]
    mask=torch.zeros((length,length),device=device,dtype=dtype)
    # Query at the last prompt token predicts the first response token.
    # BOS excluded; include all other prompt positions, not oracle-selected
    # changed tokens. Never unmask a future key.
    region=(q>=prompt_length-1)&(k>=1)&(k<prompt_length)
    return mask.masked_fill(region,bias).masked_fill(k>q,float("-inf"))


def logits(model,ids,prompt_length,bias):
    if ids.ndim!=2 or ids.size(0)!=1 or ids.dtype!=torch.long or model.training:
        raise ValueError("Single-sequence eval-only forward required")
    n=ids.size(1)
    x=model.embedding(ids)
    first=model.blocks[0]
    att=first.attention
    h=first.attention_norm(x)
    q=att.q_proj(h).reshape(1,n,att.heads,att.head_dim).transpose(1,2)
    k=att.k_proj(h).reshape(1,n,att.kv_heads,att.head_dim).transpose(1,2)
    v=att.v_proj(h).reshape(1,n,att.kv_heads,att.head_dim).transpose(1,2)
    pos=torch.arange(n,device=ids.device)
    cos,sin=rotary_cache(pos,att.head_dim,att.rope_theta,q.dtype)
    q=apply_rope(q,cos,sin)
    k=apply_rope(k,cos,sin)
    if att.heads!=att.kv_heads:
        repeats=att.heads//att.kv_heads
        k=k.repeat_interleave(repeats,dim=1)
        v=v.repeat_interleave(repeats,dim=1)
    mask=attention_mask(n,prompt_length,bias,device=q.device,dtype=q.dtype)
    attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=False,dropout_p=0.)
    x=x+att.o_proj(attended.transpose(1,2).contiguous().view(1,n,att.heads*att.head_dim))
    x=x+first.feed_forward(first.ffn_norm(x))
    for block in model.blocks[1:]:
        x,_=block(x)
    return model.lm_head(model.norm(x)).float()


def forced(model,pair,bias):
    values=[]
    max_error=0.
    for prompt in pair["prefixes"]:
        x=torch.tensor([prompt+pair["common_response_prefix"]],dtype=torch.long)
        actual=logits(model,x,len(prompt),bias)[0,-1]
        a.require(torch.isfinite(actual).all().item(),"Nonfinite forced logits")
        if bias==0.:
            orig=model(x).logits[0,-1].float()
            max_error=max(max_error,float((actual-orig).abs().max()))
            a.require(torch.allclose(actual,orig,atol=ATOL,rtol=0),"Zero-bias forward mismatch")
        values.extend(float(actual[i]) for i in pair["alternatives"])
    return {**d.score_logits(*values),"ids":pair["ids"],"family":pair["family"],
            "zero_bias_max_error":max_error if bias==0. else None}


def generate(model,tok,task,bias):
    prefix=[tok.SPECIAL["<bos>"]]+tok.encode(compact_prompt(task["task"]))
    limit=min(r.CAP,384-len(prefix))
    a.require(limit>=2,"Prompt overflow")
    ids=torch.tensor([prefix],dtype=torch.long)
    eos=tok.SPECIAL["<eos>"]
    start=time.monotonic()
    new=[]
    for _ in range(limit):
        if time.monotonic()-start>30:
            raise TimeoutError("Generation budget exhausted")
        last=logits(model,ids,len(prefix),bias)[0,-1,:tok.vocab_size]
        a.require(torch.isfinite(last).all().item(),"Nonfinite generation logits")
        token=int(last.argmax())
        new.append(token)
        ids=torch.cat((ids,torch.tensor([[token]],dtype=torch.long)),dim=1)
        if token==eos:
            break
    source=tok.decode(new)
    return {"id":task["id"],"family":task["family"],"source_sha256":a.sha(source.encode("utf8")),
            "tokens":len(new),"stop":"EOS" if new[-1]==eos else "TOKEN_LIMIT",
            "compile":"NOT_RUN","behavior":"NOT_RUN"}


def summarize_generation(rows):
    return {"rows":rows,"unique_source_hashes":len({x["source_sha256"] for x in rows}),
            "cross_family_identical_pairs":sum(x["family"]!=y["family"] and x["source_sha256"]==y["source_sha256"]
                                               for i,x in enumerate(rows) for y in rows[i+1:])}


def screen(models):
    # Exploratory mechanism screen on TRAIN only: improve BOTH choices
    # in at least five of seven fixed models. Regression is descriptive;
    # no decision may be made from its outputs or the exposed holdbacks.
    passed=[k for k,v in models.items()
            if v["bias"]["train"]["both_choices_correct"]>v["control"]["train"]["both_choices_correct"]]
    return {"decision":"TRAIN_PROBE_SIGNAL_ONLY" if len(passed)>=5 else "NO_CONSISTENT_TRAIN_PROBE_SIGNAL",
            "improved_models":passed,"required_models":5,"models":7,
            "behavior":"NOT_RUN","production_promotion":False}


def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    torch.set_num_threads(2)
    prior=verify_swap(args.swap)
    report,decision=s.audit(args.archive)
    a.require(decision["decision"]=="NO_CONSISTENT_PARAMETER_SIGNAL","Prior screen mismatch")
    tok=load_tokenizer(args.tokenizer)
    a.read_pin(args.tokenizer,a.TOKENIZER,2000000)
    probes,manifest=d.build_pairs(tok)
    a.require(manifest["manifest_sha256"]==p.PAIR_SHA and len(probes["train"])==20
              and len(probes["validation"])==3,"Probe mismatch")
    # The old suite is validated only to reconstruct existing regression
    # tasks; neither old nor fresh holdback is scored in this pilot.
    regression=suites(args.previous_holdback)["regression"]
    base=torch.load(io.BytesIO(a.read_pin(args.base,a.BASE,90*1024*1024)),map_location="cpu",weights_only=True)
    verify_checkpoint_tokenizer(base,args.tokenizer)
    cfg=ModelConfig(**base["config"])
    a.require(cfg.num_layers==4 and cfg.max_position_embeddings==384 and cfg.parameter_count()<=3000000
              and tok.vocab_size<=cfg.vocab_size,"Architecture mismatch")
    names=["base"]+[seed+"-"+arm for seed in a.SEEDS for arm in s.ARMS]
    a.require(set(prior["models"])==set(names),"Prior model coverage mismatch")
    results={}
    raw_archive=a.read_pin(args.archive,s.ARCHIVE,64*1024*1024)
    # Freeze all zero-bias equivalence checks before treatment.
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as z,torch.inference_mode():
        models={}
        for name in names:
            if name=="base":
                state=base
                diag=report["baseline"]
            else:
                seed,arm=name.split("-")
                stats=report["results"][seed][arm]
                raw=z.read(name+".pt")
                a.require(a.sha(raw)==stats["checkpoint_sha256"],"Checkpoint mismatch")
                state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
                for k,v in {"protocol":p.PROTOCOL,"base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,
                            "seed":int(seed),"arm":arm,"selected_step":stats["selected_step"],"production":False}.items():
                    a.require(state[k]==v,"Checkpoint metadata mismatch")
                diag=stats["diagnostics"]
            a.require(state["config"]==cfg.to_dict(),"Config mismatch")
            model=r.load_model(cfg,state["model"])
            model.eval()
            train=[forced(model,q,0.) for q in probes["train"]]
            valid=[forced(model,q,0.) for q in probes["validation"]]
            for key,rows in (("train",train),("validation",valid)):
                for row,old in zip(rows,diag["forced_choice"][key]["rows"]):
                    a.require(row["ids"]==old["ids"] and all(abs(row[k]-old[k])<=ATOL for k in
                              ("left_correct_minus_other","right_correct_minus_other")),"Zero-bias margin mismatch")
            gen=summarize_generation([generate(model,tok,t,0.) for t in regression])
            a.require(gen==diag["generation"]["regression"],"Zero-bias generation replay mismatch")
            results[name]={"control":{"train":d.aggregate(train),"validation":d.aggregate(valid),
                                       "regression_generation":gen}}
            models[name]=model
            print(json.dumps({"model":name,"zero_bias_replay":"PASS","regression_generations":6}),flush=True)
        for name,model in models.items():
            results[name]["bias"]={
                "train":d.aggregate([forced(model,q,BIAS) for q in probes["train"]]),
                "validation":d.aggregate([forced(model,q,BIAS) for q in probes["validation"]]),
                "regression_generation":summarize_generation([generate(model,tok,t,BIAS) for t in regression])}
            print(json.dumps({"model":name,"biased_inference":"COMPLETE"}),flush=True)
    result={"status":"FIRST_LAYER_PROMPT_BIAS_PILOT_COMPLETE","protocol":"JAVA_M3_FIRST_LAYER_PROMPT_BIAS_V1",
            "archive_sha256":s.ARCHIVE,"training_report_sha256":s.REPORT,"activation_zip_sha256":SWAP_ZIP,
            "activation_report_sha256":SWAP_REPORT,"pair_manifest_sha256":p.PAIR_SHA,
            "base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,"layer":1,"bias_logit":BIAS,
            "zero_bias_generation_matches":42,"models":results,"screen":screen(results),
            "torch_version":str(torch.__version__),"training":"NOT_RUN","compile":"NOT_RUN","behavior":"NOT_RUN",
            "independent_benchmark":"NOT_ESTABLISHED","root_cause":"NOT_ESTABLISHED","production_promotion":False}
    data=r.canonical(result)
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps({"status":result["status"],"report_sha256":a.sha(data),"screen":result["screen"],
                      "zero_bias_generation_matches":42,"models":{name:{
                          arm:{"train_both":v["train"]["both_choices_correct"],
                               "train_interaction":v["train"]["mean_interaction"],
                               "validation_both":v["validation"]["both_choices_correct"],
                               "regression_unique":v["regression_generation"]["unique_source_hashes"]}
                          for arm,v in arms.items()} for name,arms in results.items()}},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("archive","swap","base","tokenizer","previous-holdback","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
