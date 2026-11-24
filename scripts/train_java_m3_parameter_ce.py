"""Matched parameter CE experiment, CPU only; no generated Java execution."""
from __future__ import annotations
import argparse
import copy
import io
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from scripts import audit_java_m3_v2 as a
from scripts import diagnose_java_m3_parameters as d
from scripts import train_java_m3_ranking as r
from sagallm.config import ModelConfig
from sagallm.java_instruction_sft import encode_row, compact_prompt, pack, response_loss
from sagallm.java_stdio_curriculum import expected_rows, build_curriculum
from sagallm.tokenization import load_tokenizer
from sagallm.training import verify_checkpoint_tokenizer

SUITE_SHA="99c3de5b31128ecd7aa167014a3e736d883132a530ada2bad453e08f6d77317a"
PAIR_SHA="4d9b2d6eb7ff216b653953eda1f355c4cae6def22d8a9af31016e51c7daa7d0b"
PROTOCOL="JAVA_M3_PARAMETER_CE_V1"
ARMS=("control","parameter_ce")


def load_suite(path):
    suite=json.loads(a.read_pin(path,SUITE_SHA,20000))
    rows=suite["rows"]
    a.require(suite["schema"]=="sagallm-m3-parameter-development-v1"
              and suite["independent_benchmark"] is False and len(rows)==6
              and len({x["id"] for x in rows})==6,"Wrong holdback")
    for row in rows:
        a.require(len(row["cases"])==3,"Wrong holdback case count")
        family=row["family"]
        k=row["id"].rsplit("-",1)[1]
        for case in row["cases"]:
            s=case["stdin"]
            if family=="parameter_dev_negative_count":
                n,*v=map(int,s.split())
                a.require(n==len(v) and 0<=n<=20,"Bad negative count input")
                expected=str(sum(x<0 for x in v)*int(k))+"\n"
            elif family=="parameter_dev_adjacent_abs":
                n,*v=map(int,s.split())
                a.require(n==len(v) and 1<=n<=20 and all(-100<=x<=100 for x in v),"Bad adjacent input")
                expected=str(sum(abs(x-y) for x,y in zip(v,v[1:]))+int(k))+"\n"
            elif family=="parameter_dev_space_replace":
                a.require(s.isascii() and s.endswith("\n") and s.count("\n")==1 and k in ("X","Y"),"Bad text input")
                expected=s[:-1].replace(" ",k)+"\n"
            else:
                raise ValueError("Unknown new family")
            a.require(expected==case["stdout"],"New holdback oracle mismatch")
    return rows


def fresh_pair_rows(rows,tok):
    result=[]
    for i in (0,2,4):
        items=[{**x,"prefix_ids":[tok.SPECIAL["<bos>"]]+tok.encode(compact_prompt(x["task"]))}
               for x in rows[i:i+2]]
        result.append(d.verify_pair(*items,tok))
    return result


def train_pairs(lookup,tok):
    pairs=[d.verify_pair(lookup[f"train-{f}-{i:03d}"],lookup[f"train-{f}-{i+1:03d}"],tok)
           for f in d.FAMILY_NAMES for i in range(0,32,2)]
    digest=a.sha(json.dumps(pairs,sort_keys=True,separators=(",",":")).encode())
    a.require(digest==PAIR_SHA and len(pairs)==160
              and set(x for p in pairs for x in p["ids"])==set(lookup),"Pair manifest mismatch")
    return {tuple(p["ids"]):p for p in pairs}


def pair_logits(model,pair):
    # The same reference prefix is supplied to both prompts. Pad on the
    # right and gather at the last non-padding position, not the padded end.
    inputs=[p+pair["common_response_prefix"] for p in pair["prefixes"]]
    lengths=[len(v) for v in inputs]
    x=torch.zeros(2,max(lengths),dtype=torch.long)
    for i,v in enumerate(inputs):
        x[i,:len(v)]=torch.tensor(v,dtype=torch.long)
    logits=model(x).logits[torch.arange(2),torch.tensor(lengths)-1,:].float()
    return logits[:,pair["alternatives"]]


def auxiliary_loss(logits):
    if tuple(logits.shape)!=(2,2):
        raise ValueError("Expected two prompts and two alternatives")
    return F.cross_entropy(logits,torch.tensor([0,1],device=logits.device))


def fit(cfg,base,broad,lookup,pairs,schedule,selection,seed,coefficient):
    a.require(coefficient in (0.,1.),"Unknown coefficient")
    torch.manual_seed(seed)
    model=r.load_model(cfg,base)
    optimizer=torch.optim.AdamW(model.parameters(),lr=r.LR,weight_decay=0.01)
    initial=response_loss(model,selection)
    a.require(math.isfinite(initial),"Nonfinite initial validation")
    best,best_step,weights=initial,0,copy.deepcopy(model.state_dict())
    history=[{"step":0,"validation_nll":initial}]
    seen_broad=set()
    seen_stdio=set()
    total_aux=0.
    start=time.monotonic()
    for step,(bidx,ids) in enumerate(schedule,1):
        if time.monotonic()-start>300:
            raise TimeoutError("CPU arm budget exhausted")
        a.require(not seen_broad.intersection(bidx) and not seen_stdio.intersection(ids),"Repeated exposure")
        seen_broad.update(bidx)
        seen_stdio.update(ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x,y=pack([broad[i] for i in bidx]+[lookup[k] for k in ids])
        sft=model(x,labels=y).loss
        aux=auxiliary_loss(pair_logits(model,pairs[ids]))
        loss=sft+coefficient*aux
        a.require(torch.isfinite(loss).item(),"Nonfinite objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        total_aux+=float(aux.detach())
        if step in (80,160):
            current=response_loss(model,selection)
            a.require(math.isfinite(current),"Nonfinite validation")
            history.append({"step":step,"validation_nll":current})
            if current<=best-0.005:
                best,best_step,weights=current,step,copy.deepcopy(model.state_dict())
    a.require(step==160 and len(seen_broad)==len(seen_stdio)==320,"Exposure mismatch")
    return weights,{"steps":step,"selected_step":best_step,"history":history,
                    "best_validation_nll":best,"coefficient":coefficient,
                    "broad_rows_seen":len(seen_broad),"stdio_rows_seen":len(seen_stdio),
                    "auxiliary_directions":2*step,"mean_online_aux_ce":total_aux/step}


def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    torch.set_num_threads(2)
    fresh=load_suite(args.holdback)
    prior=r.frozen_suite(args.previous_holdback)
    r.pin(args.base,a.BASE,90*1024*1024)
    r.pin(args.tokenizer,a.TOKENIZER,2_000_000)
    build_curriculum(args.curriculum)
    r.pin(args.curriculum/"manifest.json",r.STDIO_SHA,8192)
    tok=load_tokenizer(args.tokenizer)
    def encode_all(rows):
        out=[encode_row(row,tok,384) for row in rows]
        a.require(all(x is not None for x in out),"Unexpected context exclusion")
        return out
    broad=encode_all(r.broad_rows())
    stdio=encode_all(expected_rows("train"))
    valid=encode_all(expected_rows("validation"))
    lookup={x["id"]:x for x in stdio}
    vlookup={x["id"]:x for x in valid}
    selection=[vlookup[k] for k in a.SELECTION]
    regression=[vlookup[k] for k in a.REGRESSION]
    old=broad+stdio+valid+prior
    old_families={f for spec in r.broad_data.SPLITS.values() for f in (*spec[0],*spec[2])}
    a.require(not ({x["family"] for x in fresh}&({x["family"] for x in old}|old_families))
              and not ({x["id"] for x in fresh}&{x["id"] for x in old})
              and not ({x["task"] for x in fresh}&{x["task"] for x in old}),
              "Fresh holdback overlaps existing rows")
    a.require(len(selection)==24 and not ({x["id"] for x in selection}&set(a.REGRESSION)),
              "Selection contamination")
    pairs=train_pairs(lookup,tok)
    groups,preflight=d.build_pairs(tok)
    a.require(preflight["manifest_sha256"]==PAIR_SHA,"Prior preflight mismatch")
    groups["fresh"]=fresh_pair_rows(fresh,tok)
    schedules={seed:r.schedule_for(broad,list(pairs),seed) for seed in r.SEEDS}
    hashes={str(s):r.sha(r.canonical([[ [broad[i]["id"] for i in b],list(ids)] for b,ids in v]))
            for s,v in schedules.items()}
    print(json.dumps({"preflight":"PASS","pair_manifest_sha256":PAIR_SHA,
                      "fresh_suite_sha256":SUITE_SHA,"schedules":hashes,
                      "pairs":len(pairs),"directions":320},sort_keys=True),flush=True)
    state=torch.load(io.BytesIO(a.read_pin(args.base,a.BASE,90*1024*1024)),map_location="cpu",weights_only=True)
    verify_checkpoint_tokenizer(state,args.tokenizer)
    cfg=ModelConfig(**state["config"])
    a.require(cfg.max_position_embeddings==384 and cfg.parameter_count()<=3_000_000
              and tok.vocab_size<=cfg.vocab_size,"Unexpected architecture")
    args.output.mkdir(parents=True)
    results={}
    frozen={}
    # All checkpoint selection finishes before any heldback scoring.
    for seed in r.SEEDS:
        results[str(seed)]={}
        for arm,coefficient in zip(ARMS,(0.,1.)):
            w,stats=fit(cfg,state["model"],broad,lookup,pairs,schedules[seed],selection,seed,coefficient)
            p=args.output/f"{seed}-{arm}.pt"
            torch.save({"model":w,"config":cfg.to_dict(),"base_sha256":a.BASE,
                        "tokenizer_sha256":a.TOKENIZER,"seed":seed,"arm":arm,
                        "protocol":PROTOCOL,"selected_step":stats["selected_step"],"production":False},p)
            stats["checkpoint_sha256"]=r.sha(p.read_bytes())
            stats["schedule_sha256"]=hashes[str(seed)]
            results[str(seed)][arm]=stats
            frozen[seed,arm]=w
            print(json.dumps({"seed":seed,"arm":arm,"steps":stats["steps"],
                              "selected_step":stats["selected_step"],"loss":stats["best_validation_nll"]}),flush=True)
    @torch.inference_mode()
    def diagnose(w):
        model=r.load_model(cfg,w)
        model.eval()
        metrics={k:d.aggregate([d.measure(model,p) for p in ps]) for k,ps in groups.items()}
        generations={k:r.inference(model,tok,items) for k,items in
                     (("regression",regression),("previous_development",prior),("fresh",fresh))}
        return {"forced_choice":metrics,"generation":generations}
    baseline=diagnose(state["model"])
    for (seed,arm),w in frozen.items():
        results[str(seed)][arm]["diagnostics"]=diagnose(w)
    report={"status":"PARAMETER_CE_CPU_TRAINING_AND_INFERENCE_COMPLETE","protocol":PROTOCOL,
            "base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,"broad_train_sha256":r.TRAIN_SHA,
            "stdio_manifest_sha256":r.STDIO_SHA,"pair_manifest_sha256":PAIR_SHA,
            "fresh_suite_sha256":SUITE_SHA,"previous_suite_sha256":a.SUITE,
            "seeds":list(r.SEEDS),"steps_per_arm":160,"batch_size":4,"learning_rate":r.LR,
            "schedule_sha256_by_seed":hashes,"selection_ids":a.SELECTION,
            "fresh_ids":[x["id"] for x in fresh],"same_aux_graph":True,
            "fresh_used_for_training_or_selection":False,"original_test_records_read":False,
            "baseline":baseline,"results":results,"torch_version":str(torch.__version__),
            "compile":"NOT_RUN","behavior":"NOT_RUN","independent_benchmark":"NOT_ESTABLISHED",
            "production_promotion":False,"root_cause":"NOT_ESTABLISHED"}
    raw=r.canonical(report)
    (args.output/"parameter-ce-report.json").write_bytes(raw)
    def brief(diag):
        return {"forced_choice":{k:{x:y for x,y in v.items() if x!="rows"} for k,v in diag["forced_choice"].items()},
                "unique_sources":{k:v["unique_source_hashes"] for k,v in diag["generation"].items()}}
    print(json.dumps({"status":report["status"],"report_sha256":r.sha(raw),
                      "baseline":brief(baseline),"arms":{s:{k:brief(v["diagnostics"]) for k,v in arms.items()}
                         for s,arms in results.items()}},sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("base","tokenizer","holdback","previous-holdback","curriculum","output"):
        p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
