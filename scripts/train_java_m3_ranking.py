"""M3 research: paired response-only SFT vs explicit conditional ranking.

CPU only. No generated Java compilation/execution, no production promotion.
The future behavioral evaluator requires separate reviewed VM authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import verify_checkpoint_tokenizer
from sagallm.java_instruction_sft import (
    V016_TOKENIZER_SHA256, compact_prompt, encode_row, interleaved_order,
    pack, response_loss,
)
from sagallm.java_stdio_curriculum import (
    TRAIN_FAMILIES, VALIDATION_FAMILIES, expected_rows, build_curriculum,
)
from sagallm import java_diverse_corpus as broad_data
from scripts.train_java_contrastive_stdio_ab import FAMILY_PAIRS
from scripts.train_java_stdio_semantic_sft import PINNED_V018
from scripts.java_m3_operation_masks import train_positions

SEEDS = (4201, 4202, 4203)
STEPS = 160
LR = 2e-5
MARGIN = 0.05
CAP = 170
TRAIN_SHA = "181346d5395efdd2ef376730321e1585d453b2b09a822302f6f2d03137e1aa8d"
STDIO_SHA = "0cea8ded8a5c03d15b2b516a86665901266b0b375095579daa8d712661316a13"
SUITE_SHA = "d641a53e6498a096fda15e92871e10db6cc2380fe1bb1be543c8f4a06aeb035a"
REGRESSION = tuple(f"validation-{f}-{i:03d}" for f in VALIDATION_FAMILIES for i in (2,3))
ARMS = ("control", "ranking")
# Predeclared before training: ambiguous positions are excluded, but every
# SFT example is retained. All five pair types must be represented.
MIN_ADMITTED_ROWS = 160


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))+"\n").encode("utf8")


def pin(path, expected, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing or unsafe pinned file")
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError("Pinned SHA mismatch")
    return data


def substitution_positions(a, b):
    """Unambiguous substitutions across ALL minimum-edit alignments.

    Unit-cost insert/delete/substitute. Keep a position only when every
    optimal path consumes it against the same different token. Never score
    insertions, deletions, identical tokens or EOS (caller excludes EOS).
    """
    n, m = len(a), len(b)
    if not (0 < n <= 384 and 0 < m <= 384):
        raise ValueError("Alignment outside bounded context")
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        d[i][0] = i
    for j in range(m+1):
        d[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1,
                          d[i-1][j-1]+(a[i-1]!=b[j-1]))
    r = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n,-1,-1):
        r[i][m] = n-i
    for j in range(m,-1,-1):
        r[n][j] = m-j
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            r[i][j] = min(r[i+1][j]+1, r[i][j+1]+1,
                          r[i+1][j+1]+(a[i]!=b[j]))
    best = d[n][m]
    result = []
    for i in range(n):
        options = set()
        for j in range(m+1):
            if d[i][j]+1+r[i+1][j] == best:
                options.add(None)
            if j<m and d[i][j]+(a[i]!=b[j])+r[i+1][j+1] == best:
                options.add(j)
        if len(options)==1:
            j=next(iter(options))
            if j is not None and a[i]!=b[j]:
                result.append(i)
    return tuple(result)


def hinge(own, wrong):
    return torch.relu(MARGIN + own - wrong)


def target(row, eos):
    ids = row["y"][row["y"] != -100].tolist()
    if len(ids)<2 or ids[-1]!=eos or eos in ids[:-1]:
        raise ValueError("Invalid reference EOS")
    return ids[:-1]


def make_aux(row, other, eos, *, tokenizer=None, mask_mode="edit-v1"):
    tgt, ref = target(row,eos), target(other,eos)
    if mask_mode=="edit-v1":
        pos=substitution_positions(tgt,ref)
    elif mask_mode=="operation-span-v2":
        if tokenizer is None:
            raise ValueError("Tokenizer required for operation spans")
        pos=train_positions(row,tgt,tokenizer)
    else:
        raise ValueError("Unknown mask protocol")
    if not pos:
        return None
    seqs=[]
    for prompt in (row["prefix_ids"],other["prefix_ids"]):
        seq=prompt+tgt
        if len(seq)-1>384:
            raise ValueError("Wrong-prompt target does not fit")
        x=torch.tensor(seq[:-1],dtype=torch.long)
        y=torch.full_like(x,-100)
        for p in pos:
            y[len(prompt)-1+p]=tgt[p]
        seqs.append({"x":x,"y":y})
    return tuple(seqs),pos


def pairs(rows,eos, *, tokenizer=None, mask_mode="edit-v1"):
    lookup={r["id"]:r for r in rows}
    if len(rows)!=320 or len(lookup)!=320:
        raise ValueError("Wrong stdio TRAIN size")
    if {r["family"] for r in rows} != set(TRAIN_FAMILIES):
        raise ValueError("Wrong stdio TRAIN families")
    table, counts, ordered = {},Counter(),[]
    for left,right in FAMILY_PAIRS:
        pair_name=left+"__"+right
        for i in range(32):
            ids=tuple(f"train-{f}-{i:03d}" for f in (left,right))
            a,b=(lookup[k] for k in ids)
            if (a["family"],b["family"]) != (left,right):
                raise ValueError("Pair family mismatch")
            ordered.append(ids)
            for x,y in ((a,b),(b,a)):
                v=make_aux(x,y,eos,tokenizer=tokenizer,mask_mode=mask_mode)
                table[x["id"]]=v
                if v is not None:
                    counts[pair_name]+=1
    if set(table)!=set(lookup):
        raise ValueError("Pair coverage mismatch")
    # Commit to the exact training-only position masks before model loading.
    mask_manifest={k:list(v[1]) if v is not None else [] for k,v in sorted(table.items())}
    coverage={left+"__"+right:counts[left+"__"+right] for left,right in FAMILY_PAIRS}
    admission={"mask_protocol":mask_mode,"admitted_rows":sum(counts.values()),"excluded_rows":320-sum(counts.values()),
               "rows_per_pair_type":coverage,
               "aligned_token_positions":sum(len(v[1]) for v in table.values() if v),
               "mask_sha256":sha(canonical(mask_manifest)),
               "minimum_admitted_rows":MIN_ADMITTED_ROWS,
               "all_five_pair_types_represented":all(v>0 for v in coverage.values())}
    admission["passed"]=admission["admitted_rows"]>=MIN_ADMITTED_ROWS and admission["all_five_pair_types_represented"]
    return table,ordered,admission


def broad_rows():
    """Regenerate TRAIN only; match the existing pinned v0.18 TRAIN bytes."""
    out=[]
    old,n,new,m=broad_data.SPLITS["train"]
    for families,count,func in ((old,n,broad_data.old_case),(new,m,broad_data.new_case)):
        for family in families:
            for i in range(count):
                ja,en,source=func(family,i)
                out.append({"id":f"{family}-{i:04d}","family":family,"messages":[
                    {"role":"user","content":broad_data._model_user_message(ja if i%2==0 else en,broad_data.BASE_SOURCE,"Main")},
                    {"role":"assistant","content":source}]})
    random.Random(broad_data.SEED+len("train")).shuffle(out)
    raw=("\n".join(broad_data._json(x) for x in out)+"\n").encode("utf8")
    if len(out)!=17080 or sha(raw)!=TRAIN_SHA:
        raise ValueError("Broad TRAIN bytes mismatch")
    return out


def oracle(row):
    """Recompute the frozen synthetic cases without Java or model outputs."""
    key=row["id"]
    k=int(key.rsplit("-",1)[1])
    for case in row["cases"]:
        raw=case["stdin"]
        if key.startswith("dev-smallest-squares-"):
            a=list(map(int,raw.split()))
            if len(a)!=3 or not all(-100<=v<=100 for v in a):
                raise ValueError("Bad three-integer case")
            v=sum(x*x for x in sorted(a)[:2])+k
        elif key.startswith("dev-adjacent-gap-"):
            n,*a=map(int,raw.split())
            if n!=len(a) or not 0<=n<=20 or not all(-100<=v<=100 for v in a):
                raise ValueError("Bad list case")
            v=sum(y-x>k for x,y in zip(a,a[1:]))
        elif key.startswith("dev-digit-transition-"):
            if not raw.isascii() or raw.count("\n")!=1 or not raw.endswith("\n"):
                raise ValueError("Bad ASCII case")
            s=raw[:-1]
            v=sum(("0"<=x<="9") != ("0"<=y<="9") for x,y in zip(s,s[1:]))+k
        else:
            raise ValueError("Unknown holdback case")
        if case["stdout"]!=str(v)+"\n":
            raise ValueError("Frozen case disagrees with oracle")


def frozen_suite(path):
    obj=json.loads(pin(path,SUITE_SHA,20000))
    rows=obj["rows"]
    if obj["schema"]!="sagallm-m3-synthetic-development-holdback-v1" or obj["independent_benchmark"] is not False:
        raise ValueError("Unexpected suite identity")
    if len(rows)!=6 or len({r["id"] for r in rows})!=6 or len({r["task"] for r in rows})!=6:
        raise ValueError("Bad holdback coverage")
    for r in rows:
        if len(r["cases"])!=3:
            raise ValueError("Bad holdback cases")
        oracle(r)
    return rows


def schedule_for(broad, ordered_pairs,seed):
    indices=interleaved_order(broad,seed)[:320]
    shuffled=list(ordered_pairs)
    random.Random(seed+2).shuffle(shuffled)
    schedule=[(tuple(indices[2*i:2*i+2]),p) for i,p in enumerate(shuffled)]
    if (len(schedule)!=160 or len(set(indices))!=320 or
        len({x for _,p in schedule for x in p})!=320):
        raise ValueError("Unequal or duplicate training exposures")
    return schedule


def per_row_nll(model, rows):
    x,y=pack(rows)
    logits=model(x).logits.float()
    losses=F.cross_entropy(logits.transpose(1,2),y,ignore_index=-100,reduction="none")
    return losses.sum(1)/(y!=-100).sum(1)


def load_model(cfg,base_state):
    model=SagaTransformer(cfg)
    model.load_state_dict(base_state,strict=True)
    return model


def fit(cfg,base_state,broad,lookup,aux,schedule,selection,seed,coefficient):
    if coefficient not in (0.0,1.0):
        raise ValueError("Unknown arm")
    torch.manual_seed(seed)
    model=load_model(cfg,base_state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    start_loss=response_loss(model,selection)
    best,best_step,best_state=start_loss,0,copy.deepcopy(model.state_dict())
    history=[{"step":0,"validation_nll":start_loss}]
    margins=[]
    start=time.monotonic()
    for step,(bidx,ids) in enumerate(schedule,1):
        if time.monotonic()-start>300:
            raise TimeoutError("CPU arm budget exhausted")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x,y=pack([broad[i] for i in bidx]+[lookup[k] for k in ids])
        sft=model(x,labels=y).loss
        matched=[aux[k][0] for k in ids if aux[k] is not None]
        if matched:
            nll=per_row_nll(model,[r for pair in matched for r in pair])
            own,wrong=nll[::2],nll[1::2]
            ranking=hinge(own,wrong).mean()
            margins.extend((wrong-own).detach().tolist())
        else:
            ranking=sft*0
        # Both arms calculate/backpropagate the identical auxiliary graph.
        loss=sft+coefficient*ranking
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step in (80,160):
            current=response_loss(model,selection)
            if not math.isfinite(current):
                raise FloatingPointError("Nonfinite validation")
            history.append({"step":step,"validation_nll":current})
            if current<=best-0.005:
                best,best_step,best_state=current,step,copy.deepcopy(model.state_dict())
    return best_state,{"steps":step,"selected_step":best_step,"history":history,
                       "best_validation_nll":best,"mean_online_train_margin":sum(margins)/len(margins),
                       "aux_rows_seen":len(margins),"coefficient":coefficient,
                       "broad_rows_seen":320,"stdio_rows_seen":320}


@torch.inference_mode()
def inference(model, tokenizer, tasks):
    model.eval()
    eos=tokenizer.SPECIAL["<eos>"]
    rows=[]
    for item in tasks:
        prefix=[tokenizer.SPECIAL["<bos>"]]+tokenizer.encode(compact_prompt(item["task"]))
        budget=min(CAP,384-len(prefix))
        if budget<2:
            raise ValueError("Prompt exceeds generation budget")
        ids=model.generate(torch.tensor([prefix],dtype=torch.long),
                           max_new_tokens=budget,temperature=0.0,eos_token_id=eos,
                           vocab_limit=tokenizer.vocab_size,
                           deadline_monotonic=time.monotonic()+25)[0,len(prefix):].tolist()
        if not ids or len(ids)>budget or (eos in ids and ids.index(eos)!=len(ids)-1):
            raise ValueError("Invalid generation stop")
        source=tokenizer.decode(ids)
        rows.append({"id":item["id"],"family":item["family"],
                     "source_sha256":sha(source.encode("utf8")),
                     "tokens":len(ids),"stop":"EOS" if ids[-1]==eos else
                     ("TOKEN_LIMIT" if len(ids)==budget else "UNEXPLAINED_SHORT_STOP"),
                     "compile":"NOT_RUN","behavior":"NOT_RUN"})
    return {"rows":rows,"unique_source_hashes":len({r["source_sha256"] for r in rows}),
            "cross_family_identical_pairs":sum(a["family"]!=b["family"] and a["source_sha256"]==b["source_sha256"]
                                              for i,a in enumerate(rows) for b in rows[i+1:])}


@torch.inference_mode()
def regression_margins(model,regression,eos):
    model.eval()
    vals=[]
    for i,row in enumerate(regression):
        aux=make_aux(row,regression[(i+2)%6],eos)
        if aux is not None:
            v=per_row_nll(model,list(aux[0])).tolist()
            vals.append(v[1]-v[0])
    return {"admitted_rows":len(vals),"mean_wrong_minus_own_nll":sum(vals)/len(vals) if vals else None,
            "positive_rows":sum(v>0 for v in vals)}


def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    # Freeze, verify and independently recompute all development case outputs
    # before training/model loading. This suite never enters gradient/selection.
    fresh=frozen_suite(args.holdback)
    pin(args.base,PINNED_V018,90*1024*1024)
    pin(args.tokenizer,V016_TOKENIZER_SHA256,2_000_000)
    build_curriculum(args.curriculum)
    pin(args.curriculum/"manifest.json",STDIO_SHA,8192)
    tok=load_tokenizer(args.tokenizer)
    eos=tok.SPECIAL["<eos>"]
    def encode_all(rows):
        encoded=[encode_row(r,tok,384) for r in rows]
        if any(r is None for r in encoded):
            raise ValueError("Unexpected training/validation admission")
        return encoded
    broad=encode_all(broad_rows())
    stdio=encode_all(expected_rows("train"))
    valid=encode_all(expected_rows("validation"))
    lookup={r["id"]:r for r in stdio}
    vlookup={r["id"]:r for r in valid}
    regression=[vlookup[k] for k in REGRESSION]
    selection=[r for r in valid if int(r["id"].rsplit("-",1)[1])>=4]
    if len(selection)!=24 or len(regression)!=6:
        raise ValueError("Selection partition mismatch")
    tf={r["family"] for r in broad+stdio}
    vf={r["family"] for r in valid}
    hf={r["family"] for r in fresh}
    all_old={f for spec in broad_data.SPLITS.values() for f in (*spec[0],*spec[2])}
    if tf&vf or hf&(tf|vf|all_old) or len(hf)!=3:
        raise ValueError("Family overlap")
    tids={r["id"] for r in broad+stdio}
    if tids&{r["id"] for r in valid+fresh} or {r["id"] for r in selection}&set(REGRESSION):
        raise ValueError("ID overlap")
    if {r["task"] for r in broad+stdio+valid}&{r["task"] for r in fresh}:
        raise ValueError("Exact task text overlap")
    aux,ordered,admission=pairs(stdio,eos,tokenizer=tok,mask_mode=args.mask_mode)
    if args.mask_mode=='operation-span-v2' and admission['admitted_rows']!=320:
        admission['passed']=False
    if not admission["passed"]:
        # Keep the preregistered gate unchanged. A blocked run publishes
        # bounded counts, not learned weights or an invented model score.
        args.output.mkdir(parents=True)
        report={"status":"BLOCKED_ALIGNMENT_ADMISSION","alignment":admission,
                "base_sha256":PINNED_V018,"tokenizer_sha256":V016_TOKENIZER_SHA256,
                "broad_train_sha256":TRAIN_SHA,"stdio_manifest_sha256":STDIO_SHA,
                "holdback_sha256":SUITE_SHA,"optimizer_steps_completed":0,
                "new_inference":"NOT_RUN","new_java_compile":"NOT_RUN",
                "new_java_behavior":"NOT_RUN","production_promotion":False,
                "independent_benchmark":"NOT_ESTABLISHED"}
        (args.output/"ranking-report.json").write_bytes(canonical(report))
        print(json.dumps(report,sort_keys=True),flush=True)
        return
    schedules={s:schedule_for(broad,ordered,s) for s in SEEDS}
    schedule_hashes={str(s):sha(canonical([[ [broad[i]["id"] for i in b],list(ids)] for b,ids in v]))
                     for s,v in schedules.items()}
    print(json.dumps({"preflight":"PASS","admission":admission,"holdback_sha256":SUITE_SHA,
                      "schedule_sha256_by_seed":schedule_hashes},sort_keys=True),flush=True)
    state=torch.load(args.base,map_location="cpu",weights_only=True)
    verify_checkpoint_tokenizer(state,args.tokenizer)
    cfg=ModelConfig(**state["config"])
    if cfg.max_position_embeddings!=384 or cfg.parameter_count()>3_000_000 or tok.vocab_size>cfg.vocab_size:
        raise ValueError("Unexpected model configuration")
    base_state=state["model"]
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    results={}
    frozen={}
    # Finish every arm/seed and select only by separate loss slice before any
    # heldback inference; no trial changes after observing holdback output.
    for seed in SEEDS:
        results[str(seed)]={}
        for arm,coefficient in zip(ARMS,(0.0,1.0)):
            weights,stats=fit(cfg,base_state,broad,lookup,aux,schedules[seed],selection,seed,coefficient)
            if stats["steps"]!=160 or stats["aux_rows_seen"]!=admission["admitted_rows"]:
                raise ValueError("Actual exposure mismatch")
            p=args.output/f"{seed}-{arm}.pt"
            torch.save({"model":weights,"config":cfg.to_dict(),"base_sha256":PINNED_V018,
                        "tokenizer_sha256":V016_TOKENIZER_SHA256,"seed":seed,"arm":arm,"mask_protocol":args.mask_mode,
                        "selected_step":stats["selected_step"],"production":False},p)
            stats["checkpoint_sha256"]=sha(p.read_bytes())
            stats["schedule_sha256"]=schedule_hashes[str(seed)]
            frozen[(seed,arm)]=weights
            results[str(seed)][arm]=stats
            print(json.dumps({"seed":seed,"arm":arm,"steps":stats["steps"],
                              "selected_step":stats["selected_step"],"loss":stats["best_validation_nll"]}),
                  flush=True)
    def diagnose(weights):
        model=load_model(cfg,weights)
        return {"regression":inference(model,tok,regression),
                "fresh_development":inference(model,tok,fresh),
                "regression_aligned_margin":regression_margins(model,regression,eos)}
    baseline=diagnose(base_state)
    for (seed,arm),weights in frozen.items():
        results[str(seed)][arm]["diagnostics"]=diagnose(weights)
    report={"status":"MATCHED_CPU_TRAINING_AND_INFERENCE_COMPLETE",
            "base_sha256":PINNED_V018,"tokenizer_sha256":V016_TOKENIZER_SHA256,
            "broad_train_sha256":TRAIN_SHA,"stdio_manifest_sha256":STDIO_SHA,
            "holdback_sha256":SUITE_SHA,"alignment":admission,
            "seeds":list(SEEDS),"steps_per_arm":STEPS,"batch_size":4,"learning_rate":LR,
            "margin":MARGIN,"mask_protocol":args.mask_mode,"regression_margin_protocol":"edit-v1 unchanged diagnostic","same_schedule_per_seed":True,"same_aux_graph":True,
            "selection_ids":[r["id"] for r in selection],"regression_ids":list(REGRESSION),
            "fresh_ids":[r["id"] for r in fresh],"baseline":baseline,"results":results,
            "original_test_records_read":False,"public_benchmark_used":False,
            "fresh_used_in_training_or_selection":False,"torch_version":str(torch.__version__),
            "new_java_compile":"NOT_RUN","new_java_behavior":"NOT_RUN",
            "behavior_gate":"BLOCKED_REVIEWED_VM_REQUIRED","independent_benchmark":"NOT_ESTABLISHED",
            "production_promotion":False,"root_cause":"NOT_ESTABLISHED"}
    (args.output/"ranking-report.json").write_bytes(canonical(report))
    print(json.dumps({"status":report["status"],"behavior_gate":report["behavior_gate"],
                      "report_sha256":sha(canonical(report)),
                      "baseline":{k:v["unique_source_hashes"] for k,v in baseline.items() if k!="regression_aligned_margin"},
                      "arms":{s:{a:{"regression_unique":v["diagnostics"]["regression"]["unique_source_hashes"],
                                     "fresh_unique":v["diagnostics"]["fresh_development"]["unique_source_hashes"],
                                     "margin":v["diagnostics"]["regression_aligned_margin"]}
                                 for a,v in arms.items()} for s,arms in results.items()}},
                     sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base","tokenizer","holdback","curriculum","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument('--mask-mode', choices=['edit-v1','operation-span-v2'], default='edit-v1')
    run(p.parse_args())


if __name__=="__main__":
    main()
