"""Fixed-checkpoint, first-divergent-token parameter binding diagnostic.

Only canonical synthetic TRAIN and already-exposed validation pairs.
Never trains, compiles, executes Java, or publishes generated sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
import math
from pathlib import Path
import zipfile

from scripts import audit_java_m3_v2 as a
from scripts.evaluate_java_m3_v2 import suites, replay

PROTOCOL = "JAVA_M3_PARAMETER_BINDING_V1"
TRAIN_PROBE_VARIANTS = ((0, 1), (14, 15))
FAMILY_NAMES = (
    "stdin_add_offset", "stdin_multiply_offset", "stdin_max_offset",
    "stdin_absolute_gap_offset", "stdin_count_greater", "stdin_even_sum_offset",
    "stdin_digit_sum_offset", "stdin_ascii_upper_suffix",
    "stdin_ascii_reverse_suffix", "stdin_clamp",
)


def first_difference(left, right):
    """Return the first unequal token at an identical response prefix."""
    if not left or not right or len(left)>384 or len(right)>384:
        raise ValueError("Invalid token sequence")
    for i,(x,y) in enumerate(zip(left,right)):
        if x!=y:
            return i,x,y
    raise ValueError("No divergent nonterminal token")


def verify_pair(left, right, tokenizer):
    """Build exact two-way forced choice at first response divergence."""
    if left["family"]!=right["family"] or left["id"]==right["id"]:
        raise ValueError("Not a same-family parameter pair")
    if left["task"]==right["task"] or left["prefix_ids"]==right["prefix_ids"]:
        raise ValueError("Prompt collision")
    refs=[tokenizer.encode(row["source"]) for row in (left,right)]
    special=set(tokenizer.SPECIAL.values())
    if any(set(ids)&special or tokenizer.decode(ids)!=row["source"]
           for ids,row in zip(refs,(left,right))):
        raise ValueError("Reference token round-trip or special token error")
    pos,l,r=first_difference(*refs)
    prefix=refs[0][:pos]
    if any(len(row["prefix_ids"])+len(prefix)>384 for row in (left,right)):
        raise ValueError("Forced-choice context overflow")
    # The canonical source templates vary only the requested parameter;
    # the first unequal token may also contain neighboring punctuation.
    return {"family":left["family"],"ids":[left["id"],right["id"]],
            "prefixes":[left["prefix_ids"],right["prefix_ids"]],
            "common_response_prefix":prefix,"alternatives":[l,r],
            "position":pos}


def score_logits(ll, lr, rl, rr):
    """Signed two-choice margins. First index prompt; second reference."""
    if not all(math.isfinite(x) for x in (ll,lr,rl,rr)):
        raise ValueError("Nonfinite logits")
    left=ll-lr
    right=rr-rl
    return {"left_correct_minus_other":left,"right_correct_minus_other":right,
            "interaction":left+right,"both_choices_correct":left>0 and right>0,
            "correct_choices":int(left>0)+int(right>0)}


def build_pairs(tok):
    from sagallm.java_stdio_curriculum import expected_rows, TRAIN_FAMILIES
    from sagallm.java_instruction_sft import encode_row
    a.require(set(TRAIN_FAMILIES)==set(FAMILY_NAMES),"Training family mismatch")
    out={}
    train_manifest=[]
    for split,families,variants in (
        ("train",FAMILY_NAMES,tuple((i,i+1) for i in range(0,32,2))),
        ("validation",a.FAMILIES,((2,3),))):
        raw=expected_rows(split)
        lookup={r["id"]:r for r in raw}
        pairs=[]
        for f in families:
            for i,j in variants:
                rows=[encode_row(lookup[f"{split}-{f}-{n:03d}"],tok,384) for n in (i,j)]
                a.require(all(row is not None for row in rows),"Reference excluded")
                p=verify_pair(*rows,tok)
                if split=="train":
                    train_manifest.append(p)
                    if (i,j) in TRAIN_PROBE_VARIANTS:
                        pairs.append(p)
                else:
                    pairs.append(p)
        out[split]=pairs
    a.require(len(train_manifest)==160 and len(out["train"])==20 and len(out["validation"])==3,
              "Unexpected diagnostic or training-pair coverage")
    ids=[i for p in train_manifest for i in p["ids"]]
    a.require(len(set(ids))==len(ids)==320,"Duplicate TRAIN exposure")
    # Hash exact sequence, target IDs and prefix structure; do not publish tokens.
    digest=a.sha(json.dumps(train_manifest,sort_keys=True,separators=(",",":")).encode())
    return out,{"pairs":160,"directional_rows":320,"families":10,
                "manifest_sha256":digest,"optimizer_steps":0}


def measure(model,pair):
    import torch
    values=[]
    for prompt in pair["prefixes"]:
        x=torch.tensor([prompt+pair["common_response_prefix"]],dtype=torch.long)
        logits=model(x).logits[0,-1,:].float()
        values.extend(float(logits[i]) for i in pair["alternatives"])
    return {**score_logits(*values),"ids":pair["ids"],"family":pair["family"],
            "common_reference_tokens":pair["position"],
            "prompt_token_counts":[len(p) for p in pair["prefixes"]]}


def aggregate(rows):
    if not rows:
        raise ValueError("Empty metric group")
    return {"pairs":len(rows),"both_choices_correct":sum(r["both_choices_correct"] for r in rows),
            "correct_choices":sum(r["correct_choices"] for r in rows),
            "positive_interactions":sum(r["interaction"]>0 for r in rows),
            "mean_interaction":sum(r["interaction"] for r in rows)/len(rows),
            "rows":rows}


def run(args):
    import torch
    from sagallm.config import ModelConfig
    from sagallm.model import SagaTransformer
    from sagallm.tokenization import load_tokenizer
    from sagallm.training import verify_checkpoint_tokenizer
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    torch.set_num_threads(2)
    report,summary=a.audit_archive(args.archive)
    task_sets=suites(args.holdback)
    # Reproduce every previously recorded generation before new scoring.
    sources,version=replay(args,report,task_sets)
    a.require(sum(len(v) for v in sources.values())==84,"Incomplete replay")
    del sources
    a.read_pin(args.tokenizer,a.TOKENIZER,2_000_000)
    tok=load_tokenizer(args.tokenizer)
    groups,preflight=build_pairs(tok)
    base=torch.load(io.BytesIO(a.read_pin(args.base,a.BASE,90*1024*1024)),
                    map_location="cpu",weights_only=True)
    verify_checkpoint_tokenizer(base,args.tokenizer)
    cfg=ModelConfig(**base["config"])
    a.require(cfg.max_position_embeddings==384 and cfg.parameter_count()<=3_000_000,
              "Unexpected model dimensions")
    results={}
    archive=a.read_pin(args.archive,a.ARCHIVE,64*1024*1024)
    with zipfile.ZipFile(io.BytesIO(archive)) as z,torch.inference_mode():
        for name in ["base"]+[s+"-"+arm for s in a.SEEDS for arm in a.ARMS]:
            if name=="base":
                state=base
            else:
                seed,arm=name.split("-")
                raw=z.read(name+".pt")
                a.require(a.sha(raw)==report["results"][seed][arm]["checkpoint_sha256"],
                          "Checkpoint changed")
                state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
                a.require(state["production"] is False and state["seed"]==int(seed)
                          and state["arm"]==arm and state["config"]==cfg.to_dict(),
                          "Checkpoint identity mismatch")
            model=SagaTransformer(cfg)
            model.load_state_dict(state["model"],strict=True)
            model.eval()
            results[name]={k:aggregate([measure(model,p) for p in ps])
                           for k,ps in groups.items()}
            results[name]["prior_generation_collisions"]={
                k:summary[name][k]["same_source_within_family_pairs"] for k in a.SPLITS}
            del model
    result={"status":"PARAMETER_BINDING_DIAGNOSTIC_COMPLETE","protocol":PROTOCOL,
            "archive_sha256":a.ARCHIVE,"prior_report_sha256":a.REPORT,
            "base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,
            "source_hash_replay_matches":84,"torch_version":version,
            "train_pair_preflight":preflight,"models":results,
            "training":"NOT_RUN","compile":"NOT_RUN","behavior":"NOT_RUN",
            "independent_benchmark":"NOT_ESTABLISHED","production_promotion":False,
            "root_cause":"NOT_ESTABLISHED"}
    data=(json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps({"status":result["status"],"report_sha256":a.sha(data),
                      "train_pair_preflight":preflight,
                      "models":{n:{k:{x:y for x,y in v.items() if x!="rows"}
                                    for k,v in d.items()} for n,d in results.items()}},
                     sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("archive","base","tokenizer","holdback","output"):
        p.add_argument("--"+n,type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
