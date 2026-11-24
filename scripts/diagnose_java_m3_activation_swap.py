"""Matched activation swaps on exposed TRAIN probes; inference only.

Hooks are local to this diagnostic. No checkpoint or production-code mutation.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import zipfile

import torch

from scripts import audit_java_m3_v2 as a
from scripts import diagnose_java_m3_parameters as d
from scripts import diagnose_java_m3_prompt_signal as s
from scripts import train_java_m3_parameter_ce as p
from scripts import train_java_m3_ranking as r

PRIOR_ZIP="46b2df5cdb251e7abb3c11796c9cc0bc92402133474e1d56e9f0c0ee4253075f"
PRIOR_REPORT="535d10f4c9873f945040ddadfbadcd5041024878d59db6fda5203cda9038ab6e"
PROTOCOL="JAVA_M3_ACTIVATION_SWAP_V1"
ATOL=1e-4
REGIONS=("changed_prompt","unchanged_prompt","final_response")


def prior_report(path):
    raw=a.read_pin(path,PRIOR_ZIP,1000000)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        a.require(z.namelist()==["prompt-signal.json"] and z.infolist()[0].file_size<=1000000,
                  "Unexpected prior archive")
        data=z.read("prompt-signal.json")
    a.require(a.sha(data)==PRIOR_REPORT,"Prior report hash mismatch")
    obj=json.loads(data)
    a.require(obj["status"]=="PARAMETER_CE_AUDIT_AND_PROMPT_SIGNAL_DIAGNOSTIC_COMPLETE"
              and obj["generation_replay_matches"]==126
              and obj["archive_sha256"]==s.ARCHIVE and obj["training_report_sha256"]==s.REPORT
              and obj["pair_manifest_sha256"]==p.PAIR_SHA
              and obj["production_promotion"] is False and obj["behavior"]=="NOT_RUN",
              "Prior evidence mismatch")
    return obj


def regions(pair):
    left,right=pair["prefixes"]
    if len(left)!=len(right):
        raise ValueError("Unequal prompt length is not position-aligned")
    common=pair["common_response_prefix"]
    if not common:
        raise ValueError("Missing response prefix")
    different=tuple(i for i,(x,y) in enumerate(zip(left,right)) if x!=y)
    same=tuple(i for i,(x,y) in enumerate(zip(left,right)) if x==y)
    if not different or not same:
        raise ValueError("Missing prompt region")
    return {"changed_prompt":different,"unchanged_prompt":same,
            "final_response":(len(left)+len(common)-1,),
            "all_prompt":tuple(range(len(left))),
            "all_positions":tuple(range(len(left)+len(common)))}


def sites(model):
    out={"embedding":model.embedding}
    for i,b in enumerate(model.blocks,1):
        out[f"block_{i}_residual"]=b
        out[f"block_{i}_attention"]=b.attention
        out[f"block_{i}_ffn"]=b.feed_forward
    return out


def swap_output(output,positions,sham=False):
    # Attention and TransformerBlock return (activation, cache). Cache is
    # None in this uncached inference diagnostic; fail on any other contract.
    tensor=output[0] if isinstance(output,tuple) else output
    if isinstance(output,tuple) and (len(output)!=2 or output[1] is not None):
        raise ValueError("Unexpected cache")
    if (tensor.ndim!=3 or tensor.shape[0]!=2 or not positions
        or len(set(positions))!=len(positions)
        or min(positions)<0 or max(positions)>=tensor.shape[1]):
        raise ValueError("Invalid activation swap")
    result=tensor.clone()
    result[:,list(positions),:]=tensor[:,list(positions),:] if sham else tensor.flip(0)[:,list(positions),:]
    return (result,None) if isinstance(output,tuple) else result


def patched_logits(model,module,x,positions,sham=False):
    calls=0
    def hook(_module,_args,output):
        nonlocal calls
        calls+=1
        return swap_output(output,positions,sham)
    handle=module.register_forward_hook(hook)
    try:
        logits=model(x).logits[:,-1,:].float()
    finally:
        handle.remove()
    a.require(calls==1 and torch.isfinite(logits).all().item(),"Invalid hook invocation")
    return logits


def preference(logits,alternatives):
    left,right=alternatives
    return [float(row[left])-float(row[right]) for row in logits]


def effect(clean,patched,alternatives):
    b=preference(clean,alternatives)
    z=preference(patched,alternatives)
    # Positive means donor A pushes recipient B towards A and donor B
    # pushes recipient A towards B. This is not a normalized recovery ratio.
    transfer=[b[0]-z[0],z[1]-b[1]]
    return {"recipient_a_donor_b_shift":transfer[0],"recipient_b_donor_a_shift":transfer[1],
            "mean_donor_directed_shift":sum(transfer)/2,
            "both_directions_positive":all(v>0 for v in transfer),
            "clean_interaction":b[0]-b[1],"patched_interaction":z[0]-z[1],
            "recipient_a_prefers_donor":z[0]<0,"recipient_b_prefers_donor":z[1]>0}


def close(actual,expected,label):
    a.require(torch.allclose(actual,expected,atol=ATOL,rtol=0),label)


@torch.inference_mode()
def measure(model,pair,previous):
    reg=regions(pair)
    common=pair["common_response_prefix"]
    x=torch.tensor([v+common for v in pair["prefixes"]],dtype=torch.long)
    baseline=model(x).logits[:,-1,:].float()
    a.require(torch.isfinite(baseline).all().item(),"Nonfinite clean logits")
    left,right=preference(baseline,pair["alternatives"])
    a.require(pair["ids"]==previous["ids"] and abs(left-previous["divergence_choice"]["left_correct_minus_other"])<=ATOL
              and abs(-right-previous["divergence_choice"]["right_correct_minus_other"])<=ATOL,
              "Prior clean margin mismatch")
    mods=sites(model)
    all_positions=reg["all_positions"]
    all_results={}
    controls=0
    for name,module in mods.items():
        # Self-patch (identity) controls every site on every pair.
        close(patched_logits(model,module,x,reg["changed_prompt"],sham=True),baseline,"Identity control failed")
        controls+=1
        residual=name=="embedding" or name.endswith("_residual")
        if residual:
            # Swapping the whole sequence at a residual boundary must
            # reproduce the opposite clean sequence, even before layer 1.
            close(patched_logits(model,module,x,all_positions),baseline.flip(0),"Full residual swap failed")
            controls+=1
        effects={}
        for key in REGIONS+(("all_prompt",) if residual else ()):
            logits=patched_logits(model,module,x,reg[key])
            if name=="embedding" and key in ("unchanged_prompt","final_response"):
                close(logits,baseline,"Shared token embedding control failed")
                controls+=1
            if name=="embedding" and key in ("changed_prompt","all_prompt"):
                close(logits,baseline.flip(0),"Changed embedding swap failed")
                controls+=1
            if name==f"block_{len(model.blocks)}_residual" and key=="final_response":
                close(logits,baseline.flip(0),"Last residual prediction position control failed")
                controls+=1
            effects[key]=effect(baseline,logits,pair["alternatives"])
        all_results[name]=effects
    # Ensure no lingering hook affected the model after intervention.
    close(model(x).logits[:,-1,:].float(),baseline,"Post-hook clean mismatch")
    controls+=1
    return {"ids":pair["ids"],"family":pair["family"],
            "prompt_tokens":len(pair["prefixes"][0]),"changed_prompt_tokens":len(reg["changed_prompt"]),
            "common_response_tokens":len(common),"control_checks":controls,
            "clean_interaction":left-right,"sites":all_results}


def aggregate(rows):
    out={}
    for site,by_region in rows[0]["sites"].items():
        out[site]={}
        for region in by_region:
            values=[row["sites"][site][region] for row in rows]
            out[site][region]={"pairs":len(values),
                "mean_donor_directed_shift":sum(v["mean_donor_directed_shift"] for v in values)/len(values),
                "both_directions_positive_pairs":sum(v["both_directions_positive"] for v in values),
                "both_recipients_prefer_donor_pairs":sum(v["recipient_a_prefers_donor"] and v["recipient_b_prefers_donor"] for v in values)}
    return out


def run(args):
    from sagallm.config import ModelConfig
    from sagallm.tokenization import load_tokenizer
    from sagallm.training import verify_checkpoint_tokenizer
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    torch.set_num_threads(2)
    old=prior_report(args.prior)
    report,decision=s.audit(args.archive)
    a.require(decision==old["preregistered_screen"],"Screen mismatch")
    a.read_pin(args.tokenizer,a.TOKENIZER,2000000)
    tok=load_tokenizer(args.tokenizer)
    groups,manifest=d.build_pairs(tok)
    a.require(manifest["manifest_sha256"]==p.PAIR_SHA,"Pair manifest mismatch")
    probes=groups["train"]
    equal=[q for q in probes if len(q["prefixes"][0])==len(q["prefixes"][1])]
    excluded=[q["ids"] for q in probes if len(q["prefixes"][0])!=len(q["prefixes"][1])]
    a.require(len(equal)==19 and len(excluded)==1,"Unexpected alignment coverage")
    base=torch.load(io.BytesIO(a.read_pin(args.base,a.BASE,90*1024*1024)),map_location="cpu",weights_only=True)
    verify_checkpoint_tokenizer(base,args.tokenizer)
    cfg=ModelConfig(**base["config"])
    a.require(cfg.num_layers==4 and cfg.max_position_embeddings==384 and cfg.parameter_count()<=3000000
              and tok.vocab_size<=cfg.vocab_size,"Architecture mismatch")
    results={}
    names=["base"]+[seed+"-"+arm for seed in a.SEEDS for arm in s.ARMS]
    a.require(set(old["models"])==set(names),"Previous model coverage mismatch")
    raw_archive=a.read_pin(args.archive,s.ARCHIVE,64*1024*1024)
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as z,torch.inference_mode():
        for name in names:
            if name=="base":
                state=base
            else:
                seed,arm=name.split("-")
                stats=report["results"][seed][arm]
                raw=z.read(name+".pt")
                a.require(a.sha(raw)==stats["checkpoint_sha256"],"Checkpoint hash mismatch")
                state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
                for k,v in {"protocol":p.PROTOCOL,"base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,
                            "seed":int(seed),"arm":arm,"selected_step":stats["selected_step"],"production":False}.items():
                    a.require(state[k]==v,"Checkpoint metadata mismatch")
            a.require(state["config"]==cfg.to_dict(),"Config mismatch")
            model=r.load_model(cfg,state["model"])
            model.eval()
            old_rows=old["models"][name]["rows"]
            a.require([row["ids"] for row in old_rows]==[q["ids"] for q in probes],"Previous probe IDs changed")
            lookup={tuple(row["ids"]):row for row in old_rows}
            rows=[measure(model,q,lookup[tuple(q["ids"])]) for q in equal]
            results[name]={"pairs":len(rows),"control_checks":sum(row["control_checks"] for row in rows),
                           "mean_clean_interaction":sum(row["clean_interaction"] for row in rows)/len(rows),
                           "summary":aggregate(rows),"rows":rows}
            print(json.dumps({"model":name,"pairs":len(rows),"control_checks":results[name]["control_checks"]},
                             sort_keys=True),flush=True)
            del model
    result={"status":"ACTIVATION_SWAP_DIAGNOSTIC_COMPLETE","protocol":PROTOCOL,
            "archive_sha256":s.ARCHIVE,"training_report_sha256":s.REPORT,
            "prior_diagnostic_zip_sha256":PRIOR_ZIP,"prior_diagnostic_report_sha256":PRIOR_REPORT,
            "tokenizer_sha256":a.TOKENIZER,"base_sha256":a.BASE,"pair_manifest_sha256":p.PAIR_SHA,
            "probe_split":"TRAIN_EXPOSED","equal_length_pairs":19,"excluded_unequal_length":excluded,
            "models":results,"matched_model_pairs":sum(v["pairs"] for v in results.values()),
            "control_checks":sum(v["control_checks"] for v in results.values()),
            "torch_version":str(torch.__version__),"training":"NOT_RUN","compile":"NOT_RUN","behavior":"NOT_RUN",
            "independent_benchmark":"NOT_ESTABLISHED","root_cause":"NOT_ESTABLISHED","production_promotion":False}
    data=r.canonical(result)
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps({"status":result["status"],"report_sha256":a.sha(data),
                      "matched_model_pairs":result["matched_model_pairs"],"control_checks":result["control_checks"],
                      "models":{name:{"mean_clean_interaction":v["mean_clean_interaction"],
                            "changed_prompt_residual":{site:regions["changed_prompt"] for site,regions in v["summary"].items()
                                                       if site=="embedding" or site.endswith("_residual")},
                            "final_response_residual":{site:regions["final_response"] for site,regions in v["summary"].items()
                                                        if site=="embedding" or site.endswith("_residual")}}
                                for name,v in results.items()}},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("archive","prior","base","tokenizer","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
