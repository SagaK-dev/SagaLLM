"""Audit the frozen parameter CE study and measure prompt signal across prefix depth.

No training, generated Java compilation/execution, or generated source publication.
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
from scripts import train_java_m3_parameter_ce as p
from scripts import train_java_m3_ranking as r

ARCHIVE="068d733b1a1059a1e904ed8931467a85aaf9f5a16968a6837ce2fb4b604bb447"
REPORT="26f143575842073c9b4119ecefcb3b50935891bba6c4c9fc60b0b0c8353a89b7"
SCHEDULES={"4201":"7a72a68174da7de20ea96daedcb9735f2030f93f115e1ac3707b5cf0234439ef",
           "4202":"75d3b7d3ab274dd021f2ac262da2a5ba3491a42473b325ef02252e3c5b373c51",
           "4203":"d8348709f3409c16ea17b0b8205b487852216af7cd15eec2b413d6f1747ac38b"}
DEPTHS=(0,25,50,75,100)
ARMS=("control","parameter_ce")


def screen(results):
    checks={}
    for seed in a.SEEDS:
        c,t=(results[seed][arm]["diagnostics"] for arm in ARMS)
        gain=t["forced_choice"]["fresh"]["both_choices_correct"]>c["forced_choice"]["fresh"]["both_choices_correct"]
        diversity=t["generation"]["fresh"]["unique_source_hashes"]>=c["generation"]["fresh"]["unique_source_hashes"]
        checks[seed]={"both_choice_gain":gain,"fresh_diversity_not_reduced":diversity,"passed":gain and diversity}
    n=sum(x["passed"] for x in checks.values())
    return {"decision":"FORCED_CHOICE_SIGNAL_ONLY" if n>=2 else "NO_CONSISTENT_PARAMETER_SIGNAL",
            "passing_seeds":n,"seed_checks":checks}


def validate_forced(g,n):
    rows=g["rows"]
    a.require(len(rows)==n and g["pairs"]==n and len({tuple(x["ids"]) for x in rows})==n,"Forced-choice coverage")
    for row in rows:
        l=row["left_correct_minus_other"]
        rr=row["right_correct_minus_other"]
        a.require(math.isfinite(l) and math.isfinite(rr),"Nonfinite margin")
        a.require(row["interaction"]==l+rr and row["both_choices_correct"]==(l>0 and rr>0)
                  and row["correct_choices"]==int(l>0)+int(rr>0),"Forced-choice inconsistency")
    expected=d.aggregate(rows)
    a.require(expected==g,"Forced-choice aggregate mismatch")


def validate_diagnostics(diag,ids):
    for k,n in (("train",20),("validation",3),("fresh",3)):
        validate_forced(diag["forced_choice"][k],n)
    for k,v in ids.items():
        a.diagnostic_counts(diag["generation"][k],v)


def audit(path):
    raw=a.read_pin(path,ARCHIVE,64*1024*1024)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        expected={"parameter-ce-report.json"}|{s+"-"+arm+".pt" for s in a.SEEDS for arm in ARMS}
        a.require(len(z.infolist())==7 and set(z.namelist())==expected,"Unexpected archive members")
        a.require(all(0<x.file_size<=16*1024*1024 for x in z.infolist()),"Unsafe archive size")
        report=z.read("parameter-ce-report.json")
        a.require(len(report)<=250000 and a.sha(report)==REPORT,"Pinned report mismatch")
        obj=json.loads(report)
        for key,value in {"status":"PARAMETER_CE_CPU_TRAINING_AND_INFERENCE_COMPLETE",
                          "protocol":p.PROTOCOL,"base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,
                          "pair_manifest_sha256":p.PAIR_SHA,"fresh_suite_sha256":p.SUITE_SHA,
                          "previous_suite_sha256":a.SUITE,"broad_train_sha256":r.TRAIN_SHA,
                          "stdio_manifest_sha256":r.STDIO_SHA,"schedule_sha256_by_seed":SCHEDULES,
                          "selection_ids":a.SELECTION,"seeds":[4201,4202,4203],"steps_per_arm":160,
                          "batch_size":4,"learning_rate":r.LR,"compile":"NOT_RUN","behavior":"NOT_RUN",
                          "independent_benchmark":"NOT_ESTABLISHED"}.items():
            a.require(obj[key]==value,"Report mismatch: "+key)
        for key in ("fresh_used_for_training_or_selection","original_test_records_read","production_promotion"):
            a.require(obj[key] is False,"Unexpected "+key)
        a.require(obj["same_aux_graph"] is True and set(obj["results"])==set(a.SEEDS),"Parity/seed failure")
        ids={"regression":a.REGRESSION,"previous_development":a.FRESH,"fresh":obj["fresh_ids"]}
        validate_diagnostics(obj["baseline"],ids)
        for seed in a.SEEDS:
            arms=obj["results"][seed]
            a.require(set(arms)==set(ARMS),"Arm coverage")
            a.require(arms["control"]["history"][0]==arms["parameter_ce"]["history"][0],"Initial NLL differs")
            for arm,v in arms.items():
                a.require(v["steps"]==160 and v["broad_rows_seen"]==v["stdio_rows_seen"]==320
                          and v["auxiliary_directions"]==320 and v["coefficient"]==float(arm=="parameter_ce")
                          and v["schedule_sha256"]==SCHEDULES[seed],"Exposure mismatch")
                h=v["history"]
                a.require([x["step"] for x in h]==[0,80,160]
                          and all(math.isfinite(x["validation_nll"]) for x in h),"Invalid history")
                best=h[0]
                for row in h[1:]:
                    if row["validation_nll"]<=best["validation_nll"]-0.005:
                        best=row
                a.require(v["selected_step"]==best["step"] and v["best_validation_nll"]==best["validation_nll"],
                          "Selection mismatch")
                a.require(a.sha(z.read(seed+"-"+arm+".pt"))==v["checkpoint_sha256"],"Checkpoint mismatch")
                validate_diagnostics(v["diagnostics"],ids)
    return obj,screen(obj["results"])


def depths(n):
    if n<4:
        raise ValueError("Prefix too short for depth comparison")
    positions=[n*q//100 for q in DEPTHS]
    if len(set(positions))!=5:
        raise ValueError("Nonunique depth positions")
    return positions


def distances(x,y):
    # Full model vocabulary, including special tokens. Centered-logit RMS
    # ignores common additive shifts; JS in nats compares normalized outputs.
    x,y=x.double(),y.double()
    if x.shape!=y.shape or x.ndim!=1 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Invalid logits")
    delta=x-y
    rms=float((delta-delta.mean()).square().mean().sqrt())
    lx,ly=x.log_softmax(0),y.log_softmax(0)
    lm=torch.logaddexp(lx,ly)-math.log(2)
    js=float(((lx.exp()*(lx-lm)).sum()+(ly.exp()*(ly-lm)).sum())/2)
    if js < -1e-12 or not math.isfinite(js):
        raise ValueError("Invalid JS")
    return {"centered_logit_rms":rms,"js_nats":max(0.,js)}


def measure(model,pair):
    prefix=pair["common_response_prefix"]
    # Causal model: forward once for each full common prefix, then gather
    # logits immediately after each chosen depth (including zero).
    outs=[]
    for prompt in pair["prefixes"]:
        seq=torch.tensor([prompt+prefix],dtype=torch.long)
        logits=model(seq).logits[0]
        outs.append([logits[len(prompt)+k-1].float() for k in depths(len(prefix))])
    out=[{"depth_percent":q,**distances(x,y)} for q,x,y in zip(DEPTHS,*outs)]
    l,r=pair["alternatives"]
    x,y=outs[0][-1],outs[1][-1]
    return {"ids":pair["ids"],"family":pair["family"],
            "prompt_lengths":[len(v) for v in pair["prefixes"]],
            "equal_prompt_length":len(pair["prefixes"][0])==len(pair["prefixes"][1]),
            "common_prefix_tokens":len(prefix),"depth_metrics":out,
            "divergence_choice":d.score_logits(float(x[l]),float(x[r]),float(y[l]),float(y[r]))}


def summarize(rows):
    if not rows:
        return {"pairs":0,"status":"NOT_AVAILABLE"}
    means=[]
    for i,q in enumerate(DEPTHS):
        means.append({"depth_percent":q,**{k:sum(x["depth_metrics"][i][k] for x in rows)/len(rows)
                                           for k in ("centered_logit_rms","js_nats")}})
    return {"pairs":len(rows),"mean_by_depth":means,
            "js_lower_at_end_pairs":sum(x["depth_metrics"][-1]["js_nats"]<x["depth_metrics"][0]["js_nats"] for x in rows),
            "rms_lower_at_end_pairs":sum(x["depth_metrics"][-1]["centered_logit_rms"]<x["depth_metrics"][0]["centered_logit_rms"] for x in rows),
            "both_choices_correct":sum(x["divergence_choice"]["both_choices_correct"] for x in rows)}


def run(args):
    from sagallm.config import ModelConfig
    from sagallm.tokenization import load_tokenizer
    from sagallm.training import verify_checkpoint_tokenizer
    from scripts.evaluate_java_m3_v2 import suites
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    torch.set_num_threads(2)
    obj,decision=audit(args.archive)
    a.read_pin(args.tokenizer,a.TOKENIZER,2_000_000)
    tok=load_tokenizer(args.tokenizer)
    old=suites(args.previous_holdback)
    fresh=p.load_suite(args.holdback)
    task_sets={"regression":old["regression"],"previous_development":old["fresh_development"],"fresh":fresh}
    groups,preflight=d.build_pairs(tok)
    a.require(preflight["manifest_sha256"]==p.PAIR_SHA,"TRAIN pair preflight mismatch")
    probes=groups["train"]
    a.require(len(probes)==20,"Probe set mismatch")
    base=torch.load(io.BytesIO(a.read_pin(args.base,a.BASE,90*1024*1024)),map_location="cpu",weights_only=True)
    verify_checkpoint_tokenizer(base,args.tokenizer)
    cfg=ModelConfig(**base["config"])
    a.require(cfg.max_position_embeddings==384 and cfg.parameter_count()<=3_000_000,"Architecture mismatch")
    results={}
    replay_count=0
    archive=a.read_pin(args.archive,ARCHIVE,64*1024*1024)
    with zipfile.ZipFile(io.BytesIO(archive)) as z,torch.inference_mode():
        for name in ["base"]+[s+"-"+arm for s in a.SEEDS for arm in ARMS]:
            if name=="base":
                state=base
                diag=obj["baseline"]
            else:
                seed,arm=name.split("-")
                stats=obj["results"][seed][arm]
                raw=z.read(name+".pt")
                a.require(a.sha(raw)==stats["checkpoint_sha256"],"Checkpoint changed")
                state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
                for k,v in {"protocol":p.PROTOCOL,"base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,
                            "seed":int(seed),"arm":arm,"selected_step":stats["selected_step"],"production":False}.items():
                    a.require(state[k]==v,"Checkpoint metadata mismatch")
                diag=stats["diagnostics"]
            a.require(state["config"]==cfg.to_dict(),"Config changed")
            model=r.load_model(cfg,state["model"])
            model.eval()
            for key,tasks in task_sets.items():
                actual=r.inference(model,tok,tasks)
                a.require(actual==diag["generation"][key],"Frozen generation mismatch")
                replay_count+=len(tasks)
            rows=[measure(model,v) for v in probes]
            # Full-prefix gathering must agree with a separately truncated
            # causal forward; guard against accidental future-token access.
            pair=probes[0]
            prompt=pair["prefixes"][0]
            k=depths(len(pair["common_response_prefix"]))[2]
            full=torch.tensor([prompt+pair["common_response_prefix"]],dtype=torch.long)
            short=full[:,:len(prompt)+k]
            a.require(torch.allclose(model(full).logits[0,len(prompt)+k-1].float(),
                                     model(short).logits[0,-1].float(),atol=1e-4,rtol=1e-4),
                      "Causal prefix consistency failure")
            # Match the earlier first-divergence forced-choice result
            # to tolerance; no post-hoc selection of favorable probes.
            for row,previous in zip(rows,diag["forced_choice"]["train"]["rows"]):
                a.require(row["ids"]==previous["ids"],"TRAIN probe ID mismatch")
                for key in ("left_correct_minus_other","right_correct_minus_other"):
                    a.require(abs(row["divergence_choice"][key]-previous[key])<=1e-4,"Margin replay mismatch")
            results[name]={"all":summarize(rows),
                           "equal_length":summarize([x for x in rows if x["equal_prompt_length"]]),
                           "unequal_length":summarize([x for x in rows if not x["equal_prompt_length"]]),
                           "rows":rows}
            del model
    a.require(replay_count==126,"Incomplete generation replay")
    result={"status":"PARAMETER_CE_AUDIT_AND_PROMPT_SIGNAL_DIAGNOSTIC_COMPLETE",
            "archive_sha256":ARCHIVE,"training_report_sha256":REPORT,"base_sha256":a.BASE,
            "tokenizer_sha256":a.TOKENIZER,"pair_manifest_sha256":p.PAIR_SHA,
            "torch_version":str(torch.__version__),"checkpoint_hashes_verified":6,
            "generation_replay_matches":replay_count,"preregistered_screen":decision,
            "probe_split":"TRAIN_EXPOSED","models":results,"training":"NOT_RUN","compile":"NOT_RUN",
            "behavior":"NOT_RUN","root_cause":"NOT_ESTABLISHED","independent_benchmark":"NOT_ESTABLISHED",
            "production_promotion":False}
    raw=r.canonical(result)
    with args.output.open("xb") as f:
        f.write(raw)
    print(json.dumps({"status":result["status"],"report_sha256":a.sha(raw),
                      "generation_replay_matches":replay_count,"screen":decision,
                      "models":{k:{x:y for x,y in v.items() if x!="rows"} for k,v in results.items()}},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("archive","base","tokenizer","holdback","previous-holdback","output"):
        parser.add_argument("--"+k,type=Path,required=True)
    run(parser.parse_args())


if __name__=="__main__":
    main()
