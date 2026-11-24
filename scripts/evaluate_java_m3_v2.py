"""Replay frozen M3 research checkpoints; default NEVER compiles or runs Java.

The optional evaluation stage is for a human-reviewed disposable Linux VM,
not hosted CI. Review metadata is an attestation, not sandbox certification.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import platform
import re
import time
import zipfile

from scripts import audit_java_m3_v2 as a


def suites(path):
    from sagallm.java_stdio_curriculum import validation_oracle_rows, expected_rows, _canonical_lines
    from sagallm.java_instruction_sft import source_task
    from scripts.train_java_m3_ranking import frozen_suite
    rows=validation_oracle_rows()
    a.require(a.sha(_canonical_lines(rows))==a.ORACLES,"Regression oracle SHA mismatch")
    oracle={r["id"]:r for r in rows}
    tasks={r["id"]:source_task(r)[0] for r in expected_rows("validation")}
    return {"regression":[{**oracle[k],"task":tasks[k]} for k in a.REGRESSION],
            "fresh_development":frozen_suite(path)}


def gate(args):
    if not args.allow_execution:
        if args.acknowledge_disposable_vm or args.review or args.review_sha256 or args.image:
            raise PermissionError("Incomplete execution opt-in")
        return None
    if (not args.acknowledge_disposable_vm or not args.review or not args.review_sha256
        or not args.image or not re.fullmatch(r"sha256:[0-9a-f]{64}",args.image)
        or not a.hex_sha(args.review_sha256)):
        raise PermissionError("Explicit VM approval, review hash and pinned OCI image required")
    if platform.system()!="Linux" or any(os.environ.get(k) for k in
            ("CI","GITHUB_ACTIONS","GITLAB_CI","JENKINS_URL","TF_BUILD")):
        raise PermissionError("Never execute model-generated Java in shared CI/non-Linux")
    record=json.loads(a.read_pin(args.review,args.review_sha256,16384))
    for k,v in {"report_sha256":a.REPORT,"archive_sha256":a.ARCHIVE,
                "regression_oracles_sha256":a.ORACLES,"fresh_suite_sha256":a.SUITE,
                "image_id":args.image,"disposable_linux_vm":True,
                "no_secrets_or_persistent_mounts":True,"restricted_oci_reviewed":True,
                "approved_synthetic_java_execution":True}.items():
        if record.get(k)!=v:
            raise PermissionError("Review scope mismatch: "+k)
    for k in ("reviewer","review_date","review_url","vm_id"):
        v=record.get(k)
        if not isinstance(v,str) or not v.strip() or len(v)>500 or v in ("PENDING","UNKNOWN"):
            raise PermissionError("Missing reviewer attestation")
    if not re.fullmatch(r"https://[^\s]+",record["review_url"]):
        raise PermissionError("Traceable review URL required")
    return args.review_sha256


def replay(args, report, suites):
    import torch
    from sagallm.config import ModelConfig
    from sagallm.model import SagaTransformer
    from sagallm.tokenization import load_tokenizer
    from sagallm.training import verify_checkpoint_tokenizer
    from sagallm.java_instruction_sft import compact_prompt
    torch.set_num_threads(2)
    a.read_pin(args.tokenizer,a.TOKENIZER,2_000_000)
    base=a.read_pin(args.base,a.BASE,90*1024*1024)
    state=torch.load(io.BytesIO(base),map_location="cpu",weights_only=True)
    verify_checkpoint_tokenizer(state,args.tokenizer)
    cfg=ModelConfig(**state["config"])
    a.require(cfg.max_position_embeddings==384 and cfg.parameter_count()<=3_000_000,
              "Unexpected architecture")
    tok=load_tokenizer(args.tokenizer)
    a.require(tok.vocab_size<=cfg.vocab_size,"Vocabulary mismatch")
    eos=tok.SPECIAL["<eos>"]
    sources={}
    archive=a.read_pin(args.archive,a.ARCHIVE,64*1024*1024)
    with zipfile.ZipFile(io.BytesIO(archive)) as z, torch.inference_mode():
        for name in ["base"]+[f"{s}-{arm}" for s in a.SEEDS for arm in a.ARMS]:
            if name=="base":
                weights=state
                diag=report["baseline"]
            else:
                seed,arm=name.split("-")
                stats=report["results"][seed][arm]
                data=z.read(name+".pt")
                a.require(a.sha(data)==stats["checkpoint_sha256"],"Checkpoint SHA mismatch")
                weights=torch.load(io.BytesIO(data),map_location="cpu",weights_only=True)
                for key,val in {"base_sha256":a.BASE,"tokenizer_sha256":a.TOKENIZER,
                                "seed":int(seed),"arm":arm,"mask_protocol":"operation-span-v2",
                                "selected_step":160,"production":False}.items():
                    a.require(weights.get(key)==val,"Checkpoint metadata mismatch: "+key)
                diag=stats["diagnostics"]
            a.require(weights["config"]==cfg.to_dict(),"Checkpoint config mismatch")
            model=SagaTransformer(cfg)
            model.load_state_dict(weights["model"],strict=True)
            model.eval()
            sources[name]={}
            for split,tasks in suites.items():
                for task,row in zip(tasks,diag[split]["rows"]):
                    a.require(task["id"]==row["id"] and task["family"]==row["family"],"Prompt identity mismatch")
                    p=[tok.SPECIAL["<bos>"]]+tok.encode(compact_prompt(task["task"]))
                    budget=min(170,384-len(p))
                    a.require(budget>=2,"Prompt exceeds context")
                    ids=model.generate(torch.tensor([p],dtype=torch.long),max_new_tokens=budget,
                                       temperature=0.0,eos_token_id=eos,vocab_limit=tok.vocab_size,
                                       deadline_monotonic=time.monotonic()+25)[0,len(p):].tolist()
                    source=tok.decode(ids)
                    a.require(len(ids)==row["tokens"] and ids[-1]==eos and eos not in ids[:-1]
                              and a.sha(source.encode("utf8"))==row["source_sha256"],"Frozen generation mismatch")
                    sources[name][task["id"]]=source
            del model
    return sources,str(torch.__version__)


def grade(source,task,image,runner):
    if not source or len(source.encode("utf8"))>4000 or "\x00" in source or "public class Main" not in source:
        return {"id":task["id"],"compile":"REJECTED","cases":["REJECTED"]*3}
    c=runner(source,"",image=image,allow_execution=True,compile_only=True,timeout=20)["status"]
    a.require(c in ("PASS","FAIL","REJECTED","NOT_RUN","RUNTIME_ERROR","TIMEOUT","OUTPUT_LIMIT"),
              "Unknown compile outcome")
    if c!="PASS":
        return {"id":task["id"],"compile":c,
                "cases":["COMPILE_FAILED" if c=="FAIL" else "UNAVAILABLE"]*3}
    cases=[]
    for case in task["cases"]:
        r=runner(source,case["stdout"],image=image,allow_execution=True,
                 stdin_data=case["stdin"],timeout=20)
        status=r["status"]
        a.require(status in ("PASS","FAIL","REJECTED","NOT_RUN","RUNTIME_ERROR","TIMEOUT","OUTPUT_LIMIT"),
                  "Unknown behavior outcome")
        if status=="FAIL":
            status="OUTPUT_MISMATCH_ZERO_EXIT" if r.get("returncode")==0 else "NONZERO_EXIT_OR_RECOMPILE_FAILURE"
        cases.append(status)
    return {"id":task["id"],"compile":c,"cases":cases}


def counts(rows):
    a.require(len(rows)==6 and all(len(r["cases"])==3 for r in rows),"Incomplete evaluation")
    return {"cases_passed":sum(r["cases"].count("PASS") for r in rows),
            "tasks_all_pass":sum(r["cases"]==["PASS"]*3 for r in rows),
            "compiled":sum(r["compile"]=="PASS" for r in rows),
            "unavailable":sum(c in ("UNAVAILABLE","NOT_RUN","RUNTIME_ERROR") for r in rows for c in r["cases"]),
            "rows":rows}


def research_gate(results, summaries):
    if any(d["unavailable"] for model in results.values() for d in model.values()):
        return {"decision":"INCONCLUSIVE_UNAVAILABLE","production_promotion":False}
    per={}
    base=results["base"]["regression"]
    for s in a.SEEDS:
        c,t=results[s+"-control"],results[s+"-ranking"]
        rg=t["regression"]["cases_passed"]>=max(c["regression"]["cases_passed"],base["cases_passed"])+2
        full=t["regression"]["tasks_all_pass"]>=max(c["regression"]["tasks_all_pass"],base["tasks_all_pass"])
        diversity=summaries[s+"-ranking"]["regression"]["unique"]>summaries[s+"-control"]["regression"]["unique"]
        fresh=t["fresh_development"]["cases_passed"]>c["fresh_development"]["cases_passed"]
        per[s]={"regression_case_gain":rg,"no_full_task_regression":full,
                "regression_diversity_gain_vs_control":diversity,"fresh_case_gain":fresh}
    # Conservative explicit operationalization: both suites' criteria must
    # hold in at least two paired seeds. No promotion even if this holds.
    passed=sum(all(d.values()) for d in per.values())>=2
    return {"decision":"SYNTHETIC_RESEARCH_SIGNAL_NOT_DEPLOYED" if passed else "NO_VALIDATED_RESEARCH_GAIN",
            "paired_seed_checks":per,"production_promotion":False}


def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    review=gate(args)
    report,summary=a.audit_archive(args.archive)
    cases=suites(args.holdback)
    diagnostics={"base":report["baseline"],**{s+"-"+arm:report["results"][s][arm]["diagnostics"]
                 for s in a.SEEDS for arm in a.ARMS}}
    bounds={m:{k:a.deterministic_ceiling(d[k]["rows"],cases[k]) for k in a.SPLITS}
            for m,d in diagnostics.items()}
    sources,version=replay(args,report,cases)
    result={"status":"FROZEN_REPLAY_COMPLETE","run_id":45373241422,"artifact_id":14909169047,
            "archive_sha256":a.ARCHIVE,"report_sha256":a.REPORT,"tokenizer_sha256":a.TOKENIZER,
            "base_sha256":a.BASE,"regression_oracles_sha256":a.ORACLES,"fresh_suite_sha256":a.SUITE,
            "checkpoint_hashes_verified":6,"source_hash_matches":sum(len(v) for v in sources.values()),
            "torch_version":version,"summaries":summary,"deterministic_bounds":bounds,
            "compile":"NOT_RUN","behavior":"NOT_RUN","independent_benchmark":"NOT_ESTABLISHED",
            "production_promotion":False,"review_authenticity":"NOT_VERIFIED_BY_SOFTWARE"}
    if review:
        from sagallm.java_behavior import oci_image_ready,run_java_in_oci
        if not oci_image_ready("docker",args.image):
            raise RuntimeError("Reviewed local OCI image not ready")
        smoke=run_java_in_oci("public class Main {public static void main(String[] a){System.out.println(7);}}",
                              "7\n",image=args.image,allow_execution=True,timeout=20)
        if smoke["status"]!="PASS":
            raise RuntimeError("Trusted OCI smoke failed")
        observed={m:{k:counts([grade(src[t["id"]],t,args.image,run_java_in_oci)
                              for t in tasks]) for k,tasks in cases.items()} for m,src in sources.items()}
        result.update({"status":"SYNTHETIC_BEHAVIOR_STUDY_COMPLETE","compile":"ATTEMPTED",
                       "behavior":"ATTEMPTED","results":observed,"review_sha256":review,
                       "image_id":args.image,"gate":research_gate(observed,summary)})
    with args.output.open("x",encoding="utf8") as f:
        json.dump(result,f,sort_keys=True,indent=2)
        f.write("\n")
    print(json.dumps({"status":result["status"],"source_hash_matches":result["source_hash_matches"],
                      "deterministic_bounds":bounds,"compile":result["compile"],"behavior":result["behavior"]},
                     sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("archive","base","tokenizer","holdback","output"):
        p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--allow-execution",action="store_true")
    p.add_argument("--acknowledge-disposable-vm",action="store_true")
    p.add_argument("--review",type=Path)
    p.add_argument("--review-sha256")
    p.add_argument("--image")
    run(p.parse_args())


if __name__=="__main__":
    main()
