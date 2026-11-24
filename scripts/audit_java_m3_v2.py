"""Exact M3 V2 artifact audit. No torch, archive extraction, or Java execution."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

ARCHIVE = "2d4d6bf653c394ddf75016b26099efcf2fc5eb09ec1edd1086d7baedbedcc97f"
REPORT = "90c8b9bb12e381c415e798d88875fc401143a0c703a3b0424cfab6c2283932cd"
BASE = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
TOKENIZER = "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"
SUITE = "d641a53e6498a096fda15e92871e10db6cc2380fe1bb1be543c8f4a06aeb035a"
ORACLES = "74ea37e0dcf69175894124a7ff982037960b5d5afb0dc0181eebe71321b400a7"
SEEDS = ("4201","4202","4203")
ARMS = ("control","ranking")
SCHEDULES = {
    "4201":"873efb881be54f317cefcaeba5693e3656ab03491529c51f3575550359638c86",
    "4202":"7fb3e319af7bdcd1d69dcc1f7f18a759cc36a6575cc42c2e3070c614480fffaa",
    "4203":"e8c816fe2b3b6a4d5c5bab444437aa84f06f7b0afc34cfbd1d0b101c70eb758b"}
FAMILIES = ("stdin_min_offset","stdin_count_odd_offset","stdin_ascii_lower_suffix")
REGRESSION = [f"validation-{f}-{i:03d}" for f in FAMILIES for i in (2,3)]
SELECTION = [f"validation-{f}-{i:03d}" for f in FAMILIES for i in range(4,12)]
FRESH = ["dev-smallest-squares-5","dev-smallest-squares-11","dev-adjacent-gap-0",
         "dev-adjacent-gap-2","dev-digit-transition-2","dev-digit-transition-7"]
SPLITS = ("regression","fresh_development")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_pin(path, expected, max_bytes):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=max_bytes:
        raise ValueError("Unsafe or missing input")
    raw=path.read_bytes()
    if sha(raw)!=expected:
        raise ValueError("Pinned SHA mismatch")
    return raw


def hex_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def require(condition, label):
    if not condition:
        raise ValueError(label)


def diagnostic_counts(d, ids):
    rows=d["rows"]
    require([r["id"] for r in rows]==ids and len(rows)==6,"Unexpected evaluation IDs")
    require(all(r["compile"]=="NOT_RUN" and r["behavior"]=="NOT_RUN"
                and r["stop"]=="EOS" and 1<=r["tokens"]<=170
                and hex_sha(r["source_sha256"]) for r in rows),"Unexpected inference evidence")
    require(all(rows[i]["family"]==rows[i+1]["family"] for i in (0,2,4))
            and len({r["family"] for r in rows})==3,"Wrong family coverage")
    hashes=[r["source_sha256"] for r in rows]
    unique=len(set(hashes))
    cross=sum(a["family"]!=b["family"] and a["source_sha256"]==b["source_sha256"]
              for i,a in enumerate(rows) for b in rows[i+1:])
    require(unique==d["unique_source_hashes"] and cross==d["cross_family_identical_pairs"],
            "Inconsistent hash metrics")
    return {"unique":unique,"cross_family_collisions":cross,
            "same_source_within_family_pairs":sum(hashes[i]==hashes[i+1] for i in (0,2,4)),
            "eos":6}


def model_diagnostics(d):
    result={k:diagnostic_counts(d[k],ids) for k,ids in zip(SPLITS,(REGRESSION,FRESH))}
    m=d["regression_aligned_margin"]
    require(m["admitted_rows"]==6 and 0<=m["positive_rows"]<=6
            and math.isfinite(m["mean_wrong_minus_own_nll"]),"Invalid margin")
    result["regression_margin"]=m["mean_wrong_minus_own_nll"]
    return result


def audit_report(r):
    for key,wanted in {"status":"MATCHED_CPU_TRAINING_AND_INFERENCE_COMPLETE",
        "base_sha256":BASE,"tokenizer_sha256":TOKENIZER,"holdback_sha256":SUITE,
        "broad_train_sha256":"181346d5395efdd2ef376730321e1585d453b2b09a822302f6f2d03137e1aa8d",
        "stdio_manifest_sha256":"0cea8ded8a5c03d15b2b516a86665901266b0b375095579daa8d712661316a13",
        "mask_protocol":"operation-span-v2","steps_per_arm":160,"batch_size":4,
        "learning_rate":2e-5,"margin":0.05,"seeds":[4201,4202,4203],
        "regression_ids":REGRESSION,"selection_ids":SELECTION,"fresh_ids":FRESH,
        "new_java_compile":"NOT_RUN","new_java_behavior":"NOT_RUN",
        "behavior_gate":"BLOCKED_REVIEWED_VM_REQUIRED","independent_benchmark":"NOT_ESTABLISHED"}.items():
        require(r[key]==wanted,"Unexpected report field "+key)
    for key in ("same_schedule_per_seed","same_aux_graph"):
        require(r[key] is True,"Missing parity attestation")
    for key in ("original_test_records_read","public_benchmark_used",
                "fresh_used_in_training_or_selection","production_promotion"):
        require(r[key] is False,"Unexpected evaluation/promotion status")
    a=r["alignment"]
    require(a["passed"] is True and a["admitted_rows"]==320 and a["excluded_rows"]==0
            and a["aligned_token_positions"]==3767 and len(a["rows_per_pair_type"])==5
            and set(a["rows_per_pair_type"].values())=={64}
            and a["mask_sha256"]=="de57942b734602b3337db5082cb856583391c12b3a5e543e37b5983f980194a2",
            "Admission mismatch")
    require(set(r["results"])==set(SEEDS),"Unexpected seeds")
    out={"base":model_diagnostics(r["baseline"])}
    for seed in SEEDS:
        arms=r["results"][seed]
        require(set(arms)==set(ARMS),"Unexpected arms")
        require(arms["control"]["history"][0]==arms["ranking"]["history"][0],"Initial loss mismatch")
        for arm in ARMS:
            v=arms[arm]
            require(v["steps"]==160 and v["selected_step"]==160 and v["aux_rows_seen"]==320
                    and v["broad_rows_seen"]==320 and v["stdio_rows_seen"]==320
                    and v["coefficient"]==(1.0 if arm=="ranking" else 0.0)
                    and v["schedule_sha256"]==SCHEDULES[seed] and hex_sha(v["checkpoint_sha256"]),
                    "Unequal or incomplete exposure")
            h=v["history"]
            require([x["step"] for x in h]==[0,80,160]
                    and all(math.isfinite(x["validation_nll"]) for x in h)
                    and h[1]["validation_nll"]<=h[0]["validation_nll"]-0.005
                    and h[2]["validation_nll"]<=h[1]["validation_nll"]-0.005
                    and v["best_validation_nll"]==h[-1]["validation_nll"],"Selection mismatch")
            out[seed+"-"+arm]={**model_diagnostics(v["diagnostics"]),
                               "validation_nll":v["best_validation_nll"],
                               "checkpoint_sha256":v["checkpoint_sha256"]}
    return out


def audit_archive(path):
    raw=read_pin(path,ARCHIVE,64*1024*1024)
    import io
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        expected={"ranking-report.json"}|{f"{s}-{a}.pt" for s in SEEDS for a in ARMS}
        require(len(z.infolist())==7 and set(z.namelist())==expected,"Unexpected archive members")
        require(all(0<x.file_size<=16*1024*1024 for x in z.infolist()),"Archive member limit")
        report_raw=z.read("ranking-report.json")
        require(len(report_raw)<=100000 and sha(report_raw)==REPORT,"Report SHA mismatch")
        report=json.loads(report_raw)
        summaries=audit_report(report)
        for s in SEEDS:
            for a in ARMS:
                require(sha(z.read(f"{s}-{a}.pt"))==report["results"][s][a]["checkpoint_sha256"],
                        "Checkpoint SHA mismatch")
    return report,summaries


def deterministic_ceiling(rows, tasks):
    """Same source+stdin cannot yield different expected outputs if deterministic.
    This is only a conditional upper bound, not a run or an accuracy estimate.
    """
    lookup={t["id"]:t for t in tasks}
    require(len(lookup)==len(tasks)==6 and set(lookup)=={r["id"] for r in rows},
            "Oracle/task mismatch")
    groups={}
    for r in rows:
        cases=lookup[r["id"]]["cases"]
        require(len(cases)==3,"Expected three cases")
        for c in cases:
            key=(r["source_sha256"],c["stdin"])
            groups.setdefault(key,Counter())[c["stdout"]]+=1
    require(sum(sum(v.values()) for v in groups.values())==18,"Case mismatch")
    upper=sum(max(v.values()) for v in groups.values())
    return {"cases":18,"conditional_correct_case_upper_bound":upper,
            "incompatible_expectation_groups":sum(len(v)>1 for v in groups.values()),
            "assumption":"DETERMINISTIC_STDOUT_FOR_IDENTICAL_SOURCE_AND_STDIN",
            "observed_passes":None}
