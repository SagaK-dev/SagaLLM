"""Reconstruct historical identifier and broad-mixture trial input metadata.

Run at one exact historical revision. No model load, optimization, compiler or Java.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

REVISIONS = {
    "797ae0d195045ee2c5177ca44c0df474876bd3fc": ("identifier", 45018649700, 14752812310, "7488f1ab75108ce8befa9b67a5b6bb380ff9802d3ca3c7a35dfc084dacad887d"),
    "49c7c55915f03d94a722951ca9b2f7918ba13d3e": ("identifier", 45018851049, 14753120733, "2d11ded150f1b52e26117d70f0c38f735b572f882c36a75da1a86cc5fb921a5d"),
    "4269121a03d06ebdc07629159e7698336d9c4dee": ("broad", 45026352157, 14756075922, "47fb1577c8932aa1182c672d54fc21c1b4d86b058360ca018ec8aab1aadd665d"),
    "f3a5a7e64fb8cb5e01e56291ff22cc34196115c6": ("broad", 45026590750, 14756001714, "986cf35b4fd91561671f9d80189875f59f49732cf3b9d1f8445c8500ff240add"),
}
ORIGINALS = {
    14750067161: "2631469620412ce526c15a7171430d380caf22db4a6bc22345926fd4eeca20a2",
    14750350753: "8fcd9dd0d1842de6479cc58f3b7ece44462c8ce227361000c48d3bd4b139759d",
    14750148284: "17eb4a91397a376eab70e8ac304db2ffec13a0883b08c720e34ca3ae9b195c53",
}
CANDIDATE = "ad18e0a328f00db2bca204c3911afd66307099e3d41ba0f0d97c964463d3c2d9"
MANIFEST = "3aae302ff32bd3737e2b0b8f06d4db089c0aebab8eba7030b213f54357e3b7cd"
SUMMARY = "715e7dc85fed81ff968f3cbda5d28716d1be90177f233697f5ef23cd9f57c5c4"
RECOVERY = "25a0d7cbacdead46d18c7261287a601bf7a2c114c8ff9a5a6ebf22a0e8ba3040"
TOKENIZER = "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"
DATA = "1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a"
BROAD = "181346d5395efdd2ef376730321e1585d453b2b09a822302f6f2d03137e1aa8d"
VALID = "e29c63a8461f"  # full SHA checked against the pinned corpus manifest
LIMIT = 2_000_000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def pin(path, digest, limit=LIMIT):
    require(path.is_file() and not path.is_symlink(), "Unsafe pinned file")
    with path.open("rb") as f:
        raw = f.read(limit+1)
    require(0 < len(raw) <= limit and sha(raw) == digest, "Pinned file mismatch")
    return raw


def member(z, name, digest, limit=LIMIT):
    matches = [m for m in z.infolist() if m.filename == name]
    require(len(matches)==1 and 0 < matches[0].file_size <= limit, "Missing or unsafe member")
    with z.open(matches[0]) as f:
        raw = f.read(limit+1)
    require(len(raw)==matches[0].file_size and sha(raw)==digest, "Pinned member mismatch")
    return raw


def original_candidates(folder):
    raws = []
    for aid, digest in ORIGINALS.items():
        path = folder / f"{aid}.zip"
        pin(path,digest)
        with zipfile.ZipFile(path) as z:
            require(len(z.infolist())==3 and set(z.namelist())==
                    {"manifest.json","audit-summary.json","train-identifier-variants.jsonl"},
                    "Original candidate archive coverage mismatch")
            raw = member(z,"train-identifier-variants.jsonl",CANDIDATE)
            manifest = member(z,"manifest.json",MANIFEST)
            summary = json.loads(member(z,"audit-summary.json",SUMMARY))
            require(summary["status"]=="CANDIDATE_TRAIN_ONLY_NOT_USED_FOR_TRAINING"
                    and summary["candidate_examples"]==120 and summary["used_for_training"] is False
                    and summary["compiler"]["java_executed"] is False, "Historical candidate summary mismatch")
            raws.append((raw,manifest))
    require(all(x==raws[0] for x in raws) and len(raws[0][0])==102959,
            "Original candidate bytes differ")
    return raws[0]


def reconstruct(args):
    revision = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    require(revision in REVISIONS,"Unpinned historical revision")
    mode,run,artifact,report_sha=REVISIONS[revision]
    pin(args.recovery_zip,RECOVERY)
    with zipfile.ZipFile(args.recovery_zip) as z:
        old=json.loads(member(z,f"reports/{run}-{artifact}-0.json",report_sha))
    raw,manifest=original_candidates(args.originals)
    from sagallm.java_identifier_curriculum import build_identifier_train_candidates, verify_identifier_train_candidates
    from sagallm.java_diverse_corpus import build_java_diverse_corpus, verify_java_diverse_corpus
    from sagallm.java_instruction_sft import read_split, interleaved_order, eval_sample
    from sagallm.tokenization import load_tokenizer
    from scripts.train_java_identifier_ab import paired_rows, baseline_row
    pin(args.tokenizer,TOKENIZER)
    tok=load_tokenizer(args.tokenizer)
    args.root.mkdir()
    candidates=args.root/"identifier"
    build_identifier_train_candidates(candidates)
    verify_identifier_train_candidates(candidates)
    require((candidates/"train-identifier-variants.jsonl").read_bytes()==raw
            and (candidates/"manifest.json").read_bytes()==manifest,"Historical generator byte mismatch")
    rows=[json.loads(line) for line in raw.splitlines()]
    require(len(rows)==len({r["id"] for r in rows})==120
            and len({r["family"] for r in rows})==10
            and all(r["id"].startswith("train-id-") for r in rows),"Candidate identity mismatch")
    canonical_rows=[baseline_row(r) for r in rows]
    require(all(a["id"]==b["id"] and a["family"]==b["family"]
                and a["messages"][0]==b["messages"][0]
                and a["messages"][1]!=b["messages"][1] for a,b in zip(rows,canonical_rows)),
            "Canonical/augmented row drift")
    control=b"".join(canonical(r)+b"\n" for r in canonical_rows)
    with (candidates/"control-paired.jsonl").open("xb") as f:
        f.write(control)
    pairs,counts=paired_rows(candidates/"train-identifier-variants.jsonl",tok,384)
    require(len(pairs)==120 and counts==old["training_pairs" if mode=="identifier" else "paired_training_counts"],
            "Reported pair admission mismatch")
    canonical_dir=args.root/"canonical"
    build_java_diverse_corpus(canonical_dir)
    verify_java_diverse_corpus(canonical_dir)
    cm=json.loads(pin(canonical_dir/"manifest.json",DATA))
    pin(canonical_dir/"train.jsonl",BROAD,90_000_000)
    # Historical verifier checks all split bytes, but only validation/TRAIN are tokenized here.
    valid,valid_counts=read_split(canonical_dir,"validation",tok,384)
    require(valid_counts==old["validation" if mode=="identifier" else "validation_counts"],
            "Validation count mismatch")
    val_ids=[r["id"] for r in eval_sample(valid,24,seed=47)]
    require(len(val_ids)==len(set(val_ids))==24,"Selection rows mismatch")
    if mode=="identifier":
        from scripts.train_java_identifier_ab import SEED, STEPS, BATCH_SIZE
        schedules=[]
        for arm,side in (("canonical",0),("identifier_augmented",1)):
            rr=[p[side] for p in pairs]
            order=interleaved_order(rr,SEED)
            batches=[[rr[order[(step*BATCH_SIZE+j)%len(order)]]["id"] for j in range(BATCH_SIZE)]
                     for step in range(STEPS)]
            freq=Counter(x for batch in batches for x in batch)
            require(len(freq)==old["results"][arm]["examples_seen"]==120 and set(freq.values())=={2},
                    "Identifier schedule exposure count mismatch")
            schedules.append(batches)
        require(schedules[0]==schedules[1],"Paired schedule mismatch")
        schedule=schedules[0]
        counts_trial={"batches":60,"rows_per_trial_arm":240,"distinct_rows_per_trial_arm":120,
                      "repeated_exposures_per_row":2}
    else:
        from scripts.train_java_broad_mix_ab import fixed_training_schedule,schedule_summary
        broad,broad_counts=read_split(canonical_dir,"train",tok,384)
        require(broad_counts==old["original_training_counts"],"Broad count mismatch")
        batches=fixed_training_schedule(broad,pairs)
        counts_trial=schedule_summary(batches,broad,pairs)
        require(counts_trial==old["schedule"],"Broad mixture schedule mismatch")
        schedule=[[[kind,(broad[i] if kind=="broad" else pairs[i][0])["id"]]
                   for kind,i in batch] for batch in batches]
    require(all(r["selected_step"]==0 for r in old["results"].values()),"Selected step drift")
    return {"schema":"sagallm-identifier-trial-input-v1","status":"RECONSTRUCTED_TRIAL_INPUT_SCOPE",
            "mode":mode,"revision":revision,"run_id":run,"artifact_id":artifact,
            "report_sha256":report_sha,"source_archives":{str(k):v for k,v in ORIGINALS.items()},
            "candidate_sha256":CANDIDATE,"candidate_bytes":len(raw),
            "candidate_manifest_sha256":MANIFEST,"control_paired_sha256":sha(control),
            "control_paired_bytes":len(control),"paired_rows":120,"train_families":10,
            "tokenizer_sha256":TOKENIZER,"canonical_manifest_sha256":DATA,
            "derived_schedule_sha256":sha(canonical(schedule)),
            "derived_selection_ids_sha256":sha(canonical(val_ids)),
            "trial_counts":counts_trial,"selected_step":0,
            "historical_schedule_digest_recorded":False,
            "source_bytes_verified_here":True,"selected_checkpoint_exposure_certified":False,
            "actual_exposure_verified":False,"training_lineage_verified":False,
            "model_loaded":False,"new_training":False,"new_inference":False,
            "java_compiled":False,"java_executed":False,"independent_evaluation":"NOT_RUN",
            "model_selection_permitted":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("originals","recovery-zip","tokenizer","root"):
        p.add_argument("--"+name,type=Path,required=True)
    print(json.dumps(reconstruct(p.parse_args()),sort_keys=True,indent=2))


if __name__=="__main__":
    main()
