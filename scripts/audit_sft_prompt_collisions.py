"""Review exact prompt collisions and plan a bounded TRAIN-only template subset.

No new train corpus is emitted; candidate membership hashes are metadata.
The historical canonical dataset and validation/test bytes are not modified.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

from sagallm.java_diverse_corpus import build_java_diverse_corpus,verify_java_diverse_corpus
from sagallm.java_instruction_sft import source_task,sha_file
from scripts.audit_canonical_sft_template_quality import LEX,signature
from scripts.run_pinned_traced_context_pilot import DATA

CAP_PER_LANGUAGE=8

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def literal_preserving_signature(source):
    return sha(canonical([m.group() for m in LEX.finditer(source) if m.lastgroup not in ("space","comment")]))
def language_variant(row):
    # Canonical generator switches language according to variant parity.
    return "ja" if int(row["id"].rsplit("-",1)[1])%2==0 else "en"

def records(root,split):
    for line in (root/f"{split}.jsonl").read_bytes().splitlines():
        row=json.loads(line);task,source=source_task(row)
        yield {"split":split,"id":row["id"],"family":row["family"],"language":language_variant(row),
               "task_sha256":sha(task.encode()),"response_sha256":sha(source.encode()),
               "literal_preserving_signature":literal_preserving_signature(source),
               "template_signature":signature(source),"row_sha256":sha(canonical(row))}

def collision_groups(rows):
    by_task=defaultdict(list)
    for r in rows:by_task[r["task_sha256"]].append(r)
    out=[]
    for h,items in sorted(by_task.items()):
        if len(items)<2:continue
        require(len({r["response_sha256"] for r in items})==len(items),"Unexpected identical response")
        # Same literal-preserving lexical sequence is evidence for formatting/comment
        # variation, not a Java semantic proof; keep all unresolved variants isolated.
        category=("LEXICAL_COMMENT_OR_SPACE_VARIANTS" if len({r["literal_preserving_signature"] for r in items})==1
                  else "DISTINCT_LEXICAL_TARGETS_REVIEW_REQUIRED")
        out.append({"task_sha256":h,"category":category,"rows":items,
                    "split_counts":dict(Counter(r["split"] for r in items)),
                    "family_counts":dict(Counter(r["family"] for r in items))})
    return out

def subset_plan(train,groups):
    excluded=set();dedup_removed=0;quarantine_removed=0
    for g in groups:
        items=[r for r in g["rows"] if r["split"]=="train"]
        if not items:continue
        # Do not pick one of distinct lexical targets without a reviewer.
        keep=(min(items,key=lambda r:r["id"])["id"] if g["category"]=="LEXICAL_COMMENT_OR_SPACE_VARIANTS"
              and len({r["split"] for r in g["rows"]})==1 else None)
        for r in items:
            if r["id"]==keep:continue
            excluded.add(r["id"])
            if keep is None:quarantine_removed+=1
            else:dedup_removed+=1
    buckets=defaultdict(list)
    for r in train:
        if r["id"] not in excluded:
            buckets[r["family"],r["template_signature"],r["language"]].append(r)
    chosen=[]
    for key,items in sorted(buckets.items()):
        # Deterministic hash ordering avoids selecting contiguous numeric variants.
        chosen.extend(sorted(items,key=lambda r:(sha(("sagallm-template-cap-v1:"+r["id"]).encode()),r["id"]))[:CAP_PER_LANGUAGE])
    require(len({r["id"] for r in chosen})==len(chosen)
            and len({r["task_sha256"] for r in chosen})==len(chosen),"Subset prompt collision")
    return {"status":"METADATA_ONLY_CANDIDATE_PLAN","candidate_count":len(chosen),
            "train_count_before":len(train),"dedup_comment_variant_rows_removed":dedup_removed,
            "unresolved_prompt_rows_quarantined":quarantine_removed,
            "template_cap_rows_omitted":len(train)-len(excluded)-len(chosen),
            "family_count":len({r["family"] for r in chosen}),
            "template_signature_count":len({r["template_signature"] for r in chosen}),
            "language_counts":dict(Counter(r["language"] for r in chosen)),
            "cap_per_family_template_language":CAP_PER_LANGUAGE,
            "membership_sha256":sha(canonical(sorted(r["row_sha256"] for r in chosen))),
            "candidate_rows":[{"id_sha256":sha(r["id"].encode()),"row_sha256":r["row_sha256"],
                               "family":r["family"],"language":r["language"],"template_signature":r["template_signature"]} for r in chosen],
            "candidate_train_file_written":False,"training_performed":False,"selection_permitted":False}

def audit(root):
    verify_java_diverse_corpus(root);require(sha_file(root/"manifest.json")==DATA,"Manifest pin mismatch")
    rows=[r for split in ("train","validation","test") for r in records(root,split)]
    require(len(rows)==17536,"Canonical coverage mismatch")
    groups=collision_groups(rows)
    require(len(groups)==28 and all(len(g["split_counts"])==1 for g in groups),"Prior collision audit mismatch")
    plan=subset_plan([r for r in rows if r["split"]=="train"],groups)
    return {"schema":"sagallm-sft-prompt-collision-review-v1","status":"PINNED_PROMPT_COLLISION_REVIEW_COMPLETE",
            "manifest_sha256":DATA,"rows_reviewed":len(rows),"collision_groups":len(groups),
            "groups_by_category":dict(Counter(g["category"] for g in groups)),
            "groups_by_split":dict(Counter(next(iter(g["split_counts"])) for g in groups)),
            "groups_by_family":dict(Counter(f for g in groups for f in g["family_counts"])),
            "collision_rows":sum(len(g["rows"]) for g in groups),"groups":groups,
            "candidate_plan":plan,"lexical_equivalence_is_heuristic":True,"semantic_correctness_verified":False,
            "test_static_source_read":True,"test_tokenization_or_scoring":False,"test_used_for_training_selection":False,
            "training_performed":False,"independent_evaluation":"NOT_RUN","rights_review":"REVIEW_REQUIRED",
            "model_selection_permitted":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),"Refuse overwrite")
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/"corpus";build_java_diverse_corpus(root);d=audit(root)
        a.output.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n")
        print(json.dumps({**{k:d[k] for k in ("status","collision_groups","groups_by_category","groups_by_split","groups_by_family","collision_rows")},
            "candidate_plan":{k:v for k,v in d["candidate_plan"].items() if k!="candidate_rows"}},sort_keys=True))
