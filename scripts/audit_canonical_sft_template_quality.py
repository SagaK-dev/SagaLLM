"""Audit canonical synthetic SFT template concentration and context admission.

Static structure only, not semantic equivalence or independent evaluation.
No TEST tokenization, model loading, Java execution, raw task/code export.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

from sagallm.java_diverse_corpus import build_java_diverse_corpus,verify_java_diverse_corpus
from sagallm.java_instruction_sft import source_task, encode_row, eval_sample, sha_file
from sagallm.tokenization import load_tokenizer
from scripts.run_pinned_traced_context_pilot import materialize_parent, DATA

# Generated Java only; a conservative lexer-like signature, not a Java parser.
LEX=re.compile(r'(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(?P<number>\b(?:0[xX][0-9A-Fa-f_]+|0[bB][01_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)[fFdDlL]?\b)|(?P<word>[A-Za-z_$][\w$]*)|(?P<space>\s+)|(?P<symbol>.)',re.S)

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def signature(source):
    parts=[]
    for m in LEX.finditer(source):
        k=m.lastgroup
        if k in ("space","comment"):continue
        parts.append("<LITERAL>" if k=="string" else "<NUMBER>" if k=="number" else m.group())
    return sha(json.dumps(parts,ensure_ascii=False,separators=(",",":")).encode())

def quantiles(values):
    if not values:return None
    a=sorted(values)
    return {"min":a[0],"median":a[(len(a)-1)//2],"p95":a[math.ceil(.95*len(a))-1],"max":a[-1]}

def audit(root,tokenizer_path):
    verified=verify_java_diverse_corpus(root)
    require(sha_file(root/"manifest.json")==DATA,"Manifest pin mismatch")
    tok=load_tokenizer(tokenizer_path)
    families=defaultdict(list);shape_index=defaultdict(set);shape_counts=Counter()
    source_hashes=set();task_hashes=defaultdict(set);task_targets=defaultdict(set);summary={}
    for split in ("train","validation","test"):
        enc=[];count=0;admission=Counter();positions=[];responses=[];shapes=Counter()
        for line in (root/f"{split}.jsonl").read_bytes().splitlines():
            row=json.loads(line);task,source=source_task(row)
            family=row["family"];key=signature(source);source_sha=sha(source.encode());task_sha=sha(task.encode())
            require(source_sha not in source_hashes,"Duplicate exact response")
            source_hashes.add(source_sha);task_hashes[task_sha].add(split);task_targets[task_sha].add(source_sha)
            shape_index[key].add((split,family));shape_counts[key]+=1;shapes[key]+=1;count+=1
            entry={"shape":key,"source_sha256":source_sha}
            if split!="test":
                # A bounded 4096 context lets the diagnostic measure full length, not truncate.
                full=encode_row(row,tok,4096)
                require(full is not None,"Unexpected oversized source")
                n=full["sequence_positions"];positions.append(n);responses.append(full["response_tokens"])
                for ctx in (192,384,512):
                    admission[ctx]+=int(n<=ctx)
                entry.update(positions=n,response_tokens=full["response_tokens"])
                if n<=384:enc.append(full)
            families[split,family].append(entry)
        summary[split]={"rows":count,"distinct_source_signatures":len(shapes),
                        "most_common_signature_rows":max(shapes.values()),
                        "tokenization_status":"NOT_RUN" if split=="test" else "MEASURED",
                        "positions":quantiles(positions),"response_tokens":quantiles(responses),
                        "accepted_by_context":dict(admission) if split!="test" else None}
        if split=="validation":
            selected=eval_sample(enc,30)
            summary[split]["existing_eval_sample"]={"rows":len(selected),
                "family_counts":dict(Counter(x["family"] for x in selected)),
                "id_sha256":[sha(x["id"].encode()) for x in selected]}
    rows=[]
    for (split,family),items in sorted(families.items()):
        shapes=Counter(x["shape"] for x in items)
        entry={"split":split,"family":family,"rows":len(items),"signatures":len(shapes),
               "max_signature_multiplicity":max(shapes.values())}
        if split!="test":
            pos=[x["positions"] for x in items]
            entry["accepted_by_context"]={ctx:sum(n<=ctx for n in pos) for ctx in (192,384,512)}
            entry["positions"]=quantiles(pos)
        rows.append(entry)
    shared=[{"signature_sha256":h,"family_split_pairs":[list(x) for x in sorted(pairs)],
             "rows":shape_counts[h]} for h,pairs in sorted(shape_index.items()) if len({s for s,f in pairs})>1]
    cross_task=sum(len(s)>1 for s in task_hashes.values())
    multi_target=sum(len(t)>1 for t in task_targets.values())
    require(sum(v["rows"] for v in summary.values())==17536 and len(rows)==110,"Corpus coverage mismatch")
    return {"schema":"sagallm-canonical-sft-quality-v1","status":"CANONICAL_TEMPLATE_TOKENIZER_AUDIT_COMPLETE",
            "manifest_sha256":DATA,"tokenizer_sha256":sha_file(tokenizer_path),"split_summary":summary,
            "family_rows":rows,"exact_response_count":len(source_hashes),
            "distinct_normalized_signatures":len(shape_index),
            "cross_split_signature_count":len(shared),"cross_split_signatures":shared,
            "exact_task_hash_cross_split_count":cross_task,"exact_task_hash_multiple_response_count":multi_target,
            "signature_is_heuristic":True,"semantic_equivalence_verified":False,
            "test_structural_source_read":True,"test_tokenization_or_scoring":False,
            "checkpoint_deserialized":False,"java_executed":False,"independent_evaluation":"NOT_RUN",
            "model_selection_permitted":False,"rights_review":"REVIEW_REQUIRED"}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parent-zip",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"Refuse overwrite")
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);materialize_parent(a.parent_zip,root)
        build_java_diverse_corpus(root/"data")
        result=audit(root/"data",root/"tokenizer.json")
        a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print(json.dumps({k:result[k] for k in ("status","split_summary","exact_response_count","distinct_normalized_signatures","cross_split_signature_count","exact_task_hash_cross_split_count","exact_task_hash_multiple_response_count")},sort_keys=True))

if __name__=="__main__":main()
