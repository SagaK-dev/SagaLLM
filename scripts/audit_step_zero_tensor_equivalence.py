"""Compare selected step-zero checkpoint tensor bytes against the exact parent.

Only SHA-pinned historical checkpoint members are loaded, with CPU weights_only.
No model instantiation, optimizer restore, inference, training or Java execution.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.audit_ranking_auxiliary import require, sha
from scripts.audit_unarchived_artifact_triage import read_evidence, gate, hexsha
from scripts.audit_v018_training import BEST

LINEAGE_ARTIFACT = 15246425930
LINEAGE_ZIP = "49df23428f59e87b965bc6138e19c2269147446c5c3742858f9ef54313c3521b"
PARENT_ARTIFACT = 14743051466
PARENT_ZIP = "b039250f6ebecb4f8631ba38050cf219eb41b53308ea8d0b13bb45f566477789"
PARENT_MEMBER = "java-context-model/java-sft-best.pt"
ARCHIVES = {
    14752812310: (45018649700, "ac16cd725a84a26e02f2c00964f236b8670529aa4ba7c35512ce60da8597246d", "identifier", "identifier-ab-evidence"),
    14753120733: (45018851049, "0679dfc57619f60b8f9a981160dbd2054970030f4faca15cc22d8af5bcca9e3c", "identifier", "identifier-ab-evidence"),
    14756075922: (45026352157, "bf142c8a4b4fa1905b505a62a5424ca2da9129f19c911b6126955245980f4fcd", "broad", "broad-mix-ab-evidence"),
    14756001714: (45026590750, "57098c53a1864db5f68a8e1307f6b37466610aaa89da03cf9b2264c0afcde079", "broad", "broad-mix-ab-evidence"),
}
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_MEMBER = 96 * 1024 * 1024
MAX_ELEMENTS = 8_000_000
MAX_TENSOR_BYTES = 64 * 1024 * 1024


def archive_bytes(path, expected):
    require(path.is_file() and not path.is_symlink(), "Unsafe archive path")
    with path.open("rb") as f:
        raw=f.read(MAX_ARCHIVE+1)
    require(0 < len(raw) <= MAX_ARCHIVE and sha(raw)==expected, "Archive pin mismatch")
    return raw


def member_bytes(z, name, digest):
    matches=[m for m in z.infolist() if m.filename==name]
    require(len(matches)==1 and not matches[0].is_dir() and not matches[0].flag_bits & 1
            and 0 < matches[0].file_size <= MAX_MEMBER, "Unsafe or absent checkpoint member")
    with z.open(matches[0]) as f:
        raw=f.read(MAX_MEMBER+1)
    require(len(raw)==matches[0].file_size and sha(raw)==digest, "Checkpoint byte digest mismatch")
    return raw


def load_state(raw):
    import torch
    # The safe loader is not a security proof; accept only the pinned repository artifacts.
    require(hasattr(torch.serialization,"get_unsafe_globals_in_checkpoint"), "Safe-loader inspection unavailable")
    require(torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(raw))==[],
            "Unapproved checkpoint globals")
    state=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    require(type(state) is dict and isinstance(state["model"],dict) and type(state["config"]) is dict,
            "Unexpected checkpoint schema")
    return state


def tensor_index(state):
    import torch
    model=state["model"]
    require(0 < len(model) <= 512, "Unexpected tensor count")
    total=0
    total_bytes=0
    result={}
    for name,t in sorted(model.items()):
        require(type(name) is str and 0 < len(name)<=300 and isinstance(t,torch.Tensor)
                and t.device.type=="cpu" and t.layout==torch.strided and not t.is_quantized,
                "Unsupported tensor member")
        total+=t.numel()
        total_bytes+=t.numel()*t.element_size()
        require(total<=MAX_ELEMENTS and total_bytes<=MAX_TENSOR_BYTES,"Tensor resource limit")
        raw=t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        require(len(raw)==t.numel()*t.element_size(),"Tensor byte length mismatch")
        result[name]={"dtype":str(t.dtype),"shape":list(t.shape),"bytes":len(raw),"sha256":sha(raw)}
    return result


def tensor_comparison(parent,candidate):
    a,b=tensor_index(parent),tensor_index(candidate)
    names=set(a)|set(b)
    differences=sorted(n for n in names if a.get(n)!=b.get(n))
    raw_a=json.dumps(a,sort_keys=True,separators=(",",":")).encode()
    raw_b=json.dumps(b,sort_keys=True,separators=(",",":")).encode()
    return {"parent_tensor_index_sha256":sha(raw_a),"candidate_tensor_index_sha256":sha(raw_b),
            "parent_tensor_count":len(a),"candidate_tensor_count":len(b),
            "different_tensor_names":differences,"all_tensor_digests_match":not differences,
            "config_equal":parent["config"]==candidate["config"],
            "parent_tensor_bytes":sum(x["bytes"] for x in a.values()),
            "candidate_tensor_bytes":sum(x["bytes"] for x in b.values())}


def expected_instances(lineage):
    gate(lineage)
    require(lineage["status"]=="DECLARED_PARENT_AND_OUTPUT_ROLES_RECONCILED"
            and lineage["step_zero_distinct_hashes"]==6 and lineage["step_zero_member_instances"]==8,
            "Prior selected-step scope mismatch")
    entries={}
    for node in lineage["recovered_outputs"]:
        if node["selected_step"]!=0:
            continue
        require(node["parent_sha256"]==BEST and node["role"]=="ARM_BEST"
                and node["checkpoint_member_bytes_verified"] is True
                and node["tensor_equality_verified"] is False and hexsha(node["sha256"]),
                "Step-zero declaration mismatch")
        for i in node["instances"]:
            aid=i["artifact_id"]
            run,digest,mode,prefix=ARCHIVES[aid]
            arm=node["arm"]
            require(arm in ("canonical","identifier_augmented")
                    and i["run_id"]==run and i["archive_sha256"]==digest
                    and i["member"]==f"{prefix}/{arm}-validation-best.pt","Member binding mismatch")
            key=(aid,arm)
            require(key not in entries,"Duplicate member declaration")
            entries[key]=(node,i)
    require(len(entries)==8 and {k[0] for k in entries}==set(ARCHIVES),"Step-zero coverage mismatch")
    return entries


def reconcile(lineage,folder):
    import torch
    entries=expected_instances(lineage)
    parent_zip=archive_bytes(folder/f"{PARENT_ARTIFACT}.zip",PARENT_ZIP)
    with zipfile.ZipFile(io.BytesIO(parent_zip)) as z:
        parent=load_state(member_bytes(z,PARENT_MEMBER,BEST))
    del parent_zip
    rows=[]
    for aid,(run,digest,mode,prefix) in ARCHIVES.items():
        raw=archive_bytes(folder/f"{aid}.zip",digest)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for arm in ("canonical","identifier_augmented"):
                node,i=entries[(aid,arm)]
                state=load_state(member_bytes(z,i["member"],node["sha256"]))
                require(state["selected_step"]==0 and state["arm"]==arm,"Embedded selected-step metadata mismatch")
                field="base_checkpoint_sha256" if mode=="identifier" else "origin_sha256"
                require(state[field]==BEST,"Embedded parent digest mismatch")
                result=tensor_comparison(parent,state)
                rows.append({"run_id":run,"artifact_id":aid,"archive_sha256":digest,
                             "member":i["member"],"checkpoint_sha256":node["sha256"],
                             "arm":arm,"mode":mode,"embedded_selected_step":0,"embedded_parent_sha256":BEST,
                             **result})
                del state
        del raw
    equal=[r for r in rows if r["all_tensor_digests_match"] and r["config_equal"]]
    return {"schema":"sagallm-step-zero-tensor-scope-v1",
            "status":"BOUNDED_STEP_ZERO_TENSOR_COMPARISON_COMPLETE",
            "torch_version":str(torch.__version__),"parent_checkpoint_sha256":BEST,
            "parent_artifact_id":PARENT_ARTIFACT,"parent_archive_sha256":PARENT_ZIP,
            "member_count":len(rows),"distinct_serialized_checkpoint_hashes":len({r["checkpoint_sha256"] for r in rows}),
            "tensor_and_config_equal_member_count":len(equal),
            "tensor_and_config_equal_distinct_hashes":len({r["checkpoint_sha256"] for r in equal}),
            "all_selected_step_zero_tensors_match_parent":len(equal)==len(rows),
            "results":rows,"checkpoint_loaded_with_weights_only":True,
            "model_instantiated":False,"new_training":False,"new_inference":False,"new_java_execution":False,
            "actual_exposure_verified":False,"training_lineage_verified":False,
            "full_exposure_inventory":"NOT_ESTABLISHED","rights_review":"REVIEW_REQUIRED",
            "independent_evaluation":"NOT_RUN","model_selection_permitted":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lineage-zip",type=Path,required=True)
    p.add_argument("--archives",type=Path,required=True)
    a=p.parse_args()
    data,_=read_evidence(a.lineage_zip,LINEAGE_ZIP,["recovered-semantic-lineage.json"])
    out=reconcile(data["recovered-semantic-lineage.json"],a.archives)
    out["lineage_evidence"]={"artifact_id":LINEAGE_ARTIFACT,"archive_sha256":LINEAGE_ZIP}
    print(json.dumps(out,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
