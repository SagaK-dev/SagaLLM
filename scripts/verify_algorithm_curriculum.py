"""Compile/execute only the exact in-tree reference curriculum sources.

No model outputs, untrusted user source, arbitrary command, or external dataset
is accepted. Fixed subprocess arguments, temporary isolated directories,
limited heap and timeouts. This is not a general-purpose Java sandbox.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from sagallm.java_algorithm_curriculum import rows,cases,oracle,canonical,sha

def require(ok,msg):
    if not ok:raise ValueError(msg)
def invoke(args,cwd,stdin=None):
    env={k:v for k,v in os.environ.items() if k in ("PATH","JAVA_HOME","LANG","LC_ALL","SYSTEMROOT")}
    p=subprocess.run(args,cwd=cwd,input=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     env=env,timeout=12,check=False)
    require(p.returncode==0 and len(p.stdout)<=16384 and len(p.stderr)<=16384,"Reference program failed")
    return p.stdout

def run():
    javac=shutil.which("javac");java=shutil.which("java")
    require(javac and java,"JDK required")
    records=list(rows());sources={}
    for r in records:
        key=(r["family"],r["variant"])
        require(len(r["source"].encode())<=4096,"Source bound")
        if key in sources:require(sources[key]==r["source"],"Language variant source drift")
        sources[key]=r["source"]
    require(len(records)==16 and len(sources)==8 and len({r["task"] for r in records})==16,"Curriculum scope mismatch")
    results=[]
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        version=invoke([javac,"-version"],root).decode().strip()
        require(version.startswith("javac 17."),"Expected Java 17")
        for (family,variant),source in sorted(sources.items()):
            wd=root/(family+"-"+variant);wd.mkdir()
            (wd/"Main.java").write_bytes(source.encode())
            invoke([javac,"-J-Xmx128m","--release","17","-encoding","UTF-8","-proc:none","Main.java"],wd)
            checks=[]
            for i,stdin in enumerate(cases(family)):
                expected=oracle(family,stdin).encode()
                actual=invoke([java,"-Xmx64m","-XX:ActiveProcessorCount=1","-cp",str(wd),"Main"],wd,stdin.encode())
                require(actual==expected,"Reference output mismatch: "+family+"/"+variant+"/"+str(i))
                checks.append({"case":i,"stdin_sha256":sha(stdin.encode()),"expected_stdout_sha256":sha(expected),"actual_stdout_sha256":sha(actual)})
            require(len(checks)==12,"Case count mismatch")
            results.append({"family":family,"variant":variant,"source_sha256":sha(source.encode()),
                            "source_bytes":len(source.encode()),"compile":"PASS","checks":checks})
    return {"schema":"sagallm-java-algorithm-curriculum-prototype-v1","status":"FIXED_REFERENCE_CURRICULUM_VERIFIED",
            "jdk":version,"families":4,"implementations":len(sources),"bilingual_records":len(records),
            "reference_compiles":len(results),"oracle_comparisons":sum(len(r["checks"]) for r in results),
            "record_index":[{"id":r["id"],"family":r["family"],"variant":r["variant"],"language":r["language"],
                             "task_sha256":sha(r["task"].encode()),"source_sha256":sha(r["source"].encode()),
                             "row_sha256":sha(canonical(r))} for r in records],
            "reference_results":results,"model_generated_code_executed":False,"arbitrary_code_sandbox":False,
            "training_performed":False,"heldout_role_assigned":False,"independent_evaluation":"NOT_RUN",
            "rights_review":"REVIEW_REQUIRED","model_selection_permitted":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),"Refuse overwrite")
    d=run()
    with a.output.open("x") as f:json.dump(d,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps({k:d[k] for k in ("status","jdk","families","implementations","bilingual_records","reference_compiles","oracle_comparisons")},sort_keys=True))
