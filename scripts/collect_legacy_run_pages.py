"""Collect bounded Actions run pages in nonoverlapping inclusive UTC windows."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import subprocess

from scripts.audit_legacy_run_discovery import stamp,normalize,unique_runs
from scripts.audit_ranking_auxiliary import require

MAX_WINDOWS=256
MAX_RUNS=50000
FIELDS=("id","run_attempt","head_sha","path","event","status","conclusion","created_at","updated_at")

def iso(dt):return dt.isoformat(timespec="seconds").replace("+00:00","Z")
def call(args):
    return json.loads(subprocess.check_output(["gh","api",*args],timeout=90))

def collect(start,cutoff,fetch):
    lo,hi=stamp(start),stamp(cutoff)
    require(lo<=hi and lo.microsecond==hi.microsecond==0,"Invalid bounds")
    todo=[(lo,hi)];windows=[];rows=[]
    while todo:
        require(len(todo)+len(windows)<=MAX_WINDOWS,"Too many windows")
        a,b=todo.pop()
        p=fetch(iso(a),iso(b),1)
        count=p["total_count"]
        require(type(count) is int and 0<=count<=MAX_RUNS,"Invalid API count")
        if count>900:
            require(a<b,"Too many runs in one second")
            mid=a+timedelta(seconds=int((b-a).total_seconds())//2)
            todo.extend([(mid+timedelta(seconds=1),b),(a,mid)])
            continue
        pages=[p]+[fetch(iso(a),iso(b),i) for i in range(2,(count+99)//100+1)]
        require(all(x["total_count"]==count and len(x["workflow_runs"])<=100 for x in pages),"Changing API window count")
        compact=[{"total_count":count,"workflow_runs":[{k:r[k] for k in FIELDS} for r in x["workflow_runs"]]} for x in pages]
        batch=[normalize(r,live=True) for x in compact for r in x["workflow_runs"]]
        require(len(batch)==count and all(a<=stamp(r["created_at"])<=b for r in batch),"Incomplete/out-of-window run set")
        unique_runs(batch);rows.extend(batch)
        require(len(rows)<=MAX_RUNS,"Aggregate run cap")
        windows.append({"start_utc":iso(a),"end_utc":iso(b),"pages":compact})
    unique_runs(rows)
    return {"schema":"sagallm-actions-windowed-pages-v1","start_utc":start,"cutoff_utc":cutoff,"windows":windows}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cutoff",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    repo="repos/SagaK-dev/SagaLLM"
    start=call([repo])["created_at"]
    def fetch(lo,hi,page):
        return call(["-X","GET",repo+"/actions/runs","-f","per_page=100","-f",f"page={page}","-f",f"created={lo}..{hi}"])
    # Concurrent deletion/visibility changes can make a scan inconsistent. Retry
    # whole snapshot twice, then fail closed; never silently accept partial data.
    for attempt in range(3):
        try:
            result=collect(start,a.cutoff,fetch)
            break
        except (ValueError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
            if attempt==2:raise
    with a.output.open("x") as f:json.dump(result,f,separators=(",",":"),sort_keys=True)
    print(json.dumps({"windows":len(result["windows"]),"run_count":sum(w["pages"][0]["total_count"] for w in result["windows"]),"start_utc":start,"cutoff_utc":a.cutoff}))
if __name__=="__main__":main()
