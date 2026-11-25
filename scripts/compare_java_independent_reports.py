"""Compare two exact pinned Java evaluation reports; offline, no execution."""
import argparse
import json
from pathlib import Path
from sagallm.java_matched_reports import compare_files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("left","right","output"):
        p.add_argument("--"+k,type=Path,required=True)
    for k in ("left-sha256","right-sha256"):
        p.add_argument("--"+k,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.is_symlink():
        raise FileExistsError("Refuse overwrite")
    result=compare_files(a.left,a.right,a.left_sha256,a.right_sha256)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    print(result["status"])


if __name__=="__main__":
    main()
