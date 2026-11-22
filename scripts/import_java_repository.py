"""Local, unapproved Java repository inventory. NEVER auto-trains this code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from sagallm.java_source_import import export_java_source_corpus

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-uri',required=True)
    p.add_argument('--revision',required=True)
    a=p.parse_args()
    print(json.dumps(export_java_source_corpus(a.source,a.output,
                source_uri=a.source_uri,revision=a.revision),indent=2))