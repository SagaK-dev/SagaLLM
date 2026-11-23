"""Non-executing pre-review audit of six pinned Java I/O candidates.

The output is a reviewer worklist, not a rights/independence certification,
and not a model evaluation. Python-only algorithmic checks. No Java runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sagallm.java_external_io_preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite existing audit evidence")
    report = run_preflight(args.root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "status": report["status"],
        "tasks": report["tasks"],
        "io_cases": report["io_cases"],
        "specification_checks_passed": report["specification_checks_passed"],
        "review_gate": report["review_gate"],
        "behavior": report["behavior"],
        "model_selection_permitted": False,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
