"""Audit a quarantined externally sourced Java I/O intake; never execute code.

Pass --upstream-root with the exact pinned upstream file snapshots to prove
the selected test UUIDs, outputs and assistant-created stdin/stdout adapters
match their source data. Without source files, only structure is checked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sagallm.java_external_io_intake import verify_intake


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Audit report already exists")
    report = verify_intake(args.root, upstream_root=args.upstream_root)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
