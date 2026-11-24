"""Run a local-only private Java benchmark intake screen; never evaluate a model."""
import argparse
import json
from pathlib import Path

from sagallm.java_private_intake import audit_private_intake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "provenance", "inventory", "exposure-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("suite", "provenance", "inventory"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse to overwrite intake evidence")
    report = audit_private_intake(
        args.suite, expected_suite_sha256=args.expected_suite_sha256,
        provenance_path=args.provenance,
        expected_provenance_sha256=args.expected_provenance_sha256,
        inventory_path=args.inventory,
        expected_inventory_sha256=args.expected_inventory_sha256,
        exposure_root=args.exposure_root,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(report["machine_status"] + " / REVIEW_REQUIRED / NOT_RUN")


if __name__ == "__main__":
    main()
