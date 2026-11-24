"""Prepare or audit private checkpoint-bound exposure shards, locally only."""
import argparse
import json
from pathlib import Path

from sagallm.java_exposure_shards import prepare_exposure_shards, audit_exposure_shards, measure_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare")
    audit = subs.add_parser("audit")
    for command in (prepare, audit):
        for name in ("inventory", "checkpoint", "tokenizer"):
            command.add_argument("--" + name, required=True, type=Path)
            command.add_argument("--expected-" + name + "-sha256", required=True)
        command.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--exposure-root", required=True, type=Path)
    prepare.add_argument("--training-code-sha", required=True)
    for name in ("suite", "provenance", "shard-root"):
        audit.add_argument("--" + name, required=True, type=Path)
    for name in ("suite", "provenance", "manifest"):
        audit.add_argument("--expected-" + name + "-sha256", required=True)
    audit.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_exposure_shards(
            args.inventory, args.expected_inventory_sha256, args.exposure_root, args.output,
            checkpoint_path=args.checkpoint, checkpoint_sha256=args.expected_checkpoint_sha256,
            tokenizer_path=args.tokenizer, tokenizer_sha256=args.expected_tokenizer_sha256,
            training_code_sha=args.training_code_sha)
    else:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("Refuse to overwrite report")
        fn = measure_audit if args.measure else audit_exposure_shards
        result = fn(args.suite, args.expected_suite_sha256,
            provenance_path=args.provenance, expected_provenance_sha256=args.expected_provenance_sha256,
            inventory_path=args.inventory, expected_inventory_sha256=args.expected_inventory_sha256,
            shard_root=args.shard_root, expected_manifest_sha256=args.expected_manifest_sha256,
            checkpoint_path=args.checkpoint, expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            tokenizer_path=args.tokenizer, expected_tokenizer_sha256=args.expected_tokenizer_sha256)
        report = {"audit": result[0], "measurement": result[1]} if args.measure else result
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
