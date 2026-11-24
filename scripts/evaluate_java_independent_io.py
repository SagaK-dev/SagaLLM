"""Validate prospective independent Java tasks and optionally run I/O in OCI.

Default is NOT_RUN. This script never invokes Java unless all separate gates
are satisfied: human-review self-attestation, exact suite SHA-256, explicit
execution opt-in, explicit disposable-VM acknowledgement, and local OCI image.
Never use host Java to execute model-generated code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sagallm.java_independent_benchmark import evaluate_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-execution", action="store_true")
    parser.add_argument("--acknowledge-disposable-vm", action="store_true")
    parser.add_argument("--expected-suite-sha256")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--expected-provenance-sha256")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--expected-inventory-sha256")
    parser.add_argument("--exposure-root", type=Path, help="Legacy; not sufficient for execution")
    parser.add_argument("--model-freeze", type=Path)
    parser.add_argument("--expected-model-freeze-sha256")
    parser.add_argument("--shard-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--candidate-freeze", type=Path)
    parser.add_argument("--expected-candidate-freeze-sha256")
    parser.add_argument("--vm-review", type=Path)
    parser.add_argument("--expected-vm-review-sha256")
    parser.add_argument("--backend", choices=["docker", "podman"], default="docker")
    parser.add_argument("--image", default="eclipse-temurin:21-jdk")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse to overwrite behavioral evidence")
    result = evaluate_suite(
        args.suite, args.candidate_dir,
        allow_execution=args.allow_execution,
        disposable_vm_acknowledged=args.acknowledge_disposable_vm,
        expected_suite_sha256=args.expected_suite_sha256,
        backend=args.backend, image=args.image,
        provenance_path=args.provenance,
        expected_provenance_sha256=args.expected_provenance_sha256,
        inventory_path=args.inventory,
        expected_inventory_sha256=args.expected_inventory_sha256,
        exposure_root=args.exposure_root,
        freeze_path=args.model_freeze,
        expected_freeze_sha256=args.expected_model_freeze_sha256,
        shard_root=args.shard_root, artifact_root=args.artifact_root,
        candidate_freeze_path=args.candidate_freeze,
        expected_candidate_freeze_sha256=args.expected_candidate_freeze_sha256,
        vm_review_path=args.vm_review,
        expected_vm_review_sha256=args.expected_vm_review_sha256,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in ("results",)
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
