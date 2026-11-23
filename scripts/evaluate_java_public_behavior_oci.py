"""Opt-in public Java I/O study inside a per-job ephemeral VM + hardened OCI.

Only the four previously compiled, output-SHA-pinned Java model generations
may run. The original six public Exercism-derived tasks are UNREVIEWED. This
experiment does not use, relabel, or relax the independent benchmark gate.

WARNING: OCI is best-effort, not a formal security isolation proof. Run on a
disposable VM without sensitive data or exposed Docker sockets. Never run
generated Java directly on the Python host or a persistent workstation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import torch

from sagallm.java_behavior import oci_image_ready, run_java_in_oci
from sagallm.java_independent_benchmark import read_suite
from sagallm.java_instruction_sft import (
    V016_TOKENIZER_SHA256, V017_SFT_BEST_SHA256, _validate_base,
    compact_prompt, extend_rope_context, restore_verified_base_weights, sha_file,
)
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import checkpoint_load
from scripts.evaluate_java_public_compile_diagnostic import (
    GENERATION_SECONDS, MAX_OUTPUT_TOKENS, PINNED_CONTEXT, PINNED_SUITE_SHA256,
    PINNED_V018_SHA256, PROMPTS, fixed_public_tasks,
)

# These SHA-256 values were measured on the successful prior compile-only
# investigation (PR #40), BEFORE generated Java was ever run. All four
# were JAVAC_PASS at that commit. Changing them requires a new study.
VERIFIED_COMPILED_OUTPUTS = {
    "v017": {
        "ex_luhn": "e13884938d9bf32bbc04837151465cd58f1eff7b72982255940d85e4a224b1d2",
        "ex_atbash_cipher": "2b513562b6c49b4b9db66f14fabb016490b41873dd5abcb280e01e44f2e54278",
    },
    "v018": {
        "ex_hamming": "118dec6fa07db5da66d9ac57e3b998db7dc536d89a687542a0a85aaa75294ac3",
        "ex_luhn": "a1389e892e8622f8dca677394427480f632f83ab11d5eab3cc6897df9f079faf",
    },
}
EXPECTED_EVALUATED_CASES = 19
SMOKE_CODE = (
    "public class Main {"
    "public static void main(String[] a) {"
    "java.util.Scanner s=new java.util.Scanner(System.in);"
    "System.out.println(s.nextLine().toUpperCase(java.util.Locale.ROOT));"
    "}}"
)


def freeze_inputs(suite_file: Path) -> tuple[dict, list[str], str]:
    ids, sha, count = fixed_public_tasks(suite_file)
    suite, actual_sha, meta = read_suite(suite_file)
    if (sha != PINNED_SUITE_SHA256 or actual_sha != sha or count != 27
            or ids != list(PROMPTS) or meta["review_gate"] != "REVIEW_REQUIRED"
            or suite["reviewers"] != []):
        raise ValueError("Pinned PUBLIC unreviewed study or task order changed")
    by_id = {item["id"]: item for item in suite["tasks"]}
    if len(by_id) != 6:
        raise ValueError("Missing or duplicate original task")
    if set(VERIFIED_COMPILED_OUTPUTS) != {"v017", "v018"}:
        raise ValueError("Unexpected model arms")
    if {x for arm in VERIFIED_COMPILED_OUTPUTS.values() for x in arm} - set(by_id):
        raise ValueError("Unregistered compiled task")
    tests = sum(
        len(by_id[task]["cases"])
        for arm in VERIFIED_COMPILED_OUTPUTS.values() for task in arm
    )
    if tests != EXPECTED_EVALUATED_CASES:
        raise ValueError("Originally pre-registered I/O coverage changed")
    return by_id, ids, sha


def frozen_model_source(model, tokenizer, model_version: str, task_id: str) -> str:
    """Regenerate a SHA-pinned source. Refuse changed bytes before running it."""
    expected_sha = VERIFIED_COMPILED_OUTPUTS[model_version][task_id]
    if model.config.max_position_embeddings != PINNED_CONTEXT:
        raise ValueError("Context changed")
    prefix = [tokenizer.SPECIAL["<bos>"]] + tokenizer.encode(
        compact_prompt(PROMPTS[task_id]))
    remaining = PINNED_CONTEXT - len(prefix)
    if remaining < 2:
        raise ValueError("Pre-registered prompt exceeds context")
    with torch.inference_mode():
        generated = model.generate(
            torch.tensor([prefix], dtype=torch.long),
            max_new_tokens=min(remaining, MAX_OUTPUT_TOKENS),
            temperature=0.0,
            eos_token_id=tokenizer.SPECIAL["<eos>"],
            vocab_limit=tokenizer.vocab_size,
            deadline_monotonic=time.monotonic() + GENERATION_SECONDS,
        )
    source = tokenizer.decode(generated[0, len(prefix):].tolist())
    if (not source or len(source.encode("utf8")) > 4000
            or "\x00" in source or "public class Main" not in source
            or hashlib.sha256(source.encode("utf8")).hexdigest() != expected_sha):
        raise ValueError("Previously compiled generated source hash mismatch")
    return source


def assess_task(source: str, task: dict, *, image: str) -> dict:
    """Run ONLY through OCI with per-case fresh containers and bounded inputs."""
    statuses = []
    for case in task["cases"]:
        outcome = run_java_in_oci(
            source, case["stdout"], backend="docker", image=image,
            stdin_data=case["stdin"], allow_execution=True, timeout=20,
        )
        status = outcome["status"]
        if status not in ("PASS", "FAIL", "TIMEOUT", "OUTPUT_LIMIT",
                          "RUNTIME_ERROR", "NOT_RUN", "REJECTED"):
            raise ValueError("Unrecognized OCI outcome")
        statuses.append(status)
    unavailable = any(code in ("NOT_RUN", "RUNTIME_ERROR") for code in statuses)
    return {
        "id": task["id"], "cases_total": len(statuses),
        "cases_passed": sum(s == "PASS" for s in statuses),
        "case_statuses": statuses,  # no raw input or stdout
        "task_status": ("UNAVAILABLE" if unavailable else
                        "PASS" if all(s == "PASS" for s in statuses) else "FAIL"),
        "generated_source_stored": False,
        "host_java_execution": False,
    }


def run(v017: Path, v018: Path, tokenizer_path: Path, suite_file: Path,
        *, image: str, allow_public_unreviewed: bool,
        acknowledge_disposable_vm: bool) -> dict:
    if not allow_public_unreviewed or not acknowledge_disposable_vm:
        raise PermissionError(
            "Explicit unreviewed-public opt-in AND disposable-VM acknowledgement required")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("OCI JDK image must be identified by its local sha256 IMAGE ID")
    tasks, ids, suite_sha = freeze_inputs(suite_file)
    if not oci_image_ready("docker", image):
        raise RuntimeError("Local OCI daemon or pinned JDK image is NOT READY")
    smoke = run_java_in_oci(
        SMOKE_CODE, "SAGALLM\n", backend="docker", image=image,
        stdin_data="sagallm\n", allow_execution=True, timeout=22,
    )
    if smoke["status"] != "PASS":
        raise RuntimeError(
            "OCI Java smoke test did not pass; abort BEFORE any generated Java: "
            + smoke["status"])
    if (not v018.is_file() or v018.is_symlink()
            or v018.stat().st_size > 90 * 1024 * 1024
            or sha_file(v018) != PINNED_V018_SHA256):
        raise ValueError("Wrong v0.18 checkpoint")
    state, original, v017_sha, tokenizer_sha = _validate_base(
        v017, tokenizer_path, require_pinned=True,
        pinned_checkpoint_sha256=V017_SFT_BEST_SHA256)
    if tokenizer_sha != V016_TOKENIZER_SHA256:
        raise ValueError("Unpinned tokenizer")
    config = extend_rope_context(original, PINNED_CONTEXT)
    tokenizer = load_tokenizer(tokenizer_path)
    torch.set_num_threads(2)
    model = SagaTransformer(config)
    model.eval()
    restore_verified_base_weights(model, state, original)

    results = {}
    for version in ("v017", "v018"):
        if version == "v018":
            checkpoint_load(v018, model)
            if model.config != config:
                raise ValueError("Changed architecture in v0.18")
            model.eval()
        cases = []
        for task_id in ids:
            if task_id not in VERIFIED_COMPILED_OUTPUTS[version]:
                continue
            code = frozen_model_source(model, tokenizer, version, task_id)
            try:
                item = assess_task(code, tasks[task_id], image=image)
            finally:
                del code
            item["source_sha256"] = VERIFIED_COMPILED_OUTPUTS[version][task_id]
            cases.append(item)
        total = sum(t["cases_total"] for t in cases)
        passed = sum(t["cases_passed"] for t in cases)
        unavailable = any(t["task_status"] == "UNAVAILABLE" for t in cases)
        results[version] = {
            "compiled_outputs_regenerated": len(cases),
            "io_cases_attempted": total,
            "io_cases_passed": passed,
            "io_cases_failing": total - passed if not unavailable else None,
            "tasks_passed_all_cases": (
                sum(t["task_status"] == "PASS" for t in cases)
                if not unavailable else None),
            "status": "PARTIAL_OR_UNAVAILABLE" if unavailable else "EVALUATED",
            "tasks": cases,
        }
    if sum(results[x]["io_cases_attempted"] for x in results) != EXPECTED_EVALUATED_CASES:
        raise ValueError("Some pinned public cases were not attempted")
    return {
        "status": "EXPLORATORY_PUBLIC_UNREVIEWED_ISOLATED_IO",
        "suite_sha256": suite_sha,
        "original_six_tasks": len(ids),
        "original_27_cases": 27,
        "matched_previous_compiles": 4,
        "attempted_cases_across_models": EXPECTED_EVALUATED_CASES,
        "v017_checkpoint_sha256": v017_sha,
        "v018_checkpoint_sha256": PINNED_V018_SHA256,
        "tokenizer_sha256": tokenizer_sha,
        "model_context": PINNED_CONTEXT,
        "decoding": "GREEDY_170_TOKENS",
        "oci_backend": "docker",
        "oci_image_id": image,
        "oci_smoke": "PASS",
        "isolation": {
            "runner": "EPHEMERAL_GITHUB_HOSTED_VM",
            "container_network": "none",
            "root_filesystem": "read_only",
            "capabilities": "all_dropped",
            "no_new_privileges": True,
            "user": "65534",
            "pid_limit": 64,
            "memory_limit_mb": 384,
            "image_pull_during_java_run": False,
            "security_proof": False,
        },
        "v017": results["v017"],
        "v018": results["v018"],
        "human_review_status": "REVIEW_REQUIRED",
        "public_pretraining_contamination_possible": True,
        "original_prompts_used_verbatim": False,
        "independent_benchmark_approval": False,
        "model_selection_permitted": False,
        "model_updated": False,
        "model_behavioral_correctness_generalized": False,
        "generated_source_persisted": False,
        "raw_stdout_or_stderr_persisted": False,
        "java_execution": "OCI_ONLY",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v017", type=Path, required=True)
    p.add_argument("--v018", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--oci-image-id", required=True)
    p.add_argument("--allow-unreviewed-public-execution", action="store_true")
    p.add_argument("--acknowledge-disposable-vm", action="store_true")
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to replace previous evidence")
    outcome = run(
        args.v017, args.v018, args.tokenizer, args.suite,
        image=args.oci_image_id,
        allow_public_unreviewed=args.allow_unreviewed_public_execution,
        acknowledge_disposable_vm=args.acknowledge_disposable_vm,
    )
    args.output.write_text(json.dumps(outcome, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf8")
    print(json.dumps({
        "status": outcome["status"],
        "v017": {k: outcome["v017"][k]
                 for k in ("io_cases_attempted", "io_cases_passed", "status",
                           "tasks_passed_all_cases")},
        "v018": {k: outcome["v018"][k]
                 for k in ("io_cases_attempted", "io_cases_passed", "status",
                           "tasks_passed_all_cases")},
        "review": "REVIEW_REQUIRED",
        "independent_benchmark_approval": False,
        "model_updated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
