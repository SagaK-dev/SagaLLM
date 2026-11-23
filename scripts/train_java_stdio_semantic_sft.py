"""Train stdin/algorithm/stdout Java SFT against v0.18; validate in OCI.

Only project-authored TRAIN examples enter optimization. All validation
families are disjoint from TRAIN and all old benchmark/test families.
Model weights can be selected by *validation response loss*, and a separate
fixed validation behavioral experiment compares frozen base vs candidate.
No public Exercism prompt or answer enters training, validation, or selection.

This experiment does not deploy a model automatically. It cannot establish
general Java competence, even when validation improves; the small curriculum
is synthetic and has known algorithmic similarity across its families.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time

import torch

from sagallm.config import ModelConfig
from sagallm.java_behavior import oci_image_ready, run_java_in_oci
from sagallm.java_diverse_corpus import verify_java_diverse_corpus
from sagallm.java_instruction_sft import (
    V016_TOKENIZER_SHA256, compact_prompt, encode_row, eval_sample,
    interleaved_order, pack, read_split, response_loss, sha_file,
)
from sagallm.java_stdio_curriculum import (
    TRAIN_FAMILIES, VALIDATION_FAMILIES, expected_rows,
    validation_oracle_rows, verify_curriculum,
)
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import checkpoint_load, verify_checkpoint_tokenizer

EXPERIMENT = "JAVA_STDIO_SEMANTIC_SFT_V1"
PINNED_V018 = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
SEED = 4201
STEPS = 160
BATCH = 4
BROAD_PER_BATCH = 2
STDIO_PER_BATCH = 2
LR = 2e-5
EVAL_AT = (80, 160)
RESPONSE_LOSS_MIN_DELTA = 0.005
BEHAVIOR_MIN_CASE_DELTA = 2
MAX_TRAIN_SECONDS = 780
VALIDATION_PROMPTS = 6
VALIDATION_CASES = 18
MAX_GEN_TOKENS = 170
SMOKE = (
    "public class Main {public static void main(String[] x){"
    "java.util.Scanner s=new java.util.Scanner(System.in);"
    "System.out.println(s.nextLine().toUpperCase(java.util.Locale.ROOT));"
    "}}"
)


def train_schedule(broad: list[dict], stdio: list[dict], *, steps: int = STEPS) -> list[tuple[tuple[str, int], ...]]:
    if (type(steps) is not int or not 1 <= steps <= STEPS
            or len(broad) < steps * BROAD_PER_BATCH
            or len(stdio) < steps * STDIO_PER_BATCH):
        raise ValueError("Not enough unique TRAIN data to run fixed schedule")
    if {item["family"] for item in stdio} != set(TRAIN_FAMILIES):
        raise ValueError("Missing/foreign stdio TRAIN families")
    if set(VALIDATION_FAMILIES) & {
        item["family"] for item in broad + stdio
    }:
        raise ValueError("Validation leaked into optimizer")
    if (len({item["id"] for item in broad}) != len(broad)
            or len({item["id"] for item in stdio}) != len(stdio)):
        raise ValueError("Repeated training record")
    b_order = interleaved_order(broad, SEED)
    s_order = interleaved_order(stdio, SEED + 1)
    return [
        tuple([("broad", i) for i in b_order[k * 2:k * 2 + 2]] +
              [("stdio", i) for i in s_order[k * 2:k * 2 + 2]])
        for k in range(steps)
    ]


def fixed_validation_samples(encoded: list[dict], oracle_rows: list[dict]) -> list[tuple[dict, dict]]:
    lookup = {item["id"]: item for item in encoded}
    cases = {item["id"]: item for item in oracle_rows}
    if len(cases) != len(oracle_rows):
        raise ValueError("Duplicate validation task")
    selected = []
    for family in VALIDATION_FAMILIES:
        for index in (0, 1):
            key = f"validation-{family}-{index:03d}"
            if key not in lookup or key not in cases:
                raise ValueError("Required validation record was excluded by context")
            if (lookup[key]["family"] != family or cases[key]["family"] != family
                    or len(cases[key]["cases"]) != 3):
                raise ValueError("Misaligned validation task and I/O cases")
            selected.append((lookup[key], cases[key]))
    if len(selected) != VALIDATION_PROMPTS:
        raise ValueError("Unexpected validation sample count")
    return selected


def _generate(model, tokenizer, row: dict) -> tuple[str | None, str]:
    prefix = row["prefix_ids"]
    remaining = model.config.max_position_embeddings - len(prefix)
    if remaining < 2:
        return None, "INSUFFICIENT_CONTEXT"
    with torch.inference_mode():
        ids = model.generate(
            torch.tensor([prefix], dtype=torch.long),
            max_new_tokens=min(remaining, MAX_GEN_TOKENS),
            temperature=0.0,
            eos_token_id=tokenizer.SPECIAL["<eos>"],
            vocab_limit=tokenizer.vocab_size,
            deadline_monotonic=time.monotonic() + 25,
        )[0, len(prefix):].tolist()
    source = tokenizer.decode(ids)
    if (not source or len(source.encode("utf8")) > 4000
            or "\x00" in source or "public class Main" not in source):
        return None, "INVALID_GENERATED_JAVA"
    return source, "GENERATED"


def behavioral_validation(model, tokenizer, selected: list[tuple[dict, dict]], image_id: str) -> dict:
    details = []
    for row, oracle in selected:
        source, generation_status = _generate(model, tokenizer, row)
        if source is None:
            details.append({
                "id": row["id"], "family": row["family"],
                "generated": generation_status, "cases_total": 3,
                "cases_oci_invoked": 0, "cases_passed": 0,
                "case_categories": ["NOT_RUN_INVALID_GENERATION"] * 3,
                "task_pass": False, "unavailable": False,
                "source_saved": False,
            })
            continue
        categories = []
        passed = 0
        unavailable = False
        for case in oracle["cases"]:
            result = run_java_in_oci(
                source, case["stdout"], backend="docker", image=image_id,
                allow_execution=True, stdin_data=case["stdin"], timeout=20,
            )
            status = result["status"]
            if status not in (
                "PASS", "FAIL", "TIMEOUT", "OUTPUT_LIMIT", "REJECTED",
                "NOT_RUN", "RUNTIME_ERROR",
            ):
                raise ValueError("Unknown OCI grader status")
            if status == "PASS":
                category = "PASS"
                passed += 1
            elif status == "FAIL":
                category = ("OUTPUT_MISMATCH_ZERO_EXIT"
                            if result.get("returncode") == 0
                            else "NONZERO_EXIT_OR_JAVAC_ERROR")
            else:
                category = status
            unavailable |= status in ("NOT_RUN", "RUNTIME_ERROR")
            categories.append(category)
        details.append({
            "id": row["id"], "family": row["family"],
            "generated": generation_status, "cases_total": 3,
            "cases_oci_invoked": len(categories), "cases_passed": passed,
            "case_categories": categories,
            "task_pass": passed == 3 and not unavailable,
            "unavailable": unavailable, "source_saved": False,
            "source_sha256": hashlib.sha256(source.encode("utf8")).hexdigest(),
        })
    if len(details) != VALIDATION_PROMPTS:
        raise ValueError("Not all required families evaluated")
    return {
        "tasks": len(details), "total_cases": VALIDATION_CASES,
        "oci_invocations": sum(item["cases_oci_invoked"] for item in details),
        "cases_passed": sum(item["cases_passed"] for item in details),
        "tasks_all_pass": sum(item["task_pass"] for item in details),
        "unavailable": any(item["unavailable"] for item in details),
        "source_persisted": False,
        "java_executed_on_python_host": False,
        "cases": details,
    }


def conservative_gate(base: dict, candidate: dict, *, selected_step: int,
                      initial_loss: float, selected_loss: float) -> dict:
    if (base["total_cases"] != candidate["total_cases"] or
            base["total_cases"] != VALIDATION_CASES):
        raise ValueError("Non-matched validation coverage")
    if (base["unavailable"] or candidate["unavailable"] or
            base["oci_invocations"] < 1 or candidate["oci_invocations"] < 1):
        return {"decision": "INCONCLUSIVE_RUNTIME_UNAVAILABLE", "validated_gain": False,
                "production_adoption": False}
    gain = candidate["cases_passed"] - base["cases_passed"]
    qualifies = (selected_step > 0
                 and initial_loss - selected_loss >= RESPONSE_LOSS_MIN_DELTA
                 and gain >= BEHAVIOR_MIN_CASE_DELTA
                 and candidate["tasks_all_pass"] >= base["tasks_all_pass"])
    return {
        "decision": ("CANDIDATE_VALIDATION_GAIN_NOT_DEPLOYED" if qualifies
                     else "NO_VALIDATED_BEHAVIOR_GAIN"),
        "compiled_and_behavioral_case_delta": gain,
        "validation_loss_delta": round(initial_loss - selected_loss, 8),
        "validated_gain": bool(qualifies),
        "production_adoption": False,
    }


def experiment(base_file: Path, tokenizer_path: Path, canonical: Path,
               stdio: Path, output: Path, *, allow_oci: bool,
               acknowledge_vm: bool, image_id: str | None) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to overwrite Java training/evaluation evidence")
    if (not base_file.is_file() or base_file.is_symlink()
            or base_file.stat().st_size > 90 * 1024 * 1024
            or sha_file(base_file) != PINNED_V018
            or not tokenizer_path.is_file() or tokenizer_path.is_symlink()
            or sha_file(tokenizer_path) != V016_TOKENIZER_SHA256):
        raise ValueError("Required pinned Java model or tokenizer not present")
    verify_java_diverse_corpus(canonical)
    metadata = verify_curriculum(stdio)
    state = torch.load(base_file, map_location="cpu", weights_only=True)
    verify_checkpoint_tokenizer(state, tokenizer_path)
    config = ModelConfig(**state["config"])
    if config.max_position_embeddings != 384 or config.parameter_count() > 3_000_000:
        raise ValueError("Unexpected SFT model configuration")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError("Tokenizer/model vocabulary mismatch")
    broad, counts = read_split(canonical, "train", tokenizer, 384)
    new_train = [
        encoded for item in expected_rows("train")
        if (encoded := encode_row(item, tokenizer, 384)) is not None
    ]
    valid = [
        encoded for item in expected_rows("validation")
        if (encoded := encode_row(item, tokenizer, 384)) is not None
    ]
    if (len(broad) < 400 or len(new_train) < STEPS * STDIO_PER_BATCH
            or len(valid) < 18):
        raise ValueError("Unbalanced or context-excluded curriculum")
    if set(item["family"] for item in valid) != set(VALIDATION_FAMILIES):
        raise ValueError("Validation family excluded")
    if {x["family"] for x in valid} & {x["family"] for x in broad + new_train}:
        raise ValueError("Validation family used for training")
    schedule = train_schedule(broad, new_train)
    selected = fixed_validation_samples(valid, validation_oracle_rows())
    loss_rows = eval_sample(valid, min(24, len(valid)), seed=47)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model = SagaTransformer(config)
    checkpoint_load(base_file, model)
    model.eval()
    before_state = copy.deepcopy(model.state_dict())
    initial_loss = response_loss(model, loss_rows)
    best_loss = initial_loss
    best_step = 0
    best_state = copy.deepcopy(before_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    history = [{"step": 0, "validation_response_loss": round(initial_loss, 8)}]
    started = time.monotonic()
    for step, indices in enumerate(schedule, start=1):
        if time.monotonic() - started > MAX_TRAIN_SECONDS:
            raise TimeoutError("Fixed CPU training budget exceeded")
        batch = [broad[k] if kind == "broad" else new_train[k]
                 for kind, k in indices]
        if len(batch) != BATCH:
            raise ValueError("Fixed minibatch schedule broken")
        x, y = pack(batch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels=y).loss
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite optimizer loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in EVAL_AT:
            current_loss = response_loss(model, loss_rows)
            history.append({
                "step": step, "validation_response_loss": round(current_loss, 8)})
            if current_loss <= best_loss - RESPONSE_LOSS_MIN_DELTA:
                best_step, best_loss = step, current_loss
                best_state = copy.deepcopy(model.state_dict())
    model.eval()
    training_seconds = round(time.monotonic() - started, 2)
    report = {
        "experiment": EXPERIMENT,
        "status": "TRAINED_VALIDATION_LOSS_SELECTED",
        "origin_checkpoint_sha256": PINNED_V018,
        "tokenizer_sha256": V016_TOKENIZER_SHA256,
        "canonical_train": counts,
        "stdio_manifest_sha256": sha_file(stdio / "manifest.json"),
        "stdio_train_accepted": len(new_train),
        "stdio_validation_accepted": len(valid),
        "stdio_train_families": list(TRAIN_FAMILIES),
        "stdio_validation_families": list(VALIDATION_FAMILIES),
        "steps": STEPS, "batch": BATCH, "lr": LR,
        "broad_exposures": STEPS * BROAD_PER_BATCH,
        "stdio_exposures": STEPS * STDIO_PER_BATCH,
        "samples_repeated_in_optimizer": False,
        "train_duration_seconds": training_seconds,
        "loss_history": history,
        "selected_step": best_step,
        "initial_validation_loss": round(initial_loss, 8),
        "selected_validation_loss": round(best_loss, 8),
        "model_promotion": False,
        "original_test_split_loaded": False,
        "public_exercism_tasks_used": False,
        "synthetic_project_authored_only": True,
        "human_reviewed_independent_benchmark": False,
        "behavioral_validation": "NOT_RUN",
        "generated_java_executed": False,
    }
    output.mkdir(parents=True)
    if best_step:
        checkpoint = output / "stdio-validation-candidate.pt"
        torch.save({
            "model": best_state, "config": config.to_dict(),
            "experiment": EXPERIMENT, "origin_sha256": PINNED_V018,
            "selected_step": best_step, "validation_loss": best_loss,
            "candidate_only_not_deployed": True,
        }, checkpoint)
        report["candidate_checkpoint_sha256"] = sha_file(checkpoint)

    if allow_oci or acknowledge_vm:
        if not (allow_oci and acknowledge_vm):
            raise PermissionError("Both OCI opt-in and ephemeral VM acknowledgment required")
        if not isinstance(image_id, str) or not re.fullmatch("sha256:[0-9a-f]{64}", image_id):
            raise ValueError("Locally pinned OCI image ID required")
        if not oci_image_ready("docker", image_id):
            raise RuntimeError("Local OCI engine or pinned JDK image not ready")
        smoke = run_java_in_oci(
            SMOKE, "PING\n", backend="docker", image=image_id,
            allow_execution=True, stdin_data="ping\n", timeout=22)
        if smoke["status"] != "PASS":
            raise RuntimeError("Trusted OCI Java stdin/stdout smoke failed BEFORE generated code")
        model.load_state_dict(before_state)
        model.eval()
        baseline = behavioral_validation(model, tokenizer, selected, image_id)
        model.load_state_dict(best_state)
        model.eval()
        candidate = behavioral_validation(model, tokenizer, selected, image_id)
        decision = conservative_gate(
            baseline, candidate, selected_step=best_step,
            initial_loss=initial_loss, selected_loss=best_loss)
        report.update({
            "status": "VALIDATION_STDIO_BEHAVIOR_EVALUATED",
            "behavioral_validation": "OCI_ATTEMPTED",
            "generated_java_executed": "BEST_EFFORT_OCI_ONLY",
            "oci_smoke": "PASS", "oci_image_id": image_id,
            "baseline_behavior": baseline, "selected_behavior": candidate,
            "gate": decision,
            "production_model_unchanged": True,
            "general_capability_established": False,
            "raw_generated_java_or_stdout_persisted": False,
        })
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--stdio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-oci", action="store_true")
    p.add_argument("--acknowledge-ephemeral-vm", action="store_true")
    p.add_argument("--oci-image-id")
    args = p.parse_args()
    result = experiment(
        args.base, args.tokenizer, args.canonical, args.stdio, args.output,
        allow_oci=args.allow_oci, acknowledge_vm=args.acknowledge_ephemeral_vm,
        image_id=args.oci_image_id,
    )
    print(json.dumps({
        "status": result["status"], "selected_step": result["selected_step"],
        "loss_history": result["loss_history"],
        "baseline_cases_passed": result.get("baseline_behavior", {}).get("cases_passed"),
        "selected_cases_passed": result.get("selected_behavior", {}).get("cases_passed"),
        "validation_gate": result.get("gate", {}).get("decision", "NOT_RUN"),
        "production_model_unchanged": True,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
