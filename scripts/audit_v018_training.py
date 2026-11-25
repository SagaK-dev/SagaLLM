"""Audit the pinned historical v0.18 SFT run without loading weights or running Java."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

REPORT = "9e86df9f571bf1196e52dfe3ab087162e46ef69fcf108e41216d8a7324318077"
BASE = "a17268d80ff443e4a0347f3a29df433556506f13537c78ce282836fa8fe4e5f9"
TOKENIZER = "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"
BEST = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
LAST = "74ae92a54ca864461ffff08d0db80191cbc88370f8a5347c099f182ea271e016"
DATA = "1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a"
PARENT_ZIP = "c25fa1833b64a620540d93067ce2c91a8c3916589dd5b565c8892661f9373581"
OUTPUT_ZIP = "b039250f6ebecb4f8631ba38050cf219eb41b53308ea8d0b13bb45f566477789"
ARCHIVED = Path("experiments/archive/reports/44994208434-v018-context-sft.json")
MAX_BYTES = 128 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(stream):
    h = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return h.hexdigest()
        size += len(block)
        require(size <= MAX_BYTES, "Input exceeds byte limit")
        h.update(block)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def integer(value):
    return type(value) is int and value >= 0


def loss(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def audit_report(raw):
    require(len(raw) <= 1024 * 1024 and hashlib.sha256(raw).hexdigest() == REPORT,
            "Historical report byte pin mismatch")
    return recount(json.loads(raw, object_pairs_hook=unique_object))


def recount(p):
    require(p["status"] == "TRAINED_CPU" and p["phase"] == "JAVA_INSTRUCTION_SFT"
            and p["termination"] == "STEP_LIMIT"
            and p["model_context_length"] == 384 and p["original_model_context_length"] == 192,
            "Unexpected training configuration")
    for key, pin in (("checkpoint_sha256_base", BASE), ("tokenizer_sha256", TOKENIZER),
                     ("best_checkpoint_sha256", BEST), ("last_checkpoint_sha256", LAST),
                     ("data_manifest_sha256", DATA)):
        require(p[key] == pin, "Report artifact pin mismatch")
    require(p["java_code_executed"] is False and p["java_behavior_test"] == "NOT_RUN"
            and p["java_coding_ability"] == "NOT_ESTABLISHED"
            and p["test_used_for_model_selection"] is False, "Unexpected evaluation claims")
    splits = p["original_split_counts"]
    require(set(splits) == {"train", "validation", "test"}, "Unexpected split coverage")
    for row in splits.values():
        keys = ("total", "accepted", "excluded_too_long", "families_accepted",
                "newly_admitted_beyond_original_context")
        require(all(integer(row[k]) for k in keys)
                and row["accepted"] + row["excluded_too_long"] == row["total"]
                and row["excluded_too_long"] == 0
                and row["families_accepted"] <= row["accepted"]
                and row["newly_admitted_beyond_original_context"] <= row["accepted"],
                "Inconsistent split counts")
    history = p["history"]
    require([x["step"] for x in history] == list(range(50, 401, 50))
            and p["optimizer_steps_requested"] == p["optimizer_steps_completed"] == 400,
            "Unexpected optimizer history")
    prior = 0
    for row in history:
        seen = row["distinct_train_examples_seen"]
        require(integer(seen) and prior <= seen <= min(4 * row["step"], splits["train"]["accepted"])
                and loss(row["validation_response_loss"]) and loss(row["last_train_response_loss"]),
                "Inconsistent training history")
        prior = seen
    require(p["distinct_training_examples_seen"] == prior, "Final exposure mismatch")
    require(loss(p["initial_validation_response_loss"]), "Invalid baseline loss")
    choices = [(p["initial_validation_response_loss"], 0, 0)]
    choices.extend((x["validation_response_loss"], x["step"], x["distinct_train_examples_seen"])
                   for x in history)
    best = min(choices)
    require(best == (p["best_validation_response_loss"], p["best_checkpoint_step"],
                     p["best_checkpoint_train_examples_seen"]), "Best selection mismatch")
    for k in ("heldout_test_response_loss_before_sft", "heldout_test_response_loss_best_sft",
              "heldout_test_response_loss_last_sft"):
        require(loss(p[k]), "Invalid heldout loss")
    counts = {}
    identities = []
    for label in ("base", "best"):
        row = p["compile_only_" + label]
        cases = row["cases"]
        require(row["status"] == "SCORED_COMPILE_ONLY" and row["behavior"] == "NOT_RUN"
                and len(cases) == row["evaluated"] == 8, "Invalid compile-only coverage")
        ids = [(x["id"], x["family"]) for x in cases]
        require(len({x[0] for x in ids}) == len(ids), "Repeated case ID")
        require(all(x["compile"] in ("PASS", "FAIL") and type(x["exact"]) is bool
                    and re.fullmatch("[0-9a-f]{64}", x["generated_sha256"]) for x in cases),
                "Invalid compile case")
        compiled = sum(x["compile"] == "PASS" for x in cases)
        exact = sum(x["exact"] for x in cases)
        require((compiled, exact) == (row["compiled"], row["exact_reference_matches"]),
                "Compile totals mismatch")
        identities.append(ids)
        counts[label] = {"compiled": compiled, "exact": exact, "evaluated": len(cases)}
    require(identities[0] == identities[1], "Unmatched compile cases")
    return {"status": "REPORT_INVARIANTS_PASS", "run_id": 44994208434,
            "run_commit": "c531d43d1d1d2e9da18a14045bfced3cfcaa2720",
            "report_sha256": REPORT, "optimizer_steps": 400,
            "best_step": best[1], "best_seen": best[2], "last_seen": prior,
            "compile_only": counts, "last_compile_only": "NOT_RECORDED",
            "context_stratified_compile": "NOT_RECORDED",
            "training_lineage_verified": False, "independent_evaluation": "NOT_RUN",
            "java_behavior": "NOT_RUN", "model_selection_permitted": False}


def audit_zip(path, archive_pin, members):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES,
            "Invalid archive")
    with path.open("rb") as stream:
        require(digest(stream) == archive_pin, "Archive byte pin mismatch")
        stream.seek(0)
        with zipfile.ZipFile(stream) as z:
            require(len(z.infolist()) <= 32, "Too many archive members")
            for name, pin in members.items():
                rows = [x for x in z.infolist() if x.filename == name]
                require(len(rows) == 1 and not rows[0].is_dir()
                        and 0 < rows[0].file_size <= MAX_BYTES, "Missing or invalid member")
                with z.open(rows[0]) as member:
                    require(digest(member) == pin, "Member byte pin mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ARCHIVED)
    parser.add_argument("--parent-zip", type=Path)
    parser.add_argument("--output-zip", type=Path)
    args = parser.parse_args()
    require((args.parent_zip is None) == (args.output_zip is None), "Both archives required")
    with args.report.open("rb") as f:
        result = audit_report(f.read(1024 * 1024 + 1))
    result["artifact_bytes"] = "NOT_CHECKED"
    if args.parent_zip is not None:
        audit_zip(args.parent_zip, PARENT_ZIP, {
            "java-sft-model/java-sft-best.pt": BASE,
            "java-sft-model/tokenizer.json": TOKENIZER,
        })
        audit_zip(args.output_zip, OUTPUT_ZIP, {
            "java-context-model/java-sft-best.pt": BEST,
            "java-context-model/java-sft-last.pt": LAST,
            "java-context-model/tokenizer.json": TOKENIZER,
            "java-context-model/train-report.json": REPORT,
            "java-context-data/manifest.json": DATA,
        })
        result["artifact_bytes"] = "PINNED_MEMBERS_VERIFIED"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
