"""Rebuild historical input/schedule metadata, never load a checkpoint or run Java.

Run inside the exact historical worktree for each mode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

REVISIONS = {
    "ab": "40dcce6fe2e4bd2ab83ea70cf5a9cd12368c0312",
    "m1": "6f36acc0cfb856e82788660acef39829f91586b4",
    "blocked": "6eb20c4d581c3aaaf5db4f91a9c4517d2c2bbaca",
}
REPORTS = {
    "ab": ("45167198398-contrastive-stdio-ab.json", "1f59f98acafc5b2939f5dc043717beb3b76dd5ff923506064377ca8a7221c468"),
    "m1": ("45369765716-m1-conditioning.json", "191a4699aeabff3a5bd36143aa3379170b0f51c8fcb0fe23aa68f7223e583e07"),
    "rerun": ("45369987553-m1-conditioning-rerun.json", "3e19d9f621c8f46011cad1560d596ad4f2f6743124757b93c8783b8012111c92"),
    "blocked": ("45372372592-m3-alignment-v1-blocked.json", "ed9e5b9fafa0b0c6c5c193ddb0ae852487346af1f145a680acc5896a5f5675d6"),
}
TOKENIZER = "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"
STDIO = "0cea8ded8a5c03d15b2b516a86665901266b0b375095579daa8d712661316a13"
BROAD = "181346d5395efdd2ef376730321e1585d453b2b09a822302f6f2d03137e1aa8d"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def pin(path, digest, limit=2_000_000):
    require(path.is_file() and not path.is_symlink(), "Unsafe pinned path")
    with path.open("rb") as f:
        body = f.read(limit + 1)
    require(0 < len(body) <= limit and sha(body) == digest, "Pinned bytes mismatch")
    return body


def report(root, mode):
    name, digest = REPORTS[mode]
    return json.loads(pin(root / name, digest))


def encoded(rows, tokenizer):
    from sagallm.java_instruction_sft import encode_row
    out = [encode_row(r, tokenizer, 384) for r in rows]
    require(all(r is not None for r in out) and len({r["id"] for r in out}) == len(out),
            "Context admission or unique-ID mismatch")
    return out


def rebuild_ab(args, tokenizer, old):
    from sagallm.java_diverse_corpus import build_java_diverse_corpus, verify_java_diverse_corpus
    from sagallm.java_stdio_curriculum import build_curriculum, expected_rows, verify_curriculum
    from sagallm.java_instruction_sft import read_split
    from scripts.train_java_stdio_semantic_sft import train_schedule
    from scripts.train_java_contrastive_stdio_ab import contrastive_schedule, schedule_evidence, fixed_behavior_only
    build_java_diverse_corpus(args.root / "broad")
    verify_java_diverse_corpus(args.root / "broad")
    pin(args.root / "broad/train.jsonl", BROAD, 90_000_000)
    build_curriculum(args.root / "stdio")
    verify_curriculum(args.root / "stdio")
    pin(args.root / "stdio/manifest.json", STDIO)
    broad, stats = read_split(args.root / "broad", "train", tokenizer, 384)
    require(stats == old["canonical_train_stats"], "Broad admission stats mismatch")
    train, valid = (encoded(expected_rows(split), tokenizer) for split in ("train", "validation"))
    require(len(train) == 320 and len(valid) == 36, "STDIO admission mismatch")
    holdback, selection = fixed_behavior_only(valid)
    holdback_ids = [r["id"] for r, _ in holdback]
    selection_ids = [r["id"] for r in selection]
    require(holdback_ids == old["holdback_behavior_task_ids"]
            and len(set(selection_ids)) == old["loss_selection_rows"] == 24
            and not set(selection_ids).intersection(holdback_ids), "Holdback/selection mismatch")
    control = train_schedule(broad, train)
    intervention = contrastive_schedule(broad, train, control)
    require(schedule_evidence(control, intervention) == old["schedule"], "Schedule parity mismatch")
    def ids(schedule):
        return [[[kind, (broad if kind == "broad" else train)[i]["id"]] for kind, i in batch]
                for batch in schedule]
    return {"schedule_declarations_match": True, "broad_train_sha256": BROAD,
            "stdio_manifest_sha256": STDIO, "canonical_train_stats": stats,
            "derived_schedule_sha256": {"control_interleaved": sha(canonical(ids(control))),
                                         "contrastive_same_variant": sha(canonical(ids(intervention)))},
            "derived_selection_ids_sha256": sha(canonical(selection_ids)),
            "selection_rows": len(selection_ids), "holdback_rows": len(holdback_ids),
            "historical_schedule_digest_recorded": False}


def rebuild_m1(args, tokenizer, old):
    from sagallm.java_stdio_curriculum import expected_rows
    from sagallm.java_instruction_sft import compact_prompt
    from scripts.diagnose_java_m1_conditioning import TASKS, token_hash
    lookup = {r["id"]: r for r in expected_rows("validation")}
    rows = encoded([lookup[k] for k in TASKS], tokenizer)
    eos, bos = tokenizer.SPECIAL["<eos>"], tokenizer.SPECIAL["<bos>"]
    metrics = []
    for r in rows:
        prefix = r["prefix_ids"]
        require(prefix[0] == bos and tokenizer.decode(prefix[1:]) == compact_prompt(r["task"]),
                "Prefix round-trip mismatch")
        metrics.append({"id": r["id"], "prefix_tokens": len(prefix), "prefix_sha256": token_hash(prefix),
                        "reference_tokens": len(tokenizer.encode(r["source"]) + [eos])})
    rerun = report(args.reports, "rerun")
    require(metrics == old["prompt_metrics"] == rerun["prompt_metrics"]
            and len({m["prefix_sha256"] for m in metrics}) == 6, "Historical prompt metric mismatch")
    return {"prompt_metrics_reconstructed": len(metrics), "report_comparisons": 12,
            "derived_prompt_metrics_sha256": sha(canonical(metrics)),
            "rerun_code_identity_checked_by_workflow": True, "generation_replayed": False}


def rebuild_blocked(args, tokenizer, old):
    from scripts import train_java_m3_ranking as rank
    rank.frozen_suite(Path("examples/java_m3_development_holdback.json"))
    rank.build_curriculum(args.root / "stdio")
    pin(args.root / "stdio/manifest.json", STDIO)
    broad = encoded(rank.broad_rows(), tokenizer)
    train = encoded(rank.expected_rows("train"), tokenizer)
    valid = encoded(rank.expected_rows("validation"), tokenizer)
    require(len(broad) == 17080 and len(train) == 320 and len(valid) == 36, "Admission mismatch")
    _, _, admission = rank.pairs(train, tokenizer.SPECIAL["<eos>"])
    require(admission == old["alignment"] and admission["passed"] is False
            and old["optimizer_steps_completed"] == 0, "Blocked mask reconstruction mismatch")
    return {"alignment": admission, "new_checkpoint_produced": False,
            "historical_mask_digest_matched": True, "broad_train_sha256": BROAD,
            "stdio_manifest_sha256": STDIO}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=REVISIONS, required=True)
    for key in ("reports", "tokenizer", "root"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(revision == REVISIONS[a.mode], "Historical revision mismatch")
    pin(a.tokenizer, TOKENIZER)
    from sagallm.tokenization import load_tokenizer
    tokenizer = load_tokenizer(a.tokenizer)
    a.root.mkdir()
    old = report(a.reports, a.mode)
    fn = {"ab": rebuild_ab, "m1": rebuild_m1, "blocked": rebuild_blocked}[a.mode]
    details = fn(a, tokenizer, old)
    print(json.dumps({"schema": "sagallm-early-input-reconstruction-v1",
                      "mode": a.mode, "revision": revision, "status": "RECONSTRUCTED_SCOPED_INPUT_METADATA",
                      "report_sha256": REPORTS[a.mode][1], "tokenizer_sha256": TOKENIZER,
                      "details": details, "model_loaded": False, "new_training": False,
                      "new_inference": False, "java_executed": False,
                      "actual_exposure_verified": False, "independent_evaluation": "NOT_RUN",
                      "model_selection_permitted": False}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
