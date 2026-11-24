"""M1: SHA-pinned, inference-only Java conditioning diagnostic.

Never compile or execute generated Java. Only aggregate metrics, hashes and
fixed categories leave memory. Prior behavior counts are historical evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

BASE = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
CANDIDATE = "379d57d6d8668995dc34fb310e0c5140041c5fd3e78b77932712785e96b01eae"
TOKENIZER = "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"
PRIOR = "1f59f98acafc5b2939f5dc043717beb3b76dd5ff923506064377ca8a7221c468"
FAMILIES = ("stdin_min_offset", "stdin_count_odd_offset", "stdin_ascii_lower_suffix")
TASKS = tuple(f"validation-{family}-{i:03d}" for family in FAMILIES for i in (2, 3))
ARMS = ("base_v018", "control_interleaved", "contrastive_same_variant")
CAP = 170
CATEGORIES = {"PASS", "OUTPUT_MISMATCH_ZERO_EXIT", "NONZERO_EXIT_OR_JAVAC_ERROR",
              "TIMEOUT", "OUTPUT_LIMIT", "REJECTED", "NOT_RUN", "RUNTIME_ERROR",
              "NOT_RUN_INVALID_GENERATION"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pinned(path: Path, expected: str, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing or unsafe pinned input")
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError("Pinned input SHA-256 mismatch")
    return data


def historical_counts(report: dict) -> dict:
    if report["experiment"] != "JAVA_CONTRASTIVE_STDIO_MATCHED_AB_V1":
        raise ValueError("Wrong historical experiment")
    if report["holdback_behavior_task_ids"] != list(TASKS):
        raise ValueError("Unexpected historical task set")
    by_arm = {"base_v018": report["baseline_behavior"],
              **{key: value["behavior"] for key, value in report["arms_results"].items()}}
    if set(by_arm) != set(ARMS):
        raise ValueError("Missing historical arm")
    result = {}
    for name in ARMS:
        b = by_arm[name]
        if [r["id"] for r in b["cases"]] != list(TASKS) or b["total_cases"] != 18:
            raise ValueError("Historical coverage mismatch")
        counts = Counter()
        for row in b["cases"]:
            cats = row["case_categories"]
            if len(cats) != 3 or not set(cats) <= CATEGORIES:
                raise ValueError("Historical categories invalid")
            if row["cases_passed"] != cats.count("PASS"):
                raise ValueError("Historical score mismatch")
            counts.update(cats)
        if counts["PASS"] != b["cases_passed"]:
            raise ValueError("Historical total mismatch")
        result[name] = {"case_categories": dict(sorted(counts.items())),
                        "cases_passed": counts["PASS"],
                        "oci_invocations": b["oci_invocations"]}
    return result


def stop_reason(ids: list[int], eos: int, budget: int) -> str:
    if not ids or not 1 <= len(ids) <= budget:
        raise ValueError("Unexpected generation length")
    if eos in ids:
        if ids.index(eos) != len(ids) - 1:
            raise ValueError("Unexpected tokens after EOS")
        return "EOS"
    return "TOKEN_LIMIT" if len(ids) == budget else "UNEXPLAINED_SHORT_STOP"


def shape(source: str) -> dict:
    # These are literal markers, not a Java lexer or a correctness proof.
    return {"main_class_marker": "public class Main" in source,
            "stdin_marker": "System.in" in source,
            "stdout_marker": "System.out" in source,
            "nul_present": "\x00" in source,
            "replacement_character_present": "\ufffd" in source,
            "raw_brace_counts_equal": source.count("{") == source.count("}"),
            "markdown_fence_present": "\x60\x60\x60" in source,
            "source_bytes": len(source.encode("utf8"))}


def token_hash(ids: list[int]) -> str:
    return digest(json.dumps(ids, separators=(",", ":")).encode("ascii"))


def common_prefix(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def probe(args) -> dict:
    import torch
    import torch.nn.functional as F
    from sagallm.config import ModelConfig
    from sagallm.model import SagaTransformer
    from sagallm.tokenization import load_tokenizer
    from sagallm.java_instruction_sft import encode_row, compact_prompt
    from sagallm.java_stdio_curriculum import expected_rows
    from sagallm.training import verify_checkpoint_tokenizer

    torch.set_num_threads(2)
    torch.manual_seed(4201)
    read_pinned(args.tokenizer, TOKENIZER, 2_000_000)
    prior = json.loads(read_pinned(args.prior, PRIOR, 100_000))
    counts = historical_counts(prior)
    tokenizer = load_tokenizer(args.tokenizer)
    raw = {r["id"]: r for r in expected_rows("validation")}
    rows = [encode_row(raw[k], tokenizer, 384) for k in TASKS]
    if any(r is None for r in rows):
        raise ValueError("Fixed validation row does not fit")
    eos, bos = tokenizer.SPECIAL["<eos>"], tokenizer.SPECIAL["<bos>"]
    prefixes = [r["prefix_ids"] for r in rows]
    if len({tuple(p) for p in prefixes}) != len(TASKS):
        raise ValueError("Distinct prompts collapsed at tokenization")
    for r in rows:
        if tokenizer.decode(r["prefix_ids"][1:]) != compact_prompt(r["task"]):
            raise ValueError("Prompt tokenizer round-trip mismatch")
        if r["prefix_ids"][0] != bos:
            raise ValueError("Unexpected BOS")
    reference_tokens = [tokenizer.encode(r["source"]) + [eos] for r in rows]
    prompt_metrics = [{"id": r["id"], "prefix_tokens": len(r["prefix_ids"]),
                       "prefix_sha256": token_hash(r["prefix_ids"]),
                       "reference_tokens": len(ref)}
                      for r, ref in zip(rows, reference_tokens)]
    outcomes = {}
    for name, path, sha in (("base_v018", args.base, BASE),
                            ("contrastive_same_variant", args.candidate, CANDIDATE)):
        read_pinned(path, sha, 90 * 1024 * 1024)
        state = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ModelConfig(**state["config"])
        if cfg.max_position_embeddings != 384 or cfg.parameter_count() > 3_000_000:
            raise ValueError("Unexpected model configuration")
        if tokenizer.vocab_size > cfg.vocab_size:
            raise ValueError("Unexpected vocabulary")
        if name == "base_v018":
            verify_checkpoint_tokenizer(state, args.tokenizer)
        elif (state.get("base_sha256") != BASE or
              state.get("experiment") != "JAVA_CONTRASTIVE_STDIO_MATCHED_AB_V1" or
              state.get("is_production_model") is not False):
            raise ValueError("Unexpected candidate provenance")
        model = SagaTransformer(cfg)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        del state
        previous = (prior["baseline_behavior"] if name == "base_v018" else
                    prior["arms_results"][name]["behavior"])
        old_hashes = {r["id"]: r.get("source_sha256") for r in previous["cases"]}
        observations, generated = [], []
        with torch.inference_mode():
            for i, row in enumerate(rows):
                prefix, target = prefixes[i], reference_tokens[i]
                budget = min(CAP, cfg.max_position_embeddings - len(prefix))
                ids = model.generate(torch.tensor([prefix], dtype=torch.long),
                                     max_new_tokens=budget, temperature=0.0,
                                     eos_token_id=eos, vocab_limit=tokenizer.vocab_size,
                                     deadline_monotonic=time.monotonic()+25)[0, len(prefix):].tolist()
                source = tokenizer.decode(ids)
                source_sha = digest(source.encode("utf8"))
                generated.append(ids)

                # Same target tokens, different prompt only (same variant,
                # next algorithm family). Length-normalized teacher forcing.
                j = (i+2) % len(rows)
                def nll(p):
                    seq = p + target
                    if len(seq)-1 > cfg.max_position_embeddings:
                        raise ValueError("Teacher-forced sequence exceeds context")
                    logits = model(torch.tensor([seq[:-1]], dtype=torch.long)).logits
                    values = logits[0, len(p)-1:, :].float()
                    v = float(F.cross_entropy(values, torch.tensor(target)))
                    if not math.isfinite(v):
                        raise ValueError("Nonfinite teacher-forced loss")
                    return v
                own, swapped = nll(prefix), nll(prefixes[j])
                observations.append({
                    "id": row["id"], "generation_budget": budget,
                    "generated_tokens": len(ids), "stop": stop_reason(ids, eos, budget),
                    "source_sha256": source_sha, "historical_source_sha_matches": source_sha == old_hashes[row["id"]],
                    "shape_markers_not_parser": shape(source),
                    "swapped_prompt_id": rows[j]["id"],
                    "own_reference_nll": round(own, 8),
                    "swapped_prompt_same_reference_nll": round(swapped, 8),
                    "swapped_minus_own_nll": round(swapped-own, 8),
                })
        cross = [{"a": TASKS[i], "b": TASKS[j],
                  "common_generated_prefix_tokens": common_prefix(generated[i], generated[j])}
                 for i in range(6) for j in range(i+1, 6) if i//2 != j//2]
        outcomes[name] = {"checkpoint_sha256": sha, "observations": observations,
                          "cross_family_prefix_comparisons": cross,
                          "unique_source_hashes": len({x["source_sha256"] for x in observations}),
                          "historical_hash_matches": sum(x["historical_source_sha_matches"] for x in observations)}
        del model
    return {"status": "INFERENCE_DIAGNOSTIC_COMPLETE",
            "historical_run_id": 45167198398, "historical_report_sha256": PRIOR,
            "historical_behavior_only": counts, "tokenizer_sha256": TOKENIZER,
            "torch_version": str(torch.__version__), "prompt_metrics": prompt_metrics,
            "fixed_task_ids": list(TASKS), "generation_cap": CAP,
            "models": outcomes, "control_checkpoint": "NOT_AVAILABLE_IN_PRIOR_ARTIFACT",
            "new_java_compilation": "NOT_RUN", "new_java_execution": "NOT_RUN",
            "test_split_accessed": False, "model_training": "NOT_RUN",
            "production_promotion": False, "independent_benchmark": "NOT_ESTABLISHED",
            "root_cause": "NOT_ESTABLISHED"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("base", "candidate", "tokenizer", "prior", "output"):
        p.add_argument("--"+flag, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite evidence")
    result = probe(args)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"],
                      "models": {k: {"unique_source_hashes": v["unique_source_hashes"],
                                     "historical_hash_matches": v["historical_hash_matches"],
                                     "stops": dict(Counter(x["stop"] for x in v["observations"])),
                                     "positive_conditioning_margins": sum(x["swapped_minus_own_nll"] > 0 for x in v["observations"])}
                                 for k,v in result["models"].items()},
                      "root_cause": result["root_cause"]}, sort_keys=True))


if __name__ == "__main__":
    main()
