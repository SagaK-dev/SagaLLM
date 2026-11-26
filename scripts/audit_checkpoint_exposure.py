"""Build a partial, report-declared checkpoint map; never an intake approval."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from scripts.audit_development_data import PINS
from scripts.audit_v018_training import BEST, BASE, DATA, require, unique_object
from scripts.audit_java_lineage import ROOT

ARCHIVE = Path("experiments/archive/reports")
DEVELOPMENT_ZIP = "dbc9702411141ee1b5b684a1bf903ad117fa6ba28080d83485d4119b64e75392"
SPECS = (
    (45373241422, "m3-ranking-v2", "90c8b9bb12e381c415e798d88875fc401143a0c703a3b0424cfab6c2283932cd",
     "babcbe130905d4acc7e4617497f0ebcea68569a9", "ranking",
     "MATCHED_CPU_TRAINING_AND_INFERENCE_COMPLETE"),
    (45377422360, "m3-parameter-ce", "26f143575842073c9b4119ecefcb3b50935891bba6c4c9fc60b0b0c8353a89b7",
     "fdf53625d8e64596748113b53fd6f4c63e2a1c74", "parameter_ce",
     "PARAMETER_CE_CPU_TRAINING_AND_INFERENCE_COMPLETE"),
)
SEEDS = (4201, 4202, 4203)
SHA = re.compile(r"[0-9a-f]{64}\Z")


def read_pinned(path, expected, limit):
    require(path.is_file() and not path.is_symlink(), "Invalid evidence path")
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    require(0 < len(raw) <= limit and hashlib.sha256(raw).hexdigest() == expected,
            "Evidence byte pin mismatch")
    return raw


def read_reconstruction(path):
    import io
    raw = read_pinned(path, DEVELOPMENT_ZIP, 1024 * 1024)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        rows = z.infolist()
        require(len(rows) == 1 and rows[0].filename == "development-data-evidence.json"
                and 0 < rows[0].file_size <= 1024 * 1024, "Unexpected evidence member")
        with z.open(rows[0]) as f:
            body = f.read(1024 * 1024 + 1)
        require(len(body) <= 1024 * 1024, "Oversized evidence member")
    data = json.loads(body, object_pairs_hook=unique_object)
    require(data["status"] == "DECLARED_DEVELOPMENT_DATA_PINS_MATCH"
            and data["pins"] == PINS and data["actual_exposure_verified"] is False
            and data["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and data["independent_evaluation"] == "NOT_RUN"
            and data["model_selection_permitted"] is False, "Reconstruction gate mismatch")
    return hashlib.sha256(body).hexdigest()


def load_reports(root):
    return [json.loads(read_pinned(root / f"{run}-{slug}.json", pin, 8 * 1024 * 1024),
                       object_pairs_hook=unique_object)
            for run, slug, pin, _, _, _ in SPECS]


def source(key, group, role, kind, sha=None):
    return {"source": key, "inventory_group": group, "role": role,
            "pin_kind": kind, "sha256": sha, "shard_ready": False}


def map_reports(reports):
    require(len(reports) == len(SPECS), "Wrong report count")
    seen = {ROOT, BASE, BEST}
    entries = []
    for report, (run, slug, pin, commit, treatment, status) in zip(reports, SPECS):
        require(report["status"] == status and report["base_sha256"] == BEST
                and report["tokenizer_sha256"] == PINS["tokenizer"]
                and report["broad_train_sha256"] == PINS["broad_train"]
                and report["stdio_manifest_sha256"] == PINS["stdio_manifest"]
                and report["seeds"] == list(SEEDS)
                and set(report["results"]) == {str(s) for s in SEEDS}
                and report["production_promotion"] is False
                and report["original_test_records_read"] is False, "Report scope mismatch")
        fresh_flag = ("fresh_used_in_training_or_selection" if treatment == "ranking"
                      else "fresh_used_for_training_or_selection")
        require(report[fresh_flag] is False, "Holdback role mismatch")
        sources = [
            source("v016_public_corpus", "pretraining", "inherited_declared", "manifest_not_in_this_map"),
            source("v017_v018_sft_train", "train", "inherited_declared", "manifest", DATA),
            source("v017_v018_sft_validation", "validation", "inherited_selection_context", "manifest", DATA),
            source("v017_v018_sft_test", "test", "inherited_evaluation_context", "manifest", DATA),
            source("broad_train", "train", "declared_training_pool", "jsonl", PINS["broad_train"]),
            source("stdio_train", "train", "declared_training_pool", "manifest", PINS["stdio_manifest"]),
            source("stdio_selection", "validation", "declared_selection_subset", "manifest", PINS["stdio_manifest"]),
        ]
        if treatment == "ranking":
            require(report["holdback_sha256"] == PINS["previous_holdback"], "Ranking holdback mismatch")
            sources.append(source("previous_holdback", "development", "reported_diagnostic",
                                  "json_suite", PINS["previous_holdback"]))
            sources.append(source("ranking_auxiliary", "train", "declared_auxiliary",
                                  "not_reconstructed_at_ranking_revision"))
        else:
            for key, field in (("previous_holdback", "previous_suite_sha256"),
                               ("fresh_holdback", "fresh_suite_sha256"),
                               ("pair_manifest", "pair_manifest_sha256")):
                require(report[field] == PINS[key], "Parameter CE pin mismatch")
            sources.extend([
                source("previous_holdback", "development", "prior_exposed_suite_reference", "json_suite", PINS["previous_holdback"]),
                source("fresh_holdback", "development", "reported_diagnostic", "json_suite", PINS["fresh_holdback"]),
                source("pair_manifest", "train", "declared_auxiliary", "derived_manifest", PINS["pair_manifest"]),
            ])
        for seed in SEEDS:
            arms = report["results"][str(seed)]
            require(set(arms) == {"control", treatment}, "Arm coverage mismatch")
            for arm in ("control", treatment):
                row = arms[arm]
                checkpoint = row["checkpoint_sha256"]
                require(isinstance(checkpoint, str) and SHA.fullmatch(checkpoint)
                        and checkpoint not in seen, "Invalid or duplicate checkpoint")
                seen.add(checkpoint)
                require(type(row["coefficient"]) is int and row["coefficient"] == int(arm == treatment)
                        and type(row["selected_step"]) is int
                        and 0 <= row["selected_step"] <= report["steps_per_arm"],
                        "Arm metadata mismatch")
                entries.append({"run_id": run, "report_sha256": pin, "training_code_sha_declared": commit,
                    "seed": seed, "arm": arm, "coefficient": row["coefficient"],
                    "selected_step": row["selected_step"], "checkpoint_sha256": checkpoint,
                    "parent_checkpoint_sha256_declared": BEST,
                    "sources": sources, "checkpoint_bytes_verified_here": False,
                    "actual_exposure_verified": False, "inventory_ready": False})
    return {"schema": "sagallm-partial-checkpoint-exposure-map-v1",
            "status": "PARTIAL_REPORT_DECLARED_MAP", "checkpoints": entries,
            "checkpoint_count": len(entries), "ancestor_checkpoint_links": [ROOT, BASE, BEST],
            "inventory_target_schema": "sagallm-private-java-exposure-v1",
            "missing_evidence": ["exact checkpoint-specific optimizer sample order",
                "exact per-split JSONL inventory and verified conversion of suites",
                "ranking auxiliary reconstruction at its own historical revision",
                "all other experiments and development probes", "rights review"],
            "full_exposure_inventory": "NOT_ESTABLISHED", "training_lineage_verified": False,
            "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reconstruction-zip", type=Path, required=True)
    p.add_argument("--reports", type=Path, default=ARCHIVE)
    args = p.parse_args()
    evidence = read_reconstruction(args.reconstruction_zip)
    result = map_reports(load_reports(args.reports))
    result["reconstruction_evidence"] = {"run_id": 45790130406, "artifact_id": 15101943417,
        "archive_sha256": DEVELOPMENT_ZIP, "member_sha256": evidence}
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
