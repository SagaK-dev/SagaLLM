"""Integrate archived declarations without claiming actual model-specific exposure."""
from scripts.audit_ranking_auxiliary import pin, parse, require
from scripts.audit_replay_activation_scope import SPECS as REPLAY_SPECS, models, hexsha
from scripts.audit_v018_training import BASE as V017, BEST, LAST, DATA, REPORT as V018_REPORT
from scripts.audit_java_lineage import ROOT

PATHS = {
    "public/train.jsonl": "pretraining",
    "public/validation.jsonl": "validation", "public/test.jsonl": "test",
    "sft/train.jsonl": "train", "sft/validation.jsonl": "validation", "sft/test.jsonl": "test",
    "stdio/train.jsonl": "train", "stdio/validation.jsonl": "validation",
    "derived/stdio_validation_oracles.jsonl": "validation",
    "derived/previous_holdback.jsonl": "development", "derived/fresh_holdback.jsonl": "development",
    "derived/ranking_auxiliary.jsonl": "train",
    "derived/parameter_ce_train.jsonl": "train",
    "derived/parameter_ce_validation.jsonl": "development",
    "derived/parameter_ce_fresh.jsonl": "development",
}
COVERAGE = {
    "v018-context-sft": "PINNED_REPORT_AND_ANCESTOR_CORPUS_SCOPE",
    "contrastive-stdio-ab": "HISTORICAL_REPORT_RECOUNT",
    "m1-conditioning": "HISTORICAL_PROMPT_AND_GENERATION_DECLARATIONS",
    "m1-conditioning-rerun": "HISTORICAL_PROMPT_AND_GENERATION_DECLARATIONS",
    "m3-alignment-v1-blocked": "BLOCKED_PREFLIGHT_REPORT_NO_NEW_CHECKPOINT",
    "m3-ranking-v2": "PRIOR_AUXILIARY_AND_SCHEDULE_RECONSTRUCTION",
    "m3-frozen-replay": "PRIOR_REPLAY_METADATA_RECOUNT",
    "m3-parameter-binding": "PRIOR_PAIR_IDENTITY_COMPARISON",
    "m3-parameter-ce": "PRIOR_PAIR_AND_SCHEDULE_RECONSTRUCTION",
    "m3-prompt-signal": "PRIOR_PROBE_IDENTITY_COMPARISON",
    "m3-activation-swap": "PRIOR_EQUAL_LENGTH_PAIR_COMPARISON",
    "m3-prompt-bias": "PRIOR_PROBE_IDENTITY_COMPARISON",
}
ANCESTOR = [
    ("public/train.jsonl", "pretraining_pool"),
    ("public/validation.jsonl", "validation_context"),
    ("public/test.jsonl", "evaluation_context"),
]
SFT = [
    ("sft/train.jsonl", "sft_training_pool"),
    ("sft/validation.jsonl", "selection_context"),
    ("sft/test.jsonl", "evaluation_context"),
]
STDIO = [
    ("sft/train.jsonl", "broad_train_pool"),
    ("stdio/train.jsonl", "stdio_train_pool"),
    ("stdio/validation.jsonl", "selection_and_development_context"),
    ("derived/stdio_validation_oracles.jsonl", "behavior_oracle_context"),
]
RANK = [
    ("derived/ranking_auxiliary.jsonl", "auxiliary_input_screening_text"),
    ("derived/previous_holdback.jsonl", "development_probe_context"),
    ("derived/parameter_ce_train.jsonl", "shared_binding_train_pair_development_context"),
    ("derived/parameter_ce_validation.jsonl", "shared_binding_pair_development_context"),
]
CE = [
    ("derived/previous_holdback.jsonl", "development_probe_context"),
    ("derived/fresh_holdback.jsonl", "development_probe_context"),
    ("derived/parameter_ce_train.jsonl", "auxiliary_input_screening_text"),
    ("derived/parameter_ce_validation.jsonl", "development_pair_screening_text"),
    ("derived/parameter_ce_fresh.jsonl", "development_pair_screening_text"),
]


def inventory_map(inventory):
    require(inventory["schema"] == "sagallm-private-java-exposure-v1", "Inventory schema mismatch")
    entries = inventory["files"]
    require(len(entries) == len(PATHS) and len({r["path"] for r in entries}) == len(PATHS)
            and {r["path"]: r["group"] for r in entries} == PATHS, "Inventory path/group drift")
    for row in entries:
        require(hexsha(row["sha256"]) and row["format"] in {"chat-jsonl", "text-jsonl"},
                "Invalid inventory source pin")
    return {r["path"]: r for r in entries}


def source_assignments(entries, pairs):
    return [{"path": p, "sha256": entries[p]["sha256"], "group": entries[p]["group"],
             "role": role, "scope": "CONSERVATIVE_DECLARATION_NOT_ACTUAL_EXPOSURE"}
            for p, role in pairs]


def archived(root, spec):
    run, slug, digest = spec
    return parse(pin(root / "reports" / f"{run}-{slug}.json", digest, 8 * 1024 * 1024))


def integrate(root, scope, inventory, ledger, early):
    from scripts.audit_remaining_historical_scope import CANDIDATE, SPECS
    entries = inventory_map(inventory)
    reports = ledger["reports"]
    require(len(reports) == len(COVERAGE) == len({r["slug"] for r in reports})
            and {r["slug"] for r in reports} == set(COVERAGE), "Report ledger coverage mismatch")
    coverage = [{"slug": r["slug"], "run_id": r["run_id"], "artifact_id": r["artifact_id"],
                 "report_sha256": r["report_sha256"], "scope": COVERAGE[r["slug"]],
                 "actual_exposure_verified": False} for r in reports]
    end = archived(root, (44994208434, "v018-context-sft", V018_REPORT))
    require(end["best_checkpoint_sha256"] == BEST and end["last_checkpoint_sha256"] == LAST
            and end["checkpoint_sha256_base"] == V017 and end["data_manifest_sha256"] == DATA
            and end["test_used_for_model_selection"] is False, "v0.18 declaration mismatch")
    rank = archived(root, REPLAY_SPECS[0])
    ce = archived(root, REPLAY_SPECS[4])
    ab = archived(root, SPECS[0])
    rank_models = models(rank, "ranking")
    ce_models = models(ce, "parameter_ce")
    require({r["model"]: r["checkpoint_sha256_declared_by_ranking"] for r in scope["ranking_models"]}
            == {k: v[0] for k, v in rank_models.items()}
            and {r["model"]: r["checkpoint_sha256_declared_by_ce"] for r in scope["ce_models"]}
            == {k: v[0] for k, v in ce_models.items()}, "Prior model scope mismatch")
    result = []
    def add(name, cp, parent, step, pairs, run, arm=None):
        require(hexsha(cp) and cp not in {r["checkpoint_sha256"] for r in result}, "Repeated checkpoint")
        result.append({"model": name, "checkpoint_sha256": cp,
                       "parent_checkpoint_sha256_declared": parent, "selected_or_last_step_declared": step,
                       "source_report_run_id": run, "arm": arm,
                       "sources": source_assignments(entries, pairs),
                       "actual_exposure_verified": False, "checkpoint_bytes_verified_here": False,
                       "inventory_ready": False})
    add("v016-best", ROOT, None, None, ANCESTOR, None)
    add("v017-best", V017, ROOT, None, ANCESTOR + SFT, None)
    add("v018-best", BEST, V017, end["best_checkpoint_step"], ANCESTOR + SFT, 44994208434)
    add("v018-last", LAST, V017, end["optimizer_steps_completed"], ANCESTOR + SFT, 44994208434)
    add("contrastive-stdio-ab-candidate", CANDIDATE, BEST,
        ab["arms_results"]["contrastive_same_variant"]["training"]["selected_step"],
        ANCESTOR + SFT + STDIO, SPECS[0][0], "contrastive_same_variant")
    for label, records, additional, spec in (("ranking", rank_models, RANK, REPLAY_SPECS[0]),
                                                ("parameter-ce", ce_models, CE, REPLAY_SPECS[4])):
        for name, (cp, _, row) in records.items():
            if name != "base":
                add(label + "-" + name, cp, BEST, row["selected_step"],
                    ANCESTOR + SFT + STDIO + additional, spec[0], name.split("-", 1)[1])
    require(len(result) == 17, "Unexpected checkpoint scope count")
    return {"schema": "sagallm-integrated-historical-exposure-scope-v1",
            "status": "ARCHIVED_REPORT_SCOPE_INTEGRATED_NOT_COMPLETE",
            "archive_manifest_sha256": ledger["archive_manifest_sha256"],
            "archive_report_count": len(coverage), "archive_report_scope": coverage,
            "inventory_sha256": scope["inventory_sha256"], "inventory_entries": len(entries),
            "checkpoint_count": len(result), "checkpoint_declarations": result,
            "unresolved_checkpoint_declarations": [
                {"model": "contrastive-stdio-ab-control", "checkpoint_sha256": None,
                 "parent_checkpoint_sha256_declared": BEST, "reason": "NOT_AVAILABLE_IN_PRIOR_ARTIFACT",
                 "sources": source_assignments(entries, ANCESTOR + SFT + STDIO)}],
            "base_probe_context": {"checkpoint_sha256": BEST,
                "sources": source_assignments(entries, [(p, "development_context_only") for p in sorted({p for p, _ in STDIO[1:] + RANK + CE})]),
                "scope": "DEVELOPMENT_CONTEXT_ONLY_NOT_GRADIENT_EXPOSURE"},
            "early_report_audit": early,
            "missing_evidence": ["unarchived or local experiment enumeration",
                "actual optimizer sample order and gradient exposure for each checkpoint",
                "exact model-specific input tensors and all probe transformations",
                "complete checkpoint-specific source rescan and rights review",
                "independent human benchmark and isolated reviewed VM"],
            "source_bytes_rechecked_here": False, "actual_exposure_verified": False,
            "training_lineage_verified": False, "full_exposure_inventory": "NOT_ESTABLISHED",
            "shards_prepared": False, "inventory_ready_for_independent_evaluation": False,
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}
