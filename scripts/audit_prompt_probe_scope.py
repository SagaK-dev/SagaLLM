"""Audit archived prompt-probe scope without inference or source recovery."""
import argparse
import io
import json
from pathlib import Path
import zipfile

from scripts.audit_ranking_auxiliary import LIMIT, TOKENIZER, parse, pin, require, sha

PRIOR_RUN = 45973926396
PRIOR_ARTIFACT = 15186012660
PRIOR_ZIP = "3553b57b58eadfea8ecd36ebedeb8ce72ad8b65136fcfc4ec1a7a58ec1c56cad"
BASE = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
PAIR = "4d9b2d6eb7ff216b653953eda1f355c4cae6def22d8a9af31016e51c7daa7d0b"
CE_ZIP = "068d733b1a1059a1e904ed8931467a85aaf9f5a16968a6837ce2fb4b604bb447"
ACTIVATION = "3d8436e4cfee22aee4918a8718a5eb270d94f7a3af832adb724ff4691e47742c"
ACTIVATION_ZIP = "e428e95300195cfcde474592f721b9be45995789bbe51b31549cc780bd1c878d"
SPECS = (
    (45377422360, "m3-parameter-ce", "26f143575842073c9b4119ecefcb3b50935891bba6c4c9fc60b0b0c8353a89b7"),
    (45378356573, "m3-prompt-signal", "535d10f4c9873f945040ddadfbadcd5041024878d59db6fda5203cda9038ab6e"),
    (45389874625, "m3-prompt-bias", "e6829df208ef4807073a27df08053c241c0871995d2f42b3c5c6f4daed6864f4"),
)
SEEDS = (4201, 4202, 4203)
ARMS = ("control", "parameter_ce")


def prior(path):
    raw = pin(path, PRIOR_ZIP)
    expected = {"scoped-parameter-ce-inventory.json", "parameter-ce-evidence.json"}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        require(len(members) == 2 and {m.filename for m in members} == expected,
                "Unexpected prior members")
        blobs = {}
        for m in members:
            require(0 < m.file_size <= LIMIT, "Prior member size")
            with archive.open(m) as stream:
                body = stream.read(LIMIT + 1)
            require(len(body) == m.file_size and len(body) <= LIMIT, "Prior size mismatch")
            blobs[m.filename] = body
    inv_raw = blobs["scoped-parameter-ce-inventory.json"]
    inv = parse(inv_raw)
    ev = parse(blobs["parameter-ce-evidence.json"])
    require(inv["schema"] == "sagallm-private-java-exposure-v1"
            and ev["status"] == "HISTORICAL_PARAMETER_CE_DECLARATIONS_RECONSTRUCTED"
            and ev["report_sha256"] == SPECS[0][2]
            and ev["tokenizer_sha256"] == TOKENIZER
            and ev["inventory_sha256"] == sha(inv_raw)
            and len(inv["files"]) == ev["inventory_entries"] == 15
            and ev["checkpoint_count"] == 6
            and ev["diagnostic_scope_checked_models"] == 7
            and ev["actual_exposure_verified"] is False
            and ev["training_lineage_verified"] is False
            and ev["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and ev["inventory_ready_for_independent_evaluation"] is False
            and ev["model_selection_permitted"] is False
            and ev["independent_evaluation"] == "NOT_RUN", "Prior evidence gate mismatch")
    require(len(ev["derived_sources"]) == 3, "Prior source count mismatch")
    for entry, name, group, count in zip(
            ev["derived_sources"], ("train", "validation", "fresh"),
            ("train", "development", "development"), (640, 12, 12)):
        require(entry["entry"] in inv["files"] and entry["rows"] == count
                and entry["entry"]["group"] == group
                and entry["entry"]["path"] == "derived/parameter_ce_" + name + ".jsonl",
                "Prior derived source declaration mismatch")
    return ev, {name: sha(body) for name, body in blobs.items()}


def load_reports(root):
    return [parse(pin(root / f"{run}-{slug}.json", digest, 8 * 1024 * 1024))
            for run, slug, digest in SPECS]


def pair_ids(group, count, prefix, lengths=False, signal=False):
    rows = group["rows"]
    require(group.get("pairs", count) == len(rows) == count, "Pair count mismatch")
    seen, result = set(), []
    for row in rows:
        ids = row["ids"]
        require(isinstance(ids, list) and len(ids) == 2 and ids[0] != ids[1]
                and all(isinstance(i, str) and i.startswith(prefix) for i in ids)
                and isinstance(row["family"], str) and bool(row["family"])
                and not seen.intersection(ids), "Pair identity invalid")
        seen.update(ids)
        entry = [ids, row["family"]]
        if lengths:
            pos = row["common_prefix_tokens" if signal else "common_reference_tokens"]
            lens = row["prompt_lengths" if signal else "prompt_token_counts"]
            require(type(pos) is int and pos >= 0 and isinstance(lens, list)
                    and len(lens) == 2 and all(type(n) is int and n > 0 for n in lens),
                    "Token length invalid")
            entry += [pos, lens]
        result.append(entry)
    return result


def generation_ids(group):
    rows = group["rows"]
    require(len(rows) == 6, "Regression generation count mismatch")
    ids = []
    for row in rows:
        require(isinstance(row["id"], str) and row["id"].startswith("validation-")
                and isinstance(row["family"], str) and row["family"]
                and row["behavior"] == row["compile"] == "NOT_RUN", "Generation scope invalid")
        ids.append([row["id"], row["family"]])
    require(len({row[0] for row in ids}) == 6, "Duplicate generation ID")
    return ids


def compare(ce, signal, bias, checkpoint_pins=None):
    for report, status in ((ce, "PARAMETER_CE_CPU_TRAINING_AND_INFERENCE_COMPLETE"),
                           (signal, "PARAMETER_CE_AUDIT_AND_PROMPT_SIGNAL_DIAGNOSTIC_COMPLETE"),
                           (bias, "FIRST_LAYER_PROMPT_BIAS_PILOT_COMPLETE")):
        require(report["status"] == status and report["base_sha256"] == BASE
                and report["tokenizer_sha256"] == TOKENIZER
                and report["pair_manifest_sha256"] == PAIR
                and report["compile"] == report["behavior"] == "NOT_RUN"
                and report["production_promotion"] is False, "Report scope mismatch")
    for report in (signal, bias):
        require(report["archive_sha256"] == CE_ZIP
                and report["training_report_sha256"] == SPECS[0][2]
                and report["training"] == "NOT_RUN"
                and report["independent_benchmark"] == "NOT_ESTABLISHED"
                and report["root_cause"] == "NOT_ESTABLISHED", "Probe reference mismatch")
    require(signal["probe_split"] == "TRAIN_EXPOSED" and signal["checkpoint_hashes_verified"] == 6
            and signal["generation_replay_matches"] == 126
            and bias["activation_report_sha256"] == ACTIVATION
            and bias["activation_zip_sha256"] == ACTIVATION_ZIP
            and bias["zero_bias_generation_matches"] == 42,
            "Probe declaration mismatch")
    require(set(ce["results"]) == {str(s) for s in SEEDS}, "CE seed coverage mismatch")
    records = {"base": (BASE, ce["baseline"])}
    for seed in SEEDS:
        arms = ce["results"][str(seed)]
        require(set(arms) == set(ARMS), "CE arm coverage mismatch")
        for arm in ARMS:
            row = arms[arm]
            records[f"{seed}-{arm}"] = (row["checkpoint_sha256"], row["diagnostics"])
    cps = [x[0] for x in records.values()]
    require(len(set(cps)) == 7 and all(isinstance(x, str) and len(x) == 64
            and all(c in "0123456789abcdef" for c in x) for x in cps), "Checkpoint pins invalid")
    if checkpoint_pins is not None:
        require(sorted(cps[1:]) == sorted(checkpoint_pins), "Prior checkpoint pins differ")
    require(set(signal["models"]) == set(bias["models"]) == set(records), "Probe model coverage mismatch")
    results = []
    for key, (cp, diag) in records.items():
        train = pair_ids(diag["forced_choice"]["train"], 20, "train-", lengths=True)
        valid = pair_ids(diag["forced_choice"]["validation"], 3, "validation-")
        gen = generation_ids(diag["generation"]["regression"])
        model = signal["models"][key]
        require(model["all"]["pairs"] == 20
                and pair_ids(model, 20, "train-", lengths=True, signal=True) == train,
                "Signal identity or length mismatch")
        modes = bias["models"][key]
        require(set(modes) == {"control", "bias"}, "Bias mode coverage mismatch")
        for mode in ("control", "bias"):
            row = modes[mode]
            require(pair_ids(row["train"], 20, "train-") == [x[:2] for x in train]
                    and pair_ids(row["validation"], 3, "validation-") == valid
                    and generation_ids(row["regression_generation"]) == gen,
                    "Bias identity mismatch")
        results.append({"model": key, "checkpoint_sha256_declared_by_ce": cp,
                        "signal_train_pairs": 20, "bias_train_pairs_per_mode": 20,
                        "bias_validation_pairs_per_mode": 3,
                        "bias_regression_rows_per_mode": 6})
    return results


def run(root, prior_zip):
    ev, hashes = prior(prior_zip)
    records = compare(*load_reports(root), checkpoint_pins=ev["checkpoint_sha256"])
    return {"schema": "sagallm-prompt-probe-scope-v1",
            "status": "ARCHIVED_PROMPT_PROBE_IDENTITIES_MATCH",
            "reports": [{"run_id": run, "slug": slug, "report_sha256": digest}
                        for run, slug, digest in SPECS],
            "prior_evidence": {"run_id": PRIOR_RUN, "artifact_id": PRIOR_ARTIFACT,
                               "archive_sha256": PRIOR_ZIP, "member_sha256": hashes},
            "inventory_sha256": ev["inventory_sha256"], "inventory_entries": 15,
            "new_inventory_entries": 0, "models": records,
            "signal_pair_comparisons": 140, "bias_pair_comparisons": 322,
            "bias_regression_row_comparisons": 84,
            "scope": "report-declared identities only; not probe tensors or actual inputs",
            "probe_scores_recomputed": False, "probe_source_bytes_reconstructed": False,
            "prior_source_bytes_rechecked_here": False, "checkpoint_bytes_verified_here": False,
            "remaining_archived_probes": ["m1-conditioning", "m1-conditioning-rerun",
                "contrastive-stdio-ab", "m3-alignment-v1-blocked", "m3-frozen-replay",
                "m3-parameter-binding", "m3-activation-swap"],
            "activation_link_verified": "DECLARATION_ONLY",
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "shards_prepared": False,
            "inventory_ready_for_independent_evaluation": False,
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("experiments/archive/reports"))
    parser.add_argument("--prior-zip", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reports, args.prior_zip), sort_keys=True, indent=2))
