"""Replay pinned admission evidence and audit complete, balanced SFT inputs.

Only original in-tree reference rows are encoded. No model loading, training,
TEST tokenization/scoring, rights approval or dataset publication.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

import torch
from sagallm.java_algorithm_curriculum import canonical, sha
from sagallm.java_ai import _model_user_message
from sagallm.java_instruction_sft import compact_prompt, encode_row, sha_file, source_task
from sagallm.tokenization import load_tokenizer
from scripts.audit_algorithm_curriculum_admission import run as replay_admission, verify_report
from scripts.run_pinned_traced_context_pilot import materialize_parent

ADMISSION_ARTIFACT = 15256607793
ADMISSION_RUN = 46129012984
ADMISSION_ZIP = "ce28b14a82f1436a6c355cf81e2cf888884001d1114f7beb116ebed6add3d12b"
ADMISSION_MEMBER = "algorithm-admission.json"
MEMBERSHIP = "9369de0c1dc2d8a034cc9a4c8f209654867699d3ff8cddbc13acf1189d2ad2b5"
CONTEXTS = (192, 256, 384, 512, 768)
EXPECTED_COUNTS = {192: 0, 256: 0, 384: 6, 512: 14, 768: 16}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_admission(path):
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == 3358 and sha_file(path) == ADMISSION_ZIP,
            "Admission archive pin mismatch")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == [ADMISSION_MEMBER], "Admission archive member drift")
        info = archive.infolist()[0]
        require(not info.is_dir() and not info.flag_bits & 1
                and 0 < info.file_size <= 65536, "Invalid admission member")
        raw = archive.read(info)
    return json.loads(raw), sha(raw)


def candidate_row(record):
    return {"id": "algorithm-v1-" + record["id"], "family": record["family"],
            "messages": [
                {"role": "user", "content": _model_user_message(
                    record["task"], "public class Main {}", "Main")},
                {"role": "assistant", "content": record["source"]},
            ]}


def check_encoding(row, tokenizer, context):
    task, source = source_task(row)
    bos, eos = (tokenizer.SPECIAL[k] for k in ("<bos>", "<eos>"))
    prefix = [bos] + tokenizer.encode(compact_prompt(task))
    response = tokenizer.encode(source) + [eos]
    ids = prefix + response
    positions = len(ids) - 1
    encoded = encode_row(row, tokenizer, context)
    if positions > context:
        require(encoded is None, "Overlong row was admitted")
        return None
    require(encoded is not None, "Complete row was excluded")
    labels = [-100] * (len(prefix) - 1) + response
    require(encoded["x"].dtype == torch.long and encoded["y"].dtype == torch.long,
            "Unexpected tensor dtype")
    require(encoded["x"].tolist() == ids[:-1] and encoded["y"].tolist() == labels,
            "Shift or assistant-only loss mask mismatch")
    require(encoded["sequence_positions"] == positions
            and encoded["response_tokens"] == len(response)
            and encoded["prefix_ids"] == prefix
            and encoded["id"] == row["id"] and encoded["family"] == row["family"],
            "Encoding metadata drift")
    require(labels[-1] == eos and labels[len(prefix)-1] == response[0],
            "First response token or terminal EOS lost")
    return {"sequence_positions": positions, "supervised_tokens": len(response),
            "masked_tokens": len(prefix) - 1,
            "encoding_sha256": sha(canonical({"x": ids[:-1], "y": labels}))}


def balance(records, contexts=CONTEXTS):
    require(len(records) == 16 and len({r["id"] for r in records}) == 16,
            "Expected sixteen distinct records")
    groups = defaultdict(list)
    for r in records:
        require(r["language"] in ("en", "ja"), "Unknown language")
        require(r["admitted_contexts"] == [c for c in contexts
                if c >= r["sequence_positions"]], "Admission is not exact and monotonic")
        groups[(r["family"], r["variant"])].append(r)
    families = sorted({f for f, _ in groups})
    require(len(families) == 4 and len(groups) == 8, "Unexpected family/variant scope")
    for f in families:
        require(sum(k[0] == f for k in groups) == 2, "Expected two implementations per family")
    for pair in groups.values():
        require(len(pair) == 2 and {r["language"] for r in pair} == {"en", "ja"}
                and len({r["source_sha256"] for r in pair}) == 1,
                "Bilingual implementation mismatch")
    summary = {}
    for c in contexts:
        admitted = [r for r in records if c in r["admitted_contexts"]]
        complete = [k for k, pair in groups.items()
                    if all(c in r["admitted_contexts"] for r in pair)]
        summary[str(c)] = {
            "admitted": len(admitted),
            "excluded_ids": sorted(r["id"] for r in records if c not in r["admitted_contexts"]),
            "language_counts": {l: sum(r["language"] == l for r in admitted) for l in ("en", "ja")},
            "family_counts": {f: sum(r["family"] == f for r in admitted) for f in families},
            "complete_bilingual_implementations": len(complete),
            "complete_families": [f for f in families if sum(k[0] == f for k in complete) == 2],
            "all_records_admitted": len(admitted) == len(records),
            "admitted_supervised_tokens": sum(r["supervised_tokens"] for r in admitted),
        }
    return summary


def run(admission_zip, reference_zip, parent_zip):
    archived, member_sha = read_admission(admission_zip)
    replay = replay_admission(reference_zip, parent_zip)
    require(archived == replay, "Pinned admission report differs from complete replay")
    require(archived["candidate_membership_sha256"] == MEMBERSHIP,
            "Candidate membership drift")
    require(all(not any(values.values()) for values in archived["canonical_match_counts"].values()),
            "Static overlap appeared")
    index = {r["id"]: r for r in archived["rows"]}
    require(len(index) == len(archived["rows"]) == 16, "Duplicate report IDs")
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialize_parent(parent_zip, root)
        tokenizer_path = root / "tokenizer.json"
        require(sha_file(tokenizer_path) == archived["tokenizer_sha256"], "Tokenizer drift")
        tokenizer = load_tokenizer(tokenizer_path)
        for record in verify_report(reference_zip):
            row = candidate_row(record)
            previous = index[row["id"]]
            require(sha(canonical(row)) == previous["candidate_row_sha256"]
                    and sha(record["source"].encode()) == previous["source_sha256"],
                    "Candidate row binding drift")
            checked = {c: check_encoding(row, tokenizer, c) for c in CONTEXTS}
            admitted = [c for c, result in checked.items() if result is not None]
            require(admitted == previous["admitted_contexts"] and admitted,
                    "Admission differs from original")
            lengths = [checked[c] for c in admitted]
            require(all(x == lengths[0] for x in lengths), "Context changed encoding")
            exact = lengths[0]
            require(exact["sequence_positions"] == previous["prompt_tokens"]
                    + previous["response_tokens_without_eos"] + 1,
                    "Position accounting drift")
            require(check_encoding(row, tokenizer, exact["sequence_positions"]) == exact
                    and check_encoding(row, tokenizer, exact["sequence_positions"] - 1) is None,
                    "Exact context boundary failed")
            records.append({"id": row["id"], "family": record["family"],
                            "variant": record["variant"], "language": record["language"],
                            "source_sha256": previous["source_sha256"],
                            "candidate_row_sha256": previous["candidate_row_sha256"],
                            "admitted_contexts": admitted, **exact})
    summary = balance(records)
    require({c: summary[str(c)]["admitted"] for c in CONTEXTS} == EXPECTED_COUNTS,
            "Historical admission counts drift")
    minimum = min(c for c in CONTEXTS if summary[str(c)]["all_records_admitted"])
    require(minimum == 768, "Expected first complete tested context 768")
    return {
        "schema": "sagallm-algorithm-context-balance-mask-v1",
        "status": "ENCODING_AND_BALANCE_AUDIT_COMPLETE",
        "admission_artifact_id": ADMISSION_ARTIFACT, "admission_run_id": ADMISSION_RUN,
        "admission_zip_sha256": ADMISSION_ZIP, "admission_member_sha256": member_sha,
        "candidate_membership_sha256": MEMBERSHIP,
        "tokenizer_sha256": archived["tokenizer_sha256"],
        "canonical_manifest_sha256": archived["canonical_manifest_sha256"],
        "tested_contexts": list(CONTEXTS), "context_summary": summary,
        "minimum_complete_tested_context": minimum,
        "exact_max_sequence_positions": max(r["sequence_positions"] for r in records),
        "rows": records, "loss_mask_checks": len(records),
        "exact_boundary_checks": len(records),
        "context_extension_feasibility": "NOT_ESTABLISHED",
        "training_admission": "NOT_AUTHORIZED",
        "semantic_overlap_verified": False, "test_tokenization_or_scoring": False,
        "training_performed": False, "weights_loaded": False, "dataset_published": False,
        "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
        "model_selection_permitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("admission-zip", "reference-zip", "parent-zip", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse overwrite")
    report = run(args.admission_zip, args.reference_zip, args.parent_zip)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, sort_keys=True, indent=2)
        out.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))
