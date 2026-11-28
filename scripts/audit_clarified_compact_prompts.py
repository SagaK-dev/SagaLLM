"""Replay compact and boundary evidence; measure clarified author-side prompts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile

from sagallm.java_algorithm_clarified_tasks import PROPOSALS, REVIEW_FOCUS
from sagallm.java_algorithm_compact_tasks import COMPACT_TASKS
from sagallm.java_algorithm_curriculum import canonical, sha
from sagallm.java_diverse_corpus import build_java_diverse_corpus, verify_java_diverse_corpus
from sagallm.java_instruction_sft import compact_prompt, encode_row, sha_file, source_task
from sagallm.tokenization import load_tokenizer
from scripts.audit_algorithm_curriculum_admission import canonical_index, require, sft_candidate, verify_report
from scripts.audit_canonical_sft_template_quality import signature
from scripts.audit_compact_algorithm_contract_boundaries import (
    bind_records, checked_cases, read_compact, CONTRACTS,
)
from scripts.run_pinned_traced_context_pilot import CONTEXT, DATA, materialize_parent
from scripts.verify_compact_algorithm_admission import verify_encoding

BOUNDARY_ARTIFACT = 15258415001
BOUNDARY_RUN = 46130826759
BOUNDARY_ZIP = "c51ac98265fb8a75f26b1d9e7ab7be54812be2d2cb82ca33ae8c4c1e2b06ea40"


def read_boundary(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == 7015
            and sha_file(path) == BOUNDARY_ZIP, "Boundary ZIP pin mismatch")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == ["compact-contract-boundaries.json"], "Boundary member drift")
        info = archive.infolist()[0]
        require(not info.is_dir() and not info.flag_bits & 1
                and 0 < info.file_size <= 131072, "Boundary member bounds")
        raw = archive.read(info)
    return json.loads(raw), sha(raw)


def bind_boundary(boundary, compact, checked, sources):
    require(boundary["status"] == "FIXED_REFERENCE_BOUNDARY_CHECKS_COMPLETE"
            and boundary["membership_sha256"] == compact["membership_sha256"]
            and boundary["compact_report_sha256"] == "b0d7596dba88acaa07735897e7ee5fac4af0e9e779914073cc9ad5cdf9c0dba1"
            and boundary["records"] == checked and boundary["contracts"] == CONTRACTS
            and boundary["reference_compiles"] == 8
            and boundary["distinct_boundary_inputs"] == 40
            and boundary["boundary_comparisons"] == 80
            and boundary["semantic_review"] == "REVIEW_REQUIRED"
            and boundary["rights_review"] == "REVIEW_REQUIRED"
            and boundary["training_performed"] is False
            and boundary["model_generated_code_executed"] is False,
            "Boundary report scope mismatch")
    seen = set()
    for result in boundary["results"]:
        key = result["family"], result["variant"]
        require(key in sources and key not in seen and result["compile"] == "PASS"
                and result["source_sha256"] == sha(sources[key].encode()), "Reference source drift")
        seen.add(key)
        expected = [{"case": name, "stdin_sha256": sha(stdin.encode()),
                     "expected_stdout_sha256": sha(stdout.encode()),
                     "actual_stdout_sha256": sha(stdout.encode())}
                    for name, stdin, stdout in checked_cases(key[0])]
        require(result["checks"] == expected, "Boundary case evidence drift")
    require(seen == set(sources), "Reference coverage mismatch")


def first_eligible(options):
    """Fixed author preference order, no silent fallback."""
    return next((row for row in options if row["eligible"]), None)


def check_option(record, task, tokenizer, index):
    candidate = sft_candidate({**record, "task": task}, "algorithm-clarified-v1-")
    require(source_task(candidate) == (task, record["source"]), "Candidate conversion drift")
    prefix = tokenizer.encode(compact_prompt(task))
    answer = tokenizer.encode(record["source"])
    require(tokenizer.decode(prefix) == compact_prompt(task)
            and tokenizer.decode(answer) == record["source"], "Tokenizer roundtrip drift")
    positions = len(prefix) + len(answer) + 1
    encoded = encode_row(candidate, tokenizer, CONTEXT)
    fit = positions <= CONTEXT
    require((encoded is not None) == fit, "Admission arithmetic mismatch")
    if fit:
        require(verify_encoding(encoded, tokenizer, task, record["source"]) == positions,
                "Encoding positions mismatch")
        require(encode_row(candidate, tokenizer, positions - 1) is None,
                "Exact context boundary mismatch")
        require(verify_encoding(encode_row(candidate, tokenizer, positions), tokenizer,
                                task, record["source"]) == positions, "Exact context rejected")
    hashes = {"task": sha(task.encode()), "source": sha(record["source"].encode()),
              "signature": signature(record["source"])}
    overlaps = {split: sorted(k for k, digest in hashes.items() if digest in entries[k])
                for split, entries in index.items()}
    return {"task_sha256": hashes["task"], "source_sha256": hashes["source"],
            "candidate_row_sha256": sha(canonical(candidate)), "positions": positions,
            "supervised_tokens": len(answer) + 1, "fit": fit, "static_matches": overlaps,
            "eligible": fit and not any(overlaps.values())}


def plan(records, compact, tokenizer, index):
    require(set(PROPOSALS) == set(REVIEW_FOCUS) and len(PROPOSALS) == 7, "Revision scope drift")
    prior = {r["id"]: r for r in compact["rows"]}
    observations = []
    selected_tasks = set()
    for record in records:
        rid = record["id"]
        old_task = COMPACT_TASKS.get(rid, record["task"])
        old = prior.pop("algorithm-compact-v1-" + rid)
        old_encoded = check_option(record, old_task, tokenizer, index)
        require(old_encoded["positions"] == old["compact_positions"]
                and old_encoded["task_sha256"] == old["task_sha256"]
                and old_encoded["source_sha256"] == old["source_sha256"]
                and old_encoded["eligible"], "Prior compact encoding mismatch")
        texts = PROPOSALS.get(rid, (old_task,))
        require(len(texts) == len(set(texts)) and all(texts), "Duplicate/empty proposal")
        if rid in PROPOSALS:
            require(old_task not in texts and record["language"] == "ja", "Invalid revision scope")
        options = []
        for i, task in enumerate(texts):
            item = check_option(record, task, tokenizer, index)
            require(item["source_sha256"] == old_encoded["source_sha256"]
                    and item["supervised_tokens"] == old_encoded["supervised_tokens"],
                    "Response changed")
            options.append({"option": i+1, **item})
        choice = first_eligible(options)
        if choice is not None:
            require(choice["task_sha256"] not in selected_tasks, "Selected task collision")
            selected_tasks.add(choice["task_sha256"])
        observations.append({"id": "algorithm-clarified-v1-"+rid, "family": record["family"],
                             "variant": record["variant"], "language": record["language"],
                             "prior_task_sha256": old["task_sha256"],
                             "prior_positions": old["compact_positions"],
                             "review_focus": REVIEW_FOCUS.get(rid, "unchanged; review original contract"),
                             "revision_requested": rid in PROPOSALS,
                             "options": options, "selected": choice})
    require(not prior and len(observations) == 16
            and sum(r["revision_requested"] for r in observations) == 7,
            "Candidate coverage mismatch")
    return observations


def run(parent_zip, reference_zip, compact_zip, boundary_zip):
    compact = read_compact(compact_zip)
    records = verify_report(reference_zip)
    checked, sources = bind_records(records, compact)
    boundary, member_sha = read_boundary(boundary_zip)
    bind_boundary(boundary, compact, checked, sources)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialize_parent(parent_zip, root)
        tok = root / "tokenizer.json"
        require(sha_file(tok) == compact["tokenizer_sha256"], "Tokenizer drift")
        dataset = root / "canonical"
        build_java_diverse_corpus(dataset)
        verify_java_diverse_corpus(dataset)
        require(sha_file(dataset/"manifest.json") == DATA, "Canonical manifest drift")
        observations = plan(records, compact, load_tokenizer(tok), canonical_index(dataset))
    selected = [r for r in observations if r["selected"] is not None]
    full = len(selected) == len(observations)
    return {"schema": "sagallm-clarified-compact-prompts-v1",
            "status": "FULL_TECHNICAL_FIT" if full else "BLOCKED_INCOMPLETE_FIT",
            "context": CONTEXT, "boundary_artifact_id": BOUNDARY_ARTIFACT,
            "boundary_run_id": BOUNDARY_RUN, "boundary_zip_sha256": BOUNDARY_ZIP,
            "boundary_member_sha256": member_sha, "prior_membership_sha256": compact["membership_sha256"],
            "tokenizer_sha256": compact["tokenizer_sha256"], "canonical_manifest_sha256": DATA,
            "historical_boundary_comparisons_reconciled": 80,
            "proposed_revisions": 7, "options_measured": sum(len(r["options"]) for r in observations),
            "selected_count": len(selected),
            "selected_by_language": dict(Counter(r["language"] for r in selected)),
            "selected_revisions": sum(r["revision_requested"] for r in selected),
            "blocked_ids": [r["id"] for r in observations if r["selected"] is None],
            "max_selected_positions": max((r["selected"]["positions"] for r in selected), default=0),
            "candidate_membership_sha256": sha(canonical(sorted(r["selected"]["candidate_row_sha256"]
                                                for r in selected))) if full else None,
            "rows": observations, "selection_scope": "AUTHOR_TECHNICAL_PROPOSAL_ONLY",
            "semantic_review": "REVIEW_REQUIRED", "rights_review": "REVIEW_REQUIRED",
            "semantic_independence": "NOT_ESTABLISHED", "training_admission": "NOT_AUTHORIZED",
            "training_performed": False, "weights_loaded": False, "dataset_published": False,
            "java_executed_in_this_stage": False, "test_tokenization_or_scoring": False,
            "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("parent-zip", "reference-zip", "compact-zip", "boundary-zip", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "Refuse overwrite")
    report = run(a.parent_zip, a.reference_zip, a.compact_zip, a.boundary_zip)
    with a.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, sort_keys=True, indent=2)
        out.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, sort_keys=True))
