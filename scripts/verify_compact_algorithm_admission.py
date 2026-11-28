"""Check development-only prompt revisions against the pinned v1 admission report."""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from sagallm.java_algorithm_compact_tasks import COMPACT_TASKS
from sagallm.java_algorithm_curriculum import canonical, sha
from sagallm.java_diverse_corpus import build_java_diverse_corpus, verify_java_diverse_corpus
from sagallm.java_instruction_sft import compact_prompt, encode_row, sha_file, source_task
from sagallm.tokenization import load_tokenizer
from scripts.audit_algorithm_curriculum_admission import (
    REPORT_SHA, canonical_index, require, sft_candidate, verify_report,
)
from scripts.audit_canonical_sft_template_quality import signature
from scripts.run_pinned_traced_context_pilot import CONTEXT, DATA, materialize_parent

BASELINE_ARCHIVE = "ce28b14a82f1436a6c355cf81e2cf888884001d1114f7beb116ebed6add3d12b"
BASELINE_REPORT = "a91edd752e57c6e5a4392b808c5b1178388571599494c84947921ca17814f367"


def read_baseline(path):
    require(path.is_file() and path.stat().st_size == 3358 and sha_file(path) == BASELINE_ARCHIVE,
            "Unexpected admission archive")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == ["algorithm-admission.json"], "Unexpected admission member")
        raw = archive.read("algorithm-admission.json")
    require(sha(raw) == BASELINE_REPORT, "Admission report digest mismatch")
    return json.loads(raw)


def verify_encoding(encoded, tokenizer, task, source):
    """Verify the complete shifted sequence and response-only supervision."""
    require(encoded is not None, "Compact task exceeds the pilot context")
    prompt = compact_prompt(task)
    prefix = [tokenizer.SPECIAL["<bos>"]] + tokenizer.encode(prompt)
    answer = tokenizer.encode(source)
    require(tokenizer.decode(prefix[1:]) == prompt and tokenizer.decode(answer) == source,
            "Tokenizer roundtrip mismatch")
    inputs = prefix + answer
    labels = [-100] * (len(prefix) - 1) + answer + [tokenizer.SPECIAL["<eos>"]]
    require(encoded["x"].tolist() == inputs and encoded["y"].tolist() == labels,
            "Incomplete sequence or incorrect response mask")
    require(encoded["sequence_positions"] == len(inputs) <= CONTEXT, "Sequence length drift")
    return len(inputs)


def compare(records, baseline, tokenizer, index):
    prior = {row["id"]: row for row in baseline["rows"]}
    require(len(records) == len(prior) == len(baseline["rows"]) == 16, "Row count drift")
    excluded = set()
    observations = []
    seen_tasks = set()
    for record in records:
        original = sft_candidate(record, "algorithm-v1-")
        old = prior.pop(original["id"])
        require(sha(canonical(original)) == old["candidate_row_sha256"], "Prior row binding drift")
        original_task, source = source_task(original)
        old_fit = encode_row(original, tokenizer, CONTEXT) is not None
        require(old_fit == (CONTEXT in old["admitted_contexts"]), "Prior admission drift")
        if not old_fit:
            excluded.add(record["id"])
        task = COMPACT_TASKS.get(record["id"], original_task)
        candidate = sft_candidate({**record, "task": task}, "algorithm-compact-v1-")
        require(source_task(candidate) == (task, source), "Task/source conversion drift")
        positions = verify_encoding(encode_row(candidate, tokenizer, CONTEXT), tokenizer, task, source)
        fingerprints = {"task": sha(task.encode()), "source": sha(source.encode()), "signature": signature(source)}
        require(fingerprints["source"] == old["source_sha256"], "Reference source drift")
        require(task not in seen_tasks, "Duplicate candidate task")
        seen_tasks.add(task)
        for split, values in index.items():
            for kind, fingerprint in fingerprints.items():
                require(fingerprint not in values[kind], f"Canonical {split} {kind} overlap")
        observations.append({
            "id": candidate["id"], "baseline_id": original["id"], "family": record["family"],
            "variant": record["variant"], "language": record["language"],
            "baseline_task_sha256": old["task_sha256"], "task_sha256": fingerprints["task"],
            "source_sha256": fingerprints["source"], "candidate_row_sha256": sha(canonical(candidate)),
            "baseline_positions": old["prompt_tokens"] + old["response_tokens_without_eos"] + 1,
            "compact_positions": positions, "task_changed": task != original_task,
        })
    require(excluded == COMPACT_TASKS.keys() and not prior, "Prompt revision scope drift")
    return observations


def run(parent_zip, reference_zip, admission_zip):
    records = verify_report(reference_zip)
    baseline = read_baseline(admission_zip)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialize_parent(parent_zip, root)
        tokenizer_path = root / "tokenizer.json"
        require(sha_file(tokenizer_path) == baseline["tokenizer_sha256"], "Tokenizer drift")
        dataset = root / "canonical"
        build_java_diverse_corpus(dataset)
        verify_java_diverse_corpus(dataset)
        require(sha_file(dataset / "manifest.json") == DATA, "Canonical corpus drift")
        observations = compare(records, baseline, load_tokenizer(tokenizer_path), canonical_index(dataset))
    by_language = {}
    for language in ("ja", "en"):
        group = [row for row in observations if row["language"] == language]
        by_language[language] = {
            "baseline_fit": sum(row["baseline_positions"] <= CONTEXT for row in group),
            "compact_fit": len(group),
            "max_positions": max(row["compact_positions"] for row in group),
        }
    return {
        "schema": "sagallm-compact-algorithm-admission-v1", "context": CONTEXT,
        "baseline_artifact_id": 15256607793, "baseline_report_sha256": BASELINE_REPORT,
        "reference_artifact_id": 15257146273, "reference_report_sha256": REPORT_SHA,
        "tokenizer_sha256": baseline["tokenizer_sha256"], "canonical_manifest_sha256": DATA,
        "by_language": by_language, "rows": observations,
        "membership_sha256": sha(canonical(sorted(row["candidate_row_sha256"] for row in observations))),
        "static_canonical_overlap": "NO_EXACT_TASK_SOURCE_OR_LEXICAL_SIGNATURE_MATCH",
        "semantic_review": "REVIEW_REQUIRED", "rights_review": "REVIEW_REQUIRED",
        "independent_evaluation": "NOT_RUN", "training_performed": False,
        "heldout_scoring": False, "model_selection_permitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-zip", "reference-zip", "admission-zip", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.parent_zip, args.reference_zip, args.admission_zip)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True, indent=2)
        out.write("\n")
    print(json.dumps(result["by_language"], sort_keys=True))
