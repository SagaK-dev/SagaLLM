"""Build a deterministic, explicitly UNAPPROVED Java human-review worklist.

This file never signs a review, checks licenses, verifies identities,
scores SagaLLM, or executes Java. Evidence refers to the pinned 6/27 intake.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from sagallm.java_external_io_preflight import run_preflight

SUITE_SHA = "d1af64a934c304f04b2cfa822280c73cb7da7b64d09e83f62b071b945a244c74"
SOURCES_SHA = "99201319a0efdc0dc3937e729e2ff552fedbb74cfd5662c8e3520a88ad9745a7"
UPSTREAM = "4812c213a2647e2034ad5ac78155ce9265cdd171"


def render(root: Path) -> str:
    root = Path(root)
    checks = run_preflight(root)
    provenance = root / "examples/java_external_io_sources.json"
    if not provenance.is_file() or provenance.is_symlink() or provenance.stat().st_size > 64000:
        raise ValueError("Unsafe provenance evidence")
    raw = provenance.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES_SHA:
        raise ValueError("Provenance changed since intake audit")
    manifest = json.loads(raw)
    if (checks["suite_sha256"] != SUITE_SHA or checks["source_revision"] != UPSTREAM
        or checks["tasks"] != 6 or checks["io_cases"] != 27
        or checks["specification_checks_passed"] != 27
        or checks["actual_independent_human_reviews"] != 0
        or checks["review_gate"] != "REVIEW_REQUIRED"
        or checks["behavior"] != "NOT_RUN"
        or checks["model_selection_permitted"] is not False
        or manifest["reviewers_attested"] != 0
        or manifest["candidate_is_not_approved"] is not True):
        raise ValueError("Evidence or approval gate changed; cannot create worklist")
    base = "https://github.com/exercism/problem-specifications/blob/" + UPSTREAM + "/"
    lines = [
        "# SagaLLM external Java task human review worklist",
        "",
        "**UNREVIEWED / REVIEW_REQUIRED / NOT_RUN** — this document is",
        "not an approval, independent benchmark result, rights clearance or",
        "authorization to execute untrusted Java.",
        "",
        "## Immutable evidence",
        "",
        "- Exercism upstream revision: §" + UPSTREAM + "§",
        "- Candidate suite SHA-256: §" + SUITE_SHA + "§",
        "- Attribution manifest SHA-256: §" + SOURCES_SHA + "§",
        "- Candidate JSON: §examples/java_external_io_INTAKE.json§",
        "- Attribution: §examples/java_external_io_sources.json§",
        "- Repository MIT license notice: §third_party_notices/exercism-problem-specifications-LICENSE.txt§",
        "- Original source metadata may credit other authors: their rights remain UNREVIEWED.",
        "- Scope: 6 publicly available exercises; 27 pinned test cases.",
        "- Two real human reviewers needed; **zero completed**.",
        "",
        "## Reviewers (DO NOT PRE-FILL)",
        "",
        "- Reviewer A: **PENDING** real identity, date, linked assessment.",
        "- Reviewer B: **PENDING** separate real identity, date, linked assessment.",
        "- Each reviewer must independently evaluate ALL 6 problems.",
        "- Reviewers must not inspect model answers or evaluation scores",
        "  before questions and exact tests are frozen.",
        "- Placeholder names or automatically generated signatures do NOT qualify.",
        "",
        "## Required checks by BOTH reviewers for each task",
        "",
        "- [ ] Rights to reuse original statement, test vectors, adapted wording;",
        "      record antecedent attributions and license evidence.",
        "- [ ] Check mapping of pinned UUIDs, original expected values,",
        "      input parsing and exact stdout contract for Java 17.",
        "- [ ] Check uncovered boundary behavior, invalid input, case, Unicode",
        "      and numeric limits for the actual intended task scope.",
        "- [ ] Compare semantics with entire training/validation/test corpus",
        "      and investigate possible exposure in pretrained sources.",
        "- [ ] Determine whether a public Exercism task can be genuinely",
        "      independent, or must be replaced by an unseen human-written task.",
        "- [ ] Publish dated, traceable, distinct task-level review records",
        "      including objections and explicit individual dispositions.",
        "",
        "## Task-by-task evidence and review blockers",
        "",
    ]
    sources = {record["task_id"]: record for record in manifest["sources"]}
    if len(sources) != 6 or len(checks["reviewer_notes"]) != 6:
        raise ValueError("Duplicate or missing task provenance")
    for item in checks["reviewer_notes"]:
        tid = item["task_id"]
        record = sources[tid]
        slug = record["exercise"]
        lines.extend([
            "### " + tid + " (" + slug + ")",
            "",
            "- Canonical cases: " + base + record["canonical_data_path"],
            "- Attribution metadata: " + base + record["metadata_path"],
            "- Credited origin: " + str(record["metadata_source"]),
            "- Credited source URL: "
            + str(record["metadata_source_url"] or "UNKNOWN - review required"),
            "- Git canonical data blob: §" + record["canonical_data_git_blob_sha1"] + "§",
            "- Git attribution metadata blob: §" + record["metadata_git_blob_sha1"] + "§",
            "- Canonical method: §" + record["test_property"] + "§",
            "- Selected cases: " + str(item["io_cases_checked"]),
            "- Original test UUIDs: " + ", ".join(
                "§" + uuid + "§" for uuid in record["selected_case_uuids"]
            ),
            "- Standalone Python checks: all selected cases agree (machine-only).",
            "- Plausible conceptual overlaps to inspect (not proven):",
        ])
        for lead in item["conceptual_overlap_review_leads"]:
            lines.append("  - §" + lead["family"] + "§ (" +
                         lead["original_split"] + " split)")
        lines.append("- Selected test-case coverage gaps:")
        for warning in item["case_coverage_warnings"]:
            lines.append("  - " + warning)
        lines.extend([
            "- Human legal/rights disposition: **PENDING**",
            "- Human Java console adapter disposition: **PENDING**",
            "- Human semantic contamination disposition: **PENDING**",
            "- Independent Reviewer A dated verdict + URL: **PENDING**",
            "- Independent Reviewer B dated verdict + URL: **PENDING**",
            "- **NOT APPROVED FOR MODEL SELECTION OR BEHAVIORAL SCORING**",
            "",
        ])
    lines.extend([
        "## Acceptance rules — never automatic",
        "",
        "Only after two genuine reviewers complete ALL task-level evidence,",
        "rights/attribution and semantics are checked, and objections resolved,",
        "consider a NEW frozen reviewed benchmark SHA-256. Public Exercism",
        "problems may not qualify as unbiased, independent testing; prefer",
        "fresh unseen rights-cleared tasks when credible independence is needed.",
        "Human review of OCI isolation is also required before optional",
        "execution on a dedicated disposable VM; never in shared CI.",
        "This worklist does not change any model or evaluation score.",
        "",
        "END: REVIEW REQUIRED / NOT_RUN — 0 HUMAN SIGN-OFFS",
        "",
    ])
    return "\n".join(lines).replace("§", chr(96))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite previous evidence")
    packet = render(args.root)
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(packet)
    print(json.dumps({
        "status": "UNREVIEWED_WORKLIST_CREATED", "tasks": 6, "io_cases": 27,
        "human_reviews": 0, "review_gate": "REVIEW_REQUIRED",
        "java_executed": False, "behavior": "NOT_RUN",
        "model_selection_permitted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
