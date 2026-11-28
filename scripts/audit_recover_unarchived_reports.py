"""Recover exact bounded report bytes from pinned historical Actions archives.

No extraction of weights, generated Java, corpus text, or executable files.
Unknown members remain explicit review items. Never load or execute archive content.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile

from scripts.audit_ranking_auxiliary import parse, require, sha
from scripts.audit_unarchived_artifact_triage import read_evidence, gate, MAX_ARCHIVE, stamp, hexsha

PRIOR_RUN = 46087794746
PRIOR_ARTIFACT = 15239309131
PRIOR_ZIP = "f06646c8fce3267c588c8b37d13e126f5a5aa7d86315fcb9126dbae2c083e5b0"
MAX_REPORT = 8 * 1024 * 1024
MAX_TOTAL_REPORT_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 256
MAX_ARTIFACTS = 150
# Exact artifact names and report paths identified in historical workflows.
ALLOWED = {
    "sagallm-v016-pinned-java-cpu": ("java-model/train-report.json",),
    "sagallm-v017-java-instruction-sft-cpu": ("java-sft-model/train-report.json",),
    "sagallm-v018-java-context-sft-cpu": ("java-context-model/train-report.json",),
    "sagallm-broad-mixture-low-lr-validation-ab": ("broad-mix-ab-evidence/ab-report.json",),
    "sagallm-java-identifier-ab-validation": ("identifier-ab-evidence/ab-report.json",),
    "sagallm-java-prompt-validation-experiment": ("java-prompt-validation-experiment.json",),
    "sagallm-m3-ranking-cpu-research": ("ranking-report.json",),
    "sagallm-contrastive-stdio-matched-validation-ab": ("contrastive-stdio-ab-evidence/ab-report.json",),
    "sagallm-public-java-isolated-behavior-unreviewed": ("exploratory-public-java-oci-io.json",),
    "sagallm-public-java-compile-diagnostic-unreviewed": ("exploratory-public-java-compile-only.json",),
    "sagallm-java-train-only-identifier-candidates": ("manifest.json", "audit-summary.json"),
    "sagallm-v018-paired-java-compile-diagnostics": ("paired-java-compile.json",),
}
# Conservative screening: raw sources, prompts, logs, credentials and corpus rows
# are not preservation targets. A block requires explicit human review, not truncation.
RAW_KEYS = {"source", "code", "generated_source", "generated_code", "java_source",
    "stdout", "stderr", "raw_stdout", "raw_stderr", "raw_source", "prompt",
    "messages", "completion", "response", "password", "secret", "api_key",
    "access_token", "authorization", "private_key", "credentials"}
SECRET = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16})")
JAVA = re.compile(r"\b(?:public\s+(?:final\s+)?class\s+\w+|public\s+static\s+void\s+main\s*\()")


def screen(raw):
    d = parse(raw)
    require(isinstance(d, dict), "Expected JSON object")
    stack = [(None, d, 0)]
    count = 0
    while stack:
        key, value, depth = stack.pop()
        count += 1
        require(count <= 500_000 and depth <= 48, "Report structural limit")
        if isinstance(value, str):
            if key in RAW_KEYS and value:
                return "RAW_FIELD_REVIEW_REQUIRED"
            if len(value) > 4096 or SECRET.search(value) or JAVA.search(value):
                return "CONTENT_REVIEW_REQUIRED"
        elif isinstance(value, dict):
            stack.extend((k.lower(), v, depth + 1) for k, v in value.items())
        elif isinstance(value, list):
            stack.extend((key, v, depth + 1) for v in value)
    return "METADATA_SCREEN_PASSED_NOT_RIGHTS_REVIEW"


def queue(prior):
    gate(prior)
    require(prior["status"] == "SCOPED_ARTIFACT_METADATA_TRIAGE", "Wrong prior scope")
    t = prior["triage"]
    rows = t["runs"]
    require(len(rows) == t["run_count"] == len({r["run_id"] for r in rows}), "Run coverage mismatch")
    all_items = [(r, a) for r in rows for a in r["artifacts"]]
    require(len(all_items) == t["artifact_count"] <= MAX_ARTIFACTS
            and len({a["artifact_id"] for _, a in all_items}) == len(all_items), "Artifact coverage mismatch")
    ready = [{"run_id": r["run_id"], "commit_sha": r["commit_sha"], **a}
             for r, a in all_items if a["status"] == "METADATA_READY_FOR_BYTE_VERIFICATION"]
    ready.sort(key=lambda a: (stamp(a["expires_at"]), a["run_id"], a["artifact_id"]))
    require(ready == t["priority_by_expiry"], "Priority queue drift")
    for a in ready:
        require(type(a["artifact_id"]) is int and a["artifact_id"] > 0
                and type(a["run_id"]) is int and a["run_id"] > 0
                and type(a["size_in_bytes"]) is int and 0 < a["size_in_bytes"] <= MAX_ARCHIVE
                and hexsha(a["api_zip_sha256"]) and a["run_binding"] == "MATCH"
                and a["archive_bytes_verified"] is False, "Invalid queue entry")
    return ready


def bounded_download(a, target):
    """Stream exact expected size to a private temp file; do not trust Content-Length."""
    endpoint = f"repos/SagaK-dev/SagaLLM/actions/artifacts/{a['artifact_id']}/zip"
    with subprocess.Popen(["gh", "api", endpoint], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL) as proc:
        try:
            total = 0
            h = hashlib.sha256()
            with target.open("xb") as out:
                while True:
                    block = proc.stdout.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    require(total <= a["size_in_bytes"], "ZIP download exceeds pinned size")
                    h.update(block)
                    out.write(block)
            require(proc.wait(timeout=30) == 0 and total == a["size_in_bytes"]
                    and h.hexdigest() == a["api_zip_sha256"], "ZIP byte pin mismatch")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def verify_zip(path, a, output):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == a["size_in_bytes"],
            "ZIP size mismatch")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    require(h.hexdigest() == a["api_zip_sha256"], "ZIP digest mismatch")
    base = {"run_id": a["run_id"], "artifact_id": a["artifact_id"], "commit_sha": a["commit_sha"],
            "artifact_name": a["name"], "archive_sha256": h.hexdigest(),
            "archive_bytes_verified": True, "expires_at": a["expires_at"]}
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        require(0 < len(members) <= MAX_MEMBERS, "ZIP member count")
        names = [m.filename for m in members]
        require(len(names) == len(set(names)), "Duplicate archive member")
        # Never use member names as local filesystem paths.
        for m in members:
            p = PurePosixPath(m.filename)
            require(not p.is_absolute() and ".." not in p.parts and "\\" not in m.filename
                    and len(m.filename) <= 300 and not m.flag_bits & 1, "Unsafe ZIP member")
        wanted = ALLOWED.get(a["name"], ())
        found, saved, pending = [], 0, []
        for index, name in enumerate(wanted):
            if name not in names:
                found.append({"member": name, "status": "EXPECTED_MEMBER_MISSING"})
                continue
            m = z.getinfo(name)
            if m.is_dir() or not 0 < m.file_size <= MAX_REPORT:
                found.append({"member": name, "status": "MEMBER_SIZE_BLOCKED"})
                continue
            with z.open(m) as f:
                raw = f.read(MAX_REPORT + 1)
            require(len(raw) == m.file_size <= MAX_REPORT, "Member size mismatch")
            info = {"member": name, "report_sha256": sha(raw), "bytes": len(raw)}
            try:
                info["status"] = screen(raw)
            except (ValueError, TypeError, RecursionError, UnicodeError):
                info["status"] = "JSON_OR_STRUCTURE_REVIEW_REQUIRED"
            if info["status"] == "METADATA_SCREEN_PASSED_NOT_RIGHTS_REVIEW":
                filename = f"{a['run_id']}-{a['artifact_id']}-{index}.json"
                pending.append((filename, raw))
                info["preserved_file"] = filename
                saved += len(raw)
            found.append(info)
        written = []
        try:
            for filename, raw in pending:
                target = output / filename
                with target.open("xb") as f:
                    written.append(target)
                    f.write(raw)
        except OSError:
            for target in written:
                target.unlink(missing_ok=True)
            raise
        return {**base, "status": "ZIP_VERIFIED", "member_count": len(members),
                "report_members": found, "preserved_bytes": saved,
                "unknown_artifact_name": not bool(wanted),
                "other_json_member_count": sum(n.endswith(".json") and n not in wanted for n in names)}


def recover(a, output):
    if stamp(a["expires_at"]) <= stamp(output[1]):
        return {"run_id": a["run_id"], "artifact_id": a["artifact_id"],
                "status": "EXPIRED_SINCE_TRIAGE", "archive_bytes_verified": False}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "archive.zip"
            bounded_download(a, p)
            return verify_zip(p, a, output[0])
    except (ValueError, OSError, zipfile.BadZipFile, subprocess.SubprocessError, RuntimeError) as e:
        # Never emit raw exception text from an untrusted ZIP or API.
        return {"run_id": a["run_id"], "artifact_id": a["artifact_id"],
                "status": "RECOVERY_BLOCKED", "error_type": type(e).__name__,
                "archive_bytes_verified": False}


def run(prior, output, observed):
    ready = queue(prior)
    output.mkdir()
    reports = output / "reports"
    reports.mkdir()
    # Independent bounded jobs, no repo writes and no model code execution.
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(lambda a: recover(a, (reports, observed)), ready))
    require([r["artifact_id"] for r in result] == [a["artifact_id"] for a in ready], "Result queue drift")
    total = sum(p.stat().st_size for p in reports.iterdir())
    require(total <= MAX_TOTAL_REPORT_BYTES, "Total preservation limit exceeded")
    members = [m for r in result for m in r.get("report_members", [])]
    preserved = [m for m in members if "preserved_file" in m]
    require(len(preserved) == len(list(reports.iterdir()))
            and all(sha((reports / m["preserved_file"]).read_bytes()) == m["report_sha256"]
                    for m in preserved), "Preserved file mismatch")
    return {"schema": "sagallm-recovered-historical-reports-v1",
            "status": "BOUNDED_RECOVERY_ATTEMPT_COMPLETE", "observed_at_utc": observed,
            "candidate_count": len(ready), "zip_verified_count": sum(r["archive_bytes_verified"] for r in result),
            "artifact_status_counts": dict(Counter(r["status"] for r in result)),
            "report_member_status_counts": dict(Counter(m["status"] for m in members)),
            "preserved_report_count": len(preserved), "preserved_bytes": total,
            "artifacts": result, "all_candidate_archives_verified": all(r["archive_bytes_verified"] for r in result),
            "all_reports_recovered": False, "actual_exposure_verified": False,
            "training_lineage_verified": False, "full_exposure_inventory": "NOT_ESTABLISHED",
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False, "new_training": False, "new_inference": False,
            "new_java_execution": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observed-at", required=True)
    a = p.parse_args()
    data, hashes = read_evidence(a.prior_zip, PRIOR_ZIP, ["unarchived-artifact-triage.json"])
    out = run(data["unarchived-artifact-triage.json"], a.output, a.observed_at)
    out["prior_evidence"] = {"run_id": PRIOR_RUN, "artifact_id": PRIOR_ARTIFACT,
                             "archive_sha256": PRIOR_ZIP, "member_sha256": hashes}
    with (a.output / "recovery-manifest.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in out.items() if k != "artifacts"}, sort_keys=True))


if __name__ == "__main__":
    main()
