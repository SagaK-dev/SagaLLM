"""Preserve explicitly pinned JSON report bytes from Actions artifacts.

No extraction of paths, weights, sources, stdout/stderr or raw logs.
Uses the reviewed manifest, GitHub metadata and exact byte hashes.
Run from a clean evidence branch; does not perform any GitHub writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile

REPO = "SagaK-dev/SagaLLM"
MAX_ARCHIVE = 128 * 1024 * 1024
MAX_REPORT = 8 * 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")
SLUG = re.compile(r"[a-z0-9-]{1,80}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_manifest(path):
    obj = json.loads(path.read_bytes())
    require(obj["schema"] == "sagallm-pinned-report-manifest-v1" and obj["repository"] == REPO,
            "Wrong manifest")
    rows = obj["entries"]
    require(0 < len(rows) <= 100, "Invalid manifest size")
    seen = set()
    for r in rows:
        require(type(r["run_id"]) is int and r["run_id"] > 0
                and type(r["artifact_id"]) is int and r["artifact_id"] > 0
                and GIT_SHA.fullmatch(r["commit_sha"]) and HEX.fullmatch(r["archive_sha256"])
                and HEX.fullmatch(r["report_sha256"]) and SLUG.fullmatch(r["slug"]), "Invalid row")
        name = PurePosixPath(r["member"])
        require(not name.is_absolute() and ".." not in name.parts and name.suffix == ".json"
                and str(name) == r["member"] and "\\" not in r["member"], "Invalid member")
        key = (r["run_id"], r["slug"])
        require(key not in seen, "Duplicate output")
        seen.add(key)
    return rows


def verify_metadata(row, meta):
    run = meta["workflow_run"]
    require(meta["id"] == row["artifact_id"] and meta["expired"] is False
            and meta["digest"] == "sha256:" + row["archive_sha256"]
            and run["id"] == row["run_id"] and run["head_sha"] == row["commit_sha"]
            and 0 < meta["size_in_bytes"] <= MAX_ARCHIVE, "Artifact metadata mismatch")


def extract_report(row, archive):
    require(archive.is_file() and not archive.is_symlink()
            and 0 < archive.stat().st_size <= MAX_ARCHIVE, "Archive size or type")
    require(sha(archive.read_bytes()) == row["archive_sha256"], "Archive SHA mismatch")
    with zipfile.ZipFile(archive) as z:
        matched = [x for x in z.infolist() if x.filename == row["member"]]
        require(len(matched) == 1 and not matched[0].is_dir()
                and 0 < matched[0].file_size <= MAX_REPORT, "Report member missing/oversize")
        raw = z.read(matched[0])
    require(sha(raw) == row["report_sha256"], "Report SHA mismatch")
    require(isinstance(json.loads(raw), dict), "Report must be JSON object")
    return raw


def gh_api(path):
    return subprocess.check_output(["gh", "api", path], timeout=120)


def archive_all(manifest, output):
    rows = load_manifest(manifest)
    require(output.is_dir() and not output.is_symlink(), "Output directory must exist")
    pending = []
    # Stage verified bytes in memory. Do not write any report before all
    # inputs pass; on transient artifact expiry leave original ledger intact.
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            artifact = row["artifact_id"]
            endpoint = f"repos/{REPO}/actions/artifacts/{artifact}"
            verify_metadata(row, json.loads(gh_api(endpoint)))
            file = Path(tmp) / "archive.zip"
            with file.open("wb") as f:
                subprocess.run(["gh", "api", endpoint + "/zip"], stdout=f, check=True, timeout=120)
            raw = extract_report(row, file)
            file.unlink()
            target = output / f'{row["run_id"]}-{row["slug"]}.json'
            if target.exists() or target.is_symlink():
                require(not target.is_symlink() and target.is_file()
                        and target.read_bytes() == raw, "Existing report conflicts; no overwrite")
            else:
                pending.append((target, raw))
            print(f'VERIFIED {row["run_id"]} {row["report_sha256"]}', flush=True)
    for target, raw in pending:
        with target.open("xb") as f:
            f.write(raw)
    print(f"ARCHIVE_VERIFIED reports={len(rows)} new={len(pending)}", flush=True)


def verify_local(manifest, output):
    rows = load_manifest(manifest)
    for row in rows:
        target = output / f'{row["run_id"]}-{row["slug"]}.json'
        require(target.is_file() and not target.is_symlink()
                and 0 < target.stat().st_size <= MAX_REPORT, "Local report missing")
        raw = target.read_bytes()
        require(sha(raw) == row["report_sha256"] and isinstance(json.loads(raw), dict),
                "Local report mismatch")
    print(f"LOCAL_REPORTS_VERIFIED reports={len(rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()
    (verify_local if args.verify_local else archive_all)(args.manifest, args.output)


if __name__ == "__main__":
    main()
