"""Reconstruct scoped development sources; not a complete exposure freeze."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from sagallm.java_private_intake import INVENTORY_SCHEMA, _inventory, _texts
from scripts.audit_ancestor_inventory import scan_file
from scripts.audit_checkpoint_exposure import read_pinned
from scripts.audit_development_data import PINS
from scripts.audit_v018_training import require, unique_object

REVISION = "fdf53625d8e64596748113b53fd6f4c63e2a1c74"
ANCESTOR_RUN = 45794803813
ANCESTOR_ARTIFACT = 15104411637
ANCESTOR_ZIP = "83c3630a100d2e8bca9a6e66589feeabc015f2df721ad22fafbd05a38b4fd8d8"
LIMIT = 2_000_000
SUITES = (
    ("previous_holdback", "java_m3_development_holdback.json",
     "sagallm-m3-synthetic-development-holdback-v1", False),
    ("fresh_holdback", "java_m3_parameter_holdback.json",
     "sagallm-m3-parameter-development-v1", True),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def ancestor(path):
    raw = read_pinned(path, ANCESTOR_ZIP, LIMIT)
    names = {"ancestor-inventory.json", "ancestor-inventory-evidence.json"}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        require(len(members) == len(names) and {m.filename for m in members} == names,
                "Unexpected ancestor members")
        blobs = {}
        for member in members:
            require(0 < member.file_size <= LIMIT, "Ancestor member size")
            with archive.open(member) as handle:
                body = handle.read(LIMIT + 1)
            require(0 < len(body) <= LIMIT, "Ancestor member size")
            blobs[member.filename] = body
    inventory = parse(blobs["ancestor-inventory.json"])
    entries = _inventory(inventory)
    report = parse(blobs["ancestor-inventory-evidence.json"])
    require(len(entries) == 6 and report["source_count"] == 6
            and report["inventory_sha256"] == sha(blobs["ancestor-inventory.json"])
            and report["status"] == "SCOPED_ANCESTOR_SOURCE_BYTES_VERIFIED"
            and report["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and report["actual_exposure_verified"] is False
            and report["inventory_ready_for_independent_evaluation"] is False
            and report["model_selection_permitted"] is False
            and report["independent_evaluation"] == "NOT_RUN",
            "Ancestor scope mismatch")
    return entries, {name: sha(body) for name, body in blobs.items()}


def text_record(row, field, value):
    require(isinstance(value, str), "Non-text exposure field")
    require(isinstance(row["id"], str) and bool(row["id"])
            and isinstance(row["family"], str) and bool(row["family"]),
            "Invalid row identity")
    return {"id": row["id"], "family": row["family"], "field": field, "text": value}


def cases(row):
    require(isinstance(row["cases"], list) and len(row["cases"]) == 3,
            "Unexpected case count")
    result = []
    for index, case in enumerate(row["cases"]):
        require(isinstance(case, dict) and set(case) == {"stdin", "stdout"},
                "Unexpected case fields")
        for field in ("stdin", "stdout"):
            result.append(text_record(row, f"case_{index}_{field}", case[field]))
    return result


def convert_suite(raw, schema, has_source):
    obj = parse(raw)
    require(isinstance(obj, dict) and set(obj) == {"schema", "independent_benchmark", "rows"}
            and obj["schema"] == schema and obj["independent_benchmark"] is False
            and isinstance(obj["rows"], list) and len(obj["rows"]) == 6,
            "Unexpected suite structure")
    result, seen = [], set()
    fields = {"id", "family", "task", "cases"} | ({"source"} if has_source else set())
    for row in obj["rows"]:
        require(isinstance(row, dict) and set(row) == fields, "Unexpected suite row")
        result.append(text_record(row, "task", row["task"]))
        require(row["id"] not in seen, "Duplicate suite ID")
        seen.add(row["id"])
        if has_source:
            result.append(text_record(row, "source", row["source"]))
        result.extend(cases(row))
    return result


def convert_oracles(raw, expected_rows):
    rows = raw.splitlines()
    require(len(rows) == expected_rows, "Oracle count mismatch")
    result, seen = [], set()
    for line in rows:
        row = parse(line)
        require(isinstance(row, dict) and set(row) == {"id", "family", "cases"},
                "Unexpected oracle fields")
        result.extend(cases(row))
        require(row["id"] not in seen, "Duplicate oracle ID")
        seen.add(row["id"])
    return result


def derived(root, name, rows, original_sha, group):
    relative = "derived/" + name + ".jsonl"
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    require(0 < len(raw) <= LIMIT, "Derived data size")
    with (root / relative).open("xb") as handle:
        handle.write(raw)
    entry = {"group": group, "path": relative, "sha256": sha(raw), "format": "text-jsonl"}
    summary = scan_file(root, relative, "text-jsonl", sha(raw), len(rows))
    return entry, {**summary, "path": relative, "group": group,
                   "original_sha256": original_sha, "conversion": "field-text-jsonl-v1"}


def build(root, historical, ancestor_zip):
    previous, member_hashes = ancestor(ancestor_zip)
    require(root.is_dir() and not root.is_symlink() and not historical.is_symlink(),
            "Unsafe root")
    manifest_raw = read_pinned(root / "stdio/manifest.json", PINS["stdio_manifest"], LIMIT)
    manifest = parse(manifest_raw)
    specs = {"train.jsonl": 320, "validation.jsonl": 36, "validation-oracles.jsonl": 36}
    require(set(manifest["files"]) == set(specs), "Unexpected STDIO file coverage")
    entries, summaries = [], []
    for name, count in specs.items():
        info = manifest["files"][name]
        require(set(info) == {"sha256", "bytes", "rows"} and info["rows"] == count,
                "STDIO manifest count mismatch")
        raw = read_pinned(root / "stdio" / name, info["sha256"], LIMIT)
        require(len(raw) == info["bytes"], "STDIO byte count mismatch")
        if name == "validation-oracles.jsonl":
            oracle_raw = raw
            continue
        group = name.removesuffix(".jsonl")
        relative = "stdio/" + name
        summary = scan_file(root, relative, "chat-jsonl", info["sha256"], count)
        entries.append({"group": group, "path": relative, "sha256": info["sha256"],
                        "format": "chat-jsonl"})
        summaries.append({**summary, "path": relative, "group": group,
                          "original_sha256": info["sha256"], "conversion": "NONE"})
    (root / "derived").mkdir()
    entry, summary = derived(root, "stdio_validation_oracles",
                             convert_oracles(oracle_raw, 36), sha(oracle_raw), "validation")
    entries.append(entry)
    summaries.append(summary)
    for key, name, schema, has_source in SUITES:
        raw = read_pinned(historical / "examples" / name, PINS[key], LIMIT)
        entry, summary = derived(root, key, convert_suite(raw, schema, has_source),
                                 PINS[key], "development")
        entries.append(entry)
        summaries.append(summary)
    inventory = {"schema": INVENTORY_SCHEMA, "files": previous + entries}
    _inventory(inventory)
    raw = canonical(inventory) + b"\n"
    return raw, {
        "schema": "sagallm-scoped-development-exposure-v1",
        "status": "SCOPED_DEVELOPMENT_SOURCE_BYTES_VERIFIED",
        "historical_revision": REVISION, "stdio_manifest_sha256": PINS["stdio_manifest"],
        "ancestor_evidence": {"run_id": ANCESTOR_RUN, "artifact_id": ANCESTOR_ARTIFACT,
                              "archive_sha256": ANCESTOR_ZIP, "member_sha256": member_hashes},
        "inventory_sha256": sha(raw), "ancestor_entries_referenced": len(previous),
        "new_entries": len(entries), "sources": summaries,
        "ancestor_source_bytes_rechecked_here": False,
        "missing_evidence": ["ranking auxiliary at ranking revision",
                             "pair manifest conversion and other probes",
                             "checkpoint-specific sample order and rights review"],
        "full_exposure_inventory": "NOT_ESTABLISHED",
        "actual_exposure_verified": False, "training_lineage_verified": False,
        "shards_prepared": False, "inventory_ready_for_independent_evaluation": False,
        "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
        "model_selection_permitted": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "historical", "ancestor_zip", "inventory_output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    raw, report = build(a.root, a.historical, a.ancestor_zip)
    with a.inventory_output.open("xb") as handle:
        handle.write(raw)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
