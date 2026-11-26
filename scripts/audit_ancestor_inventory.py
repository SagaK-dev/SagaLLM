"""Build a scoped ancestor exposure inventory, not a complete model freeze."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from sagallm.java_private_intake import INVENTORY_SCHEMA, MAX_LINE, _inventory, _texts
from scripts.audit_checkpoint_exposure import (
    ARCHIVE, DEVELOPMENT_ZIP, PINS, SHA, load_reports, map_reports, read_pinned,
)
from scripts.audit_java_lineage import ROOT_ZIP, PARENT_ZIP, read_json
from scripts.audit_reconstructed_corpora import SPLITS, verify
from scripts.audit_v018_training import DATA, require, unique_object

MAP_ZIP = "9a089d01bc02fbc32758f6b4266c2f7035d0b85a85da1abfe29971f08063679c"
MAP_RUN = 45792751412
MAP_ARTIFACT = 15103840915
MAX_FILE = 512 * 1024 * 1024
MAX_ROWS = 500_000


def read_map(path, reports):
    raw = read_pinned(path, MAP_ZIP, 1024 * 1024)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        require(len(members) == 1 and members[0].filename == "checkpoint-exposure-map.json"
                and 0 < members[0].file_size <= 1024 * 1024, "Unexpected map member")
        with z.open(members[0]) as f:
            body = f.read(1024 * 1024 + 1)
        require(len(body) <= 1024 * 1024, "Oversized map")
    data = json.loads(body, object_pairs_hook=unique_object)
    require(isinstance(data, dict), "Expected map object")
    evidence = data.pop("reconstruction_evidence")
    require(set(evidence) == {"run_id", "artifact_id", "archive_sha256", "member_sha256"}
            and evidence["run_id"] == 45790130406
            and evidence["artifact_id"] == 15101943417
            and evidence["archive_sha256"] == DEVELOPMENT_ZIP
            and isinstance(evidence["member_sha256"], str)
            and SHA.fullmatch(evidence["member_sha256"]), "Reconstruction reference mismatch")
    require(data == map_reports(reports), "Map differs from pinned report recount")
    return data, hashlib.sha256(body).hexdigest()


def scan_file(root, relative, fmt, expected_sha, expected_rows):
    path = root / relative
    require(root.is_dir() and not root.is_symlink()
            and not path.parent.is_symlink() and not path.is_symlink()
            and path.is_file(), "Unsafe source")
    require(type(expected_rows) is int and 0 < expected_rows <= MAX_ROWS,
            "Unexpected expected count")
    digest = hashlib.sha256()
    count = size = 0
    with path.open("rb") as f:
        while True:
            raw = f.readline(MAX_LINE + 1)
            if not raw:
                break
            count += 1
            size += len(raw)
            require(len(raw) <= MAX_LINE and size <= MAX_FILE and count <= MAX_ROWS,
                    "Source exceeds scan bound")
            _texts(json.loads(raw, object_pairs_hook=unique_object), fmt)
            digest.update(raw)
    require(count == expected_rows and digest.hexdigest() == expected_sha,
            "Source bytes or record count mismatch")
    return {"rows": count, "bytes": size, "sha256": digest.hexdigest()}


def build(root, public, public_sha, sft, sft_sha):
    require(sft_sha == DATA, "SFT manifest mismatch")
    require(root.is_dir() and not root.is_symlink(), "Unsafe corpus root")
    verify(root / "public", public, public_sha, False)
    verify(root / "sft", sft, sft_sha, True)
    entries, summaries = [], []
    for corpus, manifest, pin, fmt in (
        ("public", public, public_sha, "text-jsonl"),
        ("sft", sft, sft_sha, "chat-jsonl"),
    ):
        for split in SPLITS:
            group = "pretraining" if corpus == "public" and split == "train" else split
            relative = f"{corpus}/{split}.jsonl"
            expected_sha = (manifest["file_hashes"][split] if corpus == "public"
                            else manifest["files"][split]["sha256"])
            expected_rows = (manifest["splits"][split] if corpus == "public"
                             else manifest["files"][split]["count"])
            summary = scan_file(root, relative, fmt, expected_sha, expected_rows)
            entries.append({"group": group, "path": relative, "sha256": expected_sha,
                            "format": fmt})
            summaries.append({"corpus": corpus, "split": split, "group": group,
                              "manifest_sha256": pin, **summary})
    inventory = {"schema": INVENTORY_SCHEMA, "files": entries}
    _inventory(inventory)
    # Historical source bytes are not normalized or rewritten.
    raw = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode("utf-8")
    return raw, summaries


def audit(root, public_zip, sft_zip, map_zip, reports):
    mapping, member_sha = read_map(map_zip, reports)
    public, public_sha = read_json(public_zip, ROOT_ZIP, "java-corpus/manifest.json")
    sft, sft_sha = read_json(sft_zip, PARENT_ZIP, "java-sft-data/manifest.json")
    raw, summaries = build(root, public, public_sha, sft, sft_sha)
    broad_equal = sft["files"]["train"]["sha256"] == PINS["broad_train"]
    # This subset is shared by declared ancestors, not a full per-model inventory.
    result = {"schema": "sagallm-ancestor-exposure-inventory-evidence-v1",
              "status": "SCOPED_ANCESTOR_SOURCE_BYTES_VERIFIED",
              "map_evidence": {"run_id": MAP_RUN, "artifact_id": MAP_ARTIFACT,
                               "archive_sha256": MAP_ZIP, "member_sha256": member_sha},
              "inventory_sha256": hashlib.sha256(raw).hexdigest(),
              "source_count": len(summaries), "sources": summaries,
              "ancestor_checkpoint_links": mapping["ancestor_checkpoint_links"],
              "declared_descendant_checkpoints": [r["checkpoint_sha256"] for r in mapping["checkpoints"]],
              "broad_train_matches_sft_train_pin": broad_equal,
              "missing_inventory_groups": ["development"],
              "missing_evidence": ["development and STDIO source files",
                  "ranking auxiliary inputs and pair manifest conversion",
                  "other experiments and probes", "optimizer sample order", "rights review"],
              "full_exposure_inventory": "NOT_ESTABLISHED",
              "actual_exposure_verified": False, "training_lineage_verified": False,
              "checkpoint_bytes_verified_here": False, "shards_prepared": False,
              "inventory_ready_for_independent_evaluation": False,
              "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
              "model_selection_permitted": False}
    return raw, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "public_zip", "sft_zip", "map_zip", "inventory_output"):
        p.add_argument(name, type=Path)
    p.add_argument("--reports", type=Path, default=ARCHIVE)
    a = p.parse_args()
    raw, result = audit(a.root, a.public_zip, a.sft_zip, a.map_zip, load_reports(a.reports))
    with a.inventory_output.open("xb") as f:
        f.write(raw)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
