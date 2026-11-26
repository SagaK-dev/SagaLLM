"""Compare reconstructed corpus bytes against pinned historical artifact manifests."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.audit_java_lineage import read_json, ROOT_ZIP, PARENT_ZIP
from scripts.audit_v018_training import DATA, require

SPLITS = ("train", "validation", "test")


def file_hash(path):
    require(path.is_file() and not path.is_symlink(), "Missing or unsafe file")
    require(path.stat().st_size <= 512 * 1024 * 1024, "File too large")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root, manifest, pin, synthetic):
    require(not root.is_symlink() and root.is_dir(), "Unsafe corpus root")
    require(file_hash(root / "manifest.json") == pin, "Manifest mismatch")
    hashes = ({s: manifest["files"][s]["sha256"] for s in SPLITS}
              if synthetic else manifest["file_hashes"])
    require(set(hashes) == set(SPLITS), "Unexpected split inventory")
    result = {}
    for split in SPLITS:
        path = root / (split + ".jsonl")
        actual = file_hash(path)
        require(actual == hashes[split], "Split hash mismatch: " + split)
        result[split] = actual
    if not synthetic:
        for source in manifest["sources"]:
            name = source["name"]
            require(name in ("algorithms", "gson", "guava", "patterns"), "Unexpected source")
            require(file_hash(root / "LICENSES" / (name + ".txt")) ==
                    source["license_sha256"], "License hash mismatch")
    return {"manifest_sha256": pin, "split_sha256": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("public_zip", "sft_zip", "public_dir", "sft_dir"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    public, pub_sha = read_json(a.public_zip, ROOT_ZIP, "java-corpus/manifest.json")
    sft, sft_sha = read_json(a.sft_zip, PARENT_ZIP, "java-sft-data/manifest.json")
    require(sft_sha == DATA, "SFT manifest pin mismatch")
    print(json.dumps({"status": "RECONSTRUCTED_CORPUS_BYTES_MATCH",
        "public": verify(a.public_dir, public, pub_sha, False),
        "sft": verify(a.sft_dir, sft, sft_sha, True),
        "full_exposure_inventory": "NOT_ESTABLISHED",
        "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
        "model_selection_permitted": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
