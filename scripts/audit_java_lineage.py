"""Verify historical artifact links, not completeness of training exposure."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from scripts.audit_v018_training import (
    BASE, BEST, TOKENIZER, DATA, PARENT_ZIP, OUTPUT_ZIP, REPORT, LAST,
    audit_zip, audit_report, require, unique_object, MAX_BYTES, digest,
)

ROOT_ZIP = "e64f49de36921c07c4db543451e543e93b128567d3311dfea0bb4d1243917769"
ROOT = "96005da938d3317fe2ba5185b71cdbb6ed2fee6bc4b9c39e370006d5f2917d1b"
MID_LAST = "700520a9286859ed7d12a9d9aebb04eaf9933302cfdd09fbba478c452abb30d0"
SOURCES = {
    "algorithms": "f4d6708297f86629a9e415070935b903ea8b1858",
    "gson": "d9b5079eaa4567219771bba6d04289cf45c7589b",
    "guava": "36e3995499d6c3b74b1207d362fa9e19915d2589",
    "patterns": "4cabb204f240240bea319605a5ffa46ded728001",
}


def read_json(path, pin, member):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES,
            "Invalid archive")
    with path.open("rb") as f:
        require(digest(f) == pin, "Archive pin mismatch")
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            rows = [x for x in z.infolist() if x.filename == member]
            require(len(rows) == 1 and 0 < rows[0].file_size <= 1024 * 1024,
                    "Missing or oversized JSON")
            raw = z.read(rows[0])
    return json.loads(raw, object_pairs_hook=unique_object), hashlib.sha256(raw).hexdigest()


def links(root, mid, end, public_manifest, public_sha, mid_manifest_sha):
    require(root["status"] == mid["status"] == end["status"] == "TRAINED_CPU", "Training status")
    require(root["training_task"] == "JAVA_SOURCE_NEXT_TOKEN_ONLY", "Root task")
    require(root["best_checkpoint_sha256"] == mid["checkpoint_sha256_base"] == ROOT,
            "Root link mismatch")
    require(mid["best_checkpoint_sha256"] == end["checkpoint_sha256_base"] == BASE,
            "SFT link mismatch")
    require(end["best_checkpoint_sha256"] == BEST, "Final link mismatch")
    require(all(p["tokenizer_sha256"] == TOKENIZER for p in (root, mid, end)), "Tokenizer mismatch")
    require(root["dataset_manifest_sha256"] == public_sha, "Root corpus mismatch")
    require(mid["data_manifest_sha256"] == end["data_manifest_sha256"] == mid_manifest_sha == DATA,
            "SFT corpus mismatch")
    sources = public_manifest["sources"]
    require(len(sources) == len(SOURCES)
            and {s["name"]: s["revision"] for s in sources} == SOURCES, "Upstream revision mismatch")
    return {"status": "HISTORICAL_ARTIFACT_LINKS_VERIFIED",
            "checkpoint_chain": [ROOT, BASE, BEST], "tokenizer_sha256": TOKENIZER,
            "public_manifest_sha256": public_sha, "sft_manifest_sha256": DATA,
            "public_sources": SOURCES, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED",
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}


def audit(root_zip, mid_zip, end_zip):
    root, root_report_sha = read_json(root_zip, ROOT_ZIP, "java-model/train-report.json")
    mid, mid_report_sha = read_json(mid_zip, PARENT_ZIP, "java-sft-model/train-report.json")
    end, end_report_sha = read_json(end_zip, OUTPUT_ZIP, "java-context-model/train-report.json")
    public, public_sha = read_json(root_zip, ROOT_ZIP, "java-corpus/manifest.json")
    _, mid_sha = read_json(mid_zip, PARENT_ZIP, "java-sft-data/manifest.json")
    require(end_report_sha == REPORT, "Final report mismatch")
    result = links(root, mid, end, public, public_sha, mid_sha)
    audit_zip(root_zip, ROOT_ZIP, {"java-model/java-best.pt": ROOT,
        "java-model/java-last.pt": root["last_checkpoint_sha256"],
        "java-model/tokenizer.json": TOKENIZER})
    audit_zip(mid_zip, PARENT_ZIP, {"java-sft-model/java-sft-best.pt": BASE,
        "java-sft-model/java-sft-last.pt": MID_LAST, "java-sft-model/tokenizer.json": TOKENIZER})
    audit_zip(end_zip, OUTPUT_ZIP, {"java-context-model/java-sft-best.pt": BEST,
        "java-context-model/java-sft-last.pt": LAST, "java-context-model/tokenizer.json": TOKENIZER,
        "java-context-data/manifest.json": DATA})
    result["report_sha256"] = [root_report_sha, mid_report_sha, end_report_sha]
    result["run_ids"] = [44981937649, 44982940959, 44994208434]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "mid", "end"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.mid, args.end), indent=2, sort_keys=True))
