"""Cross-corpus exact and whitespace-insensitive Java source review leads."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.audit_reconstructed_corpora import verify, SPLITS
from scripts.audit_java_lineage import read_json, ROOT_ZIP, PARENT_ZIP
from scripts.audit_v018_training import DATA, require, unique_object

MAX_ROWS = 100_000
MAX_BYTES = 512 * 1024 * 1024
MAX_LINE = 2 * 1024 * 1024


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprints(text):
    require(isinstance(text, str) and bool(text.strip()), "Invalid source")
    return sha(text), sha("".join(text.split()))


def read_split(path, synthetic):
    exact, compact = {}, {}
    count = size = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(MAX_LINE + 1)
            if not raw:
                break
            count += 1
            size += len(raw)
            require(count <= MAX_ROWS and size <= MAX_BYTES and len(raw) <= MAX_LINE,
                    "Corpus scan limit exceeded")
            row = json.loads(raw, object_pairs_hook=unique_object)
            if synthetic:
                messages = row["messages"]
                require(len(messages) == 2 and messages[1]["role"] == "assistant",
                        "Unexpected SFT record")
                source = messages[1]["content"]
            else:
                source = row["text"]
                require(sha(source) == row["sha256"], "Public source hash mismatch")
            a, b = fingerprints(source)
            exact[a] = exact.get(a, 0) + 1
            compact[b] = compact.get(b, 0) + 1
    require(count > 0, "Empty split")
    return count, exact, compact


def compare(left, right):
    def matches(a, b):
        shared = a.keys() & b.keys()
        return {"shared_fingerprints": len(shared),
                "public_records": sum(a[k] for k in shared),
                "sft_records": sum(b[k] for k in shared),
                "record_pairs": sum(a[k] * b[k] for k in shared)}
    return {"public_total": left[0], "sft_total": right[0],
            "exact": matches(left[1], right[1]),
            "whitespace_insensitive": matches(left[2], right[2])}


def audit(public_zip, sft_zip, public_dir, sft_dir):
    public, pub_sha = read_json(public_zip, ROOT_ZIP, "java-corpus/manifest.json")
    sft, sft_sha = read_json(sft_zip, PARENT_ZIP, "java-sft-data/manifest.json")
    require(sft_sha == DATA, "SFT manifest pin mismatch")
    verify(public_dir, public, pub_sha, False)
    verify(sft_dir, sft, sft_sha, True)
    left = {s: read_split(public_dir / (s + ".jsonl"), False) for s in SPLITS}
    right = {s: read_split(sft_dir / (s + ".jsonl"), True) for s in SPLITS}
    require(all(left[s][0] == public["splits"][s] and
                right[s][0] == sft["files"][s]["count"] for s in SPLITS), "Count mismatch")
    matrix = []
    for p in SPLITS:
        for s in SPLITS:
            cell = compare(left[p], right[s])
            matrix.append({"public_split": p, "sft_split": s,
                           "train_eval_boundary": (p == "train") != (s == "train"), **cell})
    # Detect modifications between verification and scan.
    verify(public_dir, public, pub_sha, False)
    verify(sft_dir, sft, sft_sha, True)
    hits = any(c["train_eval_boundary"] and c["whitespace_insensitive"]["record_pairs"]
               for c in matrix)
    return {"status": "REVIEW_REQUIRED" if hits else "NO_MATCH_IN_SCOPED_SCREEN",
            "public_manifest_sha256": pub_sha, "sft_manifest_sha256": sft_sha,
            "matrix": matrix, "semantic_independence": "NOT_ESTABLISHED",
            "full_exposure_inventory": "NOT_ESTABLISHED",
            "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("public_zip", "sft_zip", "public_dir", "sft_dir"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.public_zip, a.sft_zip, a.public_dir, a.sft_dir),
                     indent=2, sort_keys=True))
