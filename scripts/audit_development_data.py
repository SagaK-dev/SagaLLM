"""Verify historical development-data pins without training or inference.

Run from the exact Parameter CE historical checkout with its dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

PINS = {
    "broad_train": "181346d5395efdd2ef376730321e1585d453b2b09a822302f6f2d03137e1aa8d",
    "stdio_manifest": "0cea8ded8a5c03d15b2b516a86665901266b0b375095579daa8d712661316a13",
    "previous_holdback": "d641a53e6498a096fda15e92871e10db6cc2380fe1bb1be543c8f4a06aeb035a",
    "fresh_holdback": "99c3de5b31128ecd7aa167014a3e736d883132a530ada2bad453e08f6d77317a",
    "pair_manifest": "4d9b2d6eb7ff216b653953eda1f355c4cae6def22d8a9af31016e51c7daa7d0b",
    "tokenizer": "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe",
}


def check(name, actual):
    if actual != PINS[name]:
        raise ValueError("Historical pin mismatch: " + name)
    return actual


def file_sha(path):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 2_000_000:
        raise ValueError("Missing or unsafe input")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(tokenizer):
    from scripts import train_java_m3_ranking as ranking
    from scripts import diagnose_java_m3_parameters as diagnostic
    from sagallm.java_stdio_curriculum import build_curriculum, verify_curriculum
    from sagallm.tokenization import load_tokenizer
    check("tokenizer", file_sha(tokenizer))
    for key, name in (("previous_holdback", "java_m3_development_holdback.json"),
                      ("fresh_holdback", "java_m3_parameter_holdback.json")):
        check(key, file_sha(Path("examples") / name))
    broad = ranking.broad_rows()
    raw = ("\n".join(ranking.broad_data._json(x) for x in broad) + "\n").encode("utf8")
    check("broad_train", hashlib.sha256(raw).hexdigest())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "stdio"
        build_curriculum(root)
        check("stdio_manifest", file_sha(root / "manifest.json"))
        stdio = verify_curriculum(root)
    _, pair = diagnostic.build_pairs(load_tokenizer(tokenizer))
    check("pair_manifest", pair["manifest_sha256"])
    return {"status": "DECLARED_DEVELOPMENT_DATA_PINS_MATCH", "pins": PINS,
            "broad_rows": len(broad), "stdio_files": stdio["files"],
            "pair_counts": pair, "actual_exposure_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED",
            "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    print(json.dumps(run(p.parse_args().tokenizer), sort_keys=True, indent=2))
