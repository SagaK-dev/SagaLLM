"""Small-data preparation primitives; no network fetches or implied usage rights."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterator

EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SECRET = re.compile(r"(?i)(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"]?[^\s'\"]{8,}")


def read_documents(path: str | Path) -> Iterator[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open(encoding="utf-8") as f:
            paragraph: list[str] = []
            for line in f:
                if not line.strip():
                    if paragraph:
                        yield ''.join(paragraph).strip()
                        paragraph.clear()
                else:
                    paragraph.append(line)
            if paragraph:
                yield ''.join(paragraph).strip()
    elif suffix in (".jsonl", ".json"):
        with path.open(encoding="utf-8") as f:
            if suffix == ".json":
                items = json.load(f)
                if isinstance(items, dict):
                    items = [items]
            else:
                items = (json.loads(line) for line in f if line.strip())
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    yield item["text"]
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("text"):
                    yield row["text"]
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Install optional dependency pyarrow for Parquet") from exc
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["text"]):
            for item in batch.column(0).to_pylist():
                if isinstance(item, str):
                    yield item
    else:
        raise ValueError(f"Unsupported dataset extension: {suffix}")


def prepare_data(source: str | Path, output: str | Path, license_id: str,
                 seed: int = 42, validation_fraction: float = 0.1) -> dict:
    """Legacy in-memory text splitting; use data shard for near-duplicate removal.

    Privacy screening is heuristic; no comprehensive detection is claimed.
    """
    if license_id.lower() in ("", "unknown", "unverified", "none"):
        raise ValueError("Provide a verified dataset license before preparation")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    from .curation import sensitive_reason

    seen = set()
    clean = []
    rejected = {"empty": 0, "duplicate": 0, "sensitive": 0}
    for text in read_documents(source):
        text = text.strip()
        if not text:
            rejected["empty"] += 1
            continue
        if sensitive_reason(text):
            rejected["sensitive"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(digest)
        clean.append(text)
    if len(clean) < 2:
        raise ValueError("At least two eligible documents required for a document-level train/validation split")
    random.Random(seed).shuffle(clean)
    val_count = min(len(clean)-1, max(1, round(len(clean)*validation_fraction)))
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    splits = {"train": clean[val_count:], "validation": clean[:val_count]}
    for name, documents in splits.items():
        (destination / f"{name}.txt").write_text("\n\n".join(documents)+"\n", encoding="utf-8")
    report = {"source": str(source), "license": license_id, "seed": seed,
              "validation_fraction": validation_fraction, "split_counts": {k: len(v) for k,v in splits.items()},
              "rejected": rejected, "unique_documents": len(clean),
              "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              "known_limitations": ["exact dedup only", "regex-based sensitive-data detection only", "in-memory dedup"]}
    (destination / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report