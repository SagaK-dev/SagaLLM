"""Local M2.4 byte-preserving exposure shards and checkpoint-bound screening.

Neither the supplied inventory nor its link to historical training is independently
authenticated. No model deserialization, network, inference or Java execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import tracemalloc

from .java_private_intake import (
    GROUPS, MAX_LINE, _inventory, _metadata, _provenance, _scan, _texts,
)
from .java_independent_benchmark import SHA_PATTERN, _unique_object, read_suite

SCHEMA = "sagallm-java-exposure-shards-v1"
MAX_SHARD_BYTES = 16 * 1024 * 1024
MAX_SHARD_ROWS = 20_000
MAX_SHARDS = 2048
MAX_TOTAL_BYTES = 8 * 1024**3
MAX_ARTIFACT_BYTES = 32 * 1024**3
MAX_MANIFEST_BYTES = 2 * 1024**2
CODE_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked input")
    return path


def _safe_source(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if (not root.is_dir() or root.is_symlink() or rel.is_absolute()
            or ".." in rel.parts or not rel.parts
            or any((root / Path(*rel.parts[:i])).is_symlink()
                   for i in range(1, len(rel.parts) + 1))):
        raise ValueError("Unsafe source path")
    return _regular(root / rel)


def _hash_file(path: Path, expected: str) -> str:
    if not isinstance(expected, str) or not SHA_PATTERN.fullmatch(expected):
        raise ValueError("Expected exact SHA-256")
    size = 0
    digest = hashlib.sha256()
    with _regular(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARTIFACT_BYTES:
                raise ValueError("Artifact byte limit exceeded")
            digest.update(chunk)
    if not size or digest.hexdigest() != expected:
        raise ValueError("Artifact SHA mismatch")
    return expected


def _pin(value):
    return isinstance(value, str) and SHA_PATTERN.fullmatch(value) is not None


def prepare_exposure_shards(inventory_path: Path, expected_inventory_sha256: str,
                            exposure_root: Path, output: Path, *,
                            checkpoint_path: Path, checkpoint_sha256: str,
                            tokenizer_path: Path, tokenizer_sha256: str,
                            training_code_sha: str) -> dict:
    """Partition declared sources exactly; do not normalize, drop or reorder rows."""
    if not isinstance(training_code_sha, str) or not CODE_SHA.fullmatch(training_code_sha):
        raise ValueError("Expected declared training code commit SHA")
    _hash_file(checkpoint_path, checkpoint_sha256)
    _hash_file(tokenizer_path, tokenizer_sha256)
    inventory, inventory_sha = _metadata(inventory_path, expected_inventory_sha256)
    entries = _inventory(inventory)
    if not entries:
        raise ValueError("No exposure sources")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to overwrite shard set")
    if not output.parent.is_dir():
        raise ValueError("Output parent must exist")
    stage = Path(tempfile.mkdtemp(prefix=".exposure-", dir=output.parent))
    try:
        total_bytes = total_shards = 0
        sources = []
        for index, entry in enumerate(entries):
            digest = hashlib.sha256()
            source_bytes = source_rows = 0
            shards = []
            handle = None
            shard_digest = None
            shard_bytes = shard_rows = 0

            def close_shard():
                nonlocal handle
                if handle is not None:
                    handle.close()
                    shards[-1].update(sha256=shard_digest.hexdigest(),
                                      bytes=shard_bytes, rows=shard_rows)
                    handle = None

            try:
                with _safe_source(exposure_root, entry["path"]).open("rb") as source:
                    while True:
                        raw = source.readline(MAX_LINE + 1)
                        if not raw:
                            break
                        if len(raw) > MAX_LINE or len(raw) > MAX_SHARD_BYTES:
                            raise ValueError("Oversized exposure row")
                        _texts(json.loads(raw, object_pairs_hook=_unique_object), entry["format"])
                        if (handle is None or shard_bytes + len(raw) > MAX_SHARD_BYTES
                                or shard_rows >= MAX_SHARD_ROWS):
                            close_shard()
                            total_shards += 1
                            if total_shards > MAX_SHARDS:
                                raise ValueError("Shard count limit exceeded")
                            name = f"source-{index:03d}-{len(shards):05d}.jsonl"
                            handle = (stage / name).open("xb")
                            shards.append({"path": name})
                            shard_digest = hashlib.sha256()
                            shard_bytes = shard_rows = 0
                        total_bytes += len(raw)
                        if total_bytes > MAX_TOTAL_BYTES:
                            raise ValueError("Total exposure byte limit exceeded")
                        handle.write(raw)
                        shard_digest.update(raw)
                        digest.update(raw)
                        shard_bytes += len(raw)
                        shard_rows += 1
                        source_bytes += len(raw)
                        source_rows += 1
            finally:
                close_shard()
            if not source_rows or digest.hexdigest() != entry["sha256"]:
                raise ValueError("Empty source or source SHA mismatch")
            sources.append({"group": entry["group"], "format": entry["format"],
                            "source_sha256": entry["sha256"], "bytes": source_bytes,
                            "rows": source_rows, "shards": shards})
        manifest = {"schema": SCHEMA, "inventory_sha256": inventory_sha,
                    "checkpoint_sha256": checkpoint_sha256,
                    "tokenizer_sha256": tokenizer_sha256,
                    "training_code_sha_declared": training_code_sha,
                    "sources": sources}
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest size limit exceeded")
        (stage / "manifest.json").write_bytes(raw)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Refuse to overwrite shard set")
        stage.rename(output)
        return {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "sources": len(sources), "shards": total_shards, "bytes": total_bytes,
                "training_lineage_verified": False}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _read_manifest(root: Path, expected: str) -> dict:
    if not _pin(expected) or not root.is_dir() or root.is_symlink():
        raise ValueError("Pinned shard directory required")
    with _regular(root / "manifest.json").open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Manifest SHA mismatch")
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    fields = {"schema", "inventory_sha256", "checkpoint_sha256", "tokenizer_sha256",
              "training_code_sha_declared", "sources"}
    if (not isinstance(manifest, dict) or set(manifest) != fields
            or manifest["schema"] != SCHEMA
            or any(not _pin(manifest[k]) for k in
                   ("inventory_sha256", "checkpoint_sha256", "tokenizer_sha256"))
            or not isinstance(manifest["training_code_sha_declared"], str)
            or not CODE_SHA.fullmatch(manifest["training_code_sha_declared"])
            or not isinstance(manifest["sources"], list)):
        raise ValueError("Malformed shard manifest")
    return manifest


def audit_exposure_shards(suite_path: Path, expected_suite_sha256: str, *,
                          provenance_path: Path, expected_provenance_sha256: str,
                          inventory_path: Path, expected_inventory_sha256: str,
                          shard_root: Path, expected_manifest_sha256: str,
                          checkpoint_path: Path, expected_checkpoint_sha256: str,
                          tokenizer_path: Path, expected_tokenizer_sha256: str) -> dict:
    suite, suite_sha, protocol = read_suite(suite_path)
    if suite_sha != expected_suite_sha256:
        raise ValueError("Suite SHA mismatch")
    provenance, provenance_sha = _metadata(provenance_path, expected_provenance_sha256)
    inventory, inventory_sha = _metadata(inventory_path, expected_inventory_sha256)
    entries = _inventory(inventory)
    manifest = _read_manifest(shard_root, expected_manifest_sha256)
    if (manifest["inventory_sha256"] != inventory_sha
            or manifest["checkpoint_sha256"] != expected_checkpoint_sha256
            or manifest["tokenizer_sha256"] != expected_tokenizer_sha256
            or not entries or len(entries) != len(manifest["sources"])):
        raise ValueError("Manifest binding mismatch")
    _hash_file(checkpoint_path, expected_checkpoint_sha256)
    _hash_file(tokenizer_path, expected_tokenizer_sha256)
    blockers = _provenance(suite, suite_sha, provenance) + protocol["review_blockers"]
    missing = sorted(set(GROUPS) - {e["group"] for e in entries})
    if missing:
        blockers.append("EXPOSURE_GROUPS_MISSING")
    expected_names = {"manifest.json"}
    seen = set()
    summaries = []
    total_bytes = total_rows = 0
    # Check declarations and file set before scanning any shard.
    for index, (source, entry) in enumerate(zip(manifest["sources"], entries)):
        if (not isinstance(source, dict)
                or set(source) != {"group", "format", "source_sha256", "bytes", "rows", "shards"}
                or source["group"] != entry["group"] or source["format"] != entry["format"]
                or source["source_sha256"] != entry["sha256"]
                or type(source["bytes"]) is not int or type(source["rows"]) is not int
                or source["bytes"] <= 0 or source["rows"] <= 0
                or not isinstance(source["shards"], list) or not source["shards"]):
            raise ValueError("Invalid source declaration")
        for ordinal, shard in enumerate(source["shards"]):
            if (not isinstance(shard, dict)
                    or set(shard) != {"path", "sha256", "bytes", "rows"}
                    or shard["path"] != f"source-{index:03d}-{ordinal:05d}.jsonl"
                    or not _pin(shard["sha256"])
                    or type(shard["bytes"]) is not int
                    or not 1 <= shard["bytes"] <= MAX_SHARD_BYTES
                    or type(shard["rows"]) is not int
                    or not 1 <= shard["rows"] <= MAX_SHARD_ROWS):
                raise ValueError("Invalid shard declaration or order")
            expected_names.add(shard["path"])
            total_bytes += shard["bytes"]
            total_rows += shard["rows"]
            if len(expected_names) - 1 > MAX_SHARDS or total_bytes > MAX_TOTAL_BYTES:
                raise ValueError("Declared scan limit exceeded")
    # Reject extra files, directories and symlinks, not just missing files.
    if {p.name for p in shard_root.iterdir()} != expected_names:
        raise ValueError("Shard file set mismatch")
    for source in manifest["sources"]:
        digest = hashlib.sha256()
        count = size = 0
        for ordinal, shard in enumerate(source["shards"]):
            # One bounded shard at a time. Hash exactly the bytes consumed by
            # the existing scanner, including source reconstruction in order.
            files, hits = _scan(shard_root, [{"group": source["group"],
                            "format": source["format"], "path": shard["path"],
                            "sha256": shard["sha256"]}], suite["tasks"],
                            source_digest=digest, expected_bytes=shard["bytes"],
                            expected_rows=shard["rows"],
                            require_final_lf=ordinal < len(source["shards"]) - 1)
            actual = files[0]
            if actual["bytes"] != shard["bytes"] or actual["records"] != shard["rows"]:
                raise ValueError("Shard byte or record count mismatch")
            count += actual["records"]
            size += actual["bytes"]
            seen.update((hit["task_index"], hit["group"], hit["kind"]) for hit in hits)
        if (count != source["rows"] or size != source["bytes"]
                or digest.hexdigest() != source["source_sha256"]):
            raise ValueError("Source reconstruction mismatch")
        summaries.append({"group": source["group"], "sha256": source["source_sha256"],
                          "bytes": size, "records": count, "shards": len(source["shards"])})
    if seen:
        blockers.append("OVERLAP_REVIEW_LEADS_FOUND")
    return {"schema": SCHEMA, "manifest_sha256": expected_manifest_sha256,
            "suite_sha256": suite_sha, "provenance_sha256": provenance_sha,
            "inventory_sha256": inventory_sha,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "tokenizer_sha256": expected_tokenizer_sha256,
            "training_code_sha_declared": manifest["training_code_sha_declared"],
            "scanned_sources": summaries, "scanned_shards": len(expected_names)-1,
            "scanned_bytes": total_bytes, "scanned_records": total_rows,
            "missing_exposure_groups": missing,
            "overlap_leads": [{"task_index": i, "group": g, "kind": k}
                              for i, g, k in sorted(seen)],
            "blockers": sorted(set(blockers)),
            "machine_status": "BLOCKED" if blockers else "PREFLIGHT_ONLY",
            "declared_inventory_bytes_verified": True,
            "training_lineage_verified": False,
            "inventory_completeness_verified": False,
            "independence_proven": False, "review_gate": "REVIEW_REQUIRED",
            "model_selection_permitted": False, "behavior": "NOT_RUN",
            "generated_java_executed": False}


def measure_audit(*args, **kwargs) -> tuple[dict, dict]:
    """Optional local measurement; tracemalloc covers Python allocations, not RSS."""
    if tracemalloc.is_tracing():
        raise RuntimeError("Measurement requires inactive tracemalloc")
    tracemalloc.start()
    start = time.perf_counter()
    try:
        report = audit_exposure_shards(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return report, {"elapsed_seconds": elapsed, "peak_python_bytes": peak,
                    "bytes_per_second": report["scanned_bytes"] / max(elapsed, 1e-9),
                    "measurement_scope": "TRACEMALLOC_NOT_PROCESS_RSS"}
