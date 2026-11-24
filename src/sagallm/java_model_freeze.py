"""Frozen model evaluation inputs, declared lineage and candidate byte pins.

Opaque artifact checks never deserialize checkpoints or execute generated Java.
A consistent declaration does not authenticate past training or inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .java_exposure_shards import CODE_SHA, _hash_file, _safe_source, audit_exposure_shards
from .java_independent_benchmark import SHA_PATTERN, _candidate
from .java_private_intake import _inventory, _metadata

SCHEMA = "sagallm-java-model-freeze-v1"
CANDIDATES = "sagallm-java-candidate-freeze-v1"
MAX_STAGES = 8
MODULES = (
    "java_independent_benchmark.py", "java_model_freeze.py",
    "java_exposure_shards.py", "java_private_intake.py",
    "java_overlap_screen.py", "java_behavior.py", "java_vm_review.py",
)


def evaluator_sha256() -> str:
    """Fingerprint the evaluator modules, not the whole Python environment."""
    root = Path(__file__).resolve().parent
    manifest = [[name, hashlib.sha256((root / name).read_bytes()).hexdigest()]
                for name in MODULES]
    return hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()


def _pin(value):
    return isinstance(value, str) and SHA_PATTERN.fullmatch(value) is not None


def _relative(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name
            or str(Path(name)) != name):
        raise ValueError("Invalid artifact path")
    return _safe_source(root, name)


def _generation(settings: dict):
    if (not isinstance(settings, dict)
            or set(settings) != {"seed", "max_new_tokens", "decoding", "prompt_path", "prompt_sha256",
                                 "generation_code_sha_declared"}
            or type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**32
            or type(settings["max_new_tokens"]) is not int
            or not 1 <= settings["max_new_tokens"] <= 2048
            or settings["decoding"] != "greedy"
            or not _pin(settings["prompt_sha256"])
            or not isinstance(settings["generation_code_sha_declared"], str)
            or not CODE_SHA.fullmatch(settings["generation_code_sha_declared"])):
        raise ValueError("Unsupported frozen generation settings")


def verify_model_freeze(suite_path: Path, suite_sha: str, *,
                        freeze_path: Path, expected_freeze_sha256: str,
                        provenance_path: Path, expected_provenance_sha256: str,
                        inventory_path: Path, expected_inventory_sha256: str,
                        shard_root: Path, artifact_root: Path, image: str) -> dict:
    freeze, freeze_sha = _metadata(freeze_path, expected_freeze_sha256)
    keys = {"schema", "suite_sha256", "provenance_sha256", "inventory_sha256",
            "shard_manifest_sha256", "tokenizer", "stages", "generation",
            "evaluator_sha256", "oci_image"}
    if (set(freeze) != keys or freeze["schema"] != SCHEMA
            or freeze["suite_sha256"] != suite_sha
            or freeze["provenance_sha256"] != expected_provenance_sha256
            or freeze["inventory_sha256"] != expected_inventory_sha256
            or not _pin(freeze["shard_manifest_sha256"])
            or freeze["evaluator_sha256"] != evaluator_sha256()
            or freeze["oci_image"] != image):
        raise ValueError("Model freeze binding mismatch")
    _generation(freeze["generation"])
    prompt = _relative(artifact_root, freeze["generation"]["prompt_path"])
    if prompt.stat().st_size > 128_000:
        raise ValueError("Prompt template too large")
    _hash_file(prompt, freeze["generation"]["prompt_sha256"])
    tokenizer = freeze["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"path", "sha256"}:
        raise ValueError("Malformed tokenizer declaration")
    tokenizer_path = _relative(artifact_root, tokenizer["path"])
    _hash_file(tokenizer_path, tokenizer["sha256"])
    inventory, _ = _metadata(inventory_path, expected_inventory_sha256)
    entries = _inventory(inventory)
    stages = freeze["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
        raise ValueError("Missing or excessive lineage stages")
    required = {i for i, entry in enumerate(entries)
                if entry["group"] in ("train", "pretraining")}
    covered = set()
    checkpoints = set()
    parent = None
    final_path = None
    lineage = []
    for index, stage in enumerate(stages):
        if (not isinstance(stage, dict)
                or set(stage) != {"kind", "checkpoint_path", "checkpoint_sha256",
                                  "parent_sha256", "training_code_sha_declared",
                                  "evidence_path", "evidence_sha256", "source_indices"}
                or stage["kind"] not in ("pretraining", "continued_pretraining", "sft")
                or (stage["kind"] == "pretraining") != (index == 0)
                or stage["parent_sha256"] != parent
                or not _pin(stage["checkpoint_sha256"])
                or stage["checkpoint_sha256"] in checkpoints
                or not isinstance(stage["training_code_sha_declared"], str)
                or not CODE_SHA.fullmatch(stage["training_code_sha_declared"])):
            raise ValueError("Broken or invalid declared checkpoint chain")
        refs = stage["source_indices"]
        group = "train" if stage["kind"] == "sft" else "pretraining"
        if (not isinstance(refs, list) or not refs
                or any(type(i) is not int or i not in required
                       or entries[i]["group"] != group for i in refs)
                or len(refs) != len(set(refs))):
            raise ValueError("Invalid stage exposure references")
        final_path = _relative(artifact_root, stage["checkpoint_path"])
        _hash_file(final_path, stage["checkpoint_sha256"])
        evidence = _relative(artifact_root, stage["evidence_path"])
        if evidence.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Stage evidence too large")
        _hash_file(evidence, stage["evidence_sha256"])
        covered.update(refs)
        checkpoints.add(stage["checkpoint_sha256"])
        parent = stage["checkpoint_sha256"]
        lineage.append({"kind": stage["kind"], "checkpoint_sha256": parent,
                        "parent_sha256": stage["parent_sha256"],
                        "training_code_sha_declared": stage["training_code_sha_declared"],
                        "evidence_sha256": stage["evidence_sha256"],
                        "source_indices": refs})
    if covered != required:
        raise ValueError("Declared training exposure not fully assigned")
    audit = audit_exposure_shards(
        suite_path, suite_sha, provenance_path=provenance_path,
        expected_provenance_sha256=expected_provenance_sha256,
        inventory_path=inventory_path, expected_inventory_sha256=expected_inventory_sha256,
        shard_root=shard_root, expected_manifest_sha256=freeze["shard_manifest_sha256"],
        checkpoint_path=final_path, expected_checkpoint_sha256=parent,
        tokenizer_path=tokenizer_path, expected_tokenizer_sha256=tokenizer["sha256"])
    if audit["training_code_sha_declared"] != stages[-1]["training_code_sha_declared"]:
        raise ValueError("Final stage code declaration differs from shard manifest")
    return {"schema": SCHEMA, "freeze_sha256": freeze_sha,
            "suite_sha256": suite_sha, "checkpoint_sha256": parent,
            "tokenizer_sha256": tokenizer["sha256"], "lineage": lineage,
            "generation": {key: value for key, value in freeze["generation"].items()
                           if key != "prompt_path"},
            "evaluator_sha256": freeze["evaluator_sha256"],
            "oci_image": image, "shard_audit": audit,
            "declared_lineage_consistent": True, "training_lineage_verified": False,
            "freeze_precedes_inference_verified": False,
            "generation_provenance_verified": False,
            "machine_status": audit["machine_status"], "blockers": audit["blockers"],
            "independence_proven": False, "model_selection_permitted": False}


def read_candidate_freeze(path: Path, expected: str, freeze_sha: str,
                          task_ids: list[str], candidate_dir: Path) -> dict:
    """Return exactly the checked source bytes decoded once; null is missing."""
    manifest, _ = _metadata(path, expected)
    if (set(manifest) != {"schema", "model_freeze_sha256", "candidates"}
            or manifest["schema"] != CANDIDATES
            or manifest["model_freeze_sha256"] != freeze_sha
            or not isinstance(manifest["candidates"], dict)
            or set(manifest["candidates"]) != set(task_ids)):
        raise ValueError("Candidate freeze binding mismatch")
    codes = {}
    for task_id in task_ids:
        expected_code = manifest["candidates"][task_id]
        source = candidate_dir / (task_id + ".java")
        if expected_code is None:
            if source.exists() or source.is_symlink():
                raise ValueError("Unexpected unpinned candidate")
            codes[task_id] = None
        else:
            if not _pin(expected_code):
                raise ValueError("Invalid candidate SHA")
            code = _candidate(source)
            if code is None or hashlib.sha256(code.encode("utf-8")).hexdigest() != expected_code:
                raise ValueError("Candidate SHA mismatch")
            codes[task_id] = code
    return codes
