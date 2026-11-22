"""Opt-in behavior evaluation of held-out original Java tasks in local OCI.

No Docker/Podman: NOT_RUN. No explicit --allow-execution: NOT_RUN.
Requires a disposable testing host; containers are NOT a security guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sagallm.java_ai import LocalCheckpointGenerator, extract_java
from sagallm.java_behavior import oci_image_ready, run_java_in_oci, task_expected_stdout
from sagallm.java_generated_data import verify_java_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(dataset: str | Path, checkpoint: str | Path, tokenizer: str | Path, *,
             backend: str = "docker", image: str = "eclipse-temurin:21-jdk",
             allow_execution: bool = False, max_examples: int = 15) -> dict:
    if not 1 <= max_examples <= 15:
        raise ValueError("max_examples must be 1..15")
    audit = verify_java_dataset(dataset)
    result = {
        "protocol": "sagallm-original-java-oci-behavior-v1",
        "status": "NOT_RUN", "dataset_status": audit["status"],
        "dataset_manifest_sha256": _sha(Path(dataset) / "manifest.json"),
        "evaluated_items": 0, "behavior_passes": None,
        "source_of_expected_output": "independent Python oracles for three synthetic held-out families",
        "risk": "OCI containers are not a security guarantee; use a disposable dedicated VM",
    }
    if not allow_execution:
        result["reason"] = "Explicit --allow-execution required; did not load model or execute Java"
        return result
    if not oci_image_ready(backend, image):
        result["reason"] = "Local OCI runtime unavailable; did not load model or execute Java"
        return result
    generator = LocalCheckpointGenerator(checkpoint, tokenizer)
    result["checkpoint_sha256"] = _sha(Path(checkpoint))
    result["tokenizer_sha256"] = _sha(Path(tokenizer))
    records = []
    test_file = Path(dataset) / "test.jsonl"
    for line in test_file.read_text(encoding="utf-8").splitlines()[:max_examples]:
        task = json.loads(line)
        prompt = "User: " + task["messages"][0]["content"] + "\nAssistant: "
        try:
            response = generator(prompt, max_new_tokens=170)
            code = extract_java(response, "Main")
        except (ValueError, RuntimeError, TimeoutError) as exc:
            outcome = {"status": "REJECTED", "behavior_match": False,
                       "reason": "Model did not produce a valid Java compilation unit"}
        else:
            expected = task_expected_stdout(task["id"], task["family"])
            outcome = run_java_in_oci(code, expected, backend=backend, image=image,
                                      allow_execution=True)
        records.append({"id": task["id"], "family": task["family"], **outcome})
    result.update({
        "status": "EVALUATED" if all(x["status"] not in ("NOT_RUN", "RUNTIME_ERROR") for x in records)
                  else "PARTIAL_OR_UNAVAILABLE",
        "evaluated_items": len(records),
        "behavior_passes": sum(x["behavior_match"] is True for x in records),
        "items": records,
        "comparability": "SYNTHETIC_FIXTURE_ONLY_NOT_PUBLIC_BENCHMARK",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--backend", choices=("docker", "podman"), default="docker")
    parser.add_argument("--image", default="eclipse-temurin:21-jdk",
                        help="Must already exist locally; never pulled")
    parser.add_argument("--allow-execution", action="store_true",
                        help="Explicitly permit model-generated Java in a local OCI container")
    parser.add_argument("--max-examples", type=int, default=15)
    parser.add_argument("--output", required=True, type=Path, help="New JSON report; refuses overwrite")
    args = parser.parse_args()
    result = evaluate(args.dataset, args.checkpoint, args.tokenizer, backend=args.backend,
                      image=args.image, allow_execution=args.allow_execution,
                      max_examples=args.max_examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()