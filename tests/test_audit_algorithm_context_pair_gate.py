import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.audit_algorithm_context_pair_gate import (
    CONTEXTS, policy_matrix, read_admission, summarize_rows,
)


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    records = []
    for family in range(4):
        for variant in range(2):
            key = f"{family}-{variant}"
            for lang in ("ja", "en"):
                ident = key + "-" + lang
                # At 384: one complete pair and one partial pair.
                positions = 330 if key == "0-0" else (
                    380 if key == "0-1" and lang == "en" else 550)
                records.append({
                    "id": ident, "family": str(family), "variant": str(variant),
                    "language": lang, "candidate_row_sha256": h("row-" + ident),
                    "task_sha256": h("task-" + ident), "source_sha256": h("source-" + key),
                    "prompt_tokens": 100, "response_tokens_without_eos": positions - 101,
                    "admitted_contexts": [n for n in CONTEXTS if positions <= n],
                    "canonical_matches": {s: dict(task=False, source=False, signature=False)
                                          for s in ("train", "validation", "test")},
                })
    return records


class PairGateTests(unittest.TestCase):
    def test_pair_coverage_and_shifted_positions(self):
        report = summarize_rows(fixture())
        small = report["by_context"]["384"]
        self.assertEqual(small["admitted_rows"], 3)
        self.assertEqual(small["complete_bilingual_pairs"], 1)
        self.assertEqual(small["partial_pairs"], 1)
        self.assertEqual(small["language_counts"], {"ja": 1, "en": 2})
        self.assertEqual(small["complete_families"], [])
        self.assertEqual(report["minimum_full_context"], 550)
        self.assertEqual(report["by_context"]["768"]["complete_bilingual_pairs"], 8)
        self.assertEqual(len(report["by_context"]["768"]["complete_families"]), 4)

    def test_exact_context_boundary(self):
        rows = fixture()
        rows[0]["response_tokens_without_eos"] = 283
        self.assertEqual(summarize_rows(rows)["pairs"][0]["members"]["ja"]["sequence_positions"], 384)
        rows[0]["response_tokens_without_eos"] += 1
        with self.assertRaises(ValueError):
            summarize_rows(rows)

    def test_reject_corrupted_metadata(self):
        mutations = (
            lambda r: r[0].update(admitted_contexts=[768]),
            lambda r: r[0].update(language="en"),
            lambda r: r[0].update(candidate_row_sha256=r[1]["candidate_row_sha256"]),
            lambda r: r[0].update(source_sha256=h("wrong")),
            lambda r: r[0]["canonical_matches"]["test"].update(task=True),
            lambda r: r[0]["canonical_matches"].pop("test"),
            lambda r: r[0].update(prompt_tokens=True),
            lambda r: r.pop(),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                rows = copy.deepcopy(fixture())
                mutate(rows)
                with self.assertRaises(ValueError):
                    summarize_rows(rows)

    def test_order_independence(self):
        rows = fixture()
        self.assertEqual(summarize_rows(rows), summarize_rows(list(reversed(rows))))

    def test_bad_archive(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "archive.zip"
            path.write_bytes(b"not an archive")
            with self.assertRaises(ValueError):
                read_admission(path)

    def test_policy_uses_existing_guard(self):
        matrix = policy_matrix()
        self.assertEqual(matrix["192"]["384"], "POLICY_ALLOWS_EXTENSION")
        self.assertEqual(matrix["192"]["512"], "BLOCKED_BY_EXISTING_POLICY")
        self.assertEqual(matrix["384"]["512"], "POLICY_ALLOWS_EXTENSION")
        self.assertEqual(matrix["384"]["768"], "BLOCKED_BY_EXISTING_POLICY")
