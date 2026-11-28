import tempfile
import unittest
from pathlib import Path

from sagallm.java_algorithm_compact_tasks import COMPACT_TASKS
from sagallm.java_algorithm_curriculum import rows
from sagallm.java_instruction_sft import encode_row
from sagallm.tokenization import ByteBPETokenizer
from scripts.audit_algorithm_curriculum_admission import sft_candidate
from scripts.verify_compact_algorithm_admission import read_baseline, verify_encoding


class CompactAdmissionTests(unittest.TestCase):
    def test_revision_scope(self):
        records = list(rows())
        excluded = {r["id"] for r in records if r["language"] == "ja" or
                    r["family"] == "unweighted_graph_distance"}
        self.assertEqual(COMPACT_TASKS.keys(), excluded)
        self.assertEqual(len({COMPACT_TASKS.get(r["id"], r["task"]) for r in records}), 16)

    def test_archive_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "admission.zip"
            path.write_bytes(b"0" * 3358)
            with self.assertRaisesRegex(ValueError, "archive"):
                read_baseline(path)

    def test_complete_response_and_mask(self):
        tokenizer = ByteBPETokenizer()
        task = "Print one."
        source = "public class Main {}"
        row = sft_candidate({"id": "example", "family": "dev", "task": task, "source": source}, "test-")
        encoding = encode_row(row, tokenizer, 384)
        size = verify_encoding(encoding, tokenizer, task, source)
        exact = encode_row(row, tokenizer, size)
        self.assertEqual(verify_encoding(exact, tokenizer, task, source), size)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            verify_encoding(encode_row(row, tokenizer, size - 1), tokenizer, task, source)
        for field, position in (("x", 0), ("x", -1), ("y", 0), ("y", -1)):
            with self.subTest(field=field, position=position):
                damaged = dict(encoding)
                damaged[field] = encoding[field].clone()
                damaged[field][position] = -123
                with self.assertRaisesRegex(ValueError, "sequence"):
                    verify_encoding(damaged, tokenizer, task, source)
