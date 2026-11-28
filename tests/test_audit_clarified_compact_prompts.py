import tempfile
import unittest
from pathlib import Path

from sagallm.java_algorithm_clarified_tasks import PROPOSALS, REVIEW_FOCUS
from sagallm.java_algorithm_compact_tasks import COMPACT_TASKS
from sagallm.java_algorithm_curriculum import rows
from sagallm.tokenization import ByteBPETokenizer
from scripts.audit_clarified_compact_prompts import first_eligible, check_option, read_boundary


class ClarifiedPromptTests(unittest.TestCase):
    def test_fixed_revision_scope(self):
        self.assertEqual(set(PROPOSALS), set(REVIEW_FOCUS))
        self.assertEqual(len(PROPOSALS), 7)
        for rid, options in PROPOSALS.items():
            self.assertTrue(rid.endswith("-ja"))
            self.assertEqual(len(options), len(set(options)))
            self.assertNotIn(COMPACT_TASKS[rid], options)

    def test_selection_preference_and_block(self):
        self.assertIsNone(first_eligible([{"eligible": False}]))
        rows_ = [{"eligible": False}, {"eligible": True, "id": 2},
                 {"eligible": True, "id": 3}]
        self.assertEqual(first_eligible(rows_)["id"], 2)

    def test_encoding_and_overlap_gate(self):
        record = {"id":"test", "family":"dev", "task":"Print one.",
                  "source":"public class Main {}"}
        tokenizer = ByteBPETokenizer()
        index = {"train":{"task":set(),"source":set(),"signature":set()}}
        result = check_option(record, record["task"], tokenizer, index)
        self.assertTrue(result["eligible"])
        index["train"]["task"].add(result["task_sha256"])
        self.assertFalse(check_option(record, record["task"], tokenizer, index)["eligible"])

    def test_overlong_response_not_truncated(self):
        record = next(rows())
        tokenizer = ByteBPETokenizer()
        result = check_option(record, "x"*240, tokenizer, {})
        self.assertFalse(result["fit"])
        self.assertFalse(result["eligible"])

    def test_wrong_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"boundary.zip"
            p.write_bytes(b"0"*7015)
            with self.assertRaisesRegex(ValueError, "pin"):
                read_boundary(p)
