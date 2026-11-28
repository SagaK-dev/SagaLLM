import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sagallm.java_algorithm_curriculum import rows, sha
from scripts.audit_algorithm_context_balance_mask import (
    balance, candidate_row, check_encoding, read_admission, CONTEXTS,
)


class ByteTokenizer:
    SPECIAL = {"<bos>": 256, "<eos>": 257}

    def encode(self, text):
        return list(text.encode("utf-8"))


class ContextBalanceTests(unittest.TestCase):
    def sample(self):
        result = []
        for i, row in enumerate(rows()):
            n = 300 if i < 6 else (450 if i < 14 else 650)
            result.append({"id": row["id"], "family": row["family"],
                           "variant": row["variant"], "language": row["language"],
                           "source_sha256": sha(row["source"].encode()),
                           "sequence_positions": n, "supervised_tokens": 100,
                           "admitted_contexts": [c for c in CONTEXTS if c >= n]})
        return result

    def test_balanced_full_and_partial(self):
        data = self.sample()
        summary = balance(data)
        self.assertEqual(summary["384"]["admitted"], 6)
        self.assertEqual(summary["384"]["complete_bilingual_implementations"], 3)
        self.assertEqual(len(summary["384"]["complete_families"]), 1)
        self.assertEqual(summary["768"]["language_counts"], {"en": 8, "ja": 8})
        self.assertEqual(summary["768"]["complete_bilingual_implementations"], 8)
        self.assertEqual(len(summary["768"]["complete_families"]), 4)
        self.assertEqual(summary["768"]["admitted_supervised_tokens"], 1600)

    def test_reject_duplicate_language_and_id(self):
        data = self.sample()
        data[1]["language"] = data[0]["language"]
        with self.assertRaises(ValueError):
            balance(data)
        data = self.sample()
        data[1]["id"] = data[0]["id"]
        with self.assertRaises(ValueError):
            balance(data)

    def test_reject_nonmonotonic_admission_and_source_drift(self):
        data = self.sample()
        data[0]["admitted_contexts"].remove(512)
        with self.assertRaises(ValueError):
            balance(data)
        data = self.sample()
        data[0]["source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            balance(data)

    def test_exact_boundary_and_mask(self):
        row = candidate_row(next(rows()))
        tokenizer = ByteTokenizer()
        result = check_encoding(row, tokenizer, 4096)
        n = result["sequence_positions"]
        self.assertEqual(result["masked_tokens"] + result["supervised_tokens"], n)
        self.assertEqual(check_encoding(row, tokenizer, n), result)
        self.assertIsNone(check_encoding(row, tokenizer, n - 1))

    def test_wrong_loss_mask_is_rejected(self):
        from sagallm.java_instruction_sft import encode_row
        row = candidate_row(next(rows()))
        tokenizer = ByteTokenizer()
        original = encode_row(row, tokenizer, 4096)
        original["y"][0] = 0
        with patch("scripts.audit_algorithm_context_balance_mask.encode_row", return_value=original):
            with self.assertRaises(ValueError):
                check_encoding(row, tokenizer, 4096)

    def test_wrong_admission_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            path.write_bytes(b"not an artifact")
            with self.assertRaises(ValueError):
                read_admission(path)
