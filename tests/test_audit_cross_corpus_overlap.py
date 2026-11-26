import json
import tempfile
import unittest
from pathlib import Path
from scripts.audit_cross_corpus_overlap import compare, read_split, sha


class CrossCorpusTests(unittest.TestCase):
    def test_exact_and_whitespace_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, s = Path(tmp)/"p", Path(tmp)/"s"
            texts = ["int x = 1;", "int x=1;", "int y=2;"]
            p.write_text("".join(json.dumps({"text": t, "sha256": sha(t)})+"\n" for t in texts))
            s.write_text("".join(json.dumps({"messages": [{"role":"user","content":"task"},
                {"role":"assistant","content":t}]})+"\n" for t in [texts[0], texts[0], "other"]))
            result = compare(read_split(p, False), read_split(s, True))
            self.assertEqual(result["exact"]["record_pairs"], 2)
            self.assertEqual(result["whitespace_insensitive"]["record_pairs"], 4)
            self.assertEqual(result["whitespace_insensitive"]["public_records"], 2)
            p.write_text(json.dumps({"text":"x","sha256":"bad"})+"\n")
            with self.assertRaises(ValueError):
                read_split(p, False)
