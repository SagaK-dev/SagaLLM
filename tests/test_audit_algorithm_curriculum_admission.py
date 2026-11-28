import tempfile
import unittest
from pathlib import Path
from sagallm.java_algorithm_curriculum import rows
from sagallm.java_ai import _model_user_message
from sagallm.java_instruction_sft import source_task,compact_prompt
from scripts.audit_algorithm_curriculum_admission import verify_report

class AdmissionTests(unittest.TestCase):
    def test_all_records_convert_to_legacy_sft_prompt(self):
        records=list(rows())
        self.assertEqual(len(records),16)
        for r in records:
            row={"messages":[{"role":"user","content":_model_user_message(r["task"],"public class Main {}","Main")},
                              {"role":"assistant","content":r["source"]}]}
            self.assertEqual(source_task(row),(r["task"],r["source"]))
            self.assertIn(r["task"],compact_prompt(r["task"]))
    def test_wrong_reference_archive_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"report.zip";p.write_bytes(b"invalid")
            with self.assertRaises(ValueError):verify_report(p)
