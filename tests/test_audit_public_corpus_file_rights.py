import unittest
from scripts.audit_public_corpus_file_rights import signals, NOTICE

class FileRightsSignalsTests(unittest.TestCase):
    def test_headers_not_legal_decisions(self):
        self.assertFalse(signals("// SPDX-License-Identifier: MIT\npackage x;","MIT")["possible_non_root_license_signal"])
        self.assertTrue(signals("// SPDX-License-Identifier: GPL-3.0-only\npackage x;","MIT")["possible_non_root_license_signal"])
        self.assertTrue(signals("/* Apache License */\npackage x;","MIT")["possible_non_root_license_signal"])
        self.assertFalse(signals("package x;\n// Apache License","MIT")["header_license_signal_present"])
        self.assertFalse(signals("// Copyright 2020\npackage x;","MIT")["header_license_signal_present"])
    def test_notice_candidates(self):
        for name in ("LICENSE","LICENSE.md","NOTICE.txt","COPYING","third-party-notices.txt"):
            self.assertIsNotNone(NOTICE.fullmatch(name))
        self.assertIsNone(NOTICE.fullmatch("README.md"))

if __name__=="__main__":unittest.main()
