import unittest
from scripts.audit_patterns_module_license_scope import classify, NOTE

class ModuleScopeTests(unittest.TestCase):
    def test_note_is_not_a_file_level_license_claim(self):
        text="/* "+NOTE+"\nThe MIT License\nCopyright 2022 */\npackage a;"
        self.assertEqual(classify(text,"other/src/A.java")[0],"MIT_HEADER_WITH_OTHER_MODULE_DEPENDENCY_NOTE")
        self.assertEqual(classify(text,"model-view-viewmodel/src/A.java")[0],"MODULE_ZK_DEPENDENCY_REVIEW_REQUIRED")
        self.assertEqual(classify(text.replace("Copyright 2022","GNU General Public License"),"other/A.java")[0],
                         "OTHER_HEADER_SIGNAL_REVIEW_REQUIRED")
        self.assertEqual(classify("// SPDX-License-Identifier: GPL-3.0-only\npackage a;","other/A.java")[0],
                         "OTHER_HEADER_SIGNAL_REVIEW_REQUIRED")
if __name__=="__main__":unittest.main()
