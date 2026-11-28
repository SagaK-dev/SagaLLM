import unittest
from scripts.audit_sft_prompt_collisions import collision_groups,subset_plan,literal_preserving_signature

class PromptCollisionTests(unittest.TestCase):
    def test_comment_variants_preserve_literals(self):
        self.assertEqual(literal_preserving_signature("int x=2; // first"),literal_preserving_signature("int x = 2; /* second */"))
        self.assertNotEqual(literal_preserving_signature("int x=2;"),literal_preserving_signature("int x=3;"))
    def test_quarantine_distinct_targets(self):
        def row(i,sig):
            return {"split":"train","id":f"a-{i:04d}","family":"a","language":"ja",
                    "task_sha256":"same","response_sha256":str(i),"literal_preserving_signature":sig,
                    "template_signature":"shape","row_sha256":str(i)}
        rows=[row(0,"one"),row(2,"two")]
        g=collision_groups(rows)
        self.assertEqual(g[0]["category"],"DISTINCT_LEXICAL_TARGETS_REVIEW_REQUIRED")
        self.assertEqual(subset_plan(rows,g)["unresolved_prompt_rows_quarantined"],2)
        self.assertEqual(subset_plan(rows,g)["candidate_count"],0)
    def test_deduplicate_only_comment_variants(self):
        rows=[{"split":"train","id":f"a-{i:04d}","family":"a","language":"ja",
               "task_sha256":"same","response_sha256":str(i),"literal_preserving_signature":"same",
               "template_signature":"shape","row_sha256":str(i)} for i in range(2)]
        p=subset_plan(rows,collision_groups(rows))
        self.assertEqual(p["candidate_count"],1)
        self.assertEqual(p["dedup_comment_variant_rows_removed"],1)
if __name__=="__main__":unittest.main()
