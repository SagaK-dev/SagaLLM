import unittest
from scripts.audit_canonical_sft_template_quality import signature,quantiles

class TemplateQualityTests(unittest.TestCase):
    def test_literals_and_comments(self):
        self.assertEqual(signature('int a=12;String s="//";'),signature('/* note */ int a=32;String s="abc";'))
        self.assertNotEqual(signature('int a=12;'),signature('int b=12;'))
        self.assertNotEqual(signature('int a=12;'),signature('int a=12+2;'))
    def test_quantile(self):
        self.assertEqual(quantiles([4,3,2,1]),{"min":1,"median":2,"p95":4,"max":4})
        self.assertIsNone(quantiles([]))
if __name__=="__main__":unittest.main()
