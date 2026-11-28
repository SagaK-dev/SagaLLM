import unittest
from scripts import audit_public_intake_provenance as a

class PublicProvenanceTests(unittest.TestCase):
    def test_reference_oracles(self):
        for slug,stdin,stdout in (
            ("hamming","A\nT\n","1\n"),("rna-transcription","ACGT\n","UGCA\n"),
            ("luhn","059\n","true\n"),("luhn","0\n","false\n"),
            ("luhn","05a9\n","false\n"),("roman-numerals","3999\n","MMMCMXCIX\n"),
            ("allergies","3 eggs\n","true\n"),("allergies","2 eggs\n","false\n"),
            ("atbash-cipher","Testing,1 2 3, testing.\n","gvhgr mt123 gvhgr mt\n")):
            self.assertEqual(a.oracle(slug,stdin),stdout)
        with self.assertRaises(ValueError):
            a.oracle("hamming","A\nAA\n")
        with self.assertRaises(ValueError):
            a.oracle("roman-numerals","0\n")

    def test_canonical_adaptation(self):
        self.assertEqual(a.adapt("allergies",{"input":{"score":2,"item":"eggs"},
                  "property":"allergicTo","expected":False}),{"stdin":"2 eggs\n","stdout":"false\n"})
        self.assertIsNone(a.adapt("allergies",{"input":{"score":2},"property":"list","expected":["peanuts"]}))
        self.assertEqual(a.flatten([{"cases":[{"uuid":"a"}]},{"uuid":"b"}]),[{"uuid":"a"},{"uuid":"b"}])

if __name__=="__main__":
    unittest.main()
