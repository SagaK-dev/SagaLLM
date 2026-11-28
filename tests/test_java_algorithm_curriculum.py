import unittest
from sagallm.java_algorithm_curriculum import rows,cases,oracle,SPECS

class AlgorithmCurriculumTests(unittest.TestCase):
    def test_scope(self):
        r=list(rows());self.assertEqual(len(r),16)
        self.assertEqual(len({x["source"] for x in r}),8)
        self.assertEqual(len({x["task"] for x in r}),16)
        for family in SPECS:
            self.assertEqual(len(list(cases(family))),12)
    def test_oracles(self):
        self.assertEqual(oracle("range_sum_queries","3 2\n1 -5 7\n0 2\n1 1\n"),"3\n-5\n")
        self.assertEqual(oracle("interval_union_length","3\n-2 1\n0 3\n4 6\n"),"7\n")
        self.assertEqual(oracle("edit_distance","kitten sitting"),"3\n")
        self.assertEqual(oracle("unweighted_graph_distance","4 3\n0 1\n1 2\n2 3\n0 3\n"),"3\n")
        self.assertEqual(oracle("unweighted_graph_distance","4 1\n0 1\n0 3\n"),"-1\n")
if __name__=="__main__":unittest.main()
