import copy
import unittest
from scripts import audit_java_lineage as a


class LineageTests(unittest.TestCase):
    def test_links_and_rejections(self):
        common = {"status": "TRAINED_CPU", "tokenizer_sha256": a.TOKENIZER}
        root = dict(common, training_task="JAVA_SOURCE_NEXT_TOKEN_ONLY",
                    best_checkpoint_sha256=a.ROOT, dataset_manifest_sha256="a"*64)
        mid = dict(common, checkpoint_sha256_base=a.ROOT,
                   best_checkpoint_sha256=a.BASE, data_manifest_sha256=a.DATA)
        end = dict(common, checkpoint_sha256_base=a.BASE,
                   best_checkpoint_sha256=a.BEST, data_manifest_sha256=a.DATA)
        public = {"sources": [{"name": k, "revision": v} for k,v in a.SOURCES.items()]}
        args = [root, mid, end, public, "a"*64, a.DATA]
        self.assertFalse(a.links(*args)["training_lineage_verified"])
        for index, key, value in [(0,"best_checkpoint_sha256",a.BASE),
                (1,"checkpoint_sha256_base",a.BEST), (2,"tokenizer_sha256","b"*64),
                (2,"data_manifest_sha256","b"*64), (3,"sources",[])]:
            changed = copy.deepcopy(args)
            changed[index][key] = value
            with self.assertRaises(ValueError):
                a.links(*changed)
