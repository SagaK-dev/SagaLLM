"""Local Java repository inventory never gives itself permission to train."""
import hashlib
import json
from pathlib import Path
import pytest
from sagallm.java_source_import import inventory_java_tree, export_java_source_corpus

SHA='0123456789abcdef0123456789abcdef01234567'


def _repo(root):
    root.mkdir()
    (root/'A.java').write_text('public class A { public int square(int a){return a*a;} }')
    (root/'B.java').write_text('public class B { public String hello(){return "hello";} }')
    (root/'generated').mkdir();(root/'generated'/'C.java').write_text('class C {}')
    (root/'private.java').write_text('class Secrets { String email = "person@example.org"; }')
    (root/'dupe.java').write_text((root/'A.java').read_text())


def test_inventory_filters_suspicious_and_duplicate_source(tmp_path):
    root=tmp_path/'repo';_repo(root)
    scan=inventory_java_tree(root)
    assert scan['safe_files']==2
    assert scan['rejections']=={'duplicate':1,'excluded_directory':1,'sensitive_email':1}
    assert [x['relative_path'] for x in scan['records']]==['A.java','B.java']


def test_export_is_unapproved_even_with_source_revision(tmp_path):
    root=tmp_path/'repo';_repo(root)
    target=tmp_path/'export'
    result=export_java_source_corpus(root,target,source_uri='https://example.org/project',revision=SHA)
    assert result['java_files']==2 and result['review_status']=='PENDING'
    assert result['training_authorized'] is False
    meta=json.loads((target/'manifest.json').read_text())
    source=target/'java-documents.jsonl'
    assert meta['java_documents_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert [json.loads(x)['source_path'] for x in source.read_text().splitlines()]==['A.java','B.java']
    with pytest.raises(FileExistsError):
        export_java_source_corpus(root,target,source_uri='https://example.org/project',revision=SHA)


def test_refuses_unreviewed_license_claim_and_unpinned_revision(tmp_path):
    root=tmp_path/'repo';_repo(root)
    with pytest.raises(ValueError,match='pinned'):
        export_java_source_corpus(root,tmp_path/'export',source_uri='repo',revision='main')
    with pytest.raises(ValueError,match='Do not claim'):
        export_java_source_corpus(root,tmp_path/'export',source_uri='repo',revision=SHA,
                                  license_id='Apache-2.0')
    assert not (tmp_path/'export').exists()


def test_symlink_exclusion(tmp_path):
    root=tmp_path/'repo';_repo(root)
    (root/'link.java').symlink_to(root/'A.java')
    assert inventory_java_tree(root)['rejections']['unsafe_or_large']==1
    (tmp_path/'linkdir').symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        inventory_java_tree(tmp_path/'linkdir')