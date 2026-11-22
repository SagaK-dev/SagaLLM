"""Original diverse Java curriculum: provenance, leakage, compilation and training."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from sagallm.java_diverse_corpus import (
    NEW_TRAIN, NEW_VALIDATION, NEW_TEST, SCHEMA, SPLITS,
    build_java_diverse_corpus, new_case, verify_java_diverse_corpus,
)
from sagallm.java_large_corpus import TRAIN_FAMILIES, VALIDATION_FAMILIES, TEST_FAMILIES


def test_canonical_corpus_has_exact_splits_and_zero_duplicate_sources(tmp_path):
    root=tmp_path/'dataset'
    summary=build_java_diverse_corpus(root)
    assert summary['counts']=={'train':17080,'validation':228,'test':228}
    assert summary['unique_sources']==17536
    assert summary['family_disjoint'] is True
    assert summary['independent_algorithms'] is False
    assert verify_java_diverse_corpus(root)['status']=='PASS'
    assert json.loads((root/'manifest.json').read_text())['schema']==SCHEMA
    actual={}
    for split in SPLITS:
        actual[split]={json.loads(x)['family'] for x in (root/(split+'.jsonl')).read_text().splitlines()}
    assert all(not actual[a]&actual[b] for a,b in (('train','test'),('train','validation'),('validation','test')))
    assert len(actual['train'])==90 and len(actual['test'])==10 and len(actual['validation'])==10


def test_dataset_refuses_overwrite_and_tampering(tmp_path):
    root=tmp_path/'data'
    build_java_diverse_corpus(root)
    with pytest.raises(FileExistsError):build_java_diverse_corpus(root)
    file=root/'train.jsonl'
    file.write_bytes(file.read_bytes().replace(b'public class Main',b'public class Maen',1))
    with pytest.raises(ValueError,match='SHA-256'):
        verify_java_diverse_corpus(root)


def test_manifest_forgery_does_not_bypass_recomputation(tmp_path):
    root=tmp_path/'data'
    build_java_diverse_corpus(root)
    file=root/'test.jsonl'
    file.write_bytes(file.read_bytes().replace(b'public class Main',b'public class Maen',1))
    meta=root/'manifest.json'
    m=json.loads(meta.read_text())
    m['files']['test']['sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    meta.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Noncanonical'):
        verify_java_diverse_corpus(root)


def test_rejects_claim_of_preapproved_license_and_symlink(tmp_path):
    root=tmp_path/'data';build_java_diverse_corpus(root)
    meta=root/'manifest.json';m=json.loads(meta.read_text())
    m['license_attestation']='APPROVED'
    meta.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='provenance'):
        verify_java_diverse_corpus(root)
    meta.unlink();meta.symlink_to('/etc/passwd')
    with pytest.raises(ValueError,match='unsafe'):
        verify_java_diverse_corpus(root)


def test_new_java_families_compile(tmp_path):
    if not shutil.which('javac'):pytest.skip('JDK not installed')
    all_families=NEW_TRAIN+NEW_VALIDATION+NEW_TEST
    files=[]
    for i,f in enumerate(all_families):
        for variant in (0,11,23):
            cls=f'Case{i}V{variant}'
            text=new_case(f,variant)[2].replace('public class Main',f'public class {cls}',1)
            file=tmp_path/(cls+'.java');file.write_text(text,encoding='utf8')
            files.append(str(file))
    r=subprocess.run(['javac','-proc:none','-encoding','UTF-8','-d',str(tmp_path),*files],
                     capture_output=True,text=True,timeout=100)
    assert r.returncode==0,r.stderr[:2500]
    assert len(files)==120


def test_trusted_reference_java_outputs_match_independent_oracles(tmp_path):
    if not shutil.which('javac') or not shutil.which('java'):pytest.skip('JDK unavailable')
    families={
        'count_divisors':lambda i: str(sum((31+7*i)%k==0 for k in range(1,32+7*i))),
        'perfect_square':lambda i: str((int((31+7*i)**.5)**2)==(31+7*i)).lower(),
        'reverse_digits':lambda i: str(int(str(31+7*i)[::-1])),
        'sum_squares':lambda i: str(sum(k*k for k in range(1,18+i))),
        'triangular_number':lambda i: str((31+7*i)*(32+7*i)//2),
        'hex_format':lambda i: format(31+7*i,'x'),
        'binary_format':lambda i: format(31+7*i,'b'),
        'bit_count':lambda i: str((31+7*i).bit_count()),
        'word_count':lambda i:'4',
        'array_two_sum':lambda i:'true',
    }
    for family,expected in families.items():
        source=new_case(family,3)[2]
        f=tmp_path/'Main.java';f.write_text(source,encoding='utf8')
        c=subprocess.run(['javac','-proc:none','-d',str(tmp_path),str(f)],
                         capture_output=True,text=True,timeout=12)
        assert c.returncode==0,(family,c.stderr)
        r=subprocess.run(['java','-cp',str(tmp_path),'Main'],
                         capture_output=True,text=True,timeout=5)
        assert r.returncode==0,(family,r.stderr)
        assert r.stdout.strip()==expected(3),(family,r.stdout,expected(3))


def test_family_interleaved_training_counts_gradients_correctly(tmp_path):
    root=tmp_path/'corpus';build_java_diverse_corpus(root)
    script=Path(__file__).parents[1]/'scripts'/'train_java_diverse.py'
    spec=importlib.util.spec_from_file_location('train_java_diverse',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.train(root,tmp_path/'model',steps=3,max_seconds=40,
                        hidden_size=128,layers=2,progress_every=1,compile_limit=0)
    assert result['optimizer_steps_completed']==3
    assert result['distinct_training_examples_seen']==3
    assert result['unseen_training_examples']==17077
    assert result['test_used_for_model_selection'] is False
    assert result['test_loss_evaluation_count']==1
    assert result['curriculum_order']=='family_interleaved_without_replacement'
    assert result['compile_only_heldout']['status']=='NOT_RUN'
    assert result['distinct_train_families_seen'] == 3


def test_new_family_invalid_inputs_rejected():
    with pytest.raises(ValueError):new_case('invalid_family',0)
    with pytest.raises(ValueError):new_case(NEW_TRAIN[0],-1)