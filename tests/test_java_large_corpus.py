"""Expanded corpus, byte-BPE fast-path, and bounded real training tests."""
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from sagallm.java_large_corpus import (
    SCHEMA, SPLITS, TRAIN_FAMILIES, VALIDATION_FAMILIES, TEST_FAMILIES,
    _case, build_java_large_corpus, verify_java_large_corpus,
)
from sagallm.tokenization import ByteBPETokenizer


def test_corpus_build_and_verify_all_splits(tmp_path):
    path=tmp_path/'corpus'
    result=build_java_large_corpus(path)
    expected={k:len(f)*n for k,(f,n) in SPLITS.items()}
    assert result['counts']==expected
    assert sum(expected.values())==5436
    assert result['duplicate_source_files']==0
    assert verify_java_large_corpus(path)['family_disjoint']
    assert json.loads((path/'manifest.json').read_text())['schema']==SCHEMA
    code=set()
    for split in SPLITS:
        for line in (path/f'{split}.jsonl').read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            assert len(row['messages'])==2
            source=row['messages'][1]['content']
            assert source.startswith('public class Main {')
            key=hashlib.sha256(source.encode()).hexdigest()
            assert key not in code
            code.add(key)
    assert len(code)==5436


def test_corpus_rejects_overwrite_and_tamper(tmp_path):
    path=tmp_path/'corpus';build_java_large_corpus(path)
    with pytest.raises(FileExistsError):build_java_large_corpus(path)
    p=path/'test.jsonl';p.write_text(p.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='hash'):verify_java_large_corpus(path)


def test_corpus_rejects_forged_manifest_hash(tmp_path):
    path=tmp_path/'corpus';build_java_large_corpus(path)
    p=path/'train.jsonl';text=p.read_text(encoding='utf8').replace('SagaJava_', 'Tampered_')
    p.write_text(text,encoding='utf8')
    manifest=path/'manifest.json';m=json.loads(manifest.read_text());m['files']['train']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(ValueError,match='canonical'):verify_java_large_corpus(path)


def test_corpus_rejects_symlink_and_false_rights(tmp_path):
    path=tmp_path/'corpus';build_java_large_corpus(path)
    manifest=path/'manifest.json';m=json.loads(manifest.read_text());m['license_review']='PERMITTED'
    manifest.write_text(json.dumps(m),encoding='utf8')
    with pytest.raises(ValueError,match='provenance'):verify_java_large_corpus(path)
    manifest.unlink();manifest.symlink_to(Path('/etc/passwd'))
    with pytest.raises(ValueError,match='unsafe'):verify_java_large_corpus(path)


def test_train_validation_test_families_are_disjoint():
    sets=[set(TRAIN_FAMILIES),set(VALIDATION_FAMILIES),set(TEST_FAMILIES)]
    assert len(sets[0])==58
    assert len(sets[1])==6
    assert len(sets[2])==6
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))


def test_reference_java_compiles_for_all_seventy_families(tmp_path):
    if not shutil.which('javac'):pytest.skip('JDK not installed')
    sources=[]
    for idx,family in enumerate((*TRAIN_FAMILIES,*VALIDATION_FAMILIES,*TEST_FAMILIES)):
        class_name=f'Case{idx}'
        source=_case(family,0)[2].replace('public class Main',f'public class {class_name}',1)
        file=tmp_path/(class_name+'.java');file.write_text(source,encoding='utf8');sources.append(str(file))
    p=subprocess.run(['javac','-proc:none','-d',str(tmp_path),*sources],cwd=tmp_path,
                     capture_output=True,text=True,timeout=35)
    assert p.returncode==0,p.stderr[:2000]
    assert len(sources)==70


def test_bpe_heap_encoding_matches_prior_algorithm():
    rand=random.Random(117)
    texts=['', 'こんにちは、SagaLLM', '🚀Python→Java', 'aaa aaaa a', '\x00\u3042']
    texts.extend(''.join(rand.choice('Java 01{}+-\n漢字ab') for _ in range(rand.randrange(0,90))) for _ in range(350))
    tok=ByteBPETokenizer.train(texts[:40],vocab_size=384)
    for text in texts:
        reference=[b+4 for b in text.encode('utf8')]
        while len(reference)>1:
            options=[(tok.ranks[(a,b)],(a,b)) for a,b in zip(reference,reference[1:]) if (a,b) in tok.ranks]
            if not options:break
            rank,pair=min(options);reference=tok._merge(reference,pair,rank)
        assert tok.encode(text)==reference
        assert tok.decode(reference)==text


def test_two_step_cpu_training_tracks_real_seen_count(tmp_path):
    import importlib.util
    script=Path(__file__).parents[1]/'scripts'/'train_java_expanded.py'
    spec=importlib.util.spec_from_file_location('train_java_expanded',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train=module.train
    path=tmp_path/'data';build_java_large_corpus(path)
    report=train(path,tmp_path/'out',steps=2,max_seconds=20,hidden_size=128,layers=2,
                 progress_every=1,compile_limit=0)
    assert report['optimizer_steps_completed']==2
    assert report['distinct_training_examples_seen']==2
    assert report['unseen_training_examples']==5218
    assert report['test_used_for_model_selection'] is False
    assert report['test_loss_evaluation_count']==1
    assert report['compile_only_heldout']['status']=='NOT_RUN'


def test_previous_local_checkpoint_continuation_requires_matching_tokenizer(tmp_path):
    import importlib.util
    script=Path(__file__).parents[1]/'scripts'/'train_java_expanded.py'
    spec=importlib.util.spec_from_file_location('train_java_expanded_continued',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train=module.train
    path=tmp_path/'data';build_java_large_corpus(path)
    with pytest.raises(ValueError,match='specified together'):
        train(path,tmp_path/'bad',steps=1,max_seconds=20,init_checkpoint=tmp_path/'missing.pt')
    first=tmp_path/'from-scratch'
    r1=train(path,first,steps=1,max_seconds=20,hidden_size=128,layers=2,
             progress_every=1,compile_limit=0)
    r2=train(path,tmp_path/'continued',steps=1,max_seconds=20,progress_every=1,
             compile_limit=0,init_checkpoint=first/'java-best.pt',
             init_tokenizer=first/'tokenizer.json')
    assert r2['model_weights_origin']=='PRIOR_LOCAL_SAGALLM'
    assert r2['initial_local_checkpoint_sha256']==hashlib.sha256((first/'java-best.pt').read_bytes()).hexdigest()
    assert r2['optimizer_steps_completed']==1
    assert r2['distinct_training_examples_seen']==1
    assert r2['best_checkpoint_distinct_training_examples_seen']<=1
    assert r2['test_used_for_model_selection'] is False