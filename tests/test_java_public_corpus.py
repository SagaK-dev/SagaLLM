"""Original offline source-provenance and real tiny CPU learning tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from sagallm import java_public_corpus as public


def git(root: Path, *args):
    return subprocess.run(['git', '-C', str(root), *args],check=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True).stdout.strip()


def fixture_sources(tmp_path, monkeypatch):
    sources=tmp_path/'upstreams';sources.mkdir()
    specs=[]
    for slug in ('algorithms','gson','guava','patterns'):
        root=sources/slug;root.mkdir()
        git(root,'init','-q')
        git(root,'config','user.email','unit-test@example.invalid')
        git(root,'config','user.name','SagaLLM Unit Test')
        license_text='MIT License\nCopyright 2026 Original Fixture\n'
        if slug in ('gson','guava'):
            license_text='Apache License\nVersion 2.0, January 2004\n'
        (root/'LICENSE').write_text(license_text)
        for family in ('sorts','graphs','collections','strings','numbers'):
            for i in range(2):
                if slug=='algorithms':
                    path=root/f'src/main/java/com/thealgorithms/{family}/Demo{i}.java'
                elif slug=='gson':
                    path=root/f'gson/src/main/java/com/google/gson/{family}/Demo{i}.java'
                elif slug=='guava':
                    path=root/f'guava/src/com/google/common/{family}/Demo{i}.java'
                else:
                    path=root/f'{family}/src/main/java/com/iluwatar/{family}/Demo{i}.java'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(f'package com.testing.{family};\npublic class Demo{i} '+'{\n'
                                +f'  public static int result() '+'{ return '+str(i+len(slug))+'; }\n'
                                +f'  // A different fixture from {slug} / {family}\n'+'}\n')
        # Untracked, even syntactically valid Java must never enter the source.
        extra=root/'src/main/java/Private.java';extra.parent.mkdir(parents=True,exist_ok=True)
        extra.write_text('public class Private { private String token = "never-trained"; }\n')
        git(root,'add','LICENSE')
        git(root,'add','*.java')
        git(root,'commit','-qm','Fixture source')
        expected=public.Upstream(slug,'https://example.invalid/'+slug,
                                git(root,'rev-parse','HEAD'),'LICENSE',
                                git(root,'hash-object','LICENSE'),
                                'MIT' if slug in ('algorithms','patterns') else 'Apache-2.0')
        specs.append(expected)
    monkeypatch.setattr(public,'UPSTREAMS',tuple(specs))
    return sources


def test_pinned_sources_split_hash_license_and_no_training_leak(tmp_path,monkeypatch):
    roots=fixture_sources(tmp_path,monkeypatch)
    corpus=tmp_path/'corpus'
    report=public.prepare_public_corpus(roots,corpus)
    verified=public.verify_public_corpus(corpus)
    assert verified['total']>=30
    assert report['splits']==verified['counts']
    assert len(verified['sources'])==4
    assert all(verified['counts'][s]>0 for s in ('train','validation','test'))
    assert all('never-trained' not in row['text'] for split in verified['files'].values() for row in split)
    assert len({(r['repo'],r['group']) for a in verified['files'].values() for r in a})>=12
    assert len({r['sha256'] for a in verified['files'].values() for r in a})==verified['total']
    with pytest.raises(FileExistsError):public.prepare_public_corpus(roots,corpus)


def test_pinned_revision_and_license_are_checked(tmp_path,monkeypatch):
    roots=fixture_sources(tmp_path,monkeypatch)
    with (roots/'algorithms'/'LICENSE').open('a') as f:f.write('tampered')
    with pytest.raises(ValueError,match='uncommitted'):
        public.prepare_public_corpus(roots,tmp_path/'data')
    assert not (tmp_path/'data').exists()


def test_unknown_revision_rejected(tmp_path,monkeypatch):
    roots=fixture_sources(tmp_path,monkeypatch)
    wrong=list(public.UPSTREAMS)
    x=wrong[0]
    wrong[0]=public.Upstream(x.slug,x.url,'1'*40,x.license_path,x.license_blob,x.license_id)
    monkeypatch.setattr(public,'UPSTREAMS',tuple(wrong))
    with pytest.raises(ValueError,match='revision'):
        public.prepare_public_corpus(roots,tmp_path/'bad')


def test_detects_post_export_changes_and_forged_manifest(tmp_path,monkeypatch):
    roots=fixture_sources(tmp_path,monkeypatch)
    corpus=tmp_path/'corpus';public.prepare_public_corpus(roots,corpus)
    data=corpus/'train.jsonl';data.write_bytes(data.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='SHA-256'):
        public.verify_public_corpus(corpus)
    manifest=json.loads((corpus/'manifest.json').read_text())
    manifest['file_hashes']['train']=hashlib.sha256(data.read_bytes()).hexdigest()
    (corpus/'manifest.json').write_text(json.dumps(manifest))
    # Even if forged hashes are supplied, inconsistent data split rows are rejected.
    with pytest.raises((ValueError,json.JSONDecodeError)):
        public.verify_public_corpus(corpus)


def test_detects_cross_split_forgery(tmp_path,monkeypatch):
    roots=fixture_sources(tmp_path,monkeypatch)
    corpus=tmp_path/'corpus';public.prepare_public_corpus(roots,corpus)
    train=corpus/'train.jsonl';validation=corpus/'validation.jsonl'
    rows=train.read_text().splitlines()
    original=validation.read_text()
    validation.write_text(original+rows[0]+'\n')
    manifest=json.loads((corpus/'manifest.json').read_text())
    manifest['file_hashes']['validation']=hashlib.sha256(validation.read_bytes()).hexdigest()
    manifest['splits']['validation']+=1
    (corpus/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Identical Java source'):
        public.verify_public_corpus(corpus)


def test_legacy_training_smoke_and_test_only_after_selection(tmp_path,monkeypatch):
    from scripts.train_java_public import train
    roots=fixture_sources(tmp_path,monkeypatch)
    corpus=tmp_path/'corpus';public.prepare_public_corpus(roots,corpus)
    report=train(corpus,tmp_path/'model',steps=8,max_seconds=20,batch_size=2,
                 block_size=48,per_file=2,vocab_size=260,backend='legacy',
                 lr=0.001,eval_every=10)
    assert report['optimizer_steps_completed']==8
    assert report['distinct_training_files_seen']>=4
    assert report['java_behavior_test']=='NOT_RUN'
    assert report['test_used_for_model_selection'] is False
    assert report['heldout_test_loss_once_after_selection']>0
    assert (tmp_path/'model'/'java-best.pt').exists()
    assert (tmp_path/'model'/'java-last.pt').exists()


def test_window_targets_and_nonoverlap():
    from scripts.train_java_public import make_windows
    ids=[1,4,5,6,7,8,2]
    wins=make_windows(ids,3,5)
    assert len(wins)==2
    assert wins[0][0].tolist()==[1,4,5]
    assert wins[0][1].tolist()==[4,5,6]
    assert wins[1][0].tolist()==[6,7,8]
    assert wins[1][1].tolist()==[7,8,2]