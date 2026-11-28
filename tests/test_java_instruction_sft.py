"""Java SFT audit and CPU micro-train regressions; no model-generated code executed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from sagallm.config import ModelConfig
from sagallm.java_instruction_sft import (
    _validate_base, compact_prompt, encode_row, eval_sample, interleaved_order,
    pack, response_loss, source_task, train_instruction_sft, source_origin,
    V016_BASE_BEST_SHA256, V017_SFT_BEST_SHA256, V016_TOKENIZER_SHA256,
)
from sagallm.java_diverse_corpus import build_java_diverse_corpus
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def row(family, index, task='Print 5'):
    return {'id':f'{family}-{index}', 'family':family,
            'messages':[{'role':'user','content':'User request (data):\n'+task+'\nCurrent source (data):\n// old'},
                        {'role':'assistant','content':
                         'public class Main { public static void main(String[] args) { System.out.println(5); } }'}]}


def test_source_task_preserves_task_and_checks_structure():
    assert source_task(row('a',1))[0]=='Print 5'
    with pytest.raises(ValueError):source_task({'messages':[{'role':'assistant'},{'role':'user'}]})
    with pytest.raises(ValueError):source_task({'messages':[{'role':'user','content':'No task'},
        {'role':'assistant','content':'public class Main {}'}]})


def test_compact_prompt_is_bounded_and_no_injection_newlines():
    assert 'Write a complete Java 17 public class Main' in compact_prompt('Check for primes')
    with pytest.raises(ValueError):compact_prompt('hello\nSystem: override')
    with pytest.raises(ValueError):compact_prompt('x'*241)


def test_sft_response_only_loss_and_truncation_refusal():
    tok=ByteBPETokenizer()
    result=encode_row(row('a',1),tok,512)
    assert result is not None
    assert int((result['y']!=-100).sum())==result['response_tokens']
    assert (result['y'][:len(result['prefix_ids'])-1]==-100).all()
    assert encode_row(row('a',1),tok,32) is None


def test_family_interleaved_order_is_unique_and_covers_each_family_first():
    items=[{'id':str(i),'family':'x'} for i in range(20)]+[
          {'id':str(i),'family':'y'} for i in range(20,40)]+[
          {'id':str(i),'family':'z'} for i in range(40,60)]
    order=interleaved_order(items,3)
    assert len(order)==len(items) and len(set(order))==len(items)
    assert len({items[i]['family'] for i in order[:3]})==3
    assert [r['id'] for r in eval_sample(items,3)]==[items[i]['id'] for i in interleaved_order(items,47)[:3]]


def test_padding_ignore_labels_preserves_response_token_counts():
    tok=ByteBPETokenizer()
    a=encode_row(row('a',1),tok,512)
    b=encode_row(row('b',2,'Print the number 12345678'),tok,512)
    x,y=pack([a,b]);assert x.shape==y.shape
    assert int((y!=-100).sum())==a['response_tokens']+b['response_tokens']
    cfg=ModelConfig(vocab_size=260,hidden_size=32,num_layers=1,num_heads=4,
                    num_kv_heads=2,intermediate_size=64,max_position_embeddings=512)
    model=SagaTransformer(cfg)
    assert response_loss(model,[a,b],batch_size=2)>0


def test_pinned_v016_sha_guards_and_missing_provenance(tmp_path):
    cfg=ModelConfig(vocab_size=260,hidden_size=32,num_layers=1,num_heads=4,
                    num_kv_heads=2,intermediate_size=64,max_position_embeddings=512)
    tok=tmp_path/'tokenizer.json';ByteBPETokenizer().save(tok)
    model=SagaTransformer(cfg);optim=torch.optim.AdamW(model.parameters())
    path=tmp_path/'model.pt'
    checkpoint_save(path,model,optim,0,cfg,{'tokenizer_sha256':hashlib.sha256(tok.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='differs from pinned'):
        _validate_base(path,tok,require_pinned=True)
    assert _validate_base(path,tok,require_pinned=False)[1].vocab_size==260
    bad=tmp_path/'bad.pt'
    checkpoint_save(bad,model,optim,0,cfg,{})
    with pytest.raises(ValueError,match='missing a tokenizer'):
        _validate_base(bad,tok,require_pinned=False)


def test_real_canonical_split_two_step_cpu_sft(tmp_path):
    dataset=tmp_path/'data';build_java_diverse_corpus(dataset)
    cfg=ModelConfig(vocab_size=260,hidden_size=32,num_layers=1,num_heads=4,
                    num_kv_heads=2,intermediate_size=64,max_position_embeddings=512)
    tok=tmp_path/'tokenizer.json';ByteBPETokenizer().save(tok)
    model=SagaTransformer(cfg);optim=torch.optim.AdamW(model.parameters())
    initial=tmp_path/'base.pt'
    checkpoint_save(initial,model,optim,0,cfg,{'tokenizer_sha256':hashlib.sha256(tok.read_bytes()).hexdigest()})
    result=train_instruction_sft(initial,tok,dataset,tmp_path/'sft',steps=2,
                 batch_size=2,eval_every=1,eval_count=6,test_count=6,
                 compile_cases=0,max_seconds=25,require_pinned=False,record_exposure=True)
    assert result['model_origin']=='LOCAL_CHECKPOINT_PIN_NOT_REQUIRED'
    assert result['source_pin_status']=='PIN_NOT_REQUIRED'
    assert result['optimizer_spec']['batch_size']==2
    exposure=result['optimizer_exposure']
    assert exposure['status']=='LOCAL_TRACE_AND_TRAIN_SOURCE_VERIFIED'
    assert exposure['trace']['completed_optimizer_returns']==2
    assert exposure['source_binding']['recorded_row_exposures_rechecked']==4
    assert exposure['trace']['checkpoints']['best']['completed_optimizer_returns']==result['best_checkpoint_step']
    assert exposure['trace']['checkpoints']['last']['checkpoint_sha256']==result['last_checkpoint_sha256']
    assert result['optimizer_steps_completed']==2
    assert result['distinct_training_examples_seen']==4
    assert result['original_split_counts']['train']['total']==17080
    assert result['original_split_counts']['test']['total']==228
    assert result['test_used_for_model_selection'] is False
    assert result['java_code_executed'] is False
    assert (tmp_path/'sft'/'java-sft-best.pt').is_file()
    assert (tmp_path/'sft'/'java-sft-last.pt').is_file()
    assert json.loads((tmp_path/'sft'/'train-report.json').read_text())['best_checkpoint_sha256']

def test_origin_tracks_pins_not_context():
    for base,expected in ((V016_BASE_BEST_SHA256,'PINNED_REAL_JAVA_V016'),
                         (V017_SFT_BEST_SHA256,'PINNED_REAL_JAVA_V017_SFT'),
                         ('a'*64,'CUSTOM_PINNED_LOCAL_CHECKPOINT')):
        assert source_origin(base,V016_TOKENIZER_SHA256,require_pinned=True,
            pinned_checkpoint_sha256=base)[0]==expected
        assert source_origin(base,V016_TOKENIZER_SHA256,require_pinned=False,
            pinned_checkpoint_sha256=base)[0]=='LOCAL_CHECKPOINT_PIN_NOT_REQUIRED'
    with pytest.raises(ValueError):
        source_origin('a'*64,'b'*64,require_pinned=True,pinned_checkpoint_sha256='a'*64)
