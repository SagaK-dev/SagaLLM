import pytest
import torch
from sagallm.config import ModelConfig
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import pretrain, checkpoint_load
from sagallm.model import SagaTransformer


def test_bitwise_resume_matches_uninterrupted_cpu(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('hello world, こんにちは! ' * 90, encoding='utf-8')
    tok = ByteBPETokenizer.train(['hello world, こんにちは! '*25], vocab_size=295)
    tp = tmp_path / 'tok.json'
    tok.save(tp)
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    uninterrupted = tmp_path / 'uninterrupted.pt'
    interrupted = tmp_path / 'interrupted.pt'
    opts = dict(steps=4, schedule_steps=4, block_size=4,
                batch_size=2, grad_accum=2, seed=7, device_name='cpu')
    pretrain(cfg, tp, corpus, uninterrupted, **opts)
    pretrain(cfg, tp, corpus, interrupted, **(opts | {'steps': 2}), checkpoint_every=1)
    pretrain(cfg, tp, corpus, interrupted, **opts, resume=True)
    whole = checkpoint_load(uninterrupted, SagaTransformer(cfg))
    resumed = checkpoint_load(interrupted, SagaTransformer(cfg))
    for key, weight in whole['model'].items():
        assert torch.equal(weight, resumed['model'][key]), key
    with pytest.raises(ValueError, match='schedule_total_steps'):
        pretrain(cfg, tp, corpus, interrupted, steps=5, schedule_steps=5,
                 block_size=4, batch_size=2, grad_accum=2, seed=7, resume=True)
    mismatch = tmp_path / 'mismatch.pt'
    pretrain(cfg, tp, corpus, mismatch, steps=2, schedule_steps=4,
             block_size=4, batch_size=2, grad_accum=2, seed=7)
    with pytest.raises(ValueError, match='batch_size'):
        pretrain(cfg, tp, corpus, mismatch, steps=4, schedule_steps=4,
                 block_size=4, batch_size=3, grad_accum=2, seed=7, resume=True)