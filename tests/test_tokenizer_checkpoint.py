import pytest
import torch
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.shards import sha256_file
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save, verify_checkpoint_tokenizer


def test_generation_is_limited_to_trained_tokenizer_vocab():
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg)
    ids = torch.tensor([[1, 2, 4]], dtype=torch.long)
    y = model.generate(ids, max_new_tokens=4, vocab_limit=10, temperature=0)
    assert all(0 <= tok < 10 for tok in y[0, 3:].tolist())
    with pytest.raises(ValueError, match='vocab_limit'):
        model.generate(ids, vocab_limit=cfg.vocab_size + 1)


def test_checkpoint_rejects_changed_tokenizer(tmp_path):
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    tok = ByteBPETokenizer.train(['abc abc'] * 10, vocab_size=280)
    tp = tmp_path / 'tokenizer.json'
    tok.save(tp)
    model = SagaTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters())
    cp = tmp_path / 'checkpoint.pt'
    checkpoint_save(cp, model, opt, 0, cfg, {'tokenizer_sha256': sha256_file(tp)})
    state = torch.load(cp, weights_only=True)
    verify_checkpoint_tokenizer(state, tp)
    tp.write_text(tp.read_text(encoding='utf8') + ' ', encoding='utf8')
    with pytest.raises(ValueError, match='SHA-256'):
        verify_checkpoint_tokenizer(state, tp)