from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save
from sagallm.evaluation import evaluate_language_model
import torch


def test_evaluate_returns_finite_metrics(tmp_path):
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    tok = ByteBPETokenizer.train(['hello world' * 20], vocab_size=280)
    tp = tmp_path / 'tok.json'
    tok.save(tp)
    data = tmp_path / 'heldout.txt'
    data.write_text('hello world ' * 30)
    model = SagaTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint_save(checkpoint, model, optimizer, 0, cfg, {})
    r = evaluate_language_model(checkpoint, tp, data, block_size=8, batches=2)
    assert r['perplexity'] > 1
    assert 0 <= r['next_token_accuracy'] <= 1
    assert r['evaluated_tokens'] == 32