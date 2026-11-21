import torch
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer


def test_forward_backward_and_shape():
    torch.manual_seed(10)
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg)
    x = torch.randint(4, cfg.vocab_size, (2, 6))
    result = model(x, labels=x)
    assert result.logits.shape == (2, 6, cfg.vocab_size)
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_causal_attention_no_future_leak():
    torch.manual_seed(42)
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg).eval()
    x = torch.randint(4, cfg.vocab_size, (1, 8))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        left = model(x).logits[:, :-1]
        right = model(y).logits[:, :-1]
    torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)


def test_kv_cache_matches_full_prefill():
    torch.manual_seed(2)
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg).eval()
    x = torch.randint(4, cfg.vocab_size, (2, 7))
    with torch.no_grad():
        full = model(x).logits
        past = model(x[:, :3], use_cache=True)
        rest = model(x[:, 3:], past_key_values=past.past_key_values, use_cache=True)
    torch.testing.assert_close(rest.logits, full[:, 3:], atol=2e-5, rtol=2e-5)


def test_generate_greedy_reproducible():
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg).eval()
    x = torch.tensor([[1, 6, 12]], dtype=torch.long)
    assert torch.equal(model.generate(x, max_new_tokens=4, temperature=0),
                       model.generate(x, max_new_tokens=4, temperature=0))