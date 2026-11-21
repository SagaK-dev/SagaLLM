import pytest
import torch
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def _app(tmp_path, **kwargs):
    from sagallm.api import create_app
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    tok = ByteBPETokenizer.train(['simple test'] * 20, vocab_size=280)
    tok_path = tmp_path / 'tok.json'
    tok.save(tok_path)
    model = SagaTransformer(cfg)
    optim = torch.optim.AdamW(model.parameters())
    cp = tmp_path / 'cp.pt'
    checkpoint_save(cp, model, optim, 0, cfg, {})
    return create_app(cp, tok_path, device_name='cpu', **kwargs)


def test_local_api_rate_and_payload_limits(tmp_path):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    client = TestClient(_app(tmp_path, requests_per_minute=1))
    response = client.post('/v1/completions', json={'prompt': 'Hi', 'max_tokens': 1})
    assert response.status_code == 200, response.text
    assert client.post('/v1/completions', json={'prompt': 'Hi'}).status_code == 429
    assert client.post('/v1/chat/completions', json={'messages': [{'role':'user','content':'X'*2049}]}).status_code == 422


def test_generation_cooperative_deadline():
    import time
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg)
    with pytest.raises(TimeoutError):
        model.generate(torch.tensor([[1, 5]], dtype=torch.long), max_new_tokens=2,
                       deadline_monotonic=time.monotonic() - 1)