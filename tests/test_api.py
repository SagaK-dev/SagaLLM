import pytest
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save
import torch


def test_api_auth_and_basic_completion(tmp_path):
    fastapi = pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from sagallm.api import create_app
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    tok = ByteBPETokenizer.train(['Hi there'] * 10, vocab_size=280)
    tp = tmp_path / 'tok.json'
    tok.save(tp)
    model = SagaTransformer(cfg)
    optim = torch.optim.AdamW(model.parameters())
    cp = tmp_path / 'cp.pt'
    checkpoint_save(cp, model, optim, 0, cfg, {})
    app = create_app(cp, tp, api_key='secret-token', device_name='cpu')
    client = TestClient(app)
    assert client.get('/health').status_code == 200
    assert client.post('/v1/completions', json={'prompt':'Hi'}).status_code == 401
    response = client.post('/v1/completions', headers={'Authorization':'Bearer secret-token'},
                           json={'prompt': 'Hi', 'max_tokens': 2})
    assert response.status_code == 200, response.text
    assert len(response.json()['choices']) == 1