import pytest
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer


def test_parameter_count_matches_materialized_model():
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg)
    assert cfg.parameter_count() == sum(x.numel() for x in model.parameters())
    assert model.embedding.weight is model.lm_head.weight


def test_reject_invalid_dimensions():
    cfg = ModelConfig.load('configs/model/tiny.yaml').to_dict()
    cfg['num_kv_heads'] = 3
    with pytest.raises(ValueError):
        ModelConfig(**cfg)


def test_large_configs_inspect_without_initialization():
    for name in ('1b', '3b'):
        cfg = ModelConfig.load(f'configs/model/{name}.yaml')
        assert 1_000_000_000 <= cfg.parameter_count() <= 3_100_000_000