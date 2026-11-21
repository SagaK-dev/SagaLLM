import torch
from sagallm.post_training import format_sft_example, sft_train
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def test_assistant_only_mask():
    tok = ByteBPETokenizer.train(['hi', 'answer'], vocab_size=270, min_frequency=1)
    x, y = format_sft_example([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'answer'}], tok, 128)
    assert len(x) == len(y)
    assert int((y == -100).sum()) > 0
    assert tok.decode(y[y >= 0].tolist()) == 'answer'
    assert y[-1] == tok.SPECIAL['<eos>']


def test_sft_smoke(tmp_path):
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    tok = ByteBPETokenizer.train(['hi answer hello' * 30], vocab_size=290)
    tokenizer_path = tmp_path / 'tok.json'
    tok.save(tokenizer_path)
    model = SagaTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    base = tmp_path / 'base.pt'
    checkpoint_save(base, model, optimizer, 0, cfg, {})
    data = tmp_path / 'sft.jsonl'
    data.write_text('{"messages": [{"role":"user","content":"hi"},{"role":"assistant","content":"answer"}]}\n')
    output = tmp_path / 'sft.pt'
    result = sft_train(base, tokenizer_path, data, output, steps=2)
    assert result['stage'] == 'sft'
    assert output.exists()