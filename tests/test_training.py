import torch
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import pretrain, checkpoint_load


def test_pretrain_resume_checkpoint(tmp_path):
    tok = ByteBPETokenizer.train(['hello hello hello' * 30], vocab_size=300)
    tpath = tmp_path / 'tokens.json'
    tok.save(tpath)
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('hello hello world ' * 80, encoding='utf8')
    checkpoint = tmp_path / 'model.pt'
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    first = pretrain(cfg, tpath, corpus, checkpoint, steps=2, batch_size=1, block_size=8)
    second = pretrain(cfg, tpath, corpus, checkpoint, steps=3, batch_size=1, block_size=8, resume=True)
    assert first['steps'] == 2 and second['steps'] == 3
    model = SagaTransformer(cfg)
    state = checkpoint_load(checkpoint, model)
    assert state['step'] == 3


def test_one_batch_overfit():
    torch.manual_seed(9)
    cfg = ModelConfig(vocab_size=300,hidden_size=32,num_layers=1,num_heads=4,num_kv_heads=2,intermediate_size=64,max_position_embeddings=16)
    model = SagaTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.tensor([[4, 10, 12, 4, 10, 12]], dtype=torch.long)
    y = torch.tensor([[10, 12, 4, 10, 12, 4]], dtype=torch.long)
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        loss = model(x, labels=y).loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5