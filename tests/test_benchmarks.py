"""Offline benchmark input, leakage screening and actual tiny-model scoring."""
import json
import pytest
import torch

from sagallm.benchmarks import (
    BenchmarkContaminationError, audit_contamination, evaluate_choices, read_questions,
)
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from sagallm.shards import prepare_shards
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def _fixture(tmp_path):
    tok = ByteBPETokenizer.train(['東京は日本の首都です。', 'Python is a programming language.'],
                                 vocab_size=280, min_frequency=1)
    tp = tmp_path / 'tokenizer.json'
    tok.save(tp)
    cfg = ModelConfig.load('configs/model/tiny.yaml')
    model = SagaTransformer(cfg)
    cp = tmp_path / 'tiny.pt'
    checkpoint_save(cp, model, torch.optim.AdamW(model.parameters()), 0, cfg, {})
    path = tmp_path / 'benchmark.jsonl'
    prompt = 'The capital of Japan is: '
    record = {'id': 'q1', 'prompt': prompt, 'choices': [' Tokyo', ' Osaka'], 'answer_index': 0}
    path.write_text(json.dumps(record) + '\n', encoding='utf-8')
    return cp, tp, path, prompt


def test_benchmark_is_deterministic_and_marks_unchecked_overlap(tmp_path):
    cp, tp, path, prompt = _fixture(tmp_path)
    a = evaluate_choices(cp, tp, path, license_id='CC0-1.0')
    b = evaluate_choices(cp, tp, path, license_id='CC0-1.0')
    assert a == b
    assert a['questions'] == 1 and 0 <= a['accuracy'] <= 1
    assert a['contamination']['status'] == 'NOT_CHECKED'
    assert not a['standard_benchmark_certified']
    assert len(a['results'][0]['choice_log_prob_scores']) == 2
    assert 'prompt' not in a['results'][0]
    assert a['checkpoint_sha256']


def test_contamination_blocks_scoring_and_tampered_source(tmp_path):
    cp, tp, path, prompt = _fixture(tmp_path)
    documents = tmp_path / 'train.jsonl'
    other = 'A completely unrelated document with many words and enough content.'
    documents.write_text('\n'.join(json.dumps({'text': t}) for t in [
        'A paragraph that embeds this test query: ' + prompt + ' inside it.', other,
        'Independent educational record with the same quoted query: ' + prompt + ' and added context.',
        'A third independent document with sufficiently distinctive words.',
        'A fourth independent document with different content from the rest.',
    ]), encoding='utf-8')
    prepared = tmp_path / 'prepared'
    prepare_shards(documents, prepared, 'CC0-1.0', validation_fraction=.25)
    q = read_questions(path)
    # Two different documents contain the benchmark question: at least one
    # must remain in the train split when exactly one is held out.
    audit = audit_contamination(q, prepared)
    assert audit['status'] == 'FOUND' and audit['matched_ids'] == ['q1']
    with pytest.raises(BenchmarkContaminationError):
        evaluate_choices(cp, tp, path, license_id='CC0-1.0', training_docs=prepared)
    first = next(prepared.glob('train-*.jsonl'))
    first.write_text(first.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        audit_contamination(q, prepared)


def test_benchmark_rejects_bad_records_and_licenses(tmp_path):
    cp, tp, path, prompt = _fixture(tmp_path)
    with pytest.raises(ValueError, match='license'):
        evaluate_choices(cp, tp, path, license_id='unknown')
    bad = tmp_path / 'invalid.jsonl'
    bad.write_text(json.dumps({'id':'q1','prompt':prompt,'choices':['x','y'],
                               'answer_index':True}), encoding='utf-8')
    with pytest.raises(ValueError, match='answer_index'):
        read_questions(bad)
    with pytest.raises(ValueError, match='maximum'):
        read_questions(path, max_examples=0)


def test_mmlu_style_adapter(tmp_path):
    path = tmp_path / 'mmlu.jsonl'
    path.write_text(json.dumps({'id':'math1','question':'What is 2 + 2?',
                                'choices':['2','3','4','5'],'answer':'C'}), encoding='utf-8')
    questions = read_questions(path, format='mmlu')
    assert len(questions) == 1
    assert questions[0]['answer_index'] == 2
    assert 'A. 2\nB. 3' in questions[0]['prompt']
    assert questions[0]['choices'] == [' A', ' B', ' C', ' D']


def test_short_questions_are_not_cleared_by_audit(tmp_path):
    doc = tmp_path / 'input.jsonl'
    doc.write_text('\n'.join(json.dumps({'text': t}) for t in ['some text about cats', 'some text about dogs']), encoding='utf-8')
    prep = tmp_path / 'prep'
    prepare_shards(doc, prep, 'CC0-1.0')
    audit = audit_contamination([{'id':'short','prompt':'Q?'}], prep)
    assert audit['status'] == 'INCOMPLETE'
    assert audit['unchecked_short_ids'] == ['short']


def test_checkpoint_corpus_identity_is_checked(tmp_path):
    cp, tp, path, _ = _fixture(tmp_path)
    source = tmp_path / 'corpus.jsonl'
    source.write_text('\n'.join(json.dumps({'text': t}) for t in [
        'Unique background document one about robotics.',
        'Different unrelated educational text about weather.',
        'A third distinct and well-formed sentence here.',
    ]), encoding='utf-8')
    prepared = tmp_path / 'prepared'
    prepare_shards(source, prepared, 'CC0-1.0')
    # A checkpoint generated from other corpus data cannot be cleared using
    # an unrelated (clean) training directory.
    state = torch.load(cp, map_location='cpu', weights_only=True)
    state['train_state'] = {'dataset_sha256': 'definitely-not-this-corpus'}
    torch.save(state, cp)
    with pytest.raises(ValueError, match='do not match'):
        evaluate_choices(cp, tp, path, license_id='CC0-1.0', training_docs=prepared)