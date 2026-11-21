"""MMLU-style contamination must detect raw source questions, not templates."""
import json
from dataclasses import replace
import pytest
import torch

from sagallm.benchmarks import audit_contamination, evaluate_choices, read_questions, BenchmarkContaminationError
from sagallm.config import ModelConfig
from sagallm.dataset_cards import DatasetCardError
from sagallm.model import SagaTransformer
from sagallm.shards import prepare_shards
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def _mmlu_fixture(tmp_path):
    train = tmp_path / 'train.jsonl'
    q = 'Which language is used to define this introductory sample program for a local training benchmark?'
    records = [
        'A distinct passage that refers to a reliable source for scientific experiments.',
        'A teaching note that includes a direct question: ' + q + ' Then it discusses programming.',
        'A new chapter concerning safe and reproducible machine learning experiments.',
        'A completely different lesson about the history of number theory.',
        'A separate example covering the basics of computer hardware components.',
    ]
    train.write_text('\n'.join(json.dumps({'text':x}) for x in records), encoding='utf-8')
    prep = tmp_path / 'prepared'
    prepare_shards(train, prep, 'CC0-1.0', validation_fraction=0.2)
    dataset = tmp_path / 'mmlu.jsonl'
    questions = [
        {'id':'english','subject':'language', 'question': q,
         'choices':['Python','French','Latin','Spanish'],'answer':'A'},
        {'id':'math','subject':'mathematics',
         'question': 'Which value is equal to twelve minus four in elementary arithmetic?',
         'choices':['6','7','8','9'],'answer':2},
    ]
    dataset.write_text('\n'.join(json.dumps(x) for x in questions), encoding='utf-8')
    return prep, dataset, q


def test_mmlu_leakage_scans_original_question(tmp_path):
    prep, dataset, q = _mmlu_fixture(tmp_path)
    questions = read_questions(dataset, format='mmlu')
    assert questions[0]['source_question'] == q
    assert questions[0]['source_question'] != questions[0]['prompt']
    # Construct a manifest whose training split *certainly* contains the raw
    # question, without the formatted MMLU prompt. This must be detected.
    import hashlib
    root = tmp_path / 'explicit_train'
    root.mkdir()
    f = root / 'train-00000.jsonl'
    f.write_text(json.dumps({'text': 'Course material asks: ' + q + ' End of note.'}), encoding='utf-8')
    manifest = {'format': 'sagallm-doc-shards-v1', 'splits': {'train': {'files': [
        {'path': f.name, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}]}}}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    audit = audit_contamination(questions, root)
    assert audit['status'] == 'FOUND' and 'english' in audit['matched_ids']


def test_subject_aggregation_and_card_evaluation(tmp_path):
    _, data, _ = _mmlu_fixture(tmp_path)
    tok = ByteBPETokenizer.train(['The meaning of machine learning.', 'A B C D answers.'],
                                 vocab_size=264, min_frequency=1)
    tokenizer_file = tmp_path / 'tok.json'
    tok.save(tokenizer_file)
    cfg = replace(ModelConfig.load('configs/model/tiny.yaml'), max_position_embeddings=512)
    model = SagaTransformer(cfg)
    checkpoint = tmp_path / 'model.pt'
    checkpoint_save(checkpoint, model, torch.optim.AdamW(model.parameters()), 0, cfg, {})
    report = evaluate_choices(checkpoint, tokenizer_file, data, format='mmlu', license_id='CC0-1.0')
    assert report['questions'] == 2
    assert set(report['subject_accuracy']) == {'language', 'mathematics'}
    assert len(report['wilson_95_interval']) == 2
    assert report['protocol_id'] == 'sagallm-mmlu-style-zero-shot-v1'
    assert report['standard_benchmark_certified'] is False
    assert all('source_question' not in x and 'prompt' not in x for x in report['results'])


def test_benchmark_card_must_match_data_and_purpose(tmp_path):
    import hashlib
    _, data, _ = _mmlu_fixture(tmp_path)
    tok = ByteBPETokenizer.train(['Python', 'A B C D'], vocab_size=260)
    tf = tmp_path / 'tok.json'; tok.save(tf)
    cfg = replace(ModelConfig.load('configs/model/tiny.yaml'), max_position_embeddings=512)
    model = SagaTransformer(cfg)
    cp = tmp_path / 'model.pt'; checkpoint_save(cp, model, torch.optim.AdamW(model.parameters()), 0, cfg, {})
    card = tmp_path / 'card.json'
    content = {'format':'sagallm-dataset-card-v1', 'name':'fixture',
               'source_uri':'local://SagaLLM/tests', 'source_version':'v1',
               'license_id':'CC0-1.0','license_evidence':'Fixture originally authored for tests',
               'review_status':'approved','reviewed_by':'test author',
               'reviewed_at':'2026-11-21', 'allowed_uses':['evaluate'],
               'file_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
    card.write_text(json.dumps(content))
    result = evaluate_choices(cp, tf, data, license_id='CC0-1.0',
                              format='mmlu', dataset_card_path=card)
    assert result['dataset_card_review']['purpose'] == 'evaluate'
    content['allowed_uses'] = ['train']
    card.write_text(json.dumps(content))
    with pytest.raises(DatasetCardError, match='authorize'):
        evaluate_choices(cp, tf, data, license_id='CC0-1.0',
                         format='mmlu', dataset_card_path=card)