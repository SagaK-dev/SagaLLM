"""Offline multiple-choice evaluation with optional train-corpus contamination audit.

This is a *harness* for independently licensed benchmark records, not a
bundled or verified standard benchmark. It never fetches external datasets.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
import math
import json
from pathlib import Path
from typing import Iterator

import torch
from torch.nn import functional as F

from .config import ModelConfig
from .curation import normalized_text
from .model import SagaTransformer
from .shards import safe_shard_path, sha256_file
from .tokenization import load_tokenizer
from .training import checkpoint_load, resolve_device, verify_checkpoint_tokenizer


class BenchmarkContaminationError(ValueError):
    """Benchmark prompt is present in a provided training document."""


def _valid_license(value: str) -> bool:
    return bool(value and value.strip() and value.strip().lower() not in (
        'none', 'unknown', 'unverified', 'unspecified'))


def read_questions(path: str | Path, *, format: str = 'choices', max_examples: int = 1000) -> list[dict]:
    """Read JSONL questions with size bounds; only question IDs leave this module."""
    if format not in ('choices', 'mmlu') or not (1 <= max_examples <= 10000):
        raise ValueError('Unsupported benchmark format or maximum example count')
    result, seen = [], set()
    with Path(path).open('r', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line.encode('utf-8')) > 65536:
                raise ValueError(f'Benchmark line {line_no} exceeds 64 KiB')
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f'Benchmark line {line_no} is not an object')
            if format == 'mmlu':
                question = item.get('question')
                options = item.get('choices')
                answer = item.get('answer')
                if (not isinstance(question, str) or not question.strip() or
                    not isinstance(options, list) or len(options) != 4 or
                    not all(isinstance(value, str) and value.strip() and len(value) <= 2048
                            for value in options)):
                    raise ValueError(f'Invalid MMLU-style record at line {line_no}')
                if isinstance(answer, str):
                    answer = ord(answer.upper()) - ord('A') if len(answer) == 1 and answer.upper() in 'ABCD' else -1
                item = {
                    'id': item.get('id', f'line-{line_no}'),
                    'source_question': question,
                    'subject': item.get('subject', 'unspecified'),
                    'prompt': 'Question: ' + question + '\n' + '\n'.join(
                        f'{"ABCD"[i]}. {value}' for i, value in enumerate(options)) + '\nAnswer:',
                    'choices': [' A', ' B', ' C', ' D'],
                    'answer_index': answer,
                }
            name, prompt, choices, answer = (item.get(key) for key in ('id','prompt','choices','answer_index'))
            if not isinstance(name, str) or not name or len(name) > 128 or name in seen:
                raise ValueError(f'Invalid or duplicate benchmark ID at line {line_no}')
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
                raise ValueError(f'Invalid prompt at line {line_no}')
            if not isinstance(choices, list) or not 2 <= len(choices) <= 8 or not all(
                isinstance(value, str) and value.strip() and len(value) <= 2048 for value in choices
            ) or len(set(choices)) != len(choices):
                raise ValueError(f'Invalid choices at line {line_no}')
            if type(answer) is not int or not 0 <= answer < len(choices):
                raise ValueError(f'Invalid answer_index at line {line_no}')
            subject = item.get('subject', 'unspecified')
            if not isinstance(subject, str) or not subject.strip() or len(subject) > 128:
                raise ValueError(f'Invalid subject at line {line_no}')
            seen.add(name)
            result.append({'id': name, 'prompt': prompt, 'choices': choices,
                           'answer_index': answer, 'subject': subject,
                           'source_question': question if format == 'mmlu' else prompt})
            if len(result) > max_examples:
                raise ValueError('Benchmark exceeds --max-examples')
    if not result:
        raise ValueError('Benchmark contains no questions')
    return result


def _verified_train_documents(directory: str | Path) -> Iterator[str]:
    """Stream only the train document split and verify manifest hashes."""
    root = Path(directory)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'sagallm-doc-shards-v1':
        raise ValueError('Training corpus must be a prepared document-shard directory')
    for entry in manifest['splits']['train']['files']:
        path = safe_shard_path(root, entry['path'])
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Training document shard SHA-256 mismatch')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                document = json.loads(line).get('text')
                if not isinstance(document, str):
                    raise ValueError('Invalid training document')
                yield document


def audit_contamination(questions: list[dict], training_docs: str | Path) -> dict:
    """Exact normalized prompt substring screen; not semantic/near-match detection.

    Only prompts >= 24 normalized characters are screened. Short prompts
    are *unverifiable*, not automatically cleared.
    """
    # For MMLU-style input use the *raw* question rather than the rendered
    # 'Question: ... A. ... Answer:' evaluation template. Training documents
    # may contain the original question but not the generated template.
    candidates = {q['id']: normalized_text(q.get('source_question', q['prompt']))
                  for q in questions}
    short = sorted(name for name, prompt in candidates.items() if len(prompt) < 24)
    checked = {name: prompt for name, prompt in candidates.items() if len(prompt) >= 24}
    matches: set[str] = set()
    documents = 0
    for doc in _verified_train_documents(training_docs):
        documents += 1
        normalized = normalized_text(doc)
        matches.update(name for name, prompt in checked.items() if prompt in normalized)
    return {
        'status': 'FOUND' if matches else ('INCOMPLETE' if short else 'NO_EXACT_SUBSTRING_FOUND'),
        'matched_ids': sorted(matches), 'unchecked_short_ids': short,
        'train_documents_scanned': documents,
        'method': 'NFKC+casefold+whitespace exact raw-question/prompt substring; NO semantic or paraphrase screening',
    }


@torch.inference_mode()
def _choice_score(model: SagaTransformer, tokenizer, prompt: str, choice: str,
                  device: torch.device, scoring: str) -> float:
    prefix = [tokenizer.SPECIAL['<bos>']] + tokenizer.encode(prompt)
    continuation = tokenizer.encode(choice)
    if not continuation:
        raise ValueError('Choice encoded to no tokens')
    if len(prefix) + len(continuation) - 1 > model.config.max_position_embeddings:
        raise ValueError('Benchmark prompt and choice exceed model context length')
    tokens = torch.tensor([prefix + continuation[:-1]], dtype=torch.long, device=device)
    logits = model(tokens).logits[0, len(prefix)-1: len(prefix)-1+len(continuation)]
    ids = torch.tensor(continuation, dtype=torch.long, device=device)
    logp = F.log_softmax(logits.float(), dim=-1).gather(1, ids[:, None]).sum().item()
    return float(logp / len(continuation) if scoring == 'mean' else logp)


def evaluate_choices(checkpoint: str | Path, tokenizer_path: str | Path, data_path: str | Path,
                     *, license_id: str, format: str = 'choices',
                     training_docs: str | Path | None = None,
                     allow_contaminated: bool = False, device_name: str = 'cpu',
                     max_examples: int = 1000, scoring: str = 'mean',
                     dataset_card_path: str | Path | None = None) -> dict:
    """Score each choice's continuation tokens, report reproducible accuracy.

    If no train documents are available, contamination is explicitly NOT_CHECKED.
    This result must not be labeled a certified held-out benchmark.
    """
    if scoring not in ('mean', 'sum') or not _valid_license(license_id):
        raise ValueError('Provide a reviewed benchmark license and a valid scoring mode')
    card_review = None
    if dataset_card_path is not None:
        from .dataset_cards import verify_dataset_card
        card_review = verify_dataset_card(data_path, dataset_card_path, purpose='evaluate',
                                          expected_license=license_id)
    questions = read_questions(data_path, format=format, max_examples=max_examples)
    audit = ({'status':'NOT_CHECKED', 'matched_ids': [], 'unchecked_short_ids': [],
              'method':'No --train-docs provided; corpus overlap unknown'}
             if training_docs is None else audit_contamination(questions, training_docs))
    if audit['matched_ids'] and not allow_contaminated:
        raise BenchmarkContaminationError('Training data contains benchmark prompts; refusing to report accuracy')
    checkpoint_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    # Compare the curated document manifest to the one embedded in a sharded
    # pretraining run. A user-supplied, unrelated clean corpus is NOT evidence
    # that the benchmark was absent from the corpus used for this checkpoint.
    if training_docs is not None:
        actual = checkpoint_state.get('train_state', {}).get('dataset_sha256')
        expected = (sha256_file(Path(training_docs) / 'manifest.json') + ':train:' +
                    sha256_file(tokenizer_path))
        if actual is not None and actual != expected:
            raise ValueError('Provided train documents do not match the checkpoint training corpus')
        audit['checkpoint_corpus_verified'] = (actual == expected)
        audit['checkpoint_corpus_status'] = ('VERIFIED_SHARD_SOURCE' if actual == expected
                                              else 'CHECKPOINT_CORPUS_NOT_RECORDED')
    else:
        audit['checkpoint_corpus_verified'] = False
        audit['checkpoint_corpus_status'] = 'NOT_CHECKED'
    config = ModelConfig(**checkpoint_state['config'])
    verify_checkpoint_tokenizer(checkpoint_state, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError('Tokenizer vocabulary exceeds model vocabulary')
    if config.parameter_count() > 200_000_000:
        raise ValueError('Benchmark refuses to allocate models over 200M parameters in this CPU-first harness')
    device = resolve_device(device_name)
    model = SagaTransformer(config)
    checkpoint_load(checkpoint, model)
    model.to(device).eval()
    rows, passed = [], 0
    by_subject = defaultdict(lambda: {'questions': 0, 'correct': 0})
    for q in questions:
        scores = [_choice_score(model, tokenizer, q['prompt'], option, device, scoring)
                  for option in q['choices']]
        predicted = max(range(len(scores)), key=lambda i: scores[i])
        passed += predicted == q['answer_index']
        group = by_subject[q['subject']]
        group['questions'] += 1
        group['correct'] += int(predicted == q['answer_index'])
        rows.append({'id': q['id'], 'subject': q['subject'], 'prediction': predicted, 'answer_index': q['answer_index'],
                     'correct': predicted == q['answer_index'], 'choice_log_prob_scores': scores})
    groups = {subject: {'questions': group['questions'], 'correct': group['correct'],
                        'accuracy': group['correct'] / group['questions']}
              for subject, group in sorted(by_subject.items())}
    # Wilson 95% interval is descriptive, not a guarantee for a fixed dataset.
    n, z = len(rows), 1.96
    phat = passed / n
    center = (phat + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {
        'evaluation_type': 'local_multiple_choice_loglikelihood',
        'protocol_id': 'sagallm-mmlu-style-zero-shot-v1' if format == 'mmlu'
                       else 'sagallm-choices-zero-shot-v1',
        'protocol_note': ('Not official MMLU: no prescribed subject/few-shot/split protocol; '
                          'choices are independently tokenized as suffix continuations.'),
        'num_shots': 0,
        'subject_accuracy': groups,
        'macro_subject_accuracy': sum(g['accuracy'] for g in groups.values()) / len(groups),
        'wilson_95_interval': [max(0.0, center-radius), min(1.0, center+radius)],
        'standard_benchmark_certified': False,
        'format': format, 'scoring': scoring, 'questions': len(rows),
        'accuracy': passed/len(rows), 'correct': passed,
        'checkpoint_sha256': sha256_file(checkpoint),
        'tokenizer_sha256': sha256_file(tokenizer_path),
        'benchmark_data_sha256': sha256_file(data_path),
        'benchmark_license_user_asserted': license_id,
        'dataset_card_review': card_review,
        'device': str(device), 'contamination': audit, 'results': rows,
    }