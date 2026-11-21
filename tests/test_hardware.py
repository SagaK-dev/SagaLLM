"""Offline hardware evidence tests. No physical GPU claims are made."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sagallm import hardware
from sagallm.cli import main


def test_inventory_does_not_allocate_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('hardware inventory allocated a model')
    monkeypatch.setattr(hardware, 'SagaTransformer', forbidden)
    info = hardware.inventory()
    assert info['schema'] == hardware.SCHEMA
    assert info['status'] == 'INVENTORY_ONLY'
    assert info['physical_evidence'] == 'NOT_ATTESTED'
    assert 'gloo_available' in info['distributed']
    assert 'python' in info['system']
    assert isinstance(info['accelerators'], list)


def test_tiny_cpu_training_checkpoint_and_generation():
    info = hardware.smoke(device_name='cpu')
    assert info['status'] == 'PASS_RUNTIME'
    assert info['device'] == 'cpu'
    assert info['finite_gradients']
    assert info['optimizer_updated']
    assert info['checkpoint_roundtrip']
    assert info['generated_tokens'] == 2
    assert 0 < info['parameters'] < 200_000
    assert info['physical_evidence'] == 'NOT_ATTESTED'


def test_unavailable_accelerator_never_claims_pass():
    if torch.cuda.is_available():
        pytest.skip('CUDA present; absence scenario not applicable')
    report = hardware.smoke(device_name='cuda')
    assert report['status'] == 'NOT_RUN'
    assert report['physical_evidence'] == 'NOT_ATTESTED'
    assert 'CUDA' in report['reason']


def test_unsupported_precision_is_reported_not_silently_downgraded():
    result = hardware.smoke(device_name='cpu', precision='bf16')
    assert result['status'] == 'NOT_RUN'
    assert result['precision'] == 'bf16'
    assert 'reason' in result


def test_invalid_hardware_options():
    with pytest.raises(ValueError, match='precision'):
        hardware.smoke(precision='fp16')
    with pytest.raises(ValueError, match='device'):
        hardware.smoke(device_name='remote_gpu')
    with pytest.raises(ValueError, match='timeout_seconds'):
        hardware.distributed_smoke(timeout_seconds=999)


def test_write_evidence_refuses_overwrite(tmp_path):
    target = tmp_path / 'evidence.json'
    info = hardware.run_hardware_command('inspect', output=str(target))
    assert info['report_path'] == str(target)
    stored = json.loads(target.read_text())
    assert stored['status'] == 'INVENTORY_ONLY'
    with pytest.raises(FileExistsError):
        hardware.save_report(target, {'status': 'SHOULD_NOT_REPLACE'})
    assert json.loads(target.read_text()) == stored


def test_cli_inspect_smoke_and_report(tmp_path, capsys):
    out = tmp_path / 'result.json'
    main(['hardware', 'inspect', '--output', str(out)])
    report = json.loads(capsys.readouterr().out)
    assert report['report_path'] == str(out)
    main(['hardware', 'smoke', '--device', 'cpu'])
    report = json.loads(capsys.readouterr().out)
    assert report['runtime_test']['status'] == 'PASS_RUNTIME'


def test_distributed_backend_not_run_when_unavailable(monkeypatch):
    monkeypatch.setattr(hardware.dist, 'is_available', lambda: False)
    report = hardware.distributed_smoke()
    assert report['status'] == 'NOT_RUN'


def test_distributed_worker_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(hardware.dist, 'is_available', lambda: True)
    monkeypatch.setattr(hardware.dist, 'is_gloo_available', lambda: True)
    monkeypatch.setattr(hardware.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=5))
    report = hardware.distributed_smoke()
    assert report['status'] == 'FAIL'
    assert report['exit_code'] == 5
    assert report['physical_evidence'] == 'NOT_ATTESTED'


def test_hardware_report_contains_no_false_physical_pass(tmp_path):
    report = hardware.run_hardware_command('smoke', device='cpu', output=str(tmp_path / 'cpu.json'))
    serialized = json.dumps(report)
    assert 'PASS_PHYSICAL' not in serialized
    assert report['physical_evidence'] == 'NOT_ATTESTED'
    assert report['runtime_test']['physical_evidence'] == 'NOT_ATTESTED'