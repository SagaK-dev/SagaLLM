from pathlib import Path
import yaml


def test_workflow_only_runs_cpu_tests_and_never_gpu_training():
    root = Path(__file__).resolve().parent.parent
    source = (root / '.github/workflows/cpu-ci.yml').read_text(encoding='utf-8')
    workflow = yaml.safe_load(source)
    job = workflow['jobs']['cpu-tests']
    assert job['timeout-minutes'] == 15
    assert job['runs-on'] == 'ubuntu-latest'
    assert workflow['permissions']['contents'] == 'read'
    assert any('python scripts/check_cpu.py' in step.get('run', '') for step in job['steps'])
    assert 'torch>=2.3' in source and 'download.pytorch.org/whl/cpu' in source
    assert 'self-hosted' not in source
    assert '--allow-large' not in source


def test_no_github_credentials_in_ci():
    root = Path(__file__).resolve().parent.parent
    source = (root / '.github/workflows/cpu-ci.yml').read_text(encoding='utf-8')
    assert '${{ secrets.' not in source
    assert 'pull_request_target' not in source