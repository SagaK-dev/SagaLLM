"""Local-only model export: safetensors, Transformers custom code and provenance.

Requires a trusted SagaLLM checkpoint. Never uploads or publishes models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import torch

from .config import ModelConfig
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer, FastBPETokenizer, load_tokenizer
from .training import verify_checkpoint_tokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export_transformers(
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    output: str | Path,
    *,
    allow_large: bool = False,
) -> dict:
    """Export a new standalone *local* directory, without upload or overwrite.

    Legacy Python BPE is preserved as a SagaLLM artifact, but cannot be read
    by HF AutoTokenizer. Only the optional native FastBPE is AutoTokenizer-ready.
    """
    checkpoint, tokenizer_path, output = Path(checkpoint), Path(tokenizer_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Export target already exists: {output}")
    # The operator must treat pickle-era checkpoints as trusted input even when
    # loaded with weights_only=True (resource-exhaustion attacks remain possible).
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(state, dict) or 'config' not in state or 'model' not in state:
        raise ValueError('Invalid SagaLLM checkpoint')
    config = ModelConfig(**state['config'])
    if config.parameter_count() > 200_000_000 and not allow_large:
        raise ValueError('Export above 200M parameters requires --allow-large')
    verify_checkpoint_tokenizer(state, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError('Tokenizer vocabulary exceeds model vocabulary')
    model = SagaTransformer(config).cpu().eval()
    model.load_state_dict(state['model'], strict=True)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError('Local export requires safetensors: pip install -e ".[hf-export]"') from exc

    # Standalone custom loader sources are installed with this package.
    source = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=False)
    try:
        for src, dst in (
            (source / 'model.py', 'model.py'),
            (source / 'config.py', 'config.py'),
            (source / 'hf_export' / 'configuration_sagallm.py', 'configuration_sagallm.py'),
            (source / 'hf_export' / 'modeling_sagallm.py', 'modeling_sagallm.py'),
        ):
            shutil.copyfile(src, output / dst)
        # The HF loader creates tied embedding weights internally. Leave out
        # the alias because safetensors disallows duplicate shared storage.
        tensors = {
            'transformer.' + key: tensor.detach().cpu().contiguous()
            for key, tensor in model.state_dict().items()
            if not (config.tie_word_embeddings and key == 'lm_head.weight')
        }
        save_file(tensors, str(output / 'model.safetensors'), metadata={'format': 'pt'})
        config_json = {
            **config.to_dict(),
            'model_type': 'sagallm',
            'architectures': ['SagaLLMForCausalLM'],
            'auto_map': {
                'AutoConfig': 'configuration_sagallm.SagaLLMConfig',
                'AutoModelForCausalLM': 'modeling_sagallm.SagaLLMForCausalLM',
            },
            'bos_token_id': ByteBPETokenizer.SPECIAL['<bos>'],
            'eos_token_id': ByteBPETokenizer.SPECIAL['<eos>'],
            'pad_token_id': ByteBPETokenizer.SPECIAL['<pad>'],
            'is_decoder': True,
            'sagallm_tokenizer_vocab_size': tokenizer.vocab_size,
        }
        (output / 'config.json').write_text(
            json.dumps(config_json, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        (output / 'generation_config.json').write_text(
            json.dumps({'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
                        'max_new_tokens': 32,
                        'use_cache': False,
                        'suppress_tokens': list(range(tokenizer.vocab_size, config.vocab_size))}, indent=2) + '\n', encoding='utf-8')
        # The original byte-level BPE cannot be converted to a standard HF
        # tokenizer without changing segmentation rules. Avoid a false claim.
        shutil.copyfile(tokenizer_path, output / 'sagallm_tokenizer.json')
        tokenizer_compatible = isinstance(tokenizer, FastBPETokenizer)
        if tokenizer_compatible:
            tokenizer.engine.save(str(output / 'tokenizer.json'))
            (output / 'tokenizer_config.json').write_text(json.dumps({
                'tokenizer_class': 'PreTrainedTokenizerFast',
                'model_max_length': config.max_position_embeddings,
                'bos_token': '<bos>', 'eos_token': '<eos>',
                'pad_token': '<pad>', 'unk_token': '<unk>',
            }, indent=2) + '\n', encoding='utf-8')
            (output / 'special_tokens_map.json').write_text(json.dumps({
                'bos_token': '<bos>', 'eos_token': '<eos>',
                'pad_token': '<pad>', 'unk_token': '<unk>',
            }, indent=2) + '\n', encoding='utf-8')
        report = {
            'tokenizer_identity_verified': bool(state.get('train_state', {}).get('tokenizer_sha256')),
            'format': 'sagallm_transformers_custom_v1',
            'checkpoint_sha256': _sha256(checkpoint),
            'tokenizer_sha256': _sha256(tokenizer_path),
            'parameter_count': config.parameter_count(),
            'checkpoint_step': int(state.get('step', 0)),
            'hf_model_auto_load': True,
            'hf_tokenizer_auto_load': tokenizer_compatible,
            'requires_trust_remote_code': True,
            'uploaded': False,
            'note': 'Local artifact only. Review exported Python files before trust_remote_code.',
            'files_sha256': {
                p.name: _sha256(p) for p in sorted(output.iterdir()) if p.is_file()
            },
        }
        (output / 'export_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        # Never leave an export that appears complete after a partial failure.
        shutil.rmtree(output)
        raise
    return {'output': str(output), 'parameters': report['parameter_count'],
            'hf_tokenizer_auto_load': tokenizer_compatible,
            'checkpoint_sha256': report['checkpoint_sha256']}