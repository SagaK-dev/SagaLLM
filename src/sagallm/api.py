"""Optional localhost-first inference API with bounded local resource usage.

Cooperative generation deadlines cannot interrupt a stalled native kernel. Use a
reverse proxy with enforced network timeouts and TLS for any external deployment.
"""
from collections import deque
from pathlib import Path
import secrets
from threading import BoundedSemaphore, Lock
import time
import torch
from .config import ModelConfig
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer
from .training import checkpoint_load, resolve_device, verify_checkpoint_tokenizer


def create_app(checkpoint: str | Path, tokenizer_file: str | Path, *, api_key: str | None = None,
               device_name: str = 'auto', requests_per_minute: int = 60,
               max_concurrent: int = 1, max_generation_seconds: float = 30.0):
    if requests_per_minute < 1 or max_concurrent < 1 or max_generation_seconds <= 0:
        raise ValueError('Invalid serving resource limits')
    try:
        from fastapi import FastAPI, HTTPException, Header, Depends
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install optional 'sagallm[serve]' dependencies for HTTP serving") from exc
    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = ModelConfig(**state['config'])
    verify_checkpoint_tokenizer(state, tokenizer_file)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError('Tokenizer vocabulary exceeds checkpoint vocab')
    model = SagaTransformer(config)
    checkpoint_load(checkpoint, model)
    device = resolve_device(device_name)
    model.to(device).eval()
    inference_slots = BoundedSemaphore(value=max_concurrent)
    request_times = deque()
    rate_lock = Lock()

    class CompletionRequest(BaseModel):
        model: str = 'SagaLLM'
        prompt: str = Field(min_length=1, max_length=8192)
        max_tokens: int = Field(default=16, ge=1, le=128)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    class ChatMessage(BaseModel):
        role: str = Field(min_length=1, max_length=16)
        content: str = Field(min_length=1, max_length=2048)

    class ChatRequest(BaseModel):
        model: str = 'SagaLLM'
        messages: list[ChatMessage] = Field(min_length=1, max_length=16)
        max_tokens: int = Field(default=16, ge=1, le=128)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    app = FastAPI(title='SagaLLM (experimental)', docs_url=None, redoc_url=None)

    def authenticate(authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or '', 'Bearer ' + api_key):
            raise HTTPException(status_code=401, detail='Unauthorized')

    def completion(prompt: str, limit: int, temperature: float):
        now = time.monotonic()
        with rate_lock:
            while request_times and request_times[0] <= now - 60.0:
                request_times.popleft()
            if len(request_times) >= requests_per_minute:
                raise HTTPException(status_code=429, detail='Request rate limit exceeded')
            request_times.append(now)
        if not inference_slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail='Inference concurrency limit exceeded')
        try:
            ids = tokenizer.encode(prompt, add_bos=True)
            if len(ids) > config.max_position_embeddings:
                raise HTTPException(status_code=413, detail='Prompt exceeds configured context window')
            x = torch.tensor([ids], device=device, dtype=torch.long)
            try:
                y = model.generate(x, max_new_tokens=limit, temperature=temperature,
                                   eos_token_id=tokenizer.SPECIAL['<eos>'],
                                   deadline_monotonic=time.monotonic() + max_generation_seconds,
                                   vocab_limit=tokenizer.vocab_size)
            except TimeoutError as exc:
                raise HTTPException(status_code=504, detail='Generation time budget exceeded') from exc
            generated = y[0, len(ids):].tolist()
            finished_eos = bool(generated and generated[-1] == tokenizer.SPECIAL['<eos>'])
            return tokenizer.decode(generated), len(ids), len(generated), ('stop' if finished_eos else 'length')
        finally:
            inference_slots.release()

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': 'SagaLLM', 'device': str(device)}

    @app.post('/v1/completions', dependencies=[Depends(authenticate)])
    def completions(req: CompletionRequest):
        text, prompt_tokens, generated_tokens, reason = completion(req.prompt, req.max_tokens, req.temperature)
        return {'id': 'sagallm-local', 'object': 'text_completion', 'model': 'SagaLLM',
                'choices': [{'index': 0, 'text': text, 'finish_reason': reason}],
                'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': generated_tokens,
                          'total_tokens': prompt_tokens+generated_tokens}}

    @app.post('/v1/chat/completions', dependencies=[Depends(authenticate)])
    def chat_completions(req: ChatRequest):
        if any(m.role not in {'system', 'user', 'assistant'} for m in req.messages):
            raise HTTPException(status_code=422, detail='Unknown message role')
        prompt = ''.join(f'{m.role.capitalize()}: {m.content}\n' for m in req.messages) + 'Assistant: '
        text, prompt_tokens, generated_tokens, reason = completion(prompt, req.max_tokens, req.temperature)
        return {'id': 'sagallm-local', 'object': 'chat.completion', 'model': 'SagaLLM',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text},
                             'finish_reason': reason}],
                'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': generated_tokens,
                          'total_tokens': prompt_tokens+generated_tokens}}
    return app