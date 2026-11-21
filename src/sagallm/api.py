"""Optional localhost-first inference API; deploy behind authenticated TLS proxy."""
import secrets
from pathlib import Path
import torch
from .config import ModelConfig
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer
from .training import checkpoint_load, resolve_device


def create_app(checkpoint: str | Path, tokenizer_file: str | Path, *, api_key: str | None = None,
               device_name: str = "auto"):
    try:
        from fastapi import FastAPI, HTTPException, Header, Depends
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install optional 'sagallm[serve]' dependencies for HTTP serving") from exc
    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**state["config"])
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds checkpoint vocab")
    model = SagaTransformer(config)
    checkpoint_load(checkpoint, model)
    device = resolve_device(device_name)
    model.to(device).eval()

    class CompletionRequest(BaseModel):
        model: str = "SagaLLM"
        prompt: str = Field(min_length=1, max_length=8192)
        max_tokens: int = Field(default=16, ge=1, le=128)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "SagaLLM"
        messages: list[ChatMessage] = Field(min_length=1, max_length=16)
        max_tokens: int = Field(default=16, ge=1, le=128)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    app = FastAPI(title="SagaLLM (experimental)", docs_url=None, redoc_url=None)

    def authenticate(authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or "", "Bearer " + api_key):
            raise HTTPException(status_code=401, detail="Unauthorized")

    def completion(prompt: str, limit: int, temperature: float):
        ids = tokenizer.encode(prompt, add_bos=True)
        if len(ids) > config.max_position_embeddings:
            raise HTTPException(status_code=413, detail="Prompt exceeds configured context window")
        x = torch.tensor([ids], device=device, dtype=torch.long)
        y = model.generate(x, max_new_tokens=limit, temperature=temperature, eos_token_id=tokenizer.SPECIAL["<eos>"])
        generated = y[0, len(ids):].tolist()
        return tokenizer.decode(generated), len(ids), len(generated)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "SagaLLM", "device": str(device)}

    @app.post("/v1/completions", dependencies=[Depends(authenticate)])
    def completions(req: CompletionRequest):
        text, n_prompt, n_gen = completion(req.prompt, req.max_tokens, req.temperature)
        return {"id": "sagallm-local", "object": "text_completion", "model": req.model,
                "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt+n_gen}}

    @app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
    def chat_completions(req: ChatRequest):
        if any(m.role not in {"system", "user", "assistant"} for m in req.messages):
            raise HTTPException(status_code=422, detail="Unknown message role")
        prompt = "".join(f"{m.role.capitalize()}: {m.content}\n" for m in req.messages) + "Assistant: "
        text, n_prompt, n_gen = completion(prompt, req.max_tokens, req.temperature)
        return {"id": "sagallm-local", "object": "chat.completion", "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen, "total_tokens": n_prompt+n_gen}}
    return app