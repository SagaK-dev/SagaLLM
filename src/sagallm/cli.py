"""SagaLLM command-line interface (stdlib argparse)."""
from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path
import torch
from .config import ModelConfig
from .data import prepare_data, read_documents
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer
from .training import checkpoint_load, pretrain, resolve_device


def doctor() -> dict:
    device = resolve_device()
    return {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "device": str(device), "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
            "ram_bytes": _ram(), "note": "Describes the environment running this CLI, not a remote user's PC"}


def _ram() -> int | None:
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SagaLLM: native local language model development")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    token = sub.add_parser("tokenizer")
    ts = token.add_subparsers(dest="action", required=True)
    train_tok = ts.add_parser("train")
    train_tok.add_argument("--data", required=True)
    train_tok.add_argument("--output", required=True)
    train_tok.add_argument("--vocab-size", type=int, default=512)
    token_test = ts.add_parser("check")
    token_test.add_argument("--tokenizer", required=True)
    token_test.add_argument("--text", required=True)
    data = sub.add_parser("data")
    ds = data.add_subparsers(dest="action", required=True)
    prep = ds.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--license", required=True)
    inspect = sub.add_parser("model")
    mi = inspect.add_subparsers(dest="action", required=True)
    mi_model = mi.add_parser("inspect")
    mi_model.add_argument("--config", required=True)
    training = sub.add_parser("train")
    tx = training.add_subparsers(dest="action", required=True)
    for name in ("pretrain", "resume"):
        p = tx.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--steps", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=2)
        p.add_argument("--block-size", type=int, default=32)
        p.add_argument("--grad-accum", type=int, default=1)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--device", default="auto")
        p.add_argument("--allow-large", action="store_true")
        p.add_argument("--log-path", default=None)
    sft = tx.add_parser("sft")
    sft.add_argument("--checkpoint", required=True)
    sft.add_argument("--tokenizer", required=True)
    sft.add_argument("--data", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--steps", type=int, default=4)
    sft.add_argument("--device", default="auto")
    sft.add_argument("--allow-large", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--tokenizer", required=True)
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--block-size", type=int, default=32)
    evaluation.add_argument("--batches", type=int, default=4)
    api = sub.add_parser("serve")
    api.add_argument("--checkpoint", required=True)
    api.add_argument("--tokenizer", required=True)
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--api-key", default=None)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--tokenizer", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=24)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = doctor()
    elif args.command == "tokenizer":
        if args.action == "train":
            tokenizer = ByteBPETokenizer.train(list(read_documents(args.data)), args.vocab_size)
            tokenizer.save(args.output)
            result = {"output": args.output, "vocab_size": tokenizer.vocab_size}
        else:
            tokenizer = ByteBPETokenizer.load(args.tokenizer)
            result = {"encoded": tokenizer.encode(args.text), "decoded": tokenizer.decode(tokenizer.encode(args.text))}
    elif args.command == "data":
        result = prepare_data(args.source, args.output, args.license)
    elif args.command == "model":
        config = ModelConfig.load(args.config)
        count = config.parameter_count()
        result = {"parameters": count, "fp32_weight_bytes": count*4,
                  "adamw_training_state_bytes_estimate": count*16,
                  "note": "This is parameter-state only; excludes activations and temporary buffers"}
    elif args.command == "train":
        if args.action == "sft":
            from .post_training import sft_train
            result = sft_train(args.checkpoint, args.tokenizer, args.data, args.output, steps=args.steps,
                               device_name=args.device, allow_large=args.allow_large)
        else:
            result = pretrain(ModelConfig.load(args.config), args.tokenizer, args.data, args.checkpoint,
                              steps=args.steps, batch_size=args.batch_size, block_size=args.block_size,
                              grad_accum=args.grad_accum, lr=args.lr, device_name=args.device,
                              resume=args.action == "resume", allow_large=args.allow_large, log_path=args.log_path)
    elif args.command == "evaluate":
        from .evaluation import evaluate_language_model
        result = evaluate_language_model(args.checkpoint, args.tokenizer, args.data,
                                         block_size=args.block_size, batches=args.batches)
    elif args.command == "serve":
        if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
            parser.error("Non-loopback binding requires --api-key. Use a TLS terminator for external access.")
        from .api import create_app
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install the sagallm[serve] optional dependency") from exc
        app = create_app(args.checkpoint, args.tokenizer, api_key=args.api_key)
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
        return
    else:
        tokenizer = ByteBPETokenizer.load(args.tokenizer)
        device = resolve_device(args.device)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = ModelConfig(**state["config"])
        if tokenizer.vocab_size > config.vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds checkpoint model vocabulary")
        model = SagaTransformer(config)
        checkpoint_load(args.checkpoint, model)
        model.to(device).eval()
        ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)
        new_ids = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 eos_token_id=ByteBPETokenizer.SPECIAL["<eos>"])
        result = {"text": tokenizer.decode(new_ids[0].tolist()), "generated_tokens": int(new_ids.size(1)-ids.size(1))}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()