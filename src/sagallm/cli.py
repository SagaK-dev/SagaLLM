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
from .shards import prepare_shards, tokenize_shards
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer, FastBPETokenizer, load_tokenizer
from .training import checkpoint_load, pretrain, resolve_device, verify_checkpoint_tokenizer


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
    hardware = sub.add_parser("hardware", help="Local-only hardware inventory and bounded smoke tests")
    ht = hardware.add_subparsers(dest="action", required=True)
    hw_inspect = ht.add_parser("inspect", help="Metadata only; never allocate model weights")
    hw_inspect.add_argument("--output", help="New JSON evidence file; refuses overwrite")
    hw_smoke = ht.add_parser("smoke", help="One tiny train/checkpoint/generation step")
    hw_smoke.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    hw_smoke.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    hw_smoke.add_argument("--output", help="New JSON evidence file; refuses overwrite")
    hw_dist = ht.add_parser("distributed-smoke", help="Two local CPU Gloo collective ranks, no training")
    hw_dist.add_argument("--timeout", type=int, default=40)
    hw_dist.add_argument("--output", help="New JSON evidence file; refuses overwrite")
    java = sub.add_parser("java", help="Local natural-language Java starter projects (supported templates only)")
    js = java.add_subparsers(dest="action", required=True)
    js.add_parser("templates", help="List supported basic Java tasks")
    jnew = js.add_parser("new", help="Create Java project from a supported Japanese/English request")
    jnew.add_argument("--prompt", required=True)
    jnew.add_argument("--output", required=True, help="New project directory; never overwrite")
    jnew.add_argument("--class-name", default="Main")
    jnew.add_argument("--template", choices=("hello", "calculator", "fizzbuzz", "guess", "todo", "prime", "factorial", "file", "swing"))
    jrev = js.add_parser("revise", help="Preview or apply a supported conversational revision")
    jrev.add_argument("--project", required=True)
    jrev.add_argument("--prompt", required=True)
    jrev.add_argument("--apply", action="store_true", help="Write reviewed diff; default preview only")
    jcheck = js.add_parser("check", help="Compile with javac -proc:none (does not run Java)")
    jcheck.add_argument("--project", required=True)
    jrun = js.add_parser("run", help="Explicitly compile and run trusted local Java; NOT sandboxed")
    jrun.add_argument("--project", required=True)
    jrun.add_argument("--trust-local-code", action="store_true", help="I understand Java can access local files/network")
    jrun.add_argument("--stdin", default="", help="Text to pass to the console program")
    jrun.add_argument("--timeout", type=float, default=6.0)
    jrun.add_argument("--java-arg", action="append", default=[], help="Java main method argument")
    jchat = js.add_parser("chat", help="Interactive create/revise/check loop")
    jchat.add_argument("--project", required=True)
    jchat.add_argument("--class-name", default="Main")
    jai = js.add_parser("ai-revise", help="Local SagaLLM-checkpoint Java code proposal, compile, and reviewed apply")
    jaic = js.add_parser("ai-chat", help="Interactive revision using a local SagaLLM checkpoint")
    for assistant in (jai, jaic):
        assistant.add_argument("--project", required=True, help="Existing SagaLLM Java starter project")
        assistant.add_argument("--checkpoint", required=True, help="Trusted local SagaLLM checkpoint")
        assistant.add_argument("--tokenizer", required=True, help="Matching local tokenizer file")
        assistant.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        assistant.add_argument("--max-new-tokens", type=int, default=256)
        assistant.add_argument("--attempts", type=int, default=2, help="1..3 attempts; never runs generated code")
    jai.add_argument("--prompt", required=True)
    jai.add_argument("--apply", action="store_true", help="Explicitly write a successful, compiled proposal")
    jdata = js.add_parser("dataset-build", help="Build small original Java SFT/held-out fixture")
    jdata.add_argument("--output", required=True, help="New directory; never overwrite")
    jverify = js.add_parser("dataset-verify", help="Verify Java fixture hashes and family separation")
    jverify.add_argument("--dataset", required=True)
    jeval = js.add_parser("evaluate", help="Compile-only evaluation of held-out Java generation")
    jeval.add_argument("--dataset", required=True)
    jeval.add_argument("--checkpoint", required=True)
    jeval.add_argument("--tokenizer", required=True)
    jeval.add_argument("--split", choices=("validation", "test"), default="test")
    jeval.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    jeval.add_argument("--max-examples", type=int, default=3)
    jeval.add_argument("--max-new-tokens", type=int, default=512)
    jeval.add_argument("--attempts", type=int, default=1)
    jeval.add_argument("--output", help="New JSON report; never overwritten")
    jsft = js.add_parser("toy-sft", help="Write five original toy Java instruction-response examples")
    jsft.add_argument("--output", required=True, help="New JSONL file; never overwrite")
    token = sub.add_parser("tokenizer")
    ts = token.add_subparsers(dest="action", required=True)
    train_tok = ts.add_parser("train")
    train_tok.add_argument("--data", required=True)
    train_tok.add_argument("--output", required=True)
    train_tok.add_argument("--vocab-size", type=int, default=512)
    train_tok.add_argument("--backend", choices=("python", "hf"), default="python",
                           help="hf uses optional Rust-backed Hugging Face Tokenizers")
    token_test = ts.add_parser("check")
    token_test.add_argument("--tokenizer", required=True)
    token_test.add_argument("--text", required=True)
    data = sub.add_parser("data")
    ds = data.add_subparsers(dest="action", required=True)
    prep = ds.add_parser("prepare")
    prep.add_argument("--source", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--license", required=True)
    prep.add_argument("--dataset-card", help="Reviewed, SHA-pinned dataset usage card")
    shards = ds.add_parser('shard')
    shards.add_argument('--source', required=True)
    shards.add_argument('--output', required=True)
    shards.add_argument('--license', required=True)
    shards.add_argument('--dataset-card', help='Reviewed, SHA-pinned dataset usage card')
    shards.add_argument('--max-docs-per-shard', type=int, default=1000)
    shards.add_argument('--keep-near-duplicates', action='store_true',
                        help='Disable heuristic near-duplicate filtering')
    license_check = ds.add_parser('verify-license', help='Offline review-card and SHA verification')
    license_check.add_argument('--source', required=True)
    license_check.add_argument('--card', required=True)
    license_check.add_argument('--purpose', choices=('train', 'evaluate', 'redistribute'), required=True)
    tokenize = ds.add_parser('tokenize')
    tokenize.add_argument('--prepared', required=True)
    tokenize.add_argument('--tokenizer', required=True)
    tokenize.add_argument('--output', required=True)
    tokenize.add_argument('--max-tokens-per-shard', type=int, default=100000)
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
        p.add_argument('--schedule-steps', type=int, default=None)
        p.add_argument('--checkpoint-every', type=int, default=0)
        p.add_argument('--validate-every', type=int, default=0)
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
    evaluation.add_argument('--split', choices=['train', 'validation'], default='validation')
    evaluation.add_argument('--allow-training-data', action='store_true')
    bench = sub.add_parser("benchmark", help="Offline multiple-choice benchmark harness")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--tokenizer", required=True)
    bench.add_argument("--data", required=True, help="JSONL questions (not downloaded automatically)")
    bench.add_argument("--license", required=True, help="User-reviewed benchmark dataset license")
    bench.add_argument("--dataset-card", help="Reviewed, SHA-pinned benchmark dataset card")
    bench.add_argument("--format", choices=("choices", "mmlu"), default="choices")
    bench.add_argument("--train-docs", help="Prepared document-shard directory for contamination audit")
    bench.add_argument("--allow-contaminated", action="store_true")
    bench.add_argument("--max-examples", type=int, default=1000)
    bench.add_argument("--scoring", choices=("mean", "sum"), default="mean")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--output", help="Write a JSON report (refuses overwrite)")
    mmlu = sub.add_parser("mmlu", help="Offline Hendrycks-style MMLU CSV evaluator")
    mm = mmlu.add_subparsers(dest="action", required=True)
    inv = mm.add_parser("inventory", help="Create pending-review SHA-256 inventory for local CSVs")
    inv.add_argument("--data-dir", required=True)
    inv.add_argument("--output", required=True)
    verify = mm.add_parser("verify", help="Verify SHA-pinned reviewed MMLU CSV inventory")
    verify.add_argument("--data-dir", required=True)
    verify.add_argument("--inventory", required=True)
    runm = mm.add_parser("evaluate", help="Evaluate reviewed local MMLU dev/test CSVs")
    runm.add_argument("--checkpoint", required=True)
    runm.add_argument("--tokenizer", required=True)
    runm.add_argument("--data-dir", required=True)
    runm.add_argument("--inventory", required=True)
    runm.add_argument("--subjects", help="Comma-separated local MMLU subjects; default all")
    runm.add_argument("--shots", type=int, default=5)
    runm.add_argument("--max-examples", type=int, default=100)
    runm.add_argument("--scoring", choices=("reference_next_token", "continuation"),
                      default="reference_next_token")
    runm.add_argument("--train-docs", help="Exact prepared train-document shards, if available")
    runm.add_argument("--allow-contaminated", action="store_true")
    runm.add_argument("--device", default="cpu")
    runm.add_argument("--output", help="New JSON report; refuses overwrite")
    exp = sub.add_parser("export", help="Write a local Transformers-compatible safetensors bundle")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--tokenizer", required=True)
    exp.add_argument("--output", required=True, help="New directory; never overwritten")
    exp.add_argument("--allow-large", action="store_true")
    api = sub.add_parser("serve")
    api.add_argument("--checkpoint", required=True)
    api.add_argument("--tokenizer", required=True)
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--api-key", default=None)
    api.add_argument('--requests-per-minute', type=int, default=60)
    api.add_argument('--max-concurrent', type=int, default=1)
    api.add_argument('--generation-seconds', type=float, default=30.0)
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
    elif args.command == "hardware":
        from .hardware import run_hardware_command
        result = run_hardware_command(
            args.action, device=getattr(args, "device", "cpu"),
            precision=getattr(args, "precision", "fp32"),
            timeout=getattr(args, "timeout", 40), output=args.output,
        )
    elif args.command == "java":
        from .java_vibe import TEMPLATES, create_project, propose_revision, check_project, run_project, chat
        if args.action == "templates":
            result = {"supported": TEMPLATES, "mode": "deterministic", "llm_based": False}
        elif args.action == "new":
            result = create_project(args.output, args.prompt, template=args.template, class_name=args.class_name)
        elif args.action == "revise":
            result = propose_revision(args.project, args.prompt, apply=args.apply)
        elif args.action == "check":
            result = check_project(args.project)
        elif args.action == "run":
            result = run_project(args.project, trust_local_code=args.trust_local_code,
                                 stdin=args.stdin, timeout=args.timeout, java_args=args.java_arg)
        elif args.action == "dataset-build":
            from .java_coding_eval import build_coding_fixture
            result = build_coding_fixture(args.output)
        elif args.action == "dataset-verify":
            from .java_coding_eval import verify_coding_fixture
            result = verify_coding_fixture(args.dataset)
        elif args.action == "evaluate":
            from .java_coding_eval import evaluate_java_compilation
            from .java_ai import LocalCheckpointGenerator
            from .java_coding_eval import verify_coding_fixture
            verify_coding_fixture(args.dataset)  # reject modified eval data before allocating weights
            backend = LocalCheckpointGenerator(args.checkpoint, args.tokenizer, device=args.device)
            result = evaluate_java_compilation(args.dataset, generator=backend, split=args.split,
                                               max_examples=args.max_examples,
                                               max_new_tokens=args.max_new_tokens, attempts=args.attempts)
            result["checkpoint_sha256"] = backend.checkpoint_sha256
            result["tokenizer_sha256"] = backend.tokenizer_sha256
            if args.output:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("x", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2)
        elif args.action == "toy-sft":
            from .java_ai import build_toy_java_sft
            result = build_toy_java_sft(args.output)
        elif args.action in ("ai-revise", "ai-chat"):
            from .java_ai import LocalCheckpointGenerator, generate_revision, ai_chat
            backend = LocalCheckpointGenerator(args.checkpoint, args.tokenizer, device=args.device)
            if args.action == "ai-revise":
                result = generate_revision(
                    args.project, args.prompt, generator=backend,
                    max_new_tokens=args.max_new_tokens, attempts=args.attempts,
                    apply=args.apply,
                    checkpoint_sha256=backend.checkpoint_sha256,
                    tokenizer_sha256=backend.tokenizer_sha256,
                )
            else:
                ai_chat(args.project, backend, max_new_tokens=args.max_new_tokens,
                        attempts=args.attempts, checkpoint_sha256=backend.checkpoint_sha256,
                        tokenizer_sha256=backend.tokenizer_sha256)
                return
        else:
            chat(args.project, class_name=args.class_name)
            return
    elif args.command == "tokenizer":
        if args.action == "train":
            if args.backend == "hf":
                tokenizer = FastBPETokenizer.train(read_documents(args.data), args.vocab_size)
            else:
                tokenizer = ByteBPETokenizer.train(list(read_documents(args.data)), args.vocab_size)
            tokenizer.save(args.output)
            result = {"output": args.output, "vocab_size": tokenizer.vocab_size, "backend": args.backend}
        else:
            tokenizer = load_tokenizer(args.tokenizer)
            result = {"encoded": tokenizer.encode(args.text), "decoded": tokenizer.decode(tokenizer.encode(args.text))}
    elif args.command == "data":
        if args.action == 'prepare':
            result = prepare_data(args.source, args.output, args.license,
                                  dataset_card_path=args.dataset_card)
        elif args.action == 'shard':
            result = prepare_shards(args.source, args.output, args.license,
                                    max_docs_per_shard=args.max_docs_per_shard,
                                    remove_near_duplicates=not args.keep_near_duplicates,
                                    dataset_card_path=args.dataset_card)
        elif args.action == 'verify-license':
            from .dataset_cards import verify_dataset_card
            result = verify_dataset_card(args.source, args.card, purpose=args.purpose)
        else:
            result = tokenize_shards(args.prepared, args.tokenizer, args.output,
                                     max_tokens_per_shard=args.max_tokens_per_shard)
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
                              resume=args.action == "resume", allow_large=args.allow_large, log_path=args.log_path,
                              schedule_steps=args.schedule_steps, checkpoint_every=args.checkpoint_every,
                              validate_every=args.validate_every)
    elif args.command == "evaluate":
        from .evaluation import evaluate_language_model
        result = evaluate_language_model(args.checkpoint, args.tokenizer, args.data,
                                         block_size=args.block_size, batches=args.batches,
                                         split=args.split, allow_training_data=args.allow_training_data)
    elif args.command == "benchmark":
        from .benchmarks import evaluate_choices
        result = evaluate_choices(args.checkpoint, args.tokenizer, args.data,
                                  license_id=args.license, format=args.format,
                                  training_docs=args.train_docs,
                                  allow_contaminated=args.allow_contaminated,
                                  device_name=args.device, max_examples=args.max_examples,
                                  scoring=args.scoring, dataset_card_path=args.dataset_card)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x', encoding='utf-8') as report:
                json.dump(result, report, ensure_ascii=False, indent=2)
    elif args.command == "mmlu":
        from .mmlu_reference import create_inventory_template, evaluate_mmlu
        if args.action == "inventory":
            result = create_inventory_template(args.data_dir, args.output)
        elif args.action == "verify":
            from .mmlu_reference import verify_inventory
            result = verify_inventory(args.data_dir, args.inventory)
        else:
            subjects = None if args.subjects is None else args.subjects.split(',')
            result = evaluate_mmlu(
                args.checkpoint, args.tokenizer, args.data_dir, args.inventory,
                subjects=subjects, requested_shots=args.shots, max_examples=args.max_examples,
                scoring=args.scoring, device_name=args.device, training_docs=args.train_docs,
                allow_contaminated=args.allow_contaminated)
            if args.output:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open('x', encoding='utf-8') as handle:
                    json.dump(result, handle, ensure_ascii=False, indent=2)
    elif args.command == "export":
        from .export import export_transformers
        result = export_transformers(args.checkpoint, args.tokenizer, args.output,
                                     allow_large=args.allow_large)
    elif args.command == "serve":
        if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
            parser.error("Non-loopback binding requires --api-key. Use a TLS terminator for external access.")
        from .api import create_app
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install the sagallm[serve] optional dependency") from exc
        app = create_app(args.checkpoint, args.tokenizer, api_key=args.api_key,
                         requests_per_minute=args.requests_per_minute,
                         max_concurrent=args.max_concurrent,
                         max_generation_seconds=args.generation_seconds)
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
        return
    else:
        tokenizer = load_tokenizer(args.tokenizer)
        device = resolve_device(args.device)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = ModelConfig(**state["config"])
        verify_checkpoint_tokenizer(state, args.tokenizer)
        if tokenizer.vocab_size > config.vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds checkpoint model vocabulary")
        model = SagaTransformer(config)
        checkpoint_load(args.checkpoint, model)
        model.to(device).eval()
        ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device)
        new_ids = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 eos_token_id=ByteBPETokenizer.SPECIAL["<eos>"],
                                 vocab_limit=tokenizer.vocab_size)
        result = {"text": tokenizer.decode(new_ids[0].tolist()), "generated_tokens": int(new_ids.size(1)-ids.size(1))}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.command == "java" and result.get("status") in ("FAIL", "REJECTED"):
        raise SystemExit(1)
    if args.command == "hardware" and result.get("runtime_test", {}).get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()