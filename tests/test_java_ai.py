"""Model-backed Java proposals must fail closed and never execute generated code."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from sagallm.java_ai import (
    LocalCheckpointGenerator, ai_chat, apply_revision, extract_java, generate_revision,
)
from sagallm.java_vibe import create_project

JDK = shutil.which("javac") is not None
GOOD = '''public class Main {
    public static void main(String[] args) {
        System.out.println("Hello from generated code");
    }
}
'''
BAD = '''public class Main {
    public static void main(String[] args) {
        System.out.println(
    }
}
'''


def fresh(tmp_path):
    project = tmp_path / "java"
    create_project(project, "Hello world")
    return project, project / "src/Main.java"


@pytest.mark.parametrize("value", [
    "I wrote some code, use it", "public class Other {}", "public class Main {}\npublic class Also {}",
    "package danger;\npublic class Main {}", "```python\npublic class Main {}\n``` and explanation",
    "", "x" * 40_001,
])
def test_untrusted_model_output_rejected(value):
    with pytest.raises(ValueError):
        extract_java(value, "Main")


def test_imports_fence_and_comments_allowed():
    text = '```java\nimport java.util.List;\n// a safe comment\npublic class Main {}\n```'
    assert extract_java(text, "Main").startswith("import java.util.List;")


def test_source_only_preview_has_no_writes(tmp_path):
    project, source = fresh(tmp_path)
    before = source.read_text()
    if not JDK:
        pytest.skip("JDK is required")
    queries = []
    def generate(prompt, max_new_tokens):
        queries.append(prompt)
        assert "Current source" in prompt and "new feature" in prompt
        return GOOD
    report = generate_revision(project, "new feature", generator=generate, max_new_tokens=300)
    assert report["status"] == "PREVIEW" and report["compile"]["status"] == "PASS"
    assert report["model_backend"] == "local_sagallm_checkpoint"
    assert "generated code" in report["diff"]
    assert "_candidate" not in report
    assert source.read_text() == before
    assert len(queries) == 1


@pytest.mark.skipif(not JDK, reason="JDK unavailable")
def test_apply_requires_explicit_switch_and_updates_digest(tmp_path):
    project, source = fresh(tmp_path)
    original = source.read_text()
    report = generate_revision(project, "print Hello", generator=lambda p, max_new_tokens: GOOD, apply=True)
    assert report["status"] == "APPLIED" and report["applied"] is True
    assert source.read_text() == GOOD and source.read_text() != original
    meta = json.loads((project / "sagallm-java.json").read_text())
    assert meta["revisions"] == 1
    assert meta["source_sha256"] == hashlib.sha256(GOOD.encode()).hexdigest()


@pytest.mark.skipif(not JDK, reason="JDK unavailable")
def test_compiler_guided_retry(tmp_path):
    project, source = fresh(tmp_path)
    calls = []
    def generate(prompt, max_new_tokens):
        calls.append(prompt)
        return BAD if len(calls) == 1 else GOOD
    report = generate_revision(project, "new feature", generator=generate, attempts=2)
    assert report["status"] == "PREVIEW" and report["attempts"] == 2
    assert "Prior javac diagnostics" in calls[1] and "Prior failed candidate" in calls[1]
    assert "Hello from generated code" not in source.read_text()


@pytest.mark.skipif(not JDK, reason="JDK unavailable")
def test_bad_compile_never_writes_even_with_apply(tmp_path):
    project, source = fresh(tmp_path)
    before = source.read_text()
    report = generate_revision(project, "change", generator=lambda p, max_new_tokens: BAD,
                               attempts=1, apply=True)
    assert report["status"] == "REJECTED"
    assert report["compile"]["status"] == "FAIL"
    assert source.read_text() == before


def test_missing_jdk_is_not_success(tmp_path, monkeypatch):
    import sagallm.java_vibe as vibe
    project, source = fresh(tmp_path)
    prev = source.read_text()
    original = vibe.shutil.which
    monkeypatch.setattr(vibe.shutil, "which", lambda arg: None if arg == "javac" else original(arg))
    report = generate_revision(project, "change", generator=lambda p, max_new_tokens: GOOD,
                               apply=True)
    assert report["status"] == "REJECTED" and report["compile"]["status"] == "NOT_RUN"
    assert source.read_text() == prev


def test_invalid_candidate_rejected_without_compiling(tmp_path):
    project, source = fresh(tmp_path)
    before = source.read_text()
    report = generate_revision(project, "change", generator=lambda p, max_new_tokens: "not java", apply=True)
    assert report["status"] == "REJECTED" and report["applied"] is False
    assert source.read_text() == before


@pytest.mark.skipif(not JDK, reason="JDK unavailable")
def test_stale_proposal_and_manual_edit_block_apply(tmp_path):
    project, source = fresh(tmp_path)
    preview = generate_revision(project, "change", generator=lambda p, max_new_tokens: GOOD,
                                return_candidate=True)
    assert preview["_candidate"] == GOOD
    source.write_text(source.read_text() + "// outside edit\n")
    with pytest.raises(ValueError):
        apply_revision(project, preview["_candidate"], preview["source_sha256_before"])


@pytest.mark.skipif(not JDK, reason="JDK unavailable")
def test_chat_preview_approval_applies_same_candidate(tmp_path, monkeypatch, capsys):
    project, source = fresh(tmp_path)
    replies = iter(["change the output", "APPLY", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    calls=[]
    def generate(prompt, max_new_tokens):
        calls.append(prompt);return GOOD
    ai_chat(project, generate)
    assert len(calls) == 1 and source.read_text() == GOOD
    assert "APPLIED" in capsys.readouterr().out


@pytest.mark.skipif(not JDK, reason="JDK unavailable")
def test_chat_rejection_never_writes(tmp_path, monkeypatch):
    project, source = fresh(tmp_path)
    before = source.read_text()
    replies = iter(["change the output", "no", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    ai_chat(project, lambda p, max_new_tokens: GOOD)
    assert source.read_text() == before


def test_retry_and_token_bounds(tmp_path):
    project,_=fresh(tmp_path)
    for opts in [{"attempts": 0}, {"attempts": 4}, {"max_new_tokens": 0}, {"max_new_tokens": 9999}]:
        with pytest.raises(ValueError):
            generate_revision(project, "hello", generator=lambda p, max_new_tokens: GOOD, **opts)


def test_cli_ai_requires_checkpoint(tmp_path):
    from sagallm.cli import main
    project,_=fresh(tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        main(["java", "ai-revise", "--project", str(project), "--prompt", "change output",
              "--checkpoint", str(tmp_path/"missing.pt"), "--tokenizer", str(tmp_path/"missing.json")])


def test_real_checkpoint_adapter_rejects_context_budget(tmp_path):
    """Load actual SagaTransformer checkpoint; do not claim a language capability."""
    import torch
    from sagallm.config import ModelConfig
    from sagallm.model import SagaTransformer
    from sagallm.tokenization import ByteBPETokenizer
    from sagallm.training import checkpoint_save

    tokenizer=ByteBPETokenizer()
    tok_path=tmp_path/"tokenizer.json"
    tokenizer.save(tok_path)
    cfg=ModelConfig(vocab_size=320,hidden_size=32,num_layers=1,num_heads=4,num_kv_heads=2,
                    intermediate_size=64,max_position_embeddings=64)
    model=SagaTransformer(cfg)
    opt=torch.optim.AdamW(model.parameters())
    path=tmp_path/"model.pt"
    checkpoint_save(path,model,opt,1,cfg,{"tokenizer_sha256":hashlib.sha256(tok_path.read_bytes()).hexdigest()})
    backend=LocalCheckpointGenerator(path,tok_path)
    assert backend.checkpoint_sha256==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="INSUFFICIENT_CONTEXT"):
        backend("long prompt requiring a large context",max_new_tokens=60)
    answer=backend("a",max_new_tokens=2)
    assert isinstance(answer,str)


def test_checkpoint_tokenizer_mismatch_is_blocked(tmp_path):
    import torch
    from sagallm.config import ModelConfig
    from sagallm.model import SagaTransformer
    from sagallm.tokenization import ByteBPETokenizer
    from sagallm.training import checkpoint_save
    tokenizer=ByteBPETokenizer();path=tmp_path/"tokenizer.json";tokenizer.save(path)
    cfg=ModelConfig(vocab_size=320,hidden_size=32,num_layers=1,num_heads=4,num_kv_heads=2,
                    intermediate_size=64,max_position_embeddings=64)
    model=SagaTransformer(cfg)
    ck=tmp_path/"ck.pt"
    checkpoint_save(ck,model,torch.optim.AdamW(model.parameters()),0,cfg,{"tokenizer_sha256":"0"*64})
    with pytest.raises(ValueError,match="SHA-256"):
        LocalCheckpointGenerator(ck,path)


def test_original_java_sft_examples_and_no_overwrite(tmp_path):
    from sagallm.java_ai import build_toy_java_sft, _model_request
    target = tmp_path / "toy.jsonl"
    result = build_toy_java_sft(target)
    assert result["examples"] == 5
    data = [json.loads(x) for x in target.read_text().splitlines()]
    assert len(data) == 5
    for record in data:
        user, assistant = record["messages"]
        assert user["role"] == "user" and assistant["role"] == "assistant"
        assert extract_java(assistant["content"], "Main") == assistant["content"]
        assert _model_request("task", "code", "Main").startswith("User: ")
        assert _model_request("task", "code", "Main").endswith("\nAssistant: ")
    assert "Change the FizzBuzz bound" in data[3]["messages"][0]["content"]
    with pytest.raises(FileExistsError):
        build_toy_java_sft(target)


def test_java_toy_sft_cli(tmp_path, capsys):
    from sagallm.cli import main
    target = tmp_path / "toy_sft.jsonl"
    main(["java", "toy-sft", "--output", str(target)])
    assert json.loads(capsys.readouterr().out)["examples"] == 5
    assert len(target.read_text().splitlines()) == 5


def test_real_sft_pipeline_uses_original_java_pairs(tmp_path):
    """One tiny CPU optimization is plumbing-only, not proof of Java competence."""
    import torch
    from sagallm.java_ai import build_toy_java_sft
    from sagallm.config import ModelConfig
    from sagallm.model import SagaTransformer
    from sagallm.tokenization import ByteBPETokenizer
    from sagallm.post_training import sft_train
    from sagallm.training import checkpoint_save

    toy = tmp_path / "toy.jsonl"
    build_toy_java_sft(toy)
    tokenizer = ByteBPETokenizer()
    tok_path = tmp_path / "t.json"
    tokenizer.save(tok_path)
    config = ModelConfig(vocab_size=320, hidden_size=32, num_layers=1, num_heads=4,
                         num_kv_heads=2, intermediate_size=64, max_position_embeddings=2048)
    torch.manual_seed(1)
    model = SagaTransformer(config)
    cp = tmp_path / "base.pt"
    checkpoint_save(cp, model, torch.optim.AdamW(model.parameters()), 0, config,
                    {"tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest()})
    fine = tmp_path / "sft.pt"
    result = sft_train(cp, tok_path, toy, fine, steps=1, device_name="cpu")
    assert result["steps"] == 1 and result["stage"] == "sft"
    assert fine.is_file()
    backend = LocalCheckpointGenerator(fine, tok_path, device="cpu")
    assert backend.max_context == 2048


@pytest.mark.skipif(not JDK, reason="JDK unavailable")
def test_model_edits_protect_against_template_regeneration(tmp_path):
    from sagallm.java_vibe import propose_revision
    project, source = fresh(tmp_path)
    report=generate_revision(project, "change output", generator=lambda p, max_new_tokens: GOOD,
                             apply=True, checkpoint_sha256="a"*64, tokenizer_sha256="b"*64)
    assert report["status"] == "APPLIED"
    metadata=json.loads((project/"sagallm-java.json").read_text())
    assert metadata["generated_by"] == "local_sagallm_checkpoint"
    assert metadata["last_ai_checkpoint_sha256"] == "a"*64
    assert metadata["last_ai_tokenizer_sha256"] == "b"*64
    with pytest.raises(ValueError, match="template regeneration"):
        propose_revision(project, "メッセージを「もう一度」に変更", apply=True)
    assert source.read_text() == GOOD
    again=generate_revision(project,"change again",generator=lambda p,max_new_tokens: GOOD.replace("Hello from generated code","A second version"))
    assert again["status"] == "PREVIEW"