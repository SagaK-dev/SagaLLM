"""Offline Java starter workflow, including compile/run when a local JDK exists."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from sagallm.java_vibe import (
    SCHEMA, TEMPLATES, choose_template, check_project, create_project,
    propose_revision, render, run_project,
)


@pytest.mark.parametrize("prompt,template", [
    ("Hello worldを表示して", "hello"),
    ("電卓プログラムを作って", "calculator"),
    ("FizzBuzzを15まで表示", "fizzbuzz"),
    ("数当てゲーム", "guess"),
    ("TODOリストを作って", "todo"),
    ("素数判定", "prime"),
    ("階乗", "factorial"),
    ("テキストファイル読み込み", "file"),
    ("Swingのウィンドウ", "swing"),
])
def test_intent_classification(prompt, template):
    assert choose_template(prompt) == template


def test_unrecognized_and_ambiguous_requests_are_not_fabricated():
    with pytest.raises(ValueError, match="Unsupported"):
        choose_template("JavaでシューティングゲームのAIを作って")
    with pytest.raises(ValueError, match="Ambiguous"):
        choose_template("素数か階乗を計算して")


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_java_templates_compile_when_jdk_present(template, tmp_path):
    project = tmp_path / template
    create_project(project, "Java demo", template=template)
    if shutil.which("javac") is None:
        pytest.skip("JDK is not available")
    check = check_project(project)
    assert check["status"] == "PASS", check
    assert check["compile_only"] is True


def test_new_project_no_overwrite_and_no_prompt_history(tmp_path):
    project = tmp_path / "project"
    created = create_project(project, '表示する文字列は「こんにちは世界」')
    assert created["template"] == "hello"
    meta = json.loads((project / "sagallm-java.json").read_text())
    assert meta["schema"] == SCHEMA
    assert "prompt" not in meta
    assert meta["source_sha256"] == hashlib.sha256((project / "src/Main.java").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        create_project(project, "Hello world")


def test_preview_and_reviewed_revisions(tmp_path):
    project = tmp_path / "java"
    create_project(project, "Hello world")
    source = project / "src/Main.java"
    original = source.read_text()
    preview = propose_revision(project, "メッセージを「ようこそ」に変更")
    assert preview["status"] == "PREVIEW" and "+" in preview["diff"]
    assert source.read_text() == original
    applied = propose_revision(project, "メッセージを「ようこそ」に変更", apply=True)
    assert applied["status"] == "APPLIED"
    assert "ようこそ" in source.read_text()
    propose_revision(project, "コメントを追加", apply=True)
    renamed = propose_revision(project, "クラス名をExampleに変更", apply=True)
    assert (project / "src/Example.java").exists()
    assert not source.exists()
    assert "public class Example" in (project / "src/Example.java").read_text()
    assert renamed["revisions"] == 3


def test_fizzbuzz_revisions_and_limits(tmp_path):
    project = tmp_path / "java"
    create_project(project, "FizzBuzzを25まで表示")
    assert "<= 25" in (project / "src/Main.java").read_text()
    propose_revision(project, "50までに変更", apply=True)
    assert "<= 50" in (project / "src/Main.java").read_text()
    with pytest.raises(ValueError, match="limit"):
        propose_revision(project, "10001までに変更", apply=True)
    with pytest.raises(ValueError, match="Unsupported"):
        propose_revision(project, "別のWebアプリに変えて", apply=True)


def test_detect_user_edit_and_invalid_class_name(tmp_path):
    project = tmp_path / "java"
    with pytest.raises(ValueError, match="identifier"):
        create_project(project, "Hello world", class_name="../../Bad")
    create_project(project, "Hello world")
    source = project / "src/Main.java"
    source.write_text(source.read_text() + "\n// changed externally\n")
    with pytest.raises(ValueError, match="outside SagaLLM"):
        propose_revision(project, "コメントを追加", apply=True)
    with pytest.raises(ValueError, match="outside SagaLLM"):
        run_project(project, trust_local_code=True)


def test_no_implicit_execution(tmp_path):
    project = tmp_path / "java"
    create_project(project, "Hello world")
    with pytest.raises(PermissionError, match="trust-local-code"):
        run_project(project)
    assert "class Main" in (project / "src/Main.java").read_text()


def test_java_escape_and_risky_source_rejected(tmp_path):
    p = tmp_path / "escape"
    create_project(p, '「\"; System.exit(0); //」を表示', template="hello")
    assert 'System.exit(0)' in (p / "src/Main.java").read_text()
    if shutil.which("javac"):
        assert check_project(p)["status"] == "PASS"
    with pytest.raises(ValueError, match="Unicode escape"):
        render("hello", "Main", {"message": r"text\\u000aevil"})


@pytest.mark.skipif(shutil.which("java") is None or shutil.which("javac") is None, reason="JDK not installed")
@pytest.mark.parametrize("template,stdin_text,expected", [
    ("hello", "", "こんにちは"),
    ("fizzbuzz", "", "FizzBuzz"),
    ("calculator", "5\n7\n+\n", "12.0"),
    ("prime", "13\n", "Prime"),
    ("factorial", "5\n", "120"),
    ("todo", "add 学習\nlist\nexit\n", "学習"),
])
def test_java_explicit_run(template, stdin_text, expected, tmp_path):
    project = tmp_path / template
    create_project(project, "starter", template=template)
    result = run_project(project, trust_local_code=True, stdin=stdin_text)
    assert result["status"] == "PASS", result
    assert expected in result["stdout"]


def test_missing_javac_is_not_reported_pass(tmp_path, monkeypatch):
    import sagallm.java_vibe as vibe
    project = tmp_path / "java"
    create_project(project, "Hello world")
    original = vibe.shutil.which
    monkeypatch.setattr(vibe.shutil, "which", lambda name: None if name == "javac" else original(name))
    assert check_project(project)["status"] == "NOT_RUN"


def test_invalid_rev_unmodified(tmp_path):
    project = tmp_path / "java"
    create_project(project, "数当てゲーム")
    source = project / "src/Main.java"
    content = source.read_text()
    with pytest.raises(ValueError, match="Unsupported"):
        propose_revision(project, "オンラインゲームにして", apply=True)
    assert source.read_text() == content


def test_java_cli_new_revise_check(tmp_path, capsys):
    from sagallm.cli import main
    project = tmp_path / "cli"
    main(["java", "new", "--output", str(project), "--prompt", "Hello world"])
    assert json.loads(capsys.readouterr().out)["status"] == "CREATED"
    main(["java", "revise", "--project", str(project), "--prompt", "クラス名をDemoに変更"])
    assert json.loads(capsys.readouterr().out)["status"] == "PREVIEW"
    assert (project / "src/Main.java").exists()
    main(["java", "revise", "--project", str(project), "--prompt", "クラス名をDemoに変更", "--apply"])
    assert json.loads(capsys.readouterr().out)["status"] == "APPLIED"
    main(["java", "check", "--project", str(project)])
    assert json.loads(capsys.readouterr().out)["status"] in {"PASS", "NOT_RUN"}


def test_java_chat_creates_and_revises_with_explicit_confirmation(tmp_path, monkeypatch, capsys):
    from sagallm.java_vibe import chat
    project = tmp_path / "chat"
    replies = iter(["Hello world", "メッセージを「Hello, Java!」に変更", "y", "/show", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    chat(project)
    assert '"Hello, Java!"' in (project / "src/Main.java").read_text()
    assert "PREVIEW" not in capsys.readouterr().out or "Java >" not in capsys.readouterr().out


def test_java_cli_run_requires_trust(tmp_path):
    from sagallm.cli import main
    project = tmp_path / "cli"
    create_project(project, "Hello world")
    with pytest.raises(PermissionError, match="trust-local-code"):
        main(["java", "run", "--project", str(project)])