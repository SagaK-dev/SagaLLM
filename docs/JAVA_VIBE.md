# SagaLLM Java starter workflow (v0.9)

SagaLLM now supports **simple, local natural-language-to-Java tasks**. This is a
**deterministic starter generator**, **not** a general-purpose autonomous AI coder.
It does not use a hosted LLM API, pretrained model weights or internet access.
The SagaLLM tiny model cannot reliably generate arbitrary Java from descriptions.

## Requirements

- Python 3.10+ and the usual SagaLLM dependencies.
- A JDK **17 or newer** on your `PATH` for `javac` compilation (tested with JDK 21).
- Java runtime for opt-in execution. The included templates use the Java standard library.

## Natural-language chat (interactive)

```shell
sagallm java templates
sagallm java chat --project work/first-java
```

First, describe a supported task in Japanese or English, e.g. `FizzBuzzを30まで表示して`.
The program is saved to `work/first-java/src/Main.java`.
After that, you can ask for a supported revision and see a diff before confirming:

```text
What should the Java program do? > FizzBuzzを30まで表示して
Java > 50までに変更
...diff preview...
Apply revision? [y/N] y
Java > コメントを追加
Apply revision? [y/N] y
Java > /check
Java > /show
Java > /exit
```

`/run` asks for the exact confirmation `YES` before executing code. **Java
execution is not sandboxed and may access local files or the network.**

## Noninteractive commands

```shell
# Never overwrites an existing output directory.
sagallm java new --prompt '電卓プログラムを作って' --output work/calculator
sagallm java check --project work/calculator

# Preview only; does not change a file.
sagallm java revise --project work/calculator --prompt 'クラス名をCalcに変更'
# Apply after reviewing the proposed diff.
sagallm java revise --project work/calculator --prompt 'クラス名をCalcに変更' --apply

# Explicit opt-in for executing the compiled Java program.
sagallm java run --project work/calculator --trust-local-code --stdin $'5\n7\n+\n'
```

On Windows PowerShell, enter input via the escaped multiline string
`"5`n7`n+`n"` rather than bash's `$'...'` syntax.

## Supported tasks

| Template | Recognized examples | Supported modifications |
| --- | --- | --- |
| `hello` | Hello world / 挨拶 / `「こんにちは」を表示` | Printed message, class name, comments |
| `calculator` | 電卓 / calculator | Class name, comments |
| `fizzbuzz` | FizzBuzzを20まで表示 | Upper bound `50まで`, class name, comments |
| `guess` | 数当てゲーム / guess | Class name, comments |
| `todo` | TODOリスト / タスク管理 | Class name, comments |
| `prime` | 素数判定 / prime | Class name, comments |
| `factorial` | 階乗 / factorial | Class name, comments |
| `file` | ファイル読み込み / read file | Class name, comments |
| `swing` | Swingウィンドウ / GUIボタン | Class name, comments |

Use `--template NAME` with `java new` if a request is ambiguous.
The CLI refuses tasks or revisions it does not recognize; it **does not make
up Java code or pretend to understand an arbitrary request**.

### Safety boundaries

- `java new` refuses to overwrite an existing directory.
- `java revise` shows a **unified diff** before changes; `--apply` is required.
- Project metadata tracks the Java source SHA-256; manual source edits are
  detected and block template-based revision/run to avoid overwriting edits.
- Raw prompt text is not retained in the metadata (the generated string may be
  included in the source code itself).
- `java check` compiles with `javac -proc:none` and Java 17 source compatibility;
  it does **not** execute `main`, nor does it provide an OS sandbox.
- `java run` requires `--trust-local-code`. It uses a bounded timeout and Java
  heap cap, **not** a secure sandbox. Do not run untrusted projects.
- An unavailable JDK is reported as `NOT_RUN`, not `PASS`.
- There is no auto-debug/repair loop, remote compiler, filesystem agent, IDE
  integration, or trained Java model. Freeform Java coding would require a
  capable independently trained SagaLLM model (or an explicitly chosen other
  model), additional Java evaluation datasets and far more compute than the
  current 0.8/0.9 prototypes have used.

### Verification

```shell
python -m pytest -q tests/test_java_vibe.py
python scripts/check_cpu.py
```

The GitHub CPU workflow installs JDK 21 and compiles all nine templates.
It does not upload generated Java or run any unknown external project.

## v0.10 — Local checkpoint-driven Java revisions

The optional `java ai-revise` and `java ai-chat` commands load a *trusted local*
SagaLLM `.pt` checkpoint and matching tokenizer. They do not wrap external API
calls. The usual `java new/chat/revise` starter templates still work **until**
a file has been modified by the model. Thereafter, use the model-backed
editor; template regeneration is blocked so custom edits are not discarded.

```bash
sagallm java new --prompt 'Hello world' --output work/my-java

# Generate an uncommitted candidate, compile it and show a diff.
sagallm java ai-revise --project work/my-java --prompt 'Say Hello world' \
  --checkpoint checkpoints/java-sft.pt --tokenizer data/tokenizer.json \
  --max-new-tokens 512 --attempts 2

# Interactive review / approve: type APPLY to accept precisely the displayed
# candidate; the model is not invoked again on acceptance.
sagallm java ai-chat --project work/my-java \
  --checkpoint checkpoints/java-sft.pt --tokenizer data/tokenizer.json
```

`java ai-revise --apply` is an **explicit noninteractive opt-in** to install a
compiling proposal without the interactive diff approval step. Omit it when you
want to review the change manually first. Once applied, the exact model checkpoint
and tokenizer hashes are recorded in project metadata. Raw natural-language
prompts are not saved as a separate chat history.

### Simple local SFT smoke data

```bash
sagallm java toy-sft --output data/java-toy-sft.jsonl
sagallm train sft --checkpoint checkpoints/base.pt --tokenizer data/tokenizer.json \
  --data data/java-toy-sft.jsonl --output checkpoints/java-toy-sft.pt \
  --steps 1 --device cpu
```

The five examples use project-owned deterministic Java templates. Their prompts
are encoded with the same `User: ...\nAssistant: ` framing as `format_sft_example()`.
This alone **cannot** teach a small language model to write arbitrary Java.
A checkpoint with a sufficiently long context window is required; unsupported
models reject requests with `INSUFFICIENT_CONTEXT`. Training/permission review
is always separate and explicit.

### Invariants and limitations

- Entire Java file replacement only; **one** expected public class and no
  `package` statement, dependencies, multiple files or arbitrary IDE access.
- No model output is applied if extraction or compilation fails, or if the JDK
  is absent. Compiler diagnostics are fed back only on an explicit bounded retry.
- `javac -proc:none` reduces plugin execution but is **not** an OS sandbox. The
  Java program is not run by `ai-revise` or `ai-chat`. A manually requested
  `java run --trust-local-code` is a separate unsafe opt-in.
- Checkpoint files must be from trusted sources. The local adapter refuses
  checkpoints over 512 MiB and model configs over 200 million parameters.
  No 1B/3B inference or GPU validation has been done.
- The adapter requires an appropriate tokenization/context window and does not
  assure logical correctness, unit-test passing, useful Java capability or
  consistent self-repair. Syntax compilation is not semantic validation.
- The SFT dataset is an **original synthetic smoke fixture**, not an approved
  external Java corpus or proof of learned programming ability. Before importing
  external material, use the project's dataset-license review workflow.

## v0.11 — Family-separated Java data and compile-only evaluation

See [JAVA_CODING_DATA.md](JAVA_CODING_DATA.md) for deterministic fixture creation, data integrity checks, 2-step tiny CPU SFT smoke and non-executing held-out compiler evaluation. This is not a general-purpose Java model.