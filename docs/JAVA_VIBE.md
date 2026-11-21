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