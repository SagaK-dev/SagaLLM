# Reliable Java stdin/stdout templates (EXPLICIT, NON-LLM)

**Purpose.** An actual attempted v0.18 Java stdin fine-tuning experiment
(PR #42) reduced synthetic validation response loss from **2.2634 to 0.3607**
but improved **0/18** validated Java output cases to **0/18**.
The small learned model still does not reliably implement task semantics.

To provide immediately useful Java output for a **narrow supported set**,
the explicit `sagallm java stdio` subcommand now generates **deterministic,
typed, project-authored Java**, not a new model checkpoint or an
unconstrained natural-language-to-code system.

## Example: sum two runtime input integers

```bash
sagallm java stdio --operation sum-two --offset 3 --output ./sum-program
javac -proc:none --release 17 ./sum-program/Main.java
printf '7 -3\n' | java -cp ./sum-program Main
```

Expected output:

```text
7
```

Every generated program reads actual stdin using `Scanner(System.in)`,
performs its specified calculation and writes **one exact newline-terminated
stdout answer**, with **no prompts, banners or debug messages**.

## Explicit supported operations

| Operation | stdin format | stdout |
| --- | --- | --- |
| `sum-two` | 2 signed integers | sum + `--offset` |
| `product-two` | 2 signed integers | product + offset |
| `min-two` | 2 signed integers | minimum + offset |
| `max-two` | 2 signed integers | maximum + offset |
| `absolute-gap` | 2 signed integers | absolute difference + offset |
| `count-greater` | n, then n integers | number greater than `--threshold` |
| `sum-even` | n, then n integers | even sum + offset |
| `count-odd` | n, then n integers | odd count + offset |
| `digit-sum` | one nonnegative integer | decimal digit sum + offset |
| `upper-line` | one ASCII line | uppercased text plus `--suffix` |
| `lower-line` | one ASCII line | lowercased text plus suffix |
| `reverse-line` | one ASCII line | reversed ASCII text plus suffix |
| `clamp` | one integer | restricted to `--lower-bound`/`--upper-bound` |

The `n`-prefixed operations accept 0–1000 values. Inputs and intermediate
numeric results must fit the Java signed integer types used by each
operation; Unicode grapheme reversals and unrestricted text encodings
are **not supported**. Values outside the stated input domain may throw
an error. Unsupported operations and contradictory options fail with
`ValueError`; no attempt is made to guess the user's meaning.

## Other examples

```bash
sagallm java stdio --operation count-greater --threshold 2 --output ./count
sagallm java stdio --operation lower-line --suffix _ok --output ./lower
sagallm java stdio --operation clamp --lower-bound -2 --upper-bound 2 --output ./clamp
```

The output directory must not already exist. Each command creates a
`Main.java` source file and JSON manifest with the deterministic
operation, input/output contract and SHA-256. The generator **does not**
compile or execute source; the user controls any subsequent Java execution.
Do not run Java source from untrusted origins directly on the host.

## Actual evidence and limits

CI compiles the 13 project-authored deterministic source programs using
`javac --release 17 -proc:none` and executes three known, independently
specified input/output cases per operation (**39 I/O comparisons**).
These tests are for **trusted deterministic project source only**, not
model-generated Java or a benchmark of SagaLLM's learned weights.

This is a narrow reliability feature, not evidence that v0.18 or the
v0.19 training candidate has become a generally correct Java model.
The trained candidate remains **unpromoted**. The original public
Exercism independent-review gate remains unchanged.
