"""Original, deterministic, offline Java 17 instruction/coding corpus.

Each example is a newly constructed small Java program; no third-party source
code or pretrained model is downloaded or copied. This is synthetic training
material, not evidence of Java programming competence or a legal attestation.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

from .java_ai import _model_user_message
from .java_coding_eval import BASE_SOURCE

SCHEMA = "sagallm-original-java-v1"
TRAIN_FAMILIES = (
    "hello", "add", "subtract", "multiply", "divide", "remainder", "even_odd",
    "count_up", "count_down", "sum_range", "square", "max_two", "min_two",
    "positive", "array_sum", "upper", "lower", "repeat", "table", "length",
)
VALIDATION_FAMILIES = ("factorial", "reverse", "gcd")
TEST_FAMILIES = ("fibonacci", "prime", "palindrome")
SPLITS = {"train": (TRAIN_FAMILIES, 16), "validation": (VALIDATION_FAMILIES, 5), "test": (TEST_FAMILIES, 5)}
MAX_FILE_BYTES = 8 * 1024 * 1024


def _canonical(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _java(body: str) -> str:
    return "public class Main {\n  public static void main(String[] args) {\n" + "\n".join(
        "    " + ln for ln in body.splitlines()) + "\n  }\n}\n"


def _program(family: str, idx: int) -> tuple[str, str, str]:
    """Return Japanese and English requests plus a compilable one-file answer."""
    rng = random.Random(1117 + 101 * idx + sum(ord(c) for c in family))
    a, b = rng.randint(2, 90), rng.randint(2, 40)
    n = rng.randint(3, 15)
    word = ("Saga", "Java", "学習", "Hello", "コード", "練習", "Sunny", "Blue")[idx % 8]
    quoted = json.dumps(word, ensure_ascii=False)
    if family == "hello":
        ja, en, body = f"{word}と表示して", f"Print the message {word} to the console", f"System.out.println({quoted});"
    elif family in ("add", "subtract", "multiply", "divide", "remainder"):
        operators = {"add": ("足し算", "sum", "+"), "subtract": ("引き算", "difference", "-"),
                     "multiply": ("掛け算", "product", "*"), "divide": ("整数除算", "integer quotient", "/"),
                     "remainder": ("余り", "remainder", "%")}
        jp, eng, op = operators[family]
        ja, en = f"整数{a}と{b}の{jp}を表示", f"Print the {eng} of integers {a} and {b}"
        body = f"int a = {a};\nint b = {b};\nSystem.out.println(a {op} b);"
    elif family == "even_odd":
        ja, en = f"整数{a}が偶数か奇数か表示", f"Display whether {a} is even or odd"
        body = f"int n = {a};\nSystem.out.println(n % 2 == 0 ? \"even\" : \"odd\");"
    elif family == "count_up":
        ja, en = f"1から{n}まで順番に表示", f"Print each integer from 1 through {n}"
        body = f"for (int i = 1; i <= {n}; i++) {{\n  System.out.println(i);\n}}"
    elif family == "count_down":
        ja, en = f"{n}から1まで逆順に表示", f"Count down from {n} to 1"
        body = f"for (int i = {n}; i >= 1; i--) {{\n  System.out.println(i);\n}}"
    elif family == "sum_range":
        ja, en = f"1から{n}までの合計を表示", f"Print the sum from 1 to {n}"
        body = f"int sum = 0;\nfor (int i = 1; i <= {n}; i++) {{\n  sum += i;\n}}\nSystem.out.println(sum);"
    elif family == "square":
        ja, en = f"整数{a}の二乗を表示", f"Print the square of {a}"
        body = f"int value = {a};\nSystem.out.println(value * value);"
    elif family in ("max_two", "min_two"):
        fn = "max" if family == "max_two" else "min"
        jp = "大きい" if fn == "max" else "小さい"
        ja, en = f"{a}と{b}の{jp}方を表示", f"Display the {fn}imum of {a} and {b}"
        body = f"int a = {a};\nint b = {b};\nSystem.out.println(Math.{fn}(a, b));"
    elif family == "positive":
        z = a - b - 30
        ja, en = f"整数{z}の正負とゼロを判定", f"Determine whether {z} is positive, negative, or zero"
        body = f"int n = {z};\nif (n > 0) System.out.println(\"positive\");\nelse if (n < 0) System.out.println(\"negative\");\nelse System.out.println(\"zero\");"
    elif family == "array_sum":
        arr = [rng.randint(1, 9) for _ in range(3 + idx % 3)]
        ja, en = f"配列{arr}の合計を表示", f"Print the sum of integer array {arr}"
        body = "int[] values = {" + ", ".join(map(str, arr)) + "};\nint sum = 0;\nfor (int v : values) sum += v;\nSystem.out.println(sum);"
    elif family in ("upper", "lower"):
        fn = "toUpperCase" if family == "upper" else "toLowerCase"
        jp = "大文字" if family == "upper" else "小文字"
        ja, en = f"文字列{word}を{jp}にして表示", f"Convert the string {word} to {family}case and print it"
        body = f"String text = {quoted};\nSystem.out.println(text.{fn}());"
    elif family == "repeat":
        ja, en = f"文字列{word}を{n}回表示", f"Print the word {word} exactly {n} times"
        body = f"for (int i = 0; i < {n}; i++) {{\n  System.out.println({quoted});\n}}"
    elif family == "table":
        ja, en = f"{n}の九九を1から9まで表示", f"Print the multiplication table for {n} from 1 to 9"
        body = f"for (int i = 1; i <= 9; i++) {{\n  System.out.println({n} * i);\n}}"
    elif family == "length":
        ja, en = f"文字列{word}の文字数を表示", f"Print the length of the string {word}"
        body = f"String s = {quoted};\nSystem.out.println(s.length());"
    elif family == "factorial":
        ja, en = f"{n}の階乗を計算して表示", f"Compute factorial of {n} in Java"
        body = f"long value = 1;\nfor (int i = 2; i <= {n}; i++) value *= i;\nSystem.out.println(value);"
    elif family == "reverse":
        ja, en = f"文字列{word}を逆順に表示", f"Reverse the characters in the string {word}"
        body = f"String text = {quoted};\nSystem.out.println(new StringBuilder(text).reverse());"
    elif family == "gcd":
        ja, en = f"{a}と{b}の最大公約数を表示", f"Print gcd of {a} and {b}"
        body = f"int a = {a};\nint b = {b};\nwhile (b != 0) {{\n  int t = a % b; a = b; b = t;\n}}\nSystem.out.println(a);"
    elif family == "fibonacci":
        ja, en = f"フィボナッチ数列を{n}個表示", f"Print the first {n} Fibonacci numbers"
        body = f"long a = 0;\nlong b = 1;\nfor (int i = 0; i < {n}; i++) {{\n  System.out.println(a);\n  long next = a + b; a = b; b = next;\n}}"
    elif family == "prime":
        ja, en = f"整数{a}が素数かどうか判定", f"Check if {a} is a prime number"
        body = f"int n = {a};\nboolean prime = n >= 2;\nfor (int d = 2; d * d <= n; d++) {{\n  if (n % d == 0) prime = false;\n}}\nSystem.out.println(prime);"
    elif family == "palindrome":
        ja, en = f"文字列{word}が回文か判定", f"Check whether the string {word} is a palindrome"
        body = f"String s = {quoted};\nSystem.out.println(s.equals(new StringBuilder(s).reverse().toString()));"
    else:
        raise ValueError("Unsupported original Java task family: " + family)
    return ja, en, _java(body)


def _expected() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for split, (families, count) in SPLITS.items():
        rows = []
        for family in families:
            for i in range(count):
                ja, en, code = _program(family, i)
                prompt = ja if i % 2 == 0 else en
                rows.append({"id": f"{family}-{i:02}", "family": family, "messages": [
                    {"role": "user", "content": _model_user_message(prompt, BASE_SOURCE, "Main")},
                    {"role": "assistant", "content": code},
                ]})
        # Deterministic, auditable, mixed-family order (not blocks of the same task).
        random.Random(20261122 + len(split)).shuffle(rows)
        result[split] = rows
    return result


def build_java_dataset(output: str | Path) -> dict:
    """Create a *new* directory with original Java SFT examples, no network use."""
    root = Path(output).expanduser().absolute()
    if root.exists():
        raise FileExistsError("Dataset destination already exists")
    root.mkdir(parents=True)
    try:
        content = _expected()
        files = {}
        for split, rows in content.items():
            data = "\n".join(_canonical(row) for row in rows) + "\n"
            raw = data.encode("utf-8")
            file = root / f"{split}.jsonl"
            file.write_bytes(raw)
            files[split] = {"file": file.name, "sha256": _sha(raw), "examples": len(rows),
                            "families": list(SPLITS[split][0])}
        manifest = {
            "schema": SCHEMA,
            "provenance": "newly authored deterministic Java 17 examples; no external code or corpus",
            "model_weights_used_to_create_examples": False,
            "dataset_license_review": "ORIGINAL_ASSISTANT_AUTHORED_NOT_LEGALLY_ATTESTED",
            "split_strategy": "disjoint_task_families",
            "generation_seed": 20261122,
            "files": files,
        }
        (root / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
        return {"status": "CREATED", "path": str(root), "counts": {k: len(v) for k, v in content.items()}}
    except BaseException:
        shutil.rmtree(root)
        raise


def verify_java_dataset(directory: str | Path) -> dict:
    """Integrity and split audit; hashes alone are not license verification."""
    root = Path(directory).expanduser().absolute()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 64000:
        raise ValueError("Missing or unsafe manifest")
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (meta.get("schema") != SCHEMA or meta.get("split_strategy") != "disjoint_task_families" or
        meta.get("dataset_license_review") != "ORIGINAL_ASSISTANT_AUTHORED_NOT_LEGALLY_ATTESTED" or
        meta.get("model_weights_used_to_create_examples") is not False or
        meta.get("generation_seed") != 20261122):
        raise ValueError("Unsupported dataset provenance and split claims")
    if set(meta.get("files", {})) != set(SPLITS):
        raise ValueError("Dataset inventory mismatch")
    expected = _expected()
    seen = set()
    for split, rows in expected.items():
        record = meta["files"][split]
        file = root / f"{split}.jsonl"
        if (record.get("file") != file.name or record.get("families") != list(SPLITS[split][0]) or
            file.is_symlink() or not file.is_file() or file.stat().st_size > MAX_FILE_BYTES):
            raise ValueError("Unexpected dataset file or family declaration")
        raw = file.read_bytes()
        if _sha(raw) != record.get("sha256") or record.get("examples") != len(rows):
            raise ValueError("Dataset hash or row count mismatch")
        actual = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if actual != rows:
            raise ValueError("Dataset content differs from the original generator")
        for row in actual:
            if row["id"] in seen:
                raise ValueError("Duplicate Java example ID across splits")
            seen.add(row["id"])
    return {"status": "PASS", "counts": {k: len(v) for k, v in expected.items()},
            "family_disjoint": True, "data_source": "new_original_synthetic_examples",
            "license_attested": False}