"""Java 17 stdin -> algorithm -> exact stdout training/validation curriculum.

All exercises and input/output oracles are project-authored synthetic examples.
Only TRAIN families may enter optimization; task families in VALIDATION are
disjoint and may only be used for checkpoint selection and exploratory tests.
No public Exercism task, previous heldout TEST family, or model-generated
Java answer is included. Old datasets/checkpoints stay immutable.

The user-visible task deliberately specifies *runtime stdin*, never hardcoded
sample values. Runtime inputs are varied independently for oracle checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .java_ai import _model_user_message
from .java_coding_eval import BASE_SOURCE
from .java_large_corpus import _java, TRAIN_FAMILIES as OLD_TRAIN, VALIDATION_FAMILIES as OLD_VAL, TEST_FAMILIES as OLD_TEST
from .java_diverse_corpus import NEW_TRAIN, NEW_VALIDATION, NEW_TEST

SCHEMA = "sagallm-java-stdio-semantics-v1"
TRAIN_FAMILIES = (
    "stdin_add_offset", "stdin_multiply_offset", "stdin_max_offset",
    "stdin_absolute_gap_offset", "stdin_count_greater",
    "stdin_even_sum_offset", "stdin_digit_sum_offset",
    "stdin_ascii_upper_suffix", "stdin_ascii_reverse_suffix", "stdin_clamp",
)
VALIDATION_FAMILIES = (
    "stdin_min_offset", "stdin_count_odd_offset", "stdin_ascii_lower_suffix",
)
SPLITS = {"train": (TRAIN_FAMILIES, 32), "validation": (VALIDATION_FAMILIES, 12)}
MAX_FILE_BYTES = 2_000_000


def assert_disjoint_families() -> None:
    historical = set(OLD_TRAIN) | set(OLD_VAL) | set(OLD_TEST) | set(NEW_TRAIN) | set(NEW_VALIDATION) | set(NEW_TEST)
    if (set(TRAIN_FAMILIES) & set(VALIDATION_FAMILIES)
            or len(TRAIN_FAMILIES) != len(set(TRAIN_FAMILIES))
            or len(VALIDATION_FAMILIES) != len(set(VALIDATION_FAMILIES))
            or (set(TRAIN_FAMILIES) | set(VALIDATION_FAMILIES)) & historical):
        raise ValueError("Stdio family contamination / duplicate detected")
    if any(x in family for family in TRAIN_FAMILIES + VALIDATION_FAMILIES
           for x in ("luhn", "atbash", "hamming", "rna", "roman", "allergies")):
        raise ValueError("Public benchmark family may not enter curriculum")


def _task(family: str, index: int) -> tuple[str, str]:
    if (family not in TRAIN_FAMILIES + VALIDATION_FAMILIES or
            type(index) is not int or not 0 <= index < 32):
        raise ValueError("Invalid synthetic stdin problem")
    offset = index + 1
    threshold = index - 14
    lo = -12 + index % 9
    hi = lo + 5 + index % 7
    suffix = "_" + str(offset)
    read = "java.util.Scanner in=new java.util.Scanner(System.in);\n"
    if family == "stdin_add_offset":
        task = f"Read two signed integers from stdin; print their sum plus {offset} and a newline."
        body = read + f"int a=in.nextInt(),b=in.nextInt();System.out.println(a+b+{offset});"
    elif family == "stdin_multiply_offset":
        task = f"Read two signed integers from stdin; print their product plus {offset} and a newline."
        body = read + f"int a=in.nextInt(),b=in.nextInt();System.out.println(a*b+{offset});"
    elif family == "stdin_max_offset":
        task = f"Read two signed integers from stdin; print the larger number plus {offset} and a newline."
        body = read + f"int a=in.nextInt(),b=in.nextInt();System.out.println(Math.max(a,b)+{offset});"
    elif family == "stdin_min_offset":
        task = f"Read two signed integers from stdin; print the smaller number plus {offset} and a newline."
        body = read + f"int a=in.nextInt(),b=in.nextInt();System.out.println(Math.min(a,b)+{offset});"
    elif family == "stdin_absolute_gap_offset":
        task = f"Read two signed integers from stdin; print their absolute difference plus {offset} and a newline."
        body = read + f"int a=in.nextInt(),b=in.nextInt();System.out.println(Math.abs(a-b)+{offset});"
    elif family == "stdin_count_greater":
        task = f"Read n then n signed integers from stdin; print how many are strictly greater than {threshold}, then newline."
        body = read + f"int n=in.nextInt(),total=0;for(int k=0;k<n;k++){{int v=in.nextInt();if(v>{threshold})total++;}}System.out.println(total);"
    elif family == "stdin_even_sum_offset":
        task = f"Read n then n signed integers from stdin; print their even-number sum plus {offset} and a newline."
        body = read + f"int n=in.nextInt(),total=0;for(int k=0;k<n;k++){{int v=in.nextInt();if(v%2==0)total+=v;}}System.out.println(total+{offset});"
    elif family == "stdin_count_odd_offset":
        task = f"Read n then n signed integers from stdin; print their odd-number count plus {offset} and a newline."
        body = read + f"int n=in.nextInt(),total=0;for(int k=0;k<n;k++){{int v=in.nextInt();if(v%2!=0)total++;}}System.out.println(total+{offset});"
    elif family == "stdin_digit_sum_offset":
        task = f"Read one nonnegative integer from stdin; print its decimal digit sum plus {offset} and a newline."
        body = read + f"long n=in.nextLong();long total=0;do{{total+=n%10;n/=10;}}while(n>0);System.out.println(total+{offset});"
    elif family == "stdin_clamp":
        task = f"Read one signed integer from stdin; clamp it to inclusive [{lo},{hi}] and print the result with newline."
        body = read + f"int n=in.nextInt();System.out.println(Math.max({lo},Math.min({hi},n)));"
    elif family in ("stdin_ascii_upper_suffix", "stdin_ascii_lower_suffix", "stdin_ascii_reverse_suffix"):
        if family == "stdin_ascii_upper_suffix":
            task = f"Read one ASCII text line from stdin; uppercase it, append literal {suffix}, then print a newline."
            expr = "line.toUpperCase(java.util.Locale.ROOT)"
        elif family == "stdin_ascii_lower_suffix":
            task = f"Read one ASCII text line from stdin; lowercase it, append literal {suffix}, then print a newline."
            expr = "line.toLowerCase(java.util.Locale.ROOT)"
        else:
            task = f"Read one ASCII text line from stdin; reverse its characters, append literal {suffix}, then print a newline."
            expr = "new StringBuilder(line).reverse().toString()"
        body = read + f'String line=in.nextLine();System.out.println({expr}+"{suffix}");'
    else:
        raise ValueError("Unknown synthetic family")
    if len(task) > 240:
        raise ValueError("Prompt exceeds compact SFT limit")
    source = _java(body)
    if len(source) > 1800 or "public class Main" not in source:
        raise ValueError("Bad Java Main.java reference")
    return task, source


def oracle(family: str, index: int, stdin_data: str) -> str:
    """Separate pure-Python expected stdout for bounded synthetic inputs."""
    if family not in TRAIN_FAMILIES + VALIDATION_FAMILIES or not 0 <= index < 32:
        raise ValueError("Invalid oracle family/index")
    if not isinstance(stdin_data, str) or len(stdin_data) > 120 or "\x00" in stdin_data:
        raise ValueError("Invalid bounded synthetic stdin")
    offset = index + 1
    if family.startswith("stdin_ascii_"):
        if not stdin_data.endswith("\n") or stdin_data.count("\n") != 1:
            raise ValueError("Expected exactly one input line")
        line = stdin_data[:-1]
        if not line.isascii():
            raise ValueError("ASCII text only")
        if family == "stdin_ascii_upper_suffix":
            transformed = line.upper()
        elif family == "stdin_ascii_lower_suffix":
            transformed = line.lower()
        else:
            transformed = line[::-1]
        return transformed + "_" + str(offset) + "\n"
    values = [int(value) for value in stdin_data.split()]
    if family in ("stdin_add_offset", "stdin_multiply_offset",
                  "stdin_max_offset", "stdin_min_offset", "stdin_absolute_gap_offset"):
        if len(values) != 2:
            raise ValueError("Expected exactly two signed integers")
        a, b = values
        if family == "stdin_add_offset":
            n = a + b
        elif family == "stdin_multiply_offset":
            n = a * b
        elif family == "stdin_max_offset":
            n = max(a, b)
        elif family == "stdin_min_offset":
            n = min(a, b)
        else:
            n = abs(a - b)
        return str(n + offset) + "\n"
    if family in ("stdin_count_greater", "stdin_even_sum_offset", "stdin_count_odd_offset"):
        if not values or not 0 <= values[0] <= 5 or values[0] != len(values) - 1:
            raise ValueError("Input n must match bounded item count")
        numbers = values[1:]
        if family == "stdin_count_greater":
            value = sum(v > index - 14 for v in numbers)
        elif family == "stdin_even_sum_offset":
            value = sum(v for v in numbers if v % 2 == 0) + offset
        else:
            value = sum(v % 2 != 0 for v in numbers) + offset
        return str(value) + "\n"
    if len(values) != 1:
        raise ValueError("Expected one integer")
    value = values[0]
    if family == "stdin_digit_sum_offset":
        if value < 0:
            raise ValueError("Only nonnegative input supported")
        return str(sum(int(ch) for ch in str(value)) + offset) + "\n"
    if family == "stdin_clamp":
        lo = -12 + index % 9
        hi = lo + 5 + index % 7
        return str(max(lo, min(hi, value))) + "\n"
    raise ValueError("No oracle for input family")


def case_inputs(family: str) -> tuple[str, ...]:
    if family.startswith("stdin_ascii_"):
        return ("MiXeD Case\n", "  Spaces  \n", "\n")
    if family in ("stdin_add_offset", "stdin_multiply_offset",
                  "stdin_max_offset", "stdin_min_offset", "stdin_absolute_gap_offset"):
        return ("7 -3\n", "-5 2\n", "0 0\n")
    if family in ("stdin_count_greater", "stdin_even_sum_offset", "stdin_count_odd_offset"):
        return ("0\n", "5\n3 -2 0 7 4\n", "4\n-3 -1 2 2\n")
    if family == "stdin_digit_sum_offset":
        return ("0\n", "1005\n", "789\n")
    if family == "stdin_clamp":
        return ("-30\n", "0\n", "30\n")
    raise ValueError("Unsupported family oracle inputs")


def expected_rows(split: str) -> list[dict]:
    assert_disjoint_families()
    if split not in SPLITS:
        raise ValueError("No test split exists in stdio curriculum")
    families, variants = SPLITS[split]
    result = []
    for family in families:
        for index in range(variants):
            task, source = _task(family, index)
            result.append({
                "id": f"{split}-{family}-{index:03d}",
                "family": family,
                "messages": [
                    {"role": "user", "content": _model_user_message(task, BASE_SOURCE, "Main")},
                    {"role": "assistant", "content": source},
                ],
            })
    return result


def validation_oracle_rows() -> list[dict]:
    records = []
    for family in VALIDATION_FAMILIES:
        for index in range(SPLITS["validation"][1]):
            records.append({
                "id": f"validation-{family}-{index:03d}",
                "family": family,
                "cases": [{"stdin": value, "stdout": oracle(family, index, value)}
                          for value in case_inputs(family)],
            })
    return records


def _canonical_lines(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                      for row in rows) + "\n").encode("utf8")


def _payloads() -> dict[str, bytes]:
    train = expected_rows("train")
    validation = expected_rows("validation")
    cases = validation_oracle_rows()
    return {
        "train.jsonl": _canonical_lines(train),
        "validation.jsonl": _canonical_lines(validation),
        "validation-oracles.jsonl": _canonical_lines(cases),
    }


def build_curriculum(destination: str | Path) -> dict:
    root = Path(destination)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Refusing to overwrite stdio curriculum")
    blobs = _payloads()
    if any(len(raw) > MAX_FILE_BYTES for raw in blobs.values()):
        raise ValueError("Stdio curriculum exceeds bounded file size")
    info = {
        "schema": SCHEMA,
        "origin": "PROJECT_AUTHORED_SYNTHETIC_ONLY",
        "train_families": list(TRAIN_FAMILIES),
        "validation_families": list(VALIDATION_FAMILIES),
        "test_split": "NOT_CREATED",
        "external_public_benchmark_used": False,
        "generated_java_used_as_training": False,
        "behavioral_reference_programs_trusted": True,
        "family_disjoint": True,
        "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(),
                         "bytes": len(raw), "rows": raw.count(b"\n")}
                  for name, raw in blobs.items()},
    }
    root.mkdir(parents=True)
    try:
        for name, raw in blobs.items():
            (root / name).write_bytes(raw)
        (root / "manifest.json").write_bytes(
            json.dumps(info, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8") + b"\n")
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return info


def verify_curriculum(root: str | Path) -> dict:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Missing or symlinked curriculum root")
    original = root / "manifest.json"
    if not original.is_file() or original.is_symlink() or original.stat().st_size > 8192:
        raise ValueError("Unsafe curriculum manifest")
    metadata = json.loads(original.read_text(encoding="utf8"))
    expected_blobs = _payloads()
    if set(p.name for p in root.iterdir()) != set(expected_blobs) | {"manifest.json"}:
        raise ValueError("Unknown/missing candidate corpus file")
    for name, raw in expected_blobs.items():
        file = root / name
        if (not file.is_file() or file.is_symlink()
                or file.stat().st_size > MAX_FILE_BYTES or file.read_bytes() != raw):
            raise ValueError("Stdio curriculum differs from canonical regeneration")
    rebuilt = {
        "schema": SCHEMA, "origin": "PROJECT_AUTHORED_SYNTHETIC_ONLY",
        "train_families": list(TRAIN_FAMILIES),
        "validation_families": list(VALIDATION_FAMILIES),
        "test_split": "NOT_CREATED", "external_public_benchmark_used": False,
        "generated_java_used_as_training": False,
        "behavioral_reference_programs_trusted": True, "family_disjoint": True,
        "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(),
                         "bytes": len(raw), "rows": raw.count(b"\n")}
                  for name, raw in expected_blobs.items()},
    }
    if metadata != rebuilt:
        raise ValueError("Stdio manifest does not match regenerated data")
    return {"status": "PASS", **rebuilt}
