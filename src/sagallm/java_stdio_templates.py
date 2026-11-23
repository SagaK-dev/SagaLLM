"""Explicitly supported, deterministic Java stdin -> algorithm -> stdout generator.

This is deliberately **NOT an LLM** and NOT proof that SagaLLM's learned
checkpoint can solve Java tasks. It provides a reliable, narrow alternative
while model-generated programs are still failing behavioral tests.

All accepted operations are fixed/typed, all arguments validated, and no
generated program is executed by this module. No prompt text is interpreted
as a command, no external source is injected, and no shell is invoked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil

from .java_vibe import _identifier

OPERATIONS = (
    "sum-two", "product-two", "min-two", "max-two", "absolute-gap",
    "count-greater", "sum-even", "count-odd", "digit-sum",
    "upper-line", "lower-line", "reverse-line", "clamp",
)
NUMERIC_OFFSET = {
    "sum-two", "product-two", "min-two", "max-two", "absolute-gap",
    "sum-even", "count-odd", "digit-sum",
}
LINE_OPERATIONS = {"upper-line", "lower-line", "reverse-line"}
MAX_OFFSET = 10_000
MAX_ELEMENTS = 1000


def render_stdio(operation: str, *, class_name: str = "Main",
                 offset: int = 0, threshold: int = 0,
                 lower_bound: int = -10, upper_bound: int = 10,
                 suffix: str = "") -> str:
    """Create one valid Java 17 class for an EXACT explicit supported operation."""
    if operation not in OPERATIONS:
        raise ValueError("Unsupported explicit stdio operation")
    name = _identifier(class_name)
    for key, value in (("offset", offset), ("threshold", threshold),
                       ("lower_bound", lower_bound), ("upper_bound", upper_bound)):
        if type(value) is not int or not -MAX_OFFSET <= value <= MAX_OFFSET:
            raise ValueError(f"{key} must be an integer in [-10000,10000]")
    if (lower_bound > upper_bound or
            type(suffix) is not str or
            not re.fullmatch(r"[A-Za-z0-9_]{0,24}", suffix)):
        raise ValueError("Invalid clamp bounds or unsafe ASCII suffix")
    if operation not in NUMERIC_OFFSET and offset != 0:
        raise ValueError("Unused offset must be zero")
    if operation != "count-greater" and threshold != 0:
        raise ValueError("Unused threshold must be zero")
    if operation != "clamp" and (lower_bound != -10 or upper_bound != 10):
        raise ValueError("Unused clamp bounds must remain at defaults")
    if operation not in LINE_OPERATIONS and suffix:
        raise ValueError("Suffix is only supported for line operations")

    read = 'java.util.Scanner input = new java.util.Scanner(System.in, "UTF-8");'
    pair = "int a=input.nextInt();int b=input.nextInt();"
    if operation == "sum-two":
        body = pair + f"System.out.println(a+b+({offset}));"
    elif operation == "product-two":
        body = pair + f"System.out.println(a*b+({offset}));"
    elif operation == "min-two":
        body = pair + f"System.out.println(Math.min(a,b)+({offset}));"
    elif operation == "max-two":
        body = pair + f"System.out.println(Math.max(a,b)+({offset}));"
    elif operation == "absolute-gap":
        body = pair + f"System.out.println(Math.abs(a-b)+({offset}));"
    elif operation == "count-greater":
        body = (f"int n=input.nextInt();if(n<0||n>{MAX_ELEMENTS})throw new IllegalArgumentException();"
                f"int total=0;for(int i=0;i<n;i++){{int v=input.nextInt();"
                f"if(v>{threshold})total++;}}System.out.println(total);")
    elif operation in ("sum-even", "count-odd"):
        accumulate = "total+=v;" if operation == "sum-even" else "total++;"
        condition = "v%2==0" if operation == "sum-even" else "v%2!=0"
        body = (f"int n=input.nextInt();if(n<0||n>{MAX_ELEMENTS})throw new IllegalArgumentException();"
                f"int total=0;for(int i=0;i<n;i++){{int v=input.nextInt();"
                f"if({condition}){{{accumulate}}}}}"
                f"System.out.println(total+({offset}));")
    elif operation == "digit-sum":
        body = (f"long n=input.nextLong();if(n<0)throw new IllegalArgumentException();"
                f"long total=0;do{{total+=n%10;n/=10;}}while(n>0);"
                f"System.out.println(total+({offset}));")
    elif operation == "clamp":
        body = ("int n=input.nextInt();System.out.println("
                f"Math.max({lower_bound},Math.min({upper_bound},n)));")
    else:
        if operation == "upper-line":
            expr = "line.toUpperCase(java.util.Locale.ROOT)"
        elif operation == "lower-line":
            expr = "line.toLowerCase(java.util.Locale.ROOT)"
        else:
            expr = "new StringBuilder(line).reverse().toString()"
        literal = json.dumps(suffix)
        body = f"String line=input.nextLine();System.out.println({expr}+{literal});"
    program = (
        f"public class {name} {{\n"
        "  public static void main(String[] args) {\n"
        f"    {read}\n"
        f"    {body}\n"
        "  }\n"
        "}\n"
    )
    if len(program.encode("utf8")) > 2000:
        raise ValueError("Unexpectedly large deterministic Java program")
    return program


def create_stdio_program(output: str | Path, operation: str, **options) -> dict:
    """Write a NEW directory with source/manifest; never overwrite files."""
    destination = Path(output).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Stdio output directory already exists")
    options = dict(options)
    class_name = options.get("class_name", "Main")
    source = render_stdio(operation, **options)
    destination.mkdir(parents=True)
    try:
        file = destination / (class_name + ".java")
        with file.open("x", encoding="utf8") as handle:
            handle.write(source)
        sha = hashlib.sha256(source.encode("utf8")).hexdigest()
        info = {
            "status": "CREATED",
            "mode": "DETERMINISTIC_STDIO_TEMPLATE_NOT_LLM",
            "operation": operation,
            "class_name": class_name,
            "source_file": file.name,
            "source_sha256": sha,
            "stdin": "REQUIRED_AT_RUNTIME",
            "stdout": "EXACT_NEWLINE_TERMINATED_NO_EXTRA_PROMPTS",
            "java_executed": False,
            "model_used": False,
            "options": options,
        }
        with (destination / "manifest.json").open("x", encoding="utf8") as handle:
            json.dump(info, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {**info, "directory": str(destination)}
