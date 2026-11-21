"""Local, deterministic Java starter projects for conversational prototyping.

This module is deliberately *not* an LLM code-generation backend. It handles a
small set of explicit intents; unknown requests fail rather than fabricate code.
Compilation never runs Java. Execution is separate, explicit and not sandboxed.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "sagallm-java-project-v1"
TEMPLATES: dict[str, dict[str, str]] = {
    "hello": {"label": "Print a greeting / 文字を表示", "keywords": "hello world,こんにちは,表示,出力,print,挨拶,あいさつ"},
    "calculator": {"label": "Console calculator / 四則演算", "keywords": "calculator,計算機,電卓,四則演算,足し算,割り算"},
    "fizzbuzz": {"label": "FizzBuzz loop / 条件分岐", "keywords": "fizzbuzz,fizz buzz,3と5,3の倍数"},
    "guess": {"label": "Number guessing game / 数当てゲーム", "keywords": "数当て,数字当て,guess,当てゲーム"},
    "todo": {"label": "In-memory todo list / タスク管理", "keywords": "todo,to-do,やること,タスク,TODOリスト"},
    "prime": {"label": "Prime checker / 素数判定", "keywords": "prime,素数"},
    "factorial": {"label": "Factorial calculator / 階乗", "keywords": "factorial,階乗"},
    "file": {"label": "Read a UTF-8 text file / テキスト読込", "keywords": "ファイルを読む,ファイル読み込み,read file,read text,テキストファイル"},
    "swing": {"label": "Small Swing window / GUIボタン", "keywords": "swing,gui,ウィンドウ,画面,ボタン"},
}


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", value) or len(value) > 64:
        raise ValueError("Java class name must be a simple ASCII Java identifier (max 64 characters)")
    if value in {"abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile", "while", "true", "false", "null", "_", "record", "sealed", "permits", "var", "yield"}:
        raise ValueError("Class name is a Java reserved keyword")
    return value


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError("Prompt must have 1..2000 characters")
    if any(ord(ch) < 32 and ch not in "\n\t" for ch in value):
        raise ValueError("Prompt contains unsupported control characters")
    return value.strip()


def choose_template(prompt: str) -> str:
    """Select a *supported* task; ambiguous/unrecognized requests are refused."""
    prompt = _text(prompt).lower()
    hits = [name for name, info in TEMPLATES.items()
            if any(token.lower() in prompt for token in info["keywords"].split(","))]
    # "表示/print" often describes the output of a *different* specific task.
    if "hello" in hits and len(hits) > 1:
        hits.remove("hello")
    if len(hits) > 1:
        raise ValueError(f"Ambiguous request matches {', '.join(hits)}. Supply --template explicitly.")
    if not hits:
        raise ValueError("Unsupported task. Use `sagallm java templates` to see supported simple Java projects.")
    return hits[0]


def _java_string(message: str) -> str:
    if len(message) > 200 or any(ord(c) < 32 and c not in "\n\t" for c in message):
        raise ValueError("Greeting must be printable and no longer than 200 characters")
    # JSON string quoting is compatible with Java string literals for ordinary UTF-8.
    # Reject sequences that can trigger Java's pre-lexical Unicode escape handling.
    if re.search(r"\\+u[0-9a-fA-F]{4}", message):
        raise ValueError("Unicode escape text cannot be embedded safely as a Java literal")
    return json.dumps(message, ensure_ascii=False)


def _hello_message(prompt: str) -> str:
    m = re.search(r"[「『\"]([^」』\"]{1,200})[」』\"]", prompt)
    return m.group(1) if m else "こんにちは、Java！"


def render(template: str, class_name: str, options: dict | None = None) -> str:
    """Emit bounded, self-contained Java standard-library programs."""
    if template not in TEMPLATES:
        raise ValueError("Unknown template")
    name = _identifier(class_name)
    opts = options or {}
    if template == "hello":
        body = f'''public class {name} {{
    public static void main(String[] args) {{
        System.out.println({_java_string(opts.get("message", "こんにちは、Java！"))});
    }}
}}'''
    elif template == "fizzbuzz":
        limit = opts.get("limit", 100)
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("FizzBuzz limit must be in 1..10000")
        body = f'''public class {name} {{
    public static void main(String[] args) {{
        for (int i = 1; i <= {limit}; i++) {{
            if (i % 15 == 0) System.out.println("FizzBuzz");
            else if (i % 3 == 0) System.out.println("Fizz");
            else if (i % 5 == 0) System.out.println("Buzz");
            else System.out.println(i);
        }}
    }}
}}'''
    elif template == "calculator":
        body = f'''import java.util.Locale;
import java.util.Scanner;

public class {name} {{
    public static void main(String[] args) {{
        try (Scanner scanner = new Scanner(System.in)) {{
            scanner.useLocale(Locale.US);
            System.out.println("Enter two numbers:");
            if (!scanner.hasNextDouble()) {{ System.out.println("Invalid first number"); return; }}
            double first = scanner.nextDouble();
            if (!scanner.hasNextDouble()) {{ System.out.println("Invalid second number"); return; }}
            double second = scanner.nextDouble();
            System.out.println("Choose + - * /:");
            if (!scanner.hasNext()) return;
            String operation = scanner.next();
            switch (operation) {{
                case "+": System.out.println(first + second); break;
                case "-": System.out.println(first - second); break;
                case "*": System.out.println(first * second); break;
                case "/":
                    if (second == 0) System.out.println("Cannot divide by zero");
                    else System.out.println(first / second);
                    break;
                default: System.out.println("Unknown operator");
            }}
        }}
    }}
}}'''
    elif template == "prime":
        body = f'''import java.util.Scanner;

public class {name} {{
    public static boolean isPrime(int number) {{
        if (number < 2) return false;
        for (int divisor = 2; (long) divisor * divisor <= number; divisor++) {{
            if (number % divisor == 0) return false;
        }}
        return true;
    }}

    public static void main(String[] args) {{
        try (Scanner scanner = new Scanner(System.in)) {{
            System.out.println("Enter an integer:");
            if (!scanner.hasNextInt()) {{ System.out.println("Invalid integer"); return; }}
            System.out.println(isPrime(scanner.nextInt()) ? "Prime" : "Not prime");
        }}
    }}
}}'''
    elif template == "factorial":
        body = f'''import java.math.BigInteger;
import java.util.Scanner;

public class {name} {{
    public static void main(String[] args) {{
        try (Scanner scanner = new Scanner(System.in)) {{
            System.out.println("Enter an integer 0..2000:");
            if (!scanner.hasNextInt()) {{ System.out.println("Invalid integer"); return; }}
            int n = scanner.nextInt();
            if (n < 0 || n > 2000) {{ System.out.println("Outside supported range"); return; }}
            BigInteger result = BigInteger.ONE;
            for (int i = 2; i <= n; i++) result = result.multiply(BigInteger.valueOf(i));
            System.out.println(result);
        }}
    }}
}}'''
    elif template == "guess":
        body = f'''import java.util.Random;
import java.util.Scanner;

public class {name} {{
    public static void main(String[] args) {{
        int target = new Random().nextInt(100) + 1;
        try (Scanner scanner = new Scanner(System.in)) {{
            System.out.println("Guess a number from 1 to 100 (0 to quit):");
            for (int tries = 1; tries <= 20 && scanner.hasNextLine(); tries++) {{
                String line = scanner.nextLine().trim();
                int guess;
                try {{ guess = Integer.parseInt(line); }}
                catch (NumberFormatException error) {{ System.out.println("Enter a number"); continue; }}
                if (guess == 0) break;
                if (guess == target) {{ System.out.println("Correct in " + tries + " tries!"); return; }}
                System.out.println(guess < target ? "Higher" : "Lower");
            }}
            System.out.println("Game over! The number was " + target);
        }}
    }}
}}'''
    elif template == "todo":
        body = f'''import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class {name} {{
    public static void main(String[] args) {{
        List<String> tasks = new ArrayList<>();
        try (Scanner scanner = new Scanner(System.in)) {{
            System.out.println("Commands: add TASK, list, remove NUMBER, exit");
            while (scanner.hasNextLine()) {{
                String line = scanner.nextLine().trim();
                if (line.equals("exit")) break;
                if (line.startsWith("add ") && line.length() > 4) {{
                    tasks.add(line.substring(4));
                    System.out.println("Added");
                }} else if (line.equals("list")) {{
                    for (int i = 0; i < tasks.size(); i++)
                        System.out.println((i + 1) + ": " + tasks.get(i));
                }} else if (line.startsWith("remove ")) {{
                    try {{
                        int index = Integer.parseInt(line.substring(7).trim()) - 1;
                        if (index < 0 || index >= tasks.size()) System.out.println("No such task");
                        else {{ tasks.remove(index); System.out.println("Removed"); }}
                    }} catch (NumberFormatException error) {{ System.out.println("Invalid number"); }}
                }} else System.out.println("Unknown command");
            }}
        }}
    }}
}}'''
    elif template == "file":
        body = f'''import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;

public class {name} {{
    public static void main(String[] args) {{
        if (args.length != 1) {{ System.out.println("Usage: java {name} <file.txt>"); return; }}
        try {{
            String text = Files.readString(Path.of(args[0]), StandardCharsets.UTF_8);
            System.out.println(text);
        }} catch (IOException error) {{
            System.err.println("Cannot read file: " + error.getMessage());
        }}
    }}
}}'''
    else:  # swing
        body = f'''import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

public class {name} {{
    public static void main(String[] args) {{
        SwingUtilities.invokeLater(() -> {{
            JFrame frame = new JFrame("SagaLLM Java Demo");
            JLabel label = new JLabel("Button not clicked");
            JButton button = new JButton("Click me");
            button.addActionListener(event -> label.setText("Hello from Java!"));
            JPanel panel = new JPanel();
            panel.add(button);
            panel.add(label);
            frame.setContentPane(panel);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(360, 140);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        }});
    }}
}}'''
    if opts.get("annotated"):
        label = TEMPLATES[template]["label"].split(" / ")[0]
        # Comment remains outside any executable declaration.
        first = f"// {label}. Generated from a local, deterministic SagaLLM template.\n"
        return first + body + "\n"
    return body + "\n"


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _metadata(project: Path) -> tuple[dict, Path, str]:
    file = project / "sagallm-java.json"
    if not file.is_file() or file.is_symlink():
        raise ValueError("Not a SagaLLM Java project")
    meta = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
        raise ValueError("Invalid Java project metadata")
    name = _identifier(meta.get("class_name"))
    if meta.get("template") not in TEMPLATES:
        raise ValueError("Invalid Java project template")
    source = project / "src" / (name + ".java")
    if not source.is_file() or source.is_symlink():
        raise ValueError("Project's Java source file is missing or unsafe")
    text = source.read_text(encoding="utf-8")
    if _digest(text) != meta.get("source_sha256"):
        raise ValueError("Source was changed outside SagaLLM. Review it before proceeding; no automatic overwrite or execution")
    return meta, source, text


def _save_meta(project: Path, meta: dict) -> None:
    # Metadata contains structured settings only, never raw prompts.
    (project / "sagallm-java.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_project(project_path: str | Path, prompt: str, *, template: str | None = None,
                   class_name: str = "Main") -> dict:
    prompt = _text(prompt)
    name = _identifier(class_name)
    template = template or choose_template(prompt)
    if template not in TEMPLATES:
        raise ValueError("Unknown template")
    options: dict[str, str | int | bool] = {}
    if template == "hello":
        options["message"] = _hello_message(prompt)
    if template == "fizzbuzz":
        match = re.search(r"(\d{1,5})\s*まで|(?:to|up to)\s*(\d{1,5})", prompt, re.IGNORECASE)
        if match:
            options["limit"] = int(match.group(1) or match.group(2))
    code = render(template, name, options)
    root = Path(project_path).expanduser().resolve()
    if root.exists():
        raise FileExistsError("Project directory already exists; choose a new empty destination")
    root.mkdir(parents=True)
    try:
        source = root / "src" / f"{name}.java"
        source.parent.mkdir()
        source.write_text(code, encoding="utf-8")
        _save_meta(root, {"schema": SCHEMA, "class_name": name, "template": template,
                          "options": options, "source_sha256": _digest(code),
                          "revisions": 0, "created_at": datetime.now(timezone.utc).isoformat()})
    except Exception:
        shutil.rmtree(root)
        raise
    return {"status": "CREATED", "template": template, "project": str(root),
            "java_file": str(source), "vibe_mode": "deterministic_starters_only",
            "note": "No LLM inference was used; only supported simple tasks are generated"}


def propose_revision(project_path: str | Path, prompt: str, *, apply: bool = False) -> dict:
    prompt = _text(prompt)
    project = Path(project_path).expanduser().resolve()
    meta, source, original = _metadata(project)
    opts = dict(meta.get("options", {}))
    name = meta["class_name"]
    cls = (re.search(r"(?:クラス名を|クラス名は|rename\s+(?:the\s+)?class(?:\s+to)?|class\s+name\s+to)\s*([A-Za-z_$][A-Za-z0-9_$]*)", prompt, re.IGNORECASE))
    if cls:
        name = _identifier(cls.group(1))
    elif "コメント" in prompt and any(k in prompt for k in ("追加", "付け", "入れ")) or re.search(r"add\s+comments?", prompt, re.IGNORECASE):
        opts["annotated"] = True
    elif meta["template"] == "hello":
        msg = re.search(r"(?:表示文|メッセージ|文字列)\s*を?\s*[「『\"]([^」』\"]+)[」』\"]", prompt)
        if not msg:
            msg = re.search(r"(?:message|greeting)\s+to\s+[\"]([^\"]+)[\"]", prompt, re.IGNORECASE)
        if not msg:
            raise ValueError("Unsupported revision. Examples: クラス名をDemoに変更 / コメントを追加 / メッセージを「ようこそ」に変更")
        opts["message"] = msg.group(1)
    elif meta["template"] == "fizzbuzz":
        limit = re.search(r"(?:上限を|範囲を)?\s*(\d{1,5})\s*まで|(?:limit to|up to)\s*(\d{1,5})", prompt, re.IGNORECASE)
        if not limit:
            raise ValueError("Unsupported FizzBuzz revision. Supported: 50までに変更 / クラス名をDemoに変更 / コメントを追加")
        opts["limit"] = int(limit.group(1) or limit.group(2))
    else:
        raise ValueError("Unsupported revision. Try クラス名をDemoに変更 or コメントを追加")
    result = render(meta["template"], name, opts)
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), result.splitlines(keepends=True),
                                      fromfile=str(source.name), tofile=f"{name}.java"))
    if not diff:
        return {"status": "UNCHANGED", "diff": ""}
    if not apply:
        return {"status": "PREVIEW", "diff": diff, "note": "Re-run with --apply to write the reviewed change"}
    destination = project / "src" / f"{name}.java"
    if destination != source and destination.exists():
        raise FileExistsError("Destination Java source already exists")
    # All changes are regenerated from the recorded template, never from an arbitrary model response.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=source.parent,
                                     prefix=".sagallm-new-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(result)
    try:
        temporary.replace(destination)
        if destination != source:
            source.unlink()
        meta.update({"class_name": name, "options": opts, "source_sha256": _digest(result),
                     "revisions": int(meta["revisions"]) + 1})
        _save_meta(project, meta)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "APPLIED", "diff": diff, "java_file": str(destination), "revisions": meta["revisions"]}


def _compile(source: Path, classes: Path, timeout: float) -> dict:
    javac = shutil.which("javac")
    if javac is None:
        return {"status": "NOT_RUN", "reason": "javac is not installed; install a JDK 17 or newer"}
    cmd = [javac, "-J-Xmx256m", "-proc:none", "-encoding", "UTF-8", "--release", "17",
           "-classpath", str(classes), "-d", str(classes), str(source)]
    try:
        outcome = subprocess.run(cmd, cwd=classes, capture_output=True, text=True, errors="replace",
                                 timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "reason": "javac timeout"}
    return {"status": "PASS" if outcome.returncode == 0 else "FAIL",
            "exit_code": outcome.returncode, "diagnostics": outcome.stderr[:8192]}


def check_project(project_path: str | Path, *, timeout: float = 20) -> dict:
    if not 1 <= timeout <= 60:
        raise ValueError("Compile timeout must be between 1 and 60 seconds")
    project = Path(project_path).expanduser().resolve()
    meta, source, _ = _metadata(project)
    with tempfile.TemporaryDirectory(prefix="sagallm-javac-") as directory:
        result = _compile(source, Path(directory), timeout)
    return {**result, "class_name": meta["class_name"], "compile_only": True,
            "warning": "javac -proc:none disables annotation processors; compilation is not an OS sandbox"}


def run_project(project_path: str | Path, *, trust_local_code: bool = False,
                stdin: str = "", timeout: float = 6, java_args: list[str] | None = None) -> dict:
    if not trust_local_code:
        raise PermissionError("Java execution requires --trust-local-code; generated code is NOT sandboxed")
    if not 1 <= timeout <= 15 or len(stdin) > 8192:
        raise ValueError("Runtime timeout must be 1..15 seconds and stdin <=8192 chars")
    project = Path(project_path).expanduser().resolve()
    meta, source, _ = _metadata(project)
    java = shutil.which("java")
    if java is None:
        return {"status": "NOT_RUN", "reason": "java is not installed"}
    with tempfile.TemporaryDirectory(prefix="sagallm-java-run-") as directory:
        classes = Path(directory)
        compile_info = _compile(source, classes, timeout=20)
        if compile_info["status"] != "PASS":
            return {"status": compile_info["status"], "compile": compile_info}
        try:
            proc = subprocess.run([java, "-Xmx128m", "-XX:ActiveProcessorCount=2", "-cp", str(classes),
                                   meta["class_name"], *(java_args or [])],
                                  input=stdin, capture_output=True, text=True, errors="replace",
                                  cwd=project, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "reason": "Java runtime timeout"}
    return {"status": "PASS" if proc.returncode == 0 else "FAIL", "exit_code": proc.returncode,
            "stdout": proc.stdout[:8192], "stderr": proc.stderr[:8192],
            "output_truncated": len(proc.stdout) > 8192 or len(proc.stderr) > 8192,
            "warning": "Java execution can access files/network. This is NOT a sandbox."}


def chat(project_path: str | Path, *, class_name: str = "Main") -> None:
    """Explicitly interactive local chat loop; no background AI or network calls."""
    root = Path(project_path).expanduser().resolve()
    print("SagaLLM Java chat (simple supported templates, not an LLM). /help for commands")
    if not root.exists():
        prompt = input("What should the Java program do? > ")
        print(json.dumps(create_project(root, prompt, class_name=class_name), ensure_ascii=False, indent=2))
    while True:
        try:
            prompt = input("Java > ").strip()
        except EOFError:
            break
        if not prompt or prompt in ("/exit", "/quit"):
            break
        if prompt == "/help":
            print("/show, /check, /run, /exit; revisions: クラス名をDemoに変更 / コメントを追加 / メッセージを「ようこそ」に変更 / 50までに変更 (FizzBuzz)")
        elif prompt == "/show":
            _, source, code = _metadata(root)
            print(f"--- {source.name} ---\n{code}")
        elif prompt == "/check":
            print(json.dumps(check_project(root), ensure_ascii=False, indent=2))
        elif prompt == "/run":
            answer = input("Run Java with local file/network access? Type YES to proceed: ").strip()
            if answer == "YES":
                print(json.dumps(run_project(root, trust_local_code=True), ensure_ascii=False, indent=2))
            else:
                print("Execution cancelled")
        else:
            try:
                preview = propose_revision(root, prompt)
                print(preview.get("diff", "No changes"))
                if preview["status"] == "PREVIEW" and input("Apply revision? [y/N] ").strip().lower() == "y":
                    print(json.dumps(propose_revision(root, prompt, apply=True), ensure_ascii=False, indent=2))
            except (OSError, ValueError) as exc:
                print(f"Cannot perform request: {exc}")