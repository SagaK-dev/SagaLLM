"""Deterministic original Java 17 coding corpus for offline education.

All source programs here are project-authored, parameterized examples; **not**
independent human-authored programs or evidence of Java coding proficiency.
Whole task families remain segregated across train, validation and test.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

from .java_ai import _model_user_message
from .java_coding_eval import BASE_SOURCE

SCHEMA = "sagallm-original-java-expanded-v1"
TRAIN_FAMILIES = (
    "add", "subtract", "multiply", "divide", "modulo", "absolute", "square", "cube",
    "minimum", "maximum", "clamp", "temperature_c", "temperature_f", "power_two",
    "digit_sum", "digit_count", "count_up", "count_down", "sum_to", "times_table",
    "even_odd", "divisible", "sign", "leap_year", "rectangle_area", "average_three",
    "upper", "lower", "trim", "string_length", "starts_with", "ends_with",
    "contains", "replace", "substring", "repeat", "char_count", "vowel_count",
    "array_sum", "array_max", "array_min", "array_average", "array_even_count",
    "array_contains", "array_reverse", "array_sort", "array_square", "array_prefix",
    "list_add", "set_unique", "map_sum", "stack_reverse", "queue_sum", "stream_sum",
    "method_square", "parse_exception", "string_builder", "boolean_logic",
)
VALIDATION_FAMILIES = ("factorial", "gcd", "reverse_string", "list_sort", "set_intersection", "binary_search")
TEST_FAMILIES = ("fibonacci", "prime", "palindrome", "anagram", "balanced_brackets", "lcm")
SPLITS = {"train": (TRAIN_FAMILIES, 90), "validation": (VALIDATION_FAMILIES, 18), "test": (TEST_FAMILIES, 18)}
# 58*90=5220 train, plus 108 validation and 108 test -> 5,436 total.
MAX_JSONL_BYTES = 80_000_000


def _json(v: object) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _java(body: str) -> str:
    return "public class Main {\n  public static void main(String[] args) {\n" + "\n".join(
        "    " + line for line in body.splitlines()
    ) + "\n  }\n}\n"


def _case(family: str, idx: int) -> tuple[str, str, str]:
    """Return Japanese/English task requests and compilable Java source."""
    if family not in set(TRAIN_FAMILIES) | set(VALIDATION_FAMILIES) | set(TEST_FAMILIES):
        raise ValueError("Unknown task family")
    # Keep a unique first operand or string across the 90 variants of each
    # family, except where overflow could produce nonsense, e.g. factorial.
    a = 17 + idx * 7
    b = 3 + (idx * 11) % 97
    n = 2 + idx % 12
    s = f"SagaJava_{idx:04d}"
    t = f"Code_{idx:04d}"
    xs = [a, b, idx + 2, idx % 9 + 1]
    ary = ", ".join(map(str, xs))
    jp = en = body = ""

    if family in {"add", "subtract", "multiply", "divide", "modulo"}:
        op, j, e = {"add": ("+", "足し", "sum"), "subtract": ("-", "引き", "difference"),
                    "multiply": ("*", "掛け", "product"), "divide": ("/", "整数の割り", "integer quotient"),
                    "modulo": ("%", "余り", "remainder")}[family]
        jp, en = f"{a}と{b}を{j}算して表示", f"Print the {e} of {a} and {b}"
        body = f"int x = {a}, y = {b};\nSystem.out.println(x {op} y);"
    elif family in {"absolute", "square", "cube", "minimum", "maximum", "clamp", "power_two", "digit_sum", "digit_count"}:
        if family == "absolute":
            jp,en,body=f"-{a}の絶対値を表示",f"Print the absolute value of -{a}",f"System.out.println(Math.abs(-{a}));"
        elif family in ("square", "cube"):
            pow_ = "x * x" if family == "square" else "x * x * x"
            jp,en,body=f"{a}の{2 if family=='square' else 3}乗を表示",f"Print {a} raised to the power {2 if family=='square' else 3}",f"long x = {a};\nSystem.out.println({pow_});"
        elif family in ("minimum", "maximum"):
            fun = "min" if family == "minimum" else "max"
            jp,en,body=f"{a}と{b}の{'最小' if fun=='min' else '最大'}を表示",f"Print the {fun}imum of {a} and {b}",f"System.out.println(Math.{fun}({a}, {b}));"
        elif family == "clamp":
            jp,en,body=f"{a}を{b}から{b+20}までに制限",f"Clamp {a} to [{b}, {b+20}]",f"int x = {a};\nSystem.out.println(Math.max({b}, Math.min({b+20}, x)));"
        elif family == "power_two":
            jp,en,body=f"2の{n}乗を表示",f"Print 2 to the power of {n}",f"System.out.println(1 << {n});\n// variant {idx}"
        elif family == "digit_sum":
            jp,en,body=f"整数{a}の各桁の合計",f"Sum the decimal digits of {a}",f"int n = {a}, sum = 0;\nwhile (n > 0) {{ sum += n % 10; n /= 10; }}\nSystem.out.println(sum);"
        else:
            jp,en,body=f"整数{a}の桁数を表示",f"Count decimal digits in {a}",f"int n = {a}, digits = 0;\ndo {{ digits++; n /= 10; }} while (n != 0);\nSystem.out.println(digits);"
    elif family in {"temperature_c", "temperature_f", "rectangle_area", "average_three"}:
        if family == "temperature_c":
            jp,en,body=f"華氏{a}度を摂氏へ変換",f"Convert {a} Fahrenheit to Celsius",f"double f = {a};\nSystem.out.println((f - 32) * 5 / 9);"
        elif family == "temperature_f":
            jp,en,body=f"摂氏{a}度を華氏へ変換",f"Convert {a} Celsius to Fahrenheit",f"double c = {a};\nSystem.out.println(c * 9 / 5 + 32);"
        elif family == "rectangle_area":
            jp,en,body=f"縦{a}横{b}の長方形の面積を出す",f"Print the area of a {a} by {b} rectangle",f"int h = {a}, w = {b};\nSystem.out.println(h * w);"
        else:
            jp,en,body=f"{a}、{b}、{idx+2}の平均",f"Average {a}, {b}, and {idx+2}",f"System.out.println(({a} + {b} + {idx+2}) / 3.0);"
    elif family in {"count_up", "count_down", "sum_to", "times_table", "even_odd", "divisible", "sign", "leap_year"}:
        if family == "count_up":
            jp,en,body=f"0から{a}まで表示",f"Print the numbers from zero to {a}",f"for (int i=0;i<={a};i++) System.out.println(i);"
        elif family == "count_down":
            jp,en,body=f"{a}から0へカウントダウン",f"Count down from {a} to zero",f"for (int i={a};i>=0;i--) System.out.println(i);"
        elif family == "sum_to":
            jp,en,body=f"1から{a}まで合計",f"Sum integers from 1 to {a}",f"int s=0;\nfor(int i=1;i<={a};i++) s+=i;\nSystem.out.println(s);"
        elif family == "times_table":
            jp,en,body=f"{a}の掛け算表を1から9まで表示",f"Print the multiplication table of {a} to nine",f"for(int i=1;i<=9;i++) System.out.println({a} * i);"
        elif family == "even_odd":
            jp,en,body=f"{a}は偶数か奇数か",f"Check if {a} is even or odd",f"System.out.println({a} % 2 == 0 ? \"even\" : \"odd\");"
        elif family == "divisible":
            jp,en,body=f"{a}が{b}で割り切れるか",f"Check whether {a} is divisible by {b}",f"System.out.println({a} % {b} == 0);"
        elif family == "sign":
            x = a-100
            jp,en,body=f"{x}が正・負・ゼロか判定",f"Classify the sign of {x}",f"int n={x};\nSystem.out.println(n>0 ? \"positive\" : n<0 ? \"negative\" : \"zero\");"
        else:
            year = 1990 + idx
            jp,en,body=f"西暦{year}年がうるう年か",f"Is year {year} a leap year?",f"int y={year};\nSystem.out.println(y%400==0 || (y%4==0 && y%100!=0));"
    elif family in {"upper", "lower", "trim", "string_length", "starts_with", "ends_with",
                    "contains", "replace", "substring", "repeat", "char_count", "vowel_count"}:
        # ASCII-only strings keep Java String.length and Python length equivalent.
        if family == "upper":
            jp,en,body=f"文字列{s}を大文字に",f"Convert {s} to uppercase",f"System.out.println(\"{s}\".toUpperCase(java.util.Locale.ROOT));"
        elif family == "lower":
            jp,en,body=f"文字列{s}を小文字に",f"Convert {s} to lowercase",f"System.out.println(\"{s}\".toLowerCase(java.util.Locale.ROOT));"
        elif family == "trim":
            jp,en,body=f"{s}の両端の空白を除く",f"Trim spaces around {s}",f"System.out.println(\"  {s}  \".trim());"
        elif family == "string_length":
            jp,en,body=f"文字列{s}の長さを表示",f"Print the length of {s}",f"System.out.println(\"{s}\".length());"
        elif family in ("starts_with", "ends_with", "contains"):
            fun={"starts_with":"startsWith","ends_with":"endsWith","contains":"contains"}[family]
            jp,en,body=f"{s}にJavaが{'先頭' if family=='starts_with' else '末尾' if family=='ends_with' else '含まれる'}か",f"Check if {s} {family.replace('_',' ')} Java",f"System.out.println(\"{s}\".{fun}(\"Java\"));"
        elif family == "replace":
            jp,en,body=f"{s}のJavaをCodeに置換",f"Replace Java with Code in {s}",f"System.out.println(\"{s}\".replace(\"Java\", \"Code\"));"
        elif family == "substring":
            jp,en,body=f"{s}から先頭4文字を取得",f"Take the first four chars of {s}",f"System.out.println(\"{s}\".substring(0, 4));"
        elif family == "repeat":
            jp,en,body=f"{s}を{n}回連結",f"Repeat {s} {n} times",f"System.out.println(\"{s}\".repeat({n}));"
        elif family == "char_count":
            jp,en,body=f"{s}のaの出現数",f"Count a characters in {s}",f"String s=\"{s}\";\nlong c=s.chars().filter(x->x=='a').count();\nSystem.out.println(c);"
        else:
            jp,en,body=f"{s}にある母音の個数",f"Count vowels in {s}",f"String s=\"{s}\".toLowerCase();\nlong c=s.chars().filter(x->\"aeiou\".indexOf(x)>=0).count();\nSystem.out.println(c);"
    elif family in {"array_sum", "array_max", "array_min", "array_average", "array_even_count", "array_contains", "array_reverse", "array_sort", "array_square", "array_prefix"}:
        prefix=f"int[] a = {{{ary}}};\n"
        jp=en=""
        if family=="array_sum":
            jp,en,body=f"配列[{ary}]の合計",f"Sum integers [{ary}]",prefix+"int total=0; for(int x:a) total+=x;\nSystem.out.println(total);"
        elif family=="array_max":
            jp,en,body=f"配列[{ary}]の最大値",f"Find max of [{ary}]",prefix+"int m=a[0]; for(int x:a) m=Math.max(m,x);\nSystem.out.println(m);"
        elif family=="array_min":
            jp,en,body=f"配列[{ary}]の最小値",f"Find min of [{ary}]",prefix+"int m=a[0]; for(int x:a) m=Math.min(m,x);\nSystem.out.println(m);"
        elif family=="array_average":
            jp,en,body=f"配列[{ary}]の平均",f"Compute average of [{ary}]",prefix+"int total=0; for(int x:a) total+=x;\nSystem.out.println(total/(double)a.length);"
        elif family=="array_even_count":
            jp,en,body=f"配列[{ary}]の偶数を数える",f"Count even values in [{ary}]",prefix+"int c=0; for(int x:a) if(x%2==0)c++;\nSystem.out.println(c);"
        elif family=="array_contains":
            jp,en,body=f"配列[{ary}]に{b}があるか",f"Check whether [{ary}] contains {b}",prefix+f"boolean found=false; for(int x:a) if(x=={b}) found=true;\nSystem.out.println(found);"
        elif family=="array_reverse":
            jp,en,body=f"配列[{ary}]を逆順に表示",f"Print [{ary}] in reverse",prefix+"for(int i=a.length-1;i>=0;i--)System.out.println(a[i]);"
        elif family=="array_sort":
            jp,en,body=f"配列[{ary}]を昇順に整列",f"Sort [{ary}] ascending",prefix+"java.util.Arrays.sort(a);\nSystem.out.println(java.util.Arrays.toString(a));"
        elif family=="array_square":
            jp,en,body=f"配列[{ary}]の各要素を二乗",f"Square each element of [{ary}]",prefix+"for(int x:a)System.out.println(x*x);"
        else:
            jp,en,body=f"配列[{ary}]の累積和を表示",f"Print prefix sums of [{ary}]",prefix+"int sum=0;for(int x:a){sum+=x;System.out.println(sum);}"
    elif family in {"list_add", "set_unique", "map_sum", "stack_reverse", "queue_sum", "stream_sum", "method_square", "parse_exception", "string_builder", "boolean_logic"}:
        if family=="list_add":
            jp,en,body=f"{a}と{b}をArrayListに入れて合計",f"Add {a} and {b} to an ArrayList and sum",f"java.util.List<Integer> x=new java.util.ArrayList<>();\nx.add({a}); x.add({b});\nSystem.out.println(x.get(0)+x.get(1));"
        elif family=="set_unique":
            jp,en,body=f"{a}、{b}、{a}の重複を除く",f"Count distinct values {a}, {b}, {a}",f"java.util.Set<Integer> x=new java.util.HashSet<>();\nx.add({a});x.add({b});x.add({a});\nSystem.out.println(x.size());"
        elif family=="map_sum":
            jp,en,body=f"Mapに登録した{a}と{b}の合計",f"Sum {a} and {b} stored in a Map",f"java.util.Map<String,Integer> m=new java.util.HashMap<>();\nm.put(\"x\",{a});m.put(\"y\",{b});\nSystem.out.println(m.get(\"x\")+m.get(\"y\"));"
        elif family=="stack_reverse":
            jp,en,body=f"スタックへ{a}、{b}を積んで順に取り出す",f"Push {a}, {b} onto stack then pop",f"java.util.Deque<Integer> q=new java.util.ArrayDeque<>();\nq.push({a});q.push({b});\nSystem.out.println(q.pop());System.out.println(q.pop());"
        elif family=="queue_sum":
            jp,en,body=f"キューへ{a}と{b}を追加して合計",f"Queue {a} and {b} and sum",f"java.util.Queue<Integer> q=new java.util.ArrayDeque<>();\nq.add({a});q.add({b});\nSystem.out.println(q.remove()+q.remove());"
        elif family=="stream_sum":
            jp,en,body=f"Streamで配列[{ary}]の和を計算",f"Use an IntStream to sum [{ary}]",f"int[] a={{{ary}}};\nSystem.out.println(java.util.Arrays.stream(a).sum());"
        elif family=="method_square":
            jp,en,body=f"関数を使って{a}の二乗を表示",f"Use a method to square {a}",f"java.util.function.IntUnaryOperator fn=x->x*x;\nSystem.out.println(fn.applyAsInt({a}));"
        elif family=="parse_exception":
            jp,en,body=f"文字列{a}を整数に変換して表示",f"Parse the number {a} with exception handling",f"try {{ int n=Integer.parseInt(\"{a}\");System.out.println(n); }}\ncatch(NumberFormatException e) {{System.out.println(\"invalid\");}}"
        elif family=="string_builder":
            jp,en,body=f"StringBuilderで{s}と{t}を連結",f"Join {s} and {t} using StringBuilder",f"StringBuilder sb=new StringBuilder(\"{s}\");\nsb.append(\"{t}\");System.out.println(sb);"
        else:
            jp,en,body=f"{a}が正で{b}が100より小さいか",f"Check whether {a} is positive and {b} below 100",f"boolean ok=({a}>0)&&({b}<100);\nSystem.out.println(ok);"
    elif family=="factorial":
        jp,en,body=f"整数{n}の階乗を表示",f"Compute factorial of {n}",f"long p=1;for(int i=2;i<={n};i++)p*=i;\nSystem.out.println(p);\n// item {idx}"
    elif family=="gcd":
        jp,en,body=f"{a}と{b}の最大公約数",f"Compute the GCD of {a} and {b}",f"int x={a},y={b};while(y!=0){{int t=x%y;x=y;y=t;}}\nSystem.out.println(x);"
    elif family=="reverse_string":
        jp,en,body=f"{s}を逆順にする",f"Reverse {s}",f"String s=\"{s}\";System.out.println(new StringBuilder(s).reverse());"
    elif family=="list_sort":
        jp,en,body=f"List[{ary}]をソートする",f"Sort the list [{ary}]",f"java.util.List<Integer> a=new java.util.ArrayList<>(java.util.List.of({ary}));\njava.util.Collections.sort(a);System.out.println(a);"
    elif family=="set_intersection":
        jp,en,body=f"数集合[{a},{b}]と[{a},{idx+1}]の積集合",f"Intersect sets [{a},{b}] and [{a},{idx+1}]",f"java.util.Set<Integer> a=new java.util.HashSet<>(java.util.List.of({a},{b}));\na.retainAll(java.util.Set.of({a},{idx+1}));System.out.println(a.size());"
    elif family=="binary_search":
        jp,en,body=f"昇順配列[{ary}]から{a}を二分探索",f"Binary search for {a} in sorted [{ary}]",f"int[] a={{{ary}}};java.util.Arrays.sort(a);\nSystem.out.println(java.util.Arrays.binarySearch(a,{a})>=0);"
    elif family=="fibonacci":
        jp,en,body=f"最初の{n}個のフィボナッチ数",f"Print the first {n} Fibonacci numbers",f"long x=0,y=1;for(int i=0;i<{n};i++){{System.out.println(x);long z=x+y;x=y;y=z;}}\n// case {idx}"
    elif family=="prime":
        jp,en,body=f"{a}が素数かどうか",f"Check whether {a} is prime",f"int n={a};boolean ok=n>=2;\nfor(int d=2;d*d<=n;d++)if(n%d==0)ok=false;\nSystem.out.println(ok);"
    elif family=="palindrome":
        jp,en,body=f"{s}が回文か検査",f"Check if {s} is a palindrome",f"String s=\"{s}\";System.out.println(s.equals(new StringBuilder(s).reverse().toString()));"
    elif family=="anagram":
        jp,en,body=f"{s}と{s[::-1]}がアナグラムか",f"Check whether {s} and {s[::-1]} are anagrams",f"char[] a=\"{s}\".toCharArray(),b=\"{s[::-1]}\".toCharArray();\njava.util.Arrays.sort(a);java.util.Arrays.sort(b);\nSystem.out.println(java.util.Arrays.equals(a,b));"
    elif family=="balanced_brackets":
        seq="(" * (n+2*idx) + ")" * (n+2*idx)
        jp,en,body=f"長さ{len(seq)}の括弧列が対応しているか",f"Check whether {len(seq)} balanced parentheses match",f"String s=\"{seq}\";int depth=0;boolean ok=true;\nfor(char c:s.toCharArray()){{depth+=c=='('?1:-1;if(depth<0)ok=false;}}\nSystem.out.println(ok&&depth==0);\n// case {idx}"
    elif family=="lcm":
        jp,en,body=f"{a}と{b}の最小公倍数を求める",f"Find LCM of {a} and {b}",f"int x={a},y={b};int p=x,q=y;while(q!=0){{int t=p%q;p=q;q=t;}}\nSystem.out.println(x/p*y);"
    else:
        raise AssertionError(family)
    return jp,en,_java(body)


def _expected() -> dict[str, list[dict]]:
    data = {}
    for split, (families, per_family) in SPLITS.items():
        items = []
        for family in families:
            for i in range(per_family):
                ja,en,code=_case(family,i)
                instruction=ja if i%2==0 else en
                row={"id":f"{family}-{i:03d}","family":family,"messages":[
                    {"role":"user","content":_model_user_message(instruction,BASE_SOURCE,"Main")},
                    {"role":"assistant","content":code}]}
                items.append(row)
        random.Random(20261123+len(split)).shuffle(items)
        data[split]=items
    return data


def build_java_large_corpus(output: str | Path) -> dict:
    """Write all rows and SHA-pinned manifest; never overwrite files."""
    root=Path(output).expanduser().absolute()
    if root.exists():
        raise FileExistsError("Destination already exists")
    root.mkdir(parents=True)
    try:
        rows=_expected(); info={}
        seen_code=set()
        for split,items in rows.items():
            raw=("\n".join(_json(r) for r in items)+"\n").encode("utf-8")
            if len(raw)>MAX_JSONL_BYTES: raise ValueError("Dataset exceeds size cap")
            (root/f"{split}.jsonl").write_bytes(raw)
            # Track duplicate code independently; even identical prompts with
            # different answers would be a harmful data-quality regression.
            repeats=0
            for item in items:
                key=_hash_bytes(item["messages"][1]["content"].encode())
                if key in seen_code: repeats+=1
                seen_code.add(key)
            info[split]={"file":f"{split}.jsonl","sha256":_hash_bytes(raw),"examples":len(items),
                         "families":list(SPLITS[split][0]),"duplicate_source_files":repeats}
        if any(v["duplicate_source_files"] for v in info.values()):
            raise ValueError("Duplicate generated Java source detected")
        manifest={"schema":SCHEMA,"origin":"original project-authored parameterized Java 17 code; no third-party data",
                  "external_code_imported":False,"license_review":"ORIGINAL_ASSISTANT_AUTHORED_NOT_LEGALLY_ATTESTED",
                  "split_policy":"disjoint_task_families","generation_seed":20261123,"files":info,
                  "model_training_claim":"Only examples actually visited by an optimizer step may be described as trained."}
        (root/"manifest.json").write_text(_json(manifest)+"\n",encoding="utf-8")
        return {"status":"CREATED","path":str(root),"counts":{k:len(v) for k,v in rows.items()},
                "duplicate_source_files":sum(x["duplicate_source_files"] for x in info.values())}
    except BaseException:
        shutil.rmtree(root)
        raise


def verify_java_large_corpus(directory: str | Path) -> dict:
    """Strictly regenerate from original source; reject forged hashes/manifest."""
    root=Path(directory).expanduser().absolute()
    manifest=root/"manifest.json"
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size>65536:
        raise ValueError("Missing or unsafe manifest")
    meta=json.loads(manifest.read_text(encoding="utf-8"))
    if (meta.get("schema")!=SCHEMA or meta.get("external_code_imported") is not False
        or meta.get("license_review")!="ORIGINAL_ASSISTANT_AUTHORED_NOT_LEGALLY_ATTESTED"
        or meta.get("split_policy")!="disjoint_task_families" or meta.get("generation_seed")!=20261123
        or set(meta.get("files",{}))!=set(SPLITS)):
        raise ValueError("Unrecognized or changed corpus provenance")
    expected=_expected()
    ids=set(); observed_families=[]
    for split,rows in expected.items():
        file=root/f"{split}.jsonl"; record=meta["files"][split]
        if (not file.is_file() or file.is_symlink() or file.stat().st_size>MAX_JSONL_BYTES
            or record.get("file")!=file.name or record.get("examples")!=len(rows)
            or record.get("families")!=list(SPLITS[split][0])
            or record.get("duplicate_source_files")!=0):
            raise ValueError("Invalid corpus split")
        raw=file.read_bytes()
        if _hash_bytes(raw)!=record.get("sha256"):
            raise ValueError("Corpus hash mismatch")
        actual=[json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if actual!=rows: raise ValueError("Corpus data is not the canonical generator output")
        for item in actual:
            if item["id"] in ids: raise ValueError("Duplicate IDs")
            ids.add(item["id"])
        observed_families.append(set(record["families"]))
    if any(observed_families[i]&observed_families[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError("Train/test family contamination")
    return {"status":"PASS","counts":{k:len(v) for k,v in expected.items()},
            "family_disjoint":True,"license_attested":False}