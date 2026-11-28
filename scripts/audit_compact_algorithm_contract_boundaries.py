"""Audit fixed compact curriculum contracts and execute new development boundary cases.

Only the exact reference sources bound to the pinned admission/reference reports
are executed. No model outputs, external inputs, training or independent tests.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from sagallm.java_algorithm_compact_tasks import COMPACT_TASKS
from sagallm.java_algorithm_curriculum import canonical, cases, oracle, sha
from sagallm.java_instruction_sft import sha_file
from scripts.audit_algorithm_curriculum_admission import require, sft_candidate, verify_report
from scripts.verify_algorithm_curriculum import invoke

COMPACT_ZIP = "cb76bb7ea2d41f5d64bc833a886c412bfacb95f2ed9e8c37788e5da76be0282e"
COMPACT_REPORT = "b0d7596dba88acaa07735897e7ee5fac4af0e9e779914073cc9ad5cdf9c0dba1"
ARTIFACT = 15256963954
RUN = 46129795701

# Author-side semantic inspection notes, not human approval. Exact current task
# hashes are bound by the compact admission archive; the source is unchanged.
CONTRACTS = {
    "range_sum_queries": {
        "input": "n q, n integers, q inclusive zero-based pairs",
        "bounds": "1<=n<=30; 1<=q<=20; -100<=A[i]<=100; 0<=l<=r<n",
        "output": "one inclusive sum per line, in query order",
        "variants": {"prefix": "prefix sums", "direct": "direct summation"},
        "review_note": "A[l..r] convention implies inclusive indices; numeric bound implies zero-based indices. Confirm notation with bilingual reviewers.",
    },
    "interval_union_length": {
        "input": "n, then n half-open integer intervals",
        "bounds": "0<=n<=30; -100<=l<=r<=100",
        "output": "total union length, including zero for empty union",
        "variants": {"coverage": "bounded boolean coverage", "sort_merge": "sort and merge"},
        "review_note": "The Japanese compact text retains [l,r) and bounds, but omits the explicit word integer; confirm intended integer input.",
    },
    "edit_distance": {
        "input": "two nonempty lowercase ASCII words",
        "bounds": "length 1..20; insertion/deletion/substitution cost 1",
        "output": "Levenshtein distance",
        "variants": {"matrix": "two-dimensional DP", "rolling": "rolling rows"},
        "review_note": "Regex [a-z]{1,20} conveys ASCII domain and nonempty length; confirm shorthand for rolling DP.",
    },
    "unweighted_graph_distance": {
        "input": "n m, m undirected edges u v, then a b",
        "bounds": "1<=n<=20; 0<=m<=190; vertices 0..n-1",
        "output": "minimum edge count, or -1 if unreachable",
        "variants": {"bfs": "queue BFS", "floyd": "Floyd-Warshall"},
        "review_note": "Japanese Floyd task says distance rather than explicitly minimum edge count; compact graph notation does not explicitly state unit edge cost. Human review required.",
    },
}


def read_compact(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == 3326
            and sha_file(path) == COMPACT_ZIP, "Compact archive pin mismatch")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == ["compact-admission.json"], "Compact member drift")
        info = archive.infolist()[0]
        require(not info.is_dir() and not info.flag_bits & 1
                and 0 < info.file_size <= 65536, "Invalid compact member")
        raw = archive.read(info)
    require(sha(raw) == COMPACT_REPORT, "Compact member hash mismatch")
    return json.loads(raw)


def bind_records(records, report):
    require(report["context"] == 384 and report["training_performed"] is False
            and report["semantic_review"] == "REVIEW_REQUIRED"
            and report["rights_review"] == "REVIEW_REQUIRED", "Admission scope drift")
    index = {r["id"]: r for r in report["rows"]}
    require(len(records) == len(index) == len(report["rows"]) == 16,
            "Unexpected record count")
    checked = []
    sources = {}
    for r in records:
        task = COMPACT_TASKS.get(r["id"], r["task"])
        original = sft_candidate(r, "algorithm-v1-")
        compact = sft_candidate({**r, "task": task}, "algorithm-compact-v1-")
        prior = index.pop(compact["id"])
        family, variant, language = (r[k] for k in ("family", "variant", "language"))
        require(prior["baseline_id"] == original["id"]
                and all(prior[k] == r[k] for k in ("family", "variant", "language"))
                and prior["baseline_task_sha256"] == sha(r["task"].encode())
                and prior["task_sha256"] == sha(task.encode())
                and prior["source_sha256"] == sha(r["source"].encode())
                and prior["candidate_row_sha256"] == sha(canonical(compact))
                and prior["task_changed"] == (task != r["task"])
                and 0 < prior["compact_positions"] <= 384,
                "Compact record binding mismatch")
        require(variant in CONTRACTS[family]["variants"], "Unknown contract")
        key = (family, variant)
        if key in sources:
            require(sources[key] == r["source"], "Bilingual source mismatch")
        sources[key] = r["source"]
        checked.append({"id": compact["id"], "family": family, "variant": variant,
                        "language": language, "task_sha256": prior["task_sha256"],
                        "source_sha256": prior["source_sha256"],
                        "candidate_row_sha256": prior["candidate_row_sha256"],
                        "task_changed": prior["task_changed"],
                        "positions": prior["compact_positions"],
                        "contract_review": "AUTHOR_NOTES_ONLY_HUMAN_REVIEW_REQUIRED"})
    require(not index and len(sources) == 8 and
            Counter(r["language"] for r in checked) == {"ja": 8, "en": 8}
            and sum(r["task_changed"] for r in checked) == 10,
            "Compact scope mismatch")
    membership = sha(canonical(sorted(r["candidate_row_sha256"] for r in checked)))
    require(membership == report["membership_sha256"], "Membership drift")
    return checked, sources


def range_input(values, queries):
    return f"{len(values)} {len(queries)}\n" + " ".join(map(str, values)) + "\n" + "".join(
        f"{l} {r}\n" for l, r in queries)


def interval_input(intervals):
    return f"{len(intervals)}\n" + "".join(f"{l} {r}\n" for l, r in intervals)


def graph_input(n, edges, a, b):
    return f"{n} {len(edges)}\n" + "".join(f"{u} {v}\n" for u, v in edges) + f"{a} {b}\n"


def boundary_cases(family):
    """Ten distinct cases per family, with explicit expected outputs."""
    if family == "range_sum_queries":
        yield "singleton_max", range_input([100], [(0, 0)]), "100\n"
        yield "singleton_min", range_input([-100], [(0, 0)]), "-100\n"
        yield "max_positive", range_input([100]*30, [(0, 29)]), "3000\n"
        yield "max_negative", range_input([-100]*30, [(0, 29)]), "-3000\n"
        yield "max_queries", range_input([1]*30, [(0, i) for i in range(20)]), "".join(f"{i}\n" for i in range(1,21))
        yield "cancel", range_input([100,-100]*15, [(0,29),(1,28),(1,1),(29,29)]), "0\n0\n-100\n-100\n"
        yield "order_repeat", range_input([3,-7,9], [(2,2),(0,0),(2,2),(0,2)]), "9\n3\n9\n5\n"
        yield "zero", range_input([0]*30, [(0,29),(29,29)]), "0\n0\n"
        yield "all_prefix", range_input([2,4,8,16], [(0,0),(0,1),(0,2),(0,3)]), "2\n6\n14\n30\n"
        yield "all_suffix", range_input([2,4,8,16], [(3,3),(2,3),(1,3),(0,3)]), "16\n24\n28\n30\n"
    elif family == "interval_union_length":
        yield "full_span", interval_input([(-100,100)]), "200\n"
        yield "zero_width", interval_input([(-100,-100),(0,0),(100,100)]), "0\n"
        yield "touching", interval_input([(0,1),(-1,0),(1,2)]), "3\n"
        yield "separate_ends", interval_input([(-100,-99),(99,100)]), "2\n"
        yield "nested", interval_input([(-5,5),(-4,4),(-3,3)]), "10\n"
        yield "duplicate_max_n", interval_input([(-100,100)]*30), "200\n"
        yield "max_n_disjoint", interval_input([(2*i,2*i+1) for i in range(30)]), "30\n"
        yield "bridge", interval_input([(0,2),(4,6),(1,5)]), "6\n"
        yield "negative_only", interval_input([(-9,-4),(-6,-2)]), "7\n"
        yield "mixed_zero", interval_input([(5,5),(-2,2),(7,7),(2,2)]), "4\n"
    elif family == "edit_distance":
        yield "identity_max", "abcdefghijklmnopqrst abcdefghijklmnopqrst\n", "0\n"
        yield "substitute_max", "a"*20+" "+"z"*20+"\n", "20\n"
        yield "insert_max", "a "+"a"*20+"\n", "19\n"
        yield "delete_max", "a"*20+" a\n", "19\n"
        yield "single_substitute", "a z\n", "1\n"
        yield "transpose_cost_two", "ab ba\n", "2\n"
        yield "prefix_insert", "bc abc\n", "1\n"
        yield "suffix_delete", "abcd abc\n", "1\n"
        yield "middle_insert", "ac abc\n", "1\n"
        yield "two_operations", "abc axcd\n", "2\n"
    elif family == "unweighted_graph_distance":
        path = [(i,i+1) for i in range(19)]
        yield "max_diameter", graph_input(20,path,0,19), "19\n"
        yield "reverse_path", graph_input(20,path,19,0), "19\n"
        yield "max_edges", graph_input(20,[(i,j) for i in range(20) for j in range(i+1,20)],0,19), "1\n"
        yield "no_edges", graph_input(20,[],0,19), "-1\n"
        yield "same_vertex", graph_input(20,[],19,19), "0\n"
        yield "cycle_shortcut", graph_input(5,[(0,1),(1,2),(2,3),(3,4),(4,0)],0,4), "1\n"
        yield "disconnected_components", graph_input(6,[(0,1),(1,2),(3,4),(4,5)],0,5), "-1\n"
        yield "diamond", graph_input(4,[(0,1),(0,2),(1,3),(2,3)],0,3), "2\n"
        yield "duplicate_edges", graph_input(3,[(0,1),(0,1),(1,2)],0,2), "2\n"
        yield "self_loop", graph_input(3,[(0,0),(0,1),(1,1),(1,2)],0,2), "2\n"
    else:
        raise ValueError("Unknown family")


def checked_cases(family):
    result = list(boundary_cases(family))
    require(len(result) == len({name for name, _, _ in result}) ==
            len({stdin for _, stdin, _ in result}) == 10, "Case identity drift")
    old = set(cases(family))
    for name, stdin, expected in result:
        require(stdin not in old, "Boundary input repeats original suite")
        require(oracle(family, stdin) == expected, "Explicit expectation disagrees with oracle: " + name)
    return result


def run(compact_zip, reference_zip):
    report = read_compact(compact_zip)
    records, sources = bind_records(verify_report(reference_zip), report)
    suites = {family: checked_cases(family) for family in CONTRACTS}
    javac, java = shutil.which("javac"), shutil.which("java")
    require(javac and java, "JDK required")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        version = invoke([javac,"-version"],root).decode().strip()
        require(version.startswith("javac 17."), "Java 17 required")
        for (family, variant), source in sorted(sources.items()):
            wd = root / (family + "-" + variant)
            wd.mkdir()
            (wd / "Main.java").write_bytes(source.encode())
            invoke([javac,"-J-Xmx128m","--release","17","-encoding","UTF-8","-proc:none","Main.java"],wd)
            checks = []
            for name, stdin, expected in suites[family]:
                actual = invoke([java,"-Xmx64m","-XX:ActiveProcessorCount=1","-cp",str(wd),"Main"],wd,stdin.encode())
                require(actual == expected.encode(), "Boundary stdout mismatch: " + family + "/" + variant + "/" + name)
                checks.append({"case": name, "stdin_sha256": sha(stdin.encode()),
                               "expected_stdout_sha256": sha(expected.encode()),
                               "actual_stdout_sha256": sha(actual)})
            results.append({"family": family, "variant": variant,
                            "source_sha256": sha(source.encode()), "compile": "PASS",
                            "checks": checks})
    return {
        "schema": "sagallm-compact-contract-boundaries-v1",
        "status": "FIXED_REFERENCE_BOUNDARY_CHECKS_COMPLETE", "jdk": version,
        "compact_artifact_id": ARTIFACT, "compact_run_id": RUN,
        "compact_zip_sha256": COMPACT_ZIP, "compact_report_sha256": COMPACT_REPORT,
        "membership_sha256": report["membership_sha256"], "records": records,
        "contracts": CONTRACTS, "reference_compiles": len(results),
        "distinct_boundary_inputs": sum(map(len, suites.values())),
        "boundary_comparisons": sum(len(r["checks"]) for r in results),
        "results": results, "semantic_review": "REVIEW_REQUIRED",
        "rights_review": "REVIEW_REQUIRED", "semantic_independence": "NOT_ESTABLISHED",
        "model_generated_code_executed": False, "training_performed": False,
        "heldout_scoring": False, "independent_evaluation": "NOT_RUN",
        "model_selection_permitted": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("compact-zip", "reference-zip", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse overwrite")
    report = run(args.compact_zip, args.reference_zip)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, sort_keys=True, indent=2)
        out.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "jdk", "reference_compiles",
          "distinct_boundary_inputs", "boundary_comparisons", "membership_sha256", "semantic_review")}, sort_keys=True))
