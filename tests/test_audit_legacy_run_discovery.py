from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_legacy_run_discovery as a


def row(run, path=".github/workflows/java-context-sft-cpu.yml"):
    return {"id": run, "attempt": 1, "commit_sha": "a"*40,
            "path": path, "event": "pull_request", "status": "completed",
            "conclusion": "success", "created_at": "2026-11-25T00:00:00Z",
            "updated_at": "2026-11-25T00:01:00Z"}


def api_row(run):
    r = row(run)
    r["run_attempt"] = r.pop("attempt")
    r["head_sha"] = r.pop("commit_sha")
    return r


class LegacyDiscoveryTests(unittest.TestCase):
    def test_workflow_scope_is_conservative(self):
        self.assertEqual(a.classify(".github/workflows/java-context-sft-cpu.yml"), "RESEARCH_OR_TRAINING_REVIEW")
        self.assertEqual(a.classify(".github/workflows/cpu-ci.yml"), "INFRASTRUCTURE_SCOPE_NOT_CERTIFIED")
        self.assertEqual(a.classify(".github/workflows/new-unknown.yml"), "UNKNOWN_REVIEW_REQUIRED")

    def test_compare_runs_no_completeness_claim(self):
        old = {i: row(i) for i in range(1, 17)}
        current = {**old, 17: row(17, ".github/workflows/new-unknown.yml")}
        del current[16]
        current[1] = {**old[1], "attempt": 2}
        archived = [{"run_id": i, "commit_sha": "a"*40} for i in range(1, 16)]
        result = a.compare_runs(old, current, archived)
        self.assertEqual(result["missing_old_run_ids"], [16])
        self.assertEqual(result["new_run_ids"], [17])
        self.assertEqual(result["changed_metadata_run_ids"], [1])
        self.assertEqual(result["unarchived_research_or_unknown_run_ids"], [17])
        self.assertFalse(result["all_runs_and_attempts_complete"])
        current[1] = {**old[1], "commit_sha": "b"*40}
        self.assertEqual(a.compare_runs(old, current, archived)["identity_conflicts"], [1])
        archived[0]["commit_sha"] = "b"*40
        with self.assertRaises(ValueError):
            a.compare_runs(old, current, archived)

    def test_pagination_rejects_missing_duplicate_out_of_window(self):
        cutoff = "2026-11-25T01:00:00Z"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pages.json"
            def write(rows, total):
                path.write_text(json.dumps([{"total_count": total, "workflow_runs": rows}]))
            write([api_row(1), api_row(2)], 2)
            self.assertEqual(set(a.live_pages(path, cutoff)), {1,2})
            for rows, count in (([api_row(1)], 2), ([api_row(1),api_row(1)], 2)):
                write(rows, count)
                with self.assertRaises(ValueError):
                    a.live_pages(path, cutoff)
            write([api_row(1)], 1)
            with self.assertRaises(ValueError):
                a.live_pages(path, "2026-11-24T01:00:00Z")

    def test_windowed_pages_reject_gaps_overlap_and_duplicate_ids(self):
        start="2026-11-25T00:00:00Z";cutoff="2026-11-25T00:00:02Z"
        def window(lo,hi,rows):
            return {"start_utc":lo,"end_utc":hi,"pages":[{"total_count":len(rows),"workflow_runs":rows}]}
        first=window(start,start,[api_row(1)])
        second=window("2026-11-25T00:00:01Z",cutoff,[])
        data={"schema":"sagallm-actions-windowed-pages-v1","start_utc":start,"cutoff_utc":cutoff,"windows":[first,second]}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"pages.json"
            p.write_text(json.dumps(data));self.assertEqual(set(a.live_pages(p,cutoff)),{1})
            for windows in ([first],[first,window(cutoff,cutoff,[])],[first,window(start,cutoff,[])],
                            [first,window("2026-11-25T00:00:01Z",cutoff,[{**api_row(1),"created_at":"2026-11-25T00:00:01Z"}])]):
                p.write_text(json.dumps({**data,"windows":windows}))
                with self.assertRaises(ValueError):a.live_pages(p,cutoff)

    def test_collector_bisects_large_count(self):
        from scripts.collect_legacy_run_pages import collect
        start="2026-11-25T00:00:00Z";end="2026-11-25T00:00:01Z"
        def fetch(lo,hi,page):
            if lo!=hi:return {"total_count":1001,"workflow_runs":[]}
            return {"total_count":1,"workflow_runs":[{**api_row(1 if lo==start else 2),"created_at":lo}]}
        d=collect(start,end,fetch)
        self.assertEqual(len(d["windows"]),2)
        self.assertEqual(d["windows"][0]["end_utc"],start)
        self.assertEqual(d["windows"][1]["start_utc"],end)

    def test_legacy_originals_when_staged(self):
        import os
        root = os.environ.get("LEGACY_REPORTS_DIR")
        if not root:
            self.skipTest("Exact original report worktree not staged")
        rows, data = a.legacy_reports(Path(root))
        self.assertEqual(len(rows), 3)
        self.assertEqual(a.compile_scope(data["v018-paired-compile"])["regressed_ids"],
                         ["longest_common_prefix-0025", "collatz_max-0029"])
        self.assertEqual(a.public_scope(data["public-unreviewed-behavior"])["historical_cases_recounted"], 19)
        self.assertEqual(a.stdio_scope(data["stdio-semantic-sft"])["historical_cases_recounted"], 36)


if __name__ == "__main__":
    unittest.main()
