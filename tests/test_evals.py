import json

from reply_drafter.evals import format_summary, load_run, new_run_dir, run_eval, save_run, summarize
from reply_drafter.judge import OverlapJudge
from reply_drafter.offline import RuleBasedModel


def test_full_offline_eval_run(cases, adapter, tmp_path):
    records = run_eval(cases, RuleBasedModel(), OverlapJudge(), adapter, workers=4)
    assert [r["case_id"] for r in records] == [c.case_id for c in cases]  # order preserved
    s = summarize(records)
    assert s["cases"] == 31 and sum(s["verdicts"].values()) == 31
    assert sum(t["n"] for t in s["by_topic"].values()) == 31
    assert s["errors"] == 0 and s["mean_tool_calls"] >= 1

    run_dir = new_run_dir(tmp_path, "offline-rules")
    meta = {"drafter": "offline-rules", "judge": "heuristic:overlap", "dataset": "sample", "created_at": "now"}
    save_run(run_dir, records, s, meta)
    loaded, s2, m2 = load_run(run_dir)
    assert len(loaded) == 31 and s2 == json.loads(json.dumps(s)) and m2 == meta
    html = (run_dir / "report.html").read_text()
    assert "c03#3" in html and "Sendable" in html
    assert "Drafter: offline-rules" in format_summary(s, meta)


def test_report_escapes_html(adapter, cases, tmp_path):
    case = cases[0]
    case.context = [{**case.context[0], "text": "<script>alert(1)</script> where is 1041?"}]
    records = run_eval([case], RuleBasedModel(), OverlapJudge(), adapter, workers=1)
    run_dir = new_run_dir(tmp_path, "x")
    save_run(run_dir, records, summarize(records), {"drafter": "x", "judge": "y"})
    html = (run_dir / "report.html").read_text()
    assert "<script>alert" not in html and "&lt;script&gt;" in html


def test_run_dirs_never_collide(tmp_path):
    assert new_run_dir(tmp_path, "Model/A") != new_run_dir(tmp_path, "Model/A")
