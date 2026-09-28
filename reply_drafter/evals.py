"""Replay evaluation: draft every cut point, grade it against the staff reply."""
from __future__ import annotations

import json
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .adapters.base import StoreAdapter
from .agent import draft_reply
from .dataset import Case
from .judge import SENDABLE, VERDICTS
from .llm import ChatModel


def evaluate_case(case: Case, model: ChatModel, judge, adapter: StoreAdapter) -> dict:
    draft = draft_reply(model, adapter, case.context, customer=case.customer, as_of=case.as_of)
    grade = judge.grade(case.context, draft.reply, case.reference, draft.trace)
    return {
        "case_id": case.case_id, "conversation_id": case.conversation_id, "topic": case.topic,
        "as_of": case.as_of.isoformat(), "context": case.context, "reference": case.reference,
        "draft": draft.to_dict(), "grade": {**asdict(grade), "sendable": grade.sendable},
    }


def run_eval(cases: list[Case], model: ChatModel, judge, adapter: StoreAdapter, *, workers: int = 4,
             on_item: Callable[[dict], None] | None = None) -> list[dict]:
    """Evaluate cases concurrently; results come back in dataset order."""
    def one(case):
        rec = evaluate_case(case, model, judge, adapter)
        if on_item:
            on_item(rec)
        return rec

    if workers <= 1:
        return [one(c) for c in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, cases))


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def summarize(records: list[dict]) -> dict:
    n = len(records)
    verdicts = {v: 0 for v in [*VERDICTS, "unparsed"]}
    for r in records:
        verdicts[r["grade"]["verdict"]] = verdicts.get(r["grade"]["verdict"], 0) + 1
    topics: dict[str, dict] = {}
    for r in records:
        t = topics.setdefault(r["topic"], {"n": 0, "sendable": 0, "scores": []})
        t["n"] += 1
        t["sendable"] += int(r["grade"]["sendable"])
        t["scores"].append(r["grade"]["score"])
    usage_keys = ("prompt_tokens", "completion_tokens", "cost")
    usage = {k: round(sum((r["draft"]["usage"] or {}).get(k, 0) or 0 for r in records), 6) for k in usage_keys}
    return {
        "cases": n,
        "sendable_rate": round(sum(r["grade"]["sendable"] for r in records) / n, 3) if n else None,
        "mean_score": _mean(r["grade"]["score"] for r in records),
        "wrong_rate": round(verdicts["wrong"] / n, 3) if n else None,
        "needs_human_rate": round(sum(r["draft"]["needs_human"] for r in records) / n, 3) if n else None,
        "errors": sum(1 for r in records if r["draft"]["error"]),
        "mean_tool_calls": _mean(sum(1 for t in r["draft"]["trace"] if "tool" in t) for r in records),
        "mean_seconds": _mean(r["draft"]["seconds"] for r in records),
        "verdicts": verdicts,
        "by_topic": {k: {"n": v["n"], "sendable_rate": round(v["sendable"] / v["n"], 3),
                         "mean_score": _mean(v["scores"])} for k, v in sorted(topics.items())},
        "drafter_usage": usage,
        "sendable_verdicts": sorted(SENDABLE),
    }


def new_run_dir(root: Path, model_name: str) -> Path:
    # Timestamp + model + random suffix: parallel runs never collide.
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")[:40]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = root / f"{stamp}-{slug}-{secrets.token_hex(2)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run(run_dir: Path, records: list[dict], summary: dict, meta: dict) -> None:
    from .report import render_html

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(render_html(records, summary, meta), encoding="utf-8")


def load_run(run_dir: Path) -> tuple[list[dict], dict, dict]:
    records = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return records, data["summary"], data["meta"]


def format_summary(summary: dict, meta: dict) -> str:
    lines = [
        f"Drafter: {meta.get('drafter')}   Judge: {meta.get('judge')}   Cases: {summary['cases']}",
        f"Sendable: {summary['sendable_rate']:.0%}   Mean score: {summary['mean_score']}   "
        f"Wrong: {summary['wrong_rate']:.0%}   Needs human: {summary['needs_human_rate']:.0%}   "
        f"Errors: {summary['errors']}",
        "Verdicts: " + ", ".join(f"{k}={v}" for k, v in summary["verdicts"].items() if v),
        "",
        f"{'topic':<20} {'n':>3} {'sendable':>9} {'score':>6}",
    ]
    for topic, t in summary["by_topic"].items():
        lines.append(f"{topic:<20} {t['n']:>3} {t['sendable_rate']:>9.0%} {t['mean_score'] or '-':>6}")
    return "\n".join(lines)
