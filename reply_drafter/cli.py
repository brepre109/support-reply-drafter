"""Command-line entry point: draft one reply, or run a replay eval."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .adapters.mock_shopify import MockShopifyAdapter
from .agent import draft_reply
from .dataset import all_cases, cut_points, load_conversations
from .evals import format_summary, load_run, new_run_dir, run_eval, save_run, summarize
from .judge import LLMJudge, OverlapJudge
from .llm import OpenAICompatibleModel
from .offline import RuleBasedModel

DATA = Path(__file__).resolve().parent.parent / "data"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_model(provider: str, model: str | None, prefix: str = "DRAFTER"):
    if provider == "offline":
        return RuleBasedModel()
    base = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("DRAFTER_BASE_URL")
    key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("DRAFTER_API_KEY")
    name = model or os.environ.get(f"{prefix}_MODEL")
    if not (base and key and name):
        raise SystemExit(f"error: set {prefix}_BASE_URL, {prefix}_API_KEY and {prefix}_MODEL (see .env.example), "
                         "or use --provider offline")
    return OpenAICompatibleModel(base_url=base, api_key=key, model=name)


def make_adapter(args):
    if args.adapter == "shopify":
        from .adapters.shopify_admin import ShopifyAdminAdapter

        shop, token = os.environ.get("SHOPIFY_SHOP"), os.environ.get("SHOPIFY_ACCESS_TOKEN")
        if not (shop and token):
            raise SystemExit("error: --adapter shopify needs SHOPIFY_SHOP and SHOPIFY_ACCESS_TOKEN")
        return ShopifyAdminAdapter(shop, token)
    return MockShopifyAdapter.from_file(args.store)


def cmd_draft(args) -> int:
    adapter = make_adapter(args)
    model = make_model(args.provider, args.model)
    if args.message:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        context, customer, as_of, reference = ([{"role": "customer", "at": now, "text": args.message}],
                                               {"name": args.name or "Customer", "email": args.email}, None, None)
    else:
        convs = {c["id"]: c for c in load_conversations(args.dataset)}
        if args.conversation not in convs:
            raise SystemExit(f"error: no conversation '{args.conversation}' in {args.dataset}")
        cases = cut_points(convs[args.conversation])
        case = cases[args.cut if args.cut is not None else -1]
        context, customer, as_of, reference = case.context, case.customer, case.as_of, case.reference
    d = draft_reply(model, adapter, context, customer=customer, as_of=as_of)
    if args.json:
        print(json.dumps(d.to_dict(), indent=2, default=str))
        return 0
    for m in context:
        print(f"{m['role'].upper():>8}: {m['text']}")
    for t in d.trace:
        if "tool" in t:
            print(f"    tool: {t['tool']}({json.dumps(t['args'])})")
    print(f"\n   DRAFT: {d.reply}")
    print(f"          category={d.category} confidence={d.confidence} needs_human={d.needs_human}"
          + (f" ({d.human_reason})" if d.human_reason else ""))
    if reference:
        print(f"\n   STAFF: {reference}")
    return 0


def cmd_eval(args) -> int:
    adapter = make_adapter(args)
    model = make_model(args.provider, args.model)
    if args.judge == "heuristic":
        judge = OverlapJudge()
    else:
        judge = LLMJudge(make_model("openai", args.judge_model, prefix="JUDGE"))
    cases = all_cases(load_conversations(args.dataset))
    if args.topic:
        cases = [c for c in cases if c.topic == args.topic]
    if args.seed is not None:
        random.Random(args.seed).shuffle(cases)
    if args.limit:
        cases = cases[: args.limit]
    run_dir = new_run_dir(Path(args.out), model.name)
    meta = {"drafter": model.name, "judge": judge.name, "dataset": Path(args.dataset).name, "cases": len(cases),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "version": __version__}
    print(f"Evaluating {len(cases)} case(s) with {model.name}, judged by {judge.name} -> {run_dir}", file=sys.stderr)

    def tick(rec):
        g = rec["grade"]
        print(f"  {rec['case_id']:<8} {g['verdict']:<11} score={g['score']}  {rec['topic']}", file=sys.stderr)

    records = run_eval(cases, model, judge, adapter, workers=args.workers, on_item=tick)
    summary = summarize(records)
    save_run(run_dir, records, summary, meta)
    print(format_summary(summary, meta))
    print(f"\nReport: {run_dir / 'report.html'}")
    return 0


def cmd_report(args) -> int:
    from .report import render_html

    run_dir = Path(args.run_dir)
    records, summary, meta = load_run(run_dir)
    (run_dir / "report.html").write_text(render_html(records, summary, meta), encoding="utf-8")
    print(format_summary(summary, meta))
    print(f"\nReport: {run_dir / 'report.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", choices=["openai", "offline"], default="openai",
                        help="openai = any OpenAI-compatible endpoint (DRAFTER_*); offline = rule-based stand-in")
    common.add_argument("--model", help="override DRAFTER_MODEL")
    common.add_argument("--adapter", choices=["mock", "shopify"], default="mock")
    common.add_argument("--store", default=str(DATA / "store.json"), help="mock store JSON")
    common.add_argument("--dataset", default=str(DATA / "conversations.jsonl"))
    common.add_argument("--env-file", default=".env")

    p = argparse.ArgumentParser(prog="drafter", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("draft", parents=[common], help="draft a reply for a dataset conversation or a new message")
    src = d.add_mutually_exclusive_group(required=True)
    src.add_argument("--conversation", help="conversation id from the dataset (drafts its last staff turn)")
    src.add_argument("--message", help="a new customer message")
    d.add_argument("--cut", type=int, help="which cut point of the conversation (0-based, default last)")
    d.add_argument("--email", help="customer email for --message")
    d.add_argument("--name", help="customer name for --message")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_draft)

    e = sub.add_parser("eval", parents=[common], help="replay the dataset and grade drafts against staff replies")
    e.add_argument("--judge", choices=["llm", "heuristic"], default="llm")
    e.add_argument("--judge-model", help="override JUDGE_MODEL")
    e.add_argument("--limit", type=int)
    e.add_argument("--topic")
    e.add_argument("--seed", type=int, help="shuffle cases with this seed before --limit")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--out", default="runs")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("report", help="print a run's summary and rebuild its HTML report")
    r.add_argument("run_dir")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(Path(getattr(args, "env_file", ".env") or ".env"))
    return args.func(args)
