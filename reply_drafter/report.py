"""Self-contained HTML report for an eval run (no external assets)."""
from __future__ import annotations

import html
import json

# Verdicts are ordered good -> bad, so they use the status palette; every
# colored mark is paired with a text label, never color alone.
VERDICT_STYLE = {
    "better": ("#0ca30c", "Better"),
    "equivalent": ("#0ca30c", "Equivalent"),
    "acceptable": ("#fab219", "Acceptable"),
    "worse": ("#ec835a", "Worse"),
    "wrong": ("#d03b3b", "Wrong"),
    "unparsed": ("#8a8983", "Unparsed"),
}

CSS = """
:root { --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8983; --line:#e4e3de; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --card:#232322; --ink:#ffffff; --ink2:#c3c2b7; --muted:#9a998f; --line:#3a3a37; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:0 auto; padding:28px 24px 60px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 10px; }
.meta { color:var(--ink2); font-size:13px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-top:18px; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.tile .v { font-size:28px; font-weight:650; font-variant-numeric:tabular-nums; }
.tile .k { color:var(--ink2); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.bar { display:flex; gap:2px; height:22px; margin:8px 0 6px; }
.bar span { display:block; height:100%; border-radius:4px; min-width:4px; }
.legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--ink2); font-size:13px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--ink2); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
td.num { font-variant-numeric:tabular-nums; }
.chip { display:inline-flex; align-items:center; gap:6px; font-size:12px; white-space:nowrap; color:var(--ink); }
.chip i { width:8px; height:8px; border-radius:50%; display:inline-block; }
details { background:var(--card); border:1px solid var(--line); border-radius:10px; margin:8px 0; }
summary { cursor:pointer; padding:10px 14px; display:flex; gap:12px; align-items:center; }
summary .id { font-family:ui-monospace,monospace; color:var(--ink2); font-size:12px; min-width:64px; }
summary .q { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.body { padding:4px 16px 16px; display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.body .full { grid-column:1 / -1; }
.label { color:var(--ink2); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
.box { white-space:pre-wrap; border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.msg { margin:0 0 6px; } .msg b { color:var(--ink2); font-weight:600; }
.trace { font-family:ui-monospace,monospace; font-size:12px; color:var(--ink2); white-space:pre-wrap; max-height:220px; overflow:auto; }
.flag { color:var(--ink2); font-size:12px; }
"""


def _e(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _chip(verdict: str) -> str:
    color, label = VERDICT_STYLE.get(verdict, VERDICT_STYLE["unparsed"])
    return f'<span class="chip"><i style="background:{color}"></i>{label}</span>'


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def render_html(records: list[dict], summary: dict, meta: dict) -> str:
    n = summary["cases"] or 1
    verdicts = {k: v for k, v in summary["verdicts"].items() if v}
    segments = "".join(
        f'<span title="{VERDICT_STYLE[k][1]}: {v} of {summary["cases"]} ({v / n:.0%})" '
        f'style="flex:{v};background:{VERDICT_STYLE[k][0]}"></span>' for k, v in verdicts.items())
    legend = "".join(f'<span><i style="background:{VERDICT_STYLE[k][0]}"></i>{VERDICT_STYLE[k][1]} {v}</span>'
                     for k, v in verdicts.items())
    tiles = [
        ("Sendable", _pct(summary["sendable_rate"])),
        ("Mean score (1-5)", summary["mean_score"] if summary["mean_score"] is not None else "-"),
        ("Wrong", _pct(summary["wrong_rate"])),
        ("Flagged for a human", _pct(summary["needs_human_rate"])),
        ("Tool calls / draft", summary["mean_tool_calls"] if summary["mean_tool_calls"] is not None else "-"),
        ("Cases", summary["cases"]),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div></div>'
                         for k, v in tiles)
    topic_rows = "".join(
        f"<tr><td>{_e(t)}</td><td class='num'>{d['n']}</td><td class='num'>{_pct(d['sendable_rate'])}</td>"
        f"<td class='num'>{_e(d['mean_score'] if d['mean_score'] is not None else '-')}</td></tr>"
        for t, d in summary["by_topic"].items())

    cases = []
    for r in records:
        g, d = r["grade"], r["draft"]
        last_q = next((m["text"] for m in reversed(r["context"]) if m["role"] == "customer"), "")
        convo = "".join(f'<p class="msg"><b>{_e(m["role"])}</b> <span class="flag">{_e(m["at"])}</span><br>{_e(m["text"])}</p>'
                        for m in r["context"])
        trace = "\n".join(f"{t['tool']}({json.dumps(t['args'])}) -> {json.dumps(t['result'], default=str)[:600]}"
                          for t in d["trace"] if "tool" in t) or "(no tool calls)"
        flags = []
        if d["needs_human"]:
            flags.append(f"needs human: {_e(d['human_reason'])}")
        if d["confidence"] is not None:
            flags.append(f"confidence {d['confidence']:.2f}")
        if d["error"]:
            flags.append(f"error: {_e(d['error'])}")
        issues = ""
        if g.get("factual_conflicts"):
            issues += "<br>Conflicts: " + "; ".join(_e(x) for x in g["factual_conflicts"])
        if g.get("missing"):
            issues += "<br>Missing: " + "; ".join(_e(x) for x in g["missing"])
        cases.append(f"""
<details>
  <summary><span class="id">{_e(r['case_id'])}</span>{_chip(g['verdict'])}
    <span class="flag">score {_e(g['score'] if g['score'] is not None else '-')}</span>
    <span class="q">{_e(last_q)}</span><span class="flag">{_e(r['topic'])}</span></summary>
  <div class="body">
    <div class="full"><div class="label">Conversation (as of {_e(r['as_of'])})</div><div class="box">{convo}</div></div>
    <div><div class="label">AI draft</div><div class="box">{_e(d['reply'] or '(no draft)')}</div>
      <div class="flag">{' · '.join(flags)}</div></div>
    <div><div class="label">Staff reply (reference)</div><div class="box">{_e(r['reference'])}</div></div>
    <div class="full"><div class="label">Judge ({_e(g['judge'])})</div><div>{_e(g['comment'])}{issues}</div></div>
    <div class="full"><div class="label">Tool trace</div><div class="trace">{_e(trace)}</div></div>
  </div>
</details>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reply drafter eval - {_e(meta.get('drafter'))}</title><style>{CSS}</style></head>
<body><main>
<h1>Support reply drafter: replay eval</h1>
<div class="meta">Drafter <b>{_e(meta.get('drafter'))}</b> · judge <b>{_e(meta.get('judge'))}</b> ·
dataset {_e(meta.get('dataset'))} · {_e(meta.get('created_at'))}</div>
<div class="tiles">{tiles_html}</div>
<h2>Verdicts</h2>
<div class="bar" role="img" aria-label="Verdict distribution">{segments}</div>
<div class="legend">{legend}</div>
<h2>By topic</h2>
<table><thead><tr><th>Topic</th><th>Cases</th><th>Sendable</th><th>Mean score</th></tr></thead>
<tbody>{topic_rows}</tbody></table>
<h2>Cases</h2>
{''.join(cases)}
</main></body></html>
"""
