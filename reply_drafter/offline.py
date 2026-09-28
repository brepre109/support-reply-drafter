"""A deterministic, rule-based stand-in for an LLM.

It speaks the same tool-calling protocol as a real model, so the whole
pipeline (agent loop, tools, adapters, eval harness, report) can run with no
API key: in CI, in the demo, and in tests. The replies are template-based and
deliberately plain. This is not a baseline worth comparing models against.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from .llm import ChatResult, tool_call
from .textutil import score

POLICY_HINTS = [
    (r"return|refund", "returns refunds"),
    (r"broken|cracked|damaged|wrong item|instead of", "damaged wrong items"),
    (r"code|discount|welcome\d*|coupon", "discount codes"),
    (r"wholesale|stockist|retailer", "wholesale"),
    (r"australia|canada|uk\b|international|customs|shipping cost|how much is shipping|how fast", "shipping"),
    (r"cancel|change .*address|wrong address|old address", "changing cancelling order"),
    (r"gift|receipt|prices", "gift orders"),
    (r"\bpets?\b|\bcat\b|\bdog\b|allerg|safe|tunnel", "candle care safety"),
]
PRODUCT_WORDS = re.compile(r"candle|diffuser|refill|trio|snuffer|trimmer|pumpkin|fig|cedar|lavender|sage|vegan|"
                           r"burn time|in stock|sold out|unscented", re.I)
HUMAN_WORDS = re.compile(r"broken|cracked|damaged|wrong|refund|dispute|bank|cancel|address|lost|ridiculous|photo",
                         re.I)


def _fmt_date(ts: str | None) -> str:
    if not ts:
        return "?"
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%B %-d")


def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n])


def _relevant_sentences(text: str, question: str, n: int = 2) -> str:
    """The n sentences sharing the most words with the question, kept in their original order."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    ranked = sorted(range(len(parts)), key=lambda i: (-score(question, parts[i]), i))[:n]
    return " ".join(parts[i] for i in sorted(ranked))


@dataclass
class RuleBasedModel:
    name: str = "offline-rules"

    def complete(self, messages, *, tools=None, tool_choice=None, temperature=0.2, max_tokens=1200) -> ChatResult:
        transcript = messages[1]["content"]
        customer_lines = [ln for ln in transcript.splitlines() if "] CUSTOMER:" in ln]
        last = customer_lines[-1].split("] CUSTOMER:", 1)[1].strip() if customer_lines else ""
        name_match = re.search(r"Customer: (\S+)", transcript)
        first = name_match.group(1) if name_match and name_match.group(1) != "unknown" else "there"

        results = self._tool_results(messages)
        if not results:
            return ChatResult(self._plan(last), model=self.name)
        return ChatResult(self._compose(first, last, results), model=self.name)

    @staticmethod
    def _tool_results(messages) -> list[tuple[str, object]]:
        names = {}
        for m in messages:
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
        out = []
        for m in messages:
            if m.get("role") == "tool" and m.get("content") != "submitted":
                out.append((names.get(m["tool_call_id"], "?"), json.loads(m["content"])))
        return out

    @staticmethod
    def _plan(text: str) -> dict:
        calls = []
        number = re.search(r"#?\b(\d{4})\b", text)
        if number:
            calls.append(("lookup_orders", {"order_number": number.group(1)}))
        elif re.search(r"\border\b|tracking|shipped|where is", text, re.I):
            calls.append(("lookup_orders", {}))
        for pattern, query in POLICY_HINTS:
            if re.search(pattern, text, re.I):
                calls.append(("search_policies", {"query": query}))
                break
        if PRODUCT_WORDS.search(text) and not number:
            calls.append(("search_products", {"query": text}))
        if not calls:
            calls.append(("search_policies", {"query": text}))
        msg = tool_call(*calls[0], call_id="call_0")
        for i, (name, args) in enumerate(calls[1:], start=1):
            msg["tool_calls"].append(tool_call(name, args, call_id=f"call_{i}")["tool_calls"][0])
        return msg

    @staticmethod
    def _order_sentence(o: dict) -> str:
        num = o["order_number"]
        if o["fulfillment_status"] == "CANCELLED":
            return f"Order {num} was cancelled on {_fmt_date(o['cancelled_at'])} and refunded to your original payment method."
        if not o["fulfillments"]:
            return (f"Order {num} hasn't shipped yet. We pack orders within 1-2 business days, and you'll get "
                    f"tracking by email as soon as the label is created.")
        f = o["fulfillments"][-1]
        ev = f.get("latest_event") or {}
        if f["latest_status"] == "delivered":
            return f"Order {num} was delivered on {_fmt_date(ev.get('at'))} ({ev.get('location', '')})."
        where = f" The latest scan is '{f['latest_status'].replace('_', ' ')}' in {ev.get('location')} on {_fmt_date(ev.get('at'))}." if ev else ""
        return (f"Order {num} shipped on {_fmt_date(f['shipped_at'])} with {f['carrier']} "
                f"(tracking {f['tracking_number']}).{where}")

    def _compose(self, first: str, last: str, results) -> dict:
        parts, facts = [], []
        for name, res in results:
            if name == "lookup_orders" and isinstance(res, list) and res:
                parts.append(self._order_sentence(res[0]))
                facts.append(f"lookup_orders: {res[0]['order_number']} {res[0]['fulfillment_status']}")
            elif name == "search_policies" and isinstance(res, list) and res:
                parts.append(_relevant_sentences(res[0]["text"], last))
                facts.append(f"search_policies: {res[0]['topic']}")
            elif name == "search_products" and isinstance(res, list) and res:
                p = res[0]
                stock = ", ".join(f"{v['title']} ${v['price']} ({'in stock' if v['available'] else 'sold out'})"
                                  for v in p["variants"])
                parts.append(f"{p['title']}: {_first_sentences(p['description'], 1)} {stock}.")
                facts.append(f"search_products: {p['handle']}")
        if not parts:
            parts.append("Thanks for reaching out. I'll check with the team and get back to you shortly.")
        needs_human = bool(HUMAN_WORDS.search(last))
        reply = f"Hi {first}, " + " ".join(parts)
        return tool_call("submit_reply", {
            "reply": reply, "category": "other", "confidence": 0.4 if needs_human else 0.6,
            "needs_human": needs_human, "human_reason": "keyword rule: action or approval needed" if needs_human else "",
            "facts_used": facts,
        }, call_id="call_submit")
