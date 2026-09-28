"""The drafting loop: model <-> tools until it calls submit_reply."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .adapters.base import StoreAdapter
from .llm import ChatModel
from .tools import TOOLS, ToolBox

SYSTEM = """You draft replies to customer support chats for {name}, an online store. A staff member reviews
every draft before it is sent, so accuracy matters more than sounding confident.

Voice: {voice}
Sign-off (only for emails; chats need none): {signoff}

How to work:
- Look facts up with the tools before stating them: orders, tracking, products, stock, prices, policies.
- Only state facts you found in tool results or the conversation. Never invent dates, prices, tracking
  numbers, restock dates, or policy terms. If a fact isn't available (for example a restock date), say you'll
  check with the team rather than guessing.
- You cannot change orders, issue refunds or send replacements. For those, write the reply a staff member would
  send and set needs_human=true with the reason.
- Also set needs_human=true for angry, legal or chargeback messages, and for photos you can't see.
- Keep chat replies short: 2-5 sentences, the customer's first name, the answer first.
- Finish by calling submit_reply exactly once."""

USER = """Current time: {now}
Customer: {customer}

Conversation so far (oldest first):
{transcript}

Draft the next staff reply."""

MAX_STEPS = 6


@dataclass
class Draft:
    reply: str
    category: str | None
    confidence: float | None
    needs_human: bool
    human_reason: str
    facts_used: list[str]
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def format_transcript(messages: list[dict]) -> str:
    return "\n".join(f"[{m['at']}] {m['role'].upper()}: {m['text']}" for m in messages)


def _add_usage(total: dict, usage: dict) -> None:
    for k, v in (usage or {}).items():
        if isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v


def draft_reply(
    model: ChatModel,
    adapter: StoreAdapter,
    messages: list[dict],
    *,
    customer: dict | None = None,
    as_of: datetime | None = None,
    max_steps: int = MAX_STEPS,
) -> Draft:
    """Draft the next staff reply to ``messages`` (list of {role, at, text})."""
    profile = adapter.store_profile()
    customer = customer or {}
    toolbox = ToolBox(adapter, customer_email=customer.get("email"), as_of=as_of)
    now = (as_of.isoformat() if as_of else (messages[-1]["at"] if messages else datetime.now().isoformat()))
    who = f"{customer.get('name', 'unknown')} <{customer.get('email', 'unknown')}>"
    chat = [
        # Static system prompt first: identical across drafts, so provider prompt caches can reuse it.
        {"role": "system", "content": SYSTEM.format(name=profile["name"], voice=profile.get("voice", ""),
                                                    signoff=profile.get("support_signoff", "").replace("\n", " / "))},
        {"role": "user", "content": USER.format(now=now, customer=who, transcript=format_transcript(messages))},
    ]
    usage: dict = {}
    trace: list[dict] = []
    t0 = time.monotonic()
    result: dict | None = None
    error = None

    for step in range(max_steps):
        last = step == max_steps - 1
        choice = {"type": "function", "function": {"name": "submit_reply"}} if last else "auto"
        try:
            res = model.complete(chat, tools=TOOLS, tool_choice=choice)
        except Exception as exc:
            error = f"model error: {exc}"
            break
        _add_usage(usage, res.usage)
        msg = res.message
        calls = msg.get("tool_calls") or []
        chat.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_details")})
        if not calls:
            # Some models answer in plain text. Nudge once toward the structured submit.
            chat.append({"role": "user", "content": "Call submit_reply now with your draft."})
            trace.append({"step": step, "note": "no tool call", "content": (msg.get("content") or "")[:500]})
            continue
        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "submit_reply":
                result = args
                chat.append({"role": "tool", "tool_call_id": tc["id"], "content": "submitted"})
                continue
            out = toolbox.call(name, args)
            trace.append({"step": step, "tool": name, "args": args, "result": out})
            chat.append({"role": "tool", "tool_call_id": tc["id"], "content": ToolBox.dumps(out)})
        if result is not None:
            break

    if result is None:
        return Draft(reply="", category=None, confidence=0.0, needs_human=True,
                     human_reason=error or "the model did not submit a draft", facts_used=[], trace=trace,
                     usage=usage, model=model.name, seconds=round(time.monotonic() - t0, 2), error=error)
    try:
        confidence = float(result.get("confidence")) if result.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    return Draft(
        reply=(result.get("reply") or "").strip(), category=result.get("category"), confidence=confidence,
        needs_human=bool(result.get("needs_human")), human_reason=result.get("human_reason") or "",
        facts_used=[str(f) for f in result.get("facts_used") or []], trace=trace, usage=usage, model=model.name,
        seconds=round(time.monotonic() - t0, 2),
    )
