"""Conversation datasets and replay cut points.

A dataset is JSONL, one conversation per line::

    {"id": "c01", "topic": "order-status",
     "customer": {"name": "...", "email": "..."},
     "messages": [{"role": "customer"|"staff", "at": "2026-03-08T16:02:00Z", "text": "..."}, ...]}

Every staff message that answers a customer message is a *cut point*: the
drafter sees the conversation up to (not including) that staff message, the
data adapters are frozen at the time of the customer's last message, and the
real staff reply becomes the reference answer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .adapters.mock_shopify import parse_ts


@dataclass
class Case:
    case_id: str  # "<conversation id>#<staff message index>"
    conversation_id: str
    topic: str
    customer: dict
    context: list[dict]  # messages before the staff reply
    reference: str  # what staff actually sent
    as_of: datetime  # time of the last customer message


def load_conversations(path: str | Path) -> list[dict]:
    convs = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        conv = json.loads(line)
        for key in ("id", "messages"):
            if key not in conv:
                raise ValueError(f"{path}:{n}: missing '{key}'")
        convs.append(conv)
    return convs


def cut_points(conv: dict) -> list[Case]:
    cases = []
    msgs = conv["messages"]
    for i, m in enumerate(msgs):
        if m["role"] != "staff" or i == 0 or msgs[i - 1]["role"] != "customer":
            continue
        cases.append(Case(
            case_id=f"{conv['id']}#{i}", conversation_id=conv["id"], topic=conv.get("topic", "other"),
            customer=conv.get("customer") or {}, context=msgs[:i], reference=m["text"],
            as_of=parse_ts(msgs[i - 1]["at"]),
        ))
    return cases


def all_cases(convs: list[dict]) -> list[Case]:
    return [c for conv in convs for c in cut_points(conv)]
