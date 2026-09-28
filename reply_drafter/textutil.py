"""Tiny text helpers (keyword search without a search engine dependency)."""
from __future__ import annotations

import re

STOP = set("""a an and are as at be but by can do does for from have how i if in is it its me my no not of on or
our so that the their them there this to up us was we what when where which will with you your i'm it's
please thanks hi hello hey just get got""".split())


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    out = []
    for w in words:
        if w in STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural folding: candles -> candle
        out.append(w)
    return out


def score(query: str, *fields: str) -> float:
    q = set(tokens(query))
    if not q:
        return 0.0
    hay = set()
    for f in fields:
        hay.update(tokens(f))
    return len(q & hay) / len(q)
