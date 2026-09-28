"""Judges that grade a draft against the reply staff actually sent.

``LLMJudge`` is the real grader: a separate model (keep it fixed across runs,
so scores stay comparable when you change the drafting model) with a rubric
that treats the staff reply as the reference for *facts*, not for wording.

``OverlapJudge`` is an offline heuristic for tests and smoke runs only. It
checks whether numbers, amounts and dates in the staff reply reappear in the
draft. It cannot judge tone or correctness and is not a quality metric.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .agent import format_transcript
from .llm import ChatModel
from .textutil import tokens

VERDICTS = ["better", "equivalent", "acceptable", "worse", "wrong"]
SENDABLE = {"better", "equivalent", "acceptable"}

RUBRIC = """You grade an AI-drafted customer-support reply against the reply a human staff member actually sent
in the same situation. The staff reply is the reference for FACTS the business knew at the time (a delivery date,
a restock date, a decision to replace an item), not necessarily for the best wording.

The AI looked facts up with tools; those results are listed under AI EVIDENCE. A detail found there was looked
up, not invented. Only call it a conflict if it contradicts the staff reply or the conversation.

Verdicts:
- better: more helpful or accurate than the staff reply
- equivalent: same substance
- acceptable: sendable with small edits; also use this when the draft honestly defers to a human on something
  only staff could know (e.g. an unpublished restock date)
- worse: needs real rewriting (missing the main answer, wrong tone, over-promises)
- wrong: states something false or harmful, or promises something the policy does not allow

Return JSON only:
{"verdict": "better|equivalent|acceptable|worse|wrong", "score": 1-5,
 "factual_conflicts": ["..."], "missing": ["things the staff reply covered that the draft did not"],
 "comment": "one sentence"}"""


@dataclass
class Grade:
    verdict: str
    score: int | None
    factual_conflicts: list[str]
    missing: list[str]
    comment: str
    judge: str
    usage: dict

    @property
    def sendable(self) -> bool:
        return self.verdict in SENDABLE


def _evidence_lines(trace: list[dict], limit: int = 6000) -> str:
    lines = []
    for t in trace:
        if "tool" in t:
            lines.append(f"- {t['tool']}({json.dumps(t['args'])}) -> {json.dumps(t['result'], default=str)}")
    text = "\n".join(lines) or "(no tools used)"
    return text[:limit]


def parse_grade(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[s:e + 1])
    except (ValueError, json.JSONDecodeError):
        # Salvage the essentials from truncated or chatty output.
        v = re.search(r'"verdict"\s*:\s*"(\w+)"', text)
        sc = re.search(r'"score"\s*:\s*(\d)', text)
        cm = re.search(r'"comment"\s*:\s*"([^"]*)', text)
        return {"verdict": v.group(1) if v else "unparsed", "score": int(sc.group(1)) if sc else None,
                "comment": cm.group(1) if cm else text[:300]}


class LLMJudge:
    def __init__(self, model: ChatModel):
        self.model = model
        self.name = f"llm:{model.name}"

    def grade(self, context: list[dict], draft: str, reference: str, trace: list[dict]) -> Grade:
        prompt = (f"CONVERSATION SO FAR:\n{format_transcript(context)}\n\nAI DRAFT:\n{draft or '(empty)'}\n\n"
                  f"AI EVIDENCE:\n{_evidence_lines(trace)}\n\nSTAFF REPLY THAT WAS SENT:\n{reference}")
        res = self.model.complete([{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt}],
                                  temperature=0, max_tokens=800)
        g = parse_grade(res.message.get("content") or "")
        verdict = g.get("verdict") if g.get("verdict") in VERDICTS else "unparsed"
        score = g.get("score")
        return Grade(verdict=verdict, score=int(score) if isinstance(score, (int, float)) else None,
                     factual_conflicts=list(g.get("factual_conflicts") or []), missing=list(g.get("missing") or []),
                     comment=str(g.get("comment") or ""), judge=self.name, usage=res.usage)


KEY_FACT = re.compile(r"\$\d+(?:\.\d{2})?|#\d{3,}|\b\d{6,}\b|\b\d+(?:-\d+)?\s*(?:days?|hours?|weeks?|oz|ml)\b|"
                      r"\b(?:January|February|March|April|May|June|July|August|September|October|November|"
                      r"December)\s+\d{1,2}\b", re.IGNORECASE)


class OverlapJudge:
    """Offline heuristic. See module docstring: smoke tests only."""

    name = "heuristic:overlap"

    def grade(self, context: list[dict], draft: str, reference: str, trace: list[dict]) -> Grade:
        if not draft.strip():
            return Grade("worse", 1, [], ["empty draft"], "No draft was produced.", self.name, {})
        facts = {f.lower() for f in KEY_FACT.findall(reference)}
        found = {f for f in facts if f in draft.lower()}
        ref_t, draft_t = set(tokens(reference)), set(tokens(draft))
        overlap = len(ref_t & draft_t) / max(1, len(ref_t))
        coverage = len(found) / len(facts) if facts else overlap
        blended = 0.6 * coverage + 0.4 * overlap
        score = 1 + round(4 * blended)
        verdict = "equivalent" if score >= 4 else "acceptable" if score == 3 else "worse"
        return Grade(verdict, score, [], sorted(facts - found),
                     f"{len(found)}/{len(facts)} key facts, {overlap:.0%} word overlap", self.name, {})
