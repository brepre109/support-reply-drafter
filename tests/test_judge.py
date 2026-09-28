import json

from reply_drafter.judge import LLMJudge, OverlapJudge, parse_grade
from reply_drafter.llm import ScriptedModel

CTX = [{"role": "customer", "at": "2026-03-10T11:00:00Z", "text": "How long does the diffuser last?"}]
REF = "The 100 ml Lavender Fields diffuser lasts about 8-10 weeks. A 200 ml refill is $22."


def test_llm_judge_sends_rubric_evidence_and_reference():
    verdict = {"verdict": "acceptable", "score": 4, "factual_conflicts": [], "missing": ["refill price"],
               "comment": "Correct duration, no refill price."}
    model = ScriptedModel([{"role": "assistant", "content": "```json\n" + json.dumps(verdict) + "\n```"}])
    trace = [{"tool": "get_product", "args": {"ref": "FH-DIF-LAV"}, "result": {"lasts": "about 8-10 weeks"}}]
    g = LLMJudge(model).grade(CTX, "It lasts about 8-10 weeks.", REF, trace)
    assert (g.verdict, g.score, g.sendable, g.missing) == ("acceptable", 4, True, ["refill price"])
    prompt = model.requests[0]["messages"][1]["content"]
    assert "STAFF REPLY THAT WAS SENT:\n" + REF in prompt
    assert 'get_product({"ref": "FH-DIF-LAV"})' in prompt


def test_truncated_judge_output_is_salvaged():
    assert parse_grade('{"verdict": "worse", "score": 2, "comment": "Missed the main') == {
        "verdict": "worse", "score": 2, "comment": "Missed the main"}


def test_unknown_verdict_becomes_unparsed():
    model = ScriptedModel([{"role": "assistant", "content": '{"verdict": "great", "score": 5}'}])
    g = LLMJudge(model).grade(CTX, "x", REF, [])
    assert g.verdict == "unparsed" and not g.sendable


def test_overlap_judge_counts_key_facts():
    good = OverlapJudge().grade(CTX, "It lasts about 8-10 weeks; the 200 ml refill is $22.", REF, [])
    bad = OverlapJudge().grade(CTX, "Thanks, we'll check.", REF, [])
    empty = OverlapJudge().grade(CTX, "", REF, [])
    assert good.score > bad.score and good.sendable and not bad.sendable
    assert empty.verdict == "worse" and empty.missing == ["empty draft"]
