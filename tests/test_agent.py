import json

from reply_drafter.agent import draft_reply
from reply_drafter.llm import ScriptedModel, tool_call
from reply_drafter.tools import TOOLS, ToolBox

MSGS = [{"role": "customer", "at": "2026-03-17T09:00:00Z", "text": "Did my express order 1047 ship?"}]
CUSTOMER = {"name": "Hannah Kim", "email": "hannah.kim@example.com"}
SUBMIT = {"reply": "Hi Hannah, yes, it shipped yesterday with UPS.", "category": "order_status", "confidence": 0.9,
          "needs_human": False, "facts_used": ["lookup_orders: #1047 shipped 2026-03-16"]}


def test_tool_loop_feeds_results_back_and_submits(adapter):
    from conftest import ts

    model = ScriptedModel([tool_call("lookup_orders", {"order_number": "1047"}), tool_call("submit_reply", SUBMIT)])
    d = draft_reply(model, adapter, MSGS, customer=CUSTOMER, as_of=ts("2026-03-17T09:00:00Z"))
    assert d.reply == SUBMIT["reply"] and d.category == "order_status" and d.needs_human is False
    assert d.trace[0]["tool"] == "lookup_orders"
    # the second model call saw the tool result, frozen at as_of
    second = model.requests[1]["messages"]
    tool_msg = next(m for m in second if m["role"] == "tool")
    shaped = json.loads(tool_msg["content"])
    assert shaped[0]["fulfillment_status"] == "SHIPPED"
    assert model.requests[0]["tools"] == TOOLS
    assert "Hannah Kim <hannah.kim@example.com>" in second[1]["content"]


def test_system_prompt_is_identical_across_drafts(adapter):
    a = ScriptedModel([tool_call("submit_reply", SUBMIT)])
    b = ScriptedModel([tool_call("submit_reply", SUBMIT)])
    draft_reply(a, adapter, MSGS, customer=CUSTOMER)
    draft_reply(b, adapter, [{"role": "customer", "at": "2026-04-01T00:00:00Z", "text": "other"}])
    assert a.requests[0]["messages"][0] == b.requests[0]["messages"][0]  # cacheable prefix


def test_plain_text_answer_is_nudged_then_final_step_forces_submit(adapter):
    replies = [{"role": "assistant", "content": "Sure, it shipped!"}] * 5 + [tool_call("submit_reply", SUBMIT)]
    model = ScriptedModel(replies)
    d = draft_reply(model, adapter, MSGS, customer=CUSTOMER)
    assert d.reply == SUBMIT["reply"]
    assert model.requests[1]["messages"][-1] == {"role": "user", "content": "Call submit_reply now with your draft."}
    assert model.requests[-1]["tool_choice"] == {"type": "function", "function": {"name": "submit_reply"}}
    assert model.requests[0]["tool_choice"] == "auto"


def test_no_submission_yields_flagged_empty_draft(adapter):
    model = ScriptedModel([tool_call("search_policies", {"query": "returns"})] * 6)
    d = draft_reply(model, adapter, MSGS, customer=CUSTOMER)
    assert d.reply == "" and d.needs_human and "did not submit" in d.human_reason


def test_model_error_is_captured(adapter):
    d = draft_reply(ScriptedModel([]), adapter, MSGS)
    assert d.error.startswith("model error") and d.needs_human


def test_bad_json_arguments_do_not_crash(adapter):
    bad = tool_call("lookup_orders", {})
    bad["tool_calls"][0]["function"]["arguments"] = "{not json"
    d = draft_reply(ScriptedModel([bad, tool_call("submit_reply", SUBMIT)]), adapter, MSGS, customer=CUSTOMER)
    assert d.trace[0]["args"] == {} and d.reply


def test_toolbox_defaults_to_customer_email_and_survives_adapter_errors(adapter):
    tb = ToolBox(adapter, customer_email="maya.torres@example.com")
    assert tb.call("lookup_orders", {})[0]["order_number"] == "#1041"
    assert "note" in tb.call("lookup_orders", {"order_number": "9999"})

    class Broken:
        def lookup_orders(self, **kw):
            raise TimeoutError("upstream timeout")

    assert ToolBox(Broken()).call("lookup_orders", {"order_number": "1"}) == {"error": "TimeoutError: upstream timeout"}
    assert ToolBox(adapter).call("delete_everything", {}) == {"error": "unknown tool delete_everything"}
