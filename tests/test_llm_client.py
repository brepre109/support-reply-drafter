import pytest

from reply_drafter import llm
from reply_drafter.llm import LLMError, OpenAICompatibleModel


class Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        return self._body


OK = {"model": "m", "choices": [{"message": {"role": "assistant", "content": "hi"}}], "usage": {"prompt_tokens": 5}}


@pytest.fixture
def calls(monkeypatch):
    seen, script = [], []

    def post(url, headers, json, timeout):
        seen.append({"url": url, "headers": headers, "json": json})
        return script.pop(0)

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    return seen, script


def test_request_shape(calls):
    seen, script = calls
    script.append(Resp(200, OK))
    m = OpenAICompatibleModel("https://llm.example.com/v1/", "k", "m", extra_body={"provider": {"order": ["x"]}})
    res = m.complete([{"role": "user", "content": "x"}], tools=[{"type": "function"}], tool_choice="auto")
    assert res.message["content"] == "hi" and res.usage == {"prompt_tokens": 5}
    assert seen[0]["url"] == "https://llm.example.com/v1/chat/completions"
    assert seen[0]["headers"]["Authorization"] == "Bearer k"
    body = seen[0]["json"]
    assert body["tools"] and body["tool_choice"] == "auto" and body["provider"] == {"order": ["x"]}


def test_retries_rate_limits_and_server_errors(calls):
    seen, script = calls
    script.extend([Resp(429, {}, {"Retry-After": "1"}), Resp(503, {}), Resp(200, OK)])
    assert OpenAICompatibleModel("https://x/v1", "k", "m").complete([]).message["content"] == "hi"
    assert len(seen) == 3


def test_client_errors_fail_fast(calls):
    _, script = calls
    script.append(Resp(400, {"error": "bad model"}))
    with pytest.raises(LLMError, match="HTTP 400"):
        OpenAICompatibleModel("https://x/v1", "k", "m").complete([])


def test_gives_up(calls):
    _, script = calls
    script.extend([Resp(500, {})] * 4)
    with pytest.raises(LLMError, match="giving up"):
        OpenAICompatibleModel("https://x/v1", "k", "m", max_retries=3).complete([])
