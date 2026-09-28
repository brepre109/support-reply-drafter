"""Chat-completion clients.

Anything that speaks the OpenAI ``/chat/completions`` protocol with tool
calling works: OpenAI, OpenRouter, Together, Groq, vLLM, Ollama, LiteLLM, a
corporate gateway... Only ``base_url``, ``api_key`` and ``model`` change.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    message: dict  # {"role": "assistant", "content": ..., "tool_calls": [...]}
    usage: dict = field(default_factory=dict)
    model: str = ""
    latency_s: float = 0.0


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], *, tools: list[dict] | None = None,
                 tool_choice: str | dict | None = None, temperature: float = 0.2,
                 max_tokens: int = 1200) -> ChatResult: ...


@dataclass
class OpenAICompatibleModel:
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0  # hard wall-clock limit per call
    max_retries: int = 3
    extra_body: dict = field(default_factory=dict)  # e.g. provider routing, reasoning settings

    @property
    def name(self) -> str:
        return self.model

    def _post(self, body: dict) -> requests.Response:
        # `requests` resets its read timeout on every received byte, so a slow
        # stream can exceed `timeout` many times over. Enforce a real deadline.
        box: dict = {}

        def call():
            try:
                box["resp"] = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=body, timeout=(10, self.timeout),
                )
            except Exception as exc:  # surfaced below
                box["error"] = exc

        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(self.timeout + 5)
        if t.is_alive():
            raise LLMError(f"no response from {self.model} within {self.timeout:.0f}s")
        if "error" in box:
            raise box["error"]
        return box["resp"]

    def complete(self, messages, *, tools=None, tool_choice=None, temperature=0.2, max_tokens=1200) -> ChatResult:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                **self.extra_body}
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        last = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self._post(body)
            except (requests.RequestException, LLMError) as exc:
                last = str(exc)
                time.sleep(2**attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(float(r.headers.get("Retry-After") or 2**attempt))
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            if not data.get("choices"):
                raise LLMError(f"no choices in response: {json.dumps(data)[:300]}")
            return ChatResult(message=data["choices"][0]["message"], usage=data.get("usage") or {},
                              model=data.get("model", self.model), latency_s=time.monotonic() - t0)
        raise LLMError(f"{self.model}: giving up after {self.max_retries} retries ({last})")


@dataclass
class ScriptedModel:
    """Returns pre-written assistant messages in order. For tests."""

    replies: list[dict]
    name: str = "scripted"
    requests: list[dict] = field(default_factory=list)

    def complete(self, messages, *, tools=None, tool_choice=None, temperature=0.2, max_tokens=1200) -> ChatResult:
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools,
                              "tool_choice": tool_choice})
        if not self.replies:
            raise LLMError("ScriptedModel ran out of replies")
        reply = self.replies.pop(0)
        return ChatResult(message=reply(messages) if callable(reply) else reply,
                          usage={"prompt_tokens": 100, "completion_tokens": 20}, model=self.name)


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    """Build an assistant message that calls one tool (handy for scripts and tests)."""
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}
