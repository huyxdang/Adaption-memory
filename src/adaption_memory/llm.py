"""Thin OpenAI-compatible chat client with usage tracking.

Works against Ollama / LM Studio (base_url=http://localhost:11434/v1) and
the OpenAI API alike. All eval components take an LLM instance so tests can
substitute a fake.
"""

import os
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from openai import OpenAI


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def snapshot(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "latency_ms": round(self.latency_ms, 3),
                "cost_usd": round(self.cost_usd, 8)}


class LLM:
    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024,
                 reasoning_effort: str | None = None,
                 structured_output: str = "json-schema",
                 before_call: Callable[[dict], None] | None = None,
                 on_call: Callable[[dict], None] | None = None):
        if structured_output not in {"json-schema", "json-object", "prompt-only"}:
            raise ValueError(f"unsupported structured output mode: {structured_output}")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.structured_output = structured_output
        self.before_call = before_call
        self.on_call = on_call
        self.usage = Usage()
        effective_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        host = (urlparse(effective_base_url).hostname
                if effective_base_url else "api.openai.com")
        default_api_key = (os.getenv("OPENAI_API_KEY")
                           if host == "api.openai.com" else "EMPTY")
        self.client = OpenAI(
            base_url=effective_base_url,
            api_key=api_key or default_api_key or "EMPTY",
            max_retries=5,
        )

    def chat(self, messages: list[dict], max_tokens: int | None = None,
             temperature: float | None = None,
             response_format: dict | None = None) -> str:
        kwargs = {"model": self.model, "messages": messages}
        token_limit = max_tokens or self.max_tokens
        if self.model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = token_limit
        else:
            kwargs["max_tokens"] = token_limit
        # GPT-5 reasoning models accept only their default temperature. Local
        # OpenAI-compatible models still benefit from deterministic sampling.
        if not self.model.startswith("gpt-5"):
            kwargs["temperature"] = (
                self.temperature if temperature is None else temperature
            )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if response_format is not None:
            if self.structured_output == "json-schema":
                kwargs["response_format"] = response_format
            elif self.structured_output == "json-object":
                kwargs["response_format"] = {"type": "json_object"}
        before_call = getattr(self, "before_call", None)
        if before_call:
            before_call({
                "model": self.model,
                "messages": messages,
                "max_tokens": token_limit,
            })
        started = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        self.usage.calls += 1
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        if resp.usage:
            prompt_tokens = resp.usage.prompt_tokens or 0
            completion_tokens = resp.usage.completion_tokens or 0
            details = getattr(resp.usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
            self.usage.prompt_tokens += prompt_tokens
            self.usage.completion_tokens += completion_tokens
            self.usage.reasoning_tokens += reasoning_tokens
        cost_usd = 0.0
        if self.model == "gpt-5.6-luna":
            cost_usd = (
                prompt_tokens * 0.20 / 1_000_000
                + completion_tokens * 1.20 / 1_000_000
            )
        self.usage.latency_ms += latency_ms
        self.usage.cost_usd += cost_usd
        on_call = getattr(self, "on_call", None)
        if on_call:
            on_call({
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "latency_ms": round(latency_ms, 3),
                "cost_usd": round(cost_usd, 8),
            })
        return (resp.choices[0].message.content or "").strip()
