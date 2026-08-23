"""Thin OpenAI-compatible chat client with usage tracking.

Works against Ollama / LM Studio (base_url=http://localhost:11434/v1) and
the OpenAI API alike. All eval components take an LLM instance so tests can
substitute a fake.
"""

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def snapshot(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


class LLM:
    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.usage = Usage()
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            max_retries=5,
        )

    def chat(self, messages: list[dict], max_tokens: int | None = None,
             temperature: float | None = None) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        self.usage.calls += 1
        if resp.usage:
            self.usage.prompt_tokens += resp.usage.prompt_tokens or 0
            self.usage.completion_tokens += resp.usage.completion_tokens or 0
        return (resp.choices[0].message.content or "").strip()
