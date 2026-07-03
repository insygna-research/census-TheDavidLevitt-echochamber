"""Shared implementation for OpenAI-compatible chat APIs.

OpenAI, Together.AI, and LM Studio all speak the same chat-completions
protocol; the concrete providers only differ in how the client is built.
"""

import json
import time
from typing import Optional

from .base import LLMProvider, LLMResponse, Message, ToolCall, ToolDef
from .retry import call_with_retries


class OpenAICompatProvider(LLMProvider):
    """Base provider for any OpenAI-compatible chat-completions endpoint."""

    supports_tools = True

    def __init__(self, model: str, client):
        self.model = model
        self.client = client

    def _to_wire(
        self,
        messages: list[Message],
        system_prompt: Optional[str],
    ) -> list[dict]:
        wire = []
        if system_prompt:
            wire.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                wire.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            elif msg.role == "tool":
                wire.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            else:
                wire.append({"role": msg.role, "content": msg.content})
        return wire

    def complete(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[list[ToolDef]] = None,
        tool_choice: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": self._to_wire(messages, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            if tool_choice:
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        start = time.time()
        response = call_with_retries(
            lambda: self.client.chat.completions.create(**kwargs)
        )
        latency_ms = (time.time() - start) * 1000

        choice = response.choices[0]
        tool_calls = []
        for tc in choice.message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
            )

        return LLMResponse(
            content=choice.message.content or "",
            model=getattr(response, "model", None) or self.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
        )
