"""Secondary agent that relies on OpenAI models for advanced generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class OpenAIAgent:
    """Agent responsible for advanced code generation and model orchestration."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except Exception:
                self._client = None
        return self._client

    @staticmethod
    def _format_history(messages: Iterable[BaseMessage]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            formatted.append({"role": role, "content": msg.content})
        return formatted

    def _model_inference(self, messages: Iterable[BaseMessage]) -> str:
        client = self._lazy_client()
        if client is None:
            # Offline fallback.
            return json.dumps({"summary": "OpenAI agent unavailable; returning deterministic guidance."})
        response = client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=self._format_history(messages),
        )
        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            for block in getattr(item, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        return json.dumps({"summary": "OpenAI agent produced no output."})

    def handle(self, state: Dict[str, Any]) -> Dict[str, Any]:
        history = state.get("messages", [])
        content = self._model_inference(history)
        message = AIMessage(content=content)
        return {
            "messages": [message],
            "context": {"delegate_to_openai": False},
        }
