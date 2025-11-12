"""Simple conversation orchestrator that maintains history for the OpenAI agent."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage

from .openai_agent import OpenAIAgent


class AgenticOrchestrator:
    """Stateful wrapper around the OpenAIAgent that keeps running chat history."""

    def __init__(self, agent: OpenAIAgent) -> None:
        self.agent = agent
        self._history: List[BaseMessage] = []

    def invoke(self, user_input: str) -> Dict[str, Any]:
        human_message = HumanMessage(content=user_input)
        state = {"messages": [*self._history, human_message]}
        update = self.agent.handle(state)
        self._history.append(human_message)
        self._history.extend(update.get("messages", []))
        return {
            "messages": self._history.copy(),
            "context": update.get("context", {}),
            "intermediate_steps": update.get("intermediate_steps", []),
        }

    async def ainvoke(self, user_input: str) -> Dict[str, Any]:
        human_message = HumanMessage(content=user_input)
        state = {"messages": [*self._history, human_message]}
        update = await self.agent.ahandle(state)
        self._history.append(human_message)
        self._history.extend(update.get("messages", []))
        return {
            "messages": self._history.copy(),
            "context": update.get("context", {}),
            "intermediate_steps": update.get("intermediate_steps", []),
        }

    def reset(self) -> None:
        """Clear the stored conversation history."""
        self._history.clear()

    @property
    def history(self) -> List[BaseMessage]:
        """Expose the current conversation history."""
        return self._history.copy()

