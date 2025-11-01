"""LangGraph orchestration linking the Hugging Face and OpenAI agents."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from .huggingface_agent import HuggingFaceAgent
from .openai_agent import OpenAIAgent


class AgentState(TypedDict, total=False):
    """Shared state exchanged between LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: Dict[str, Any]
    plans: list[Dict[str, Any]]
    tool_results: list[Dict[str, Any]]


class AgenticOrchestrator:
    """Builds the agent graph and provides an ergonomic interface for execution."""

    def __init__(self, huggingface_agent: HuggingFaceAgent, openai_agent: OpenAIAgent) -> None:
        self.huggingface_agent = huggingface_agent
        self.openai_agent = openai_agent
        self._graph_app = self._build_graph()

    def _build_graph(self) -> CompiledGraph:
        graph = StateGraph(AgentState)
        graph.add_node("huggingface", self._huggingface_node)
        graph.add_node("openai", self._openai_node)
        graph.add_edge(START, "huggingface")
        graph.add_conditional_edges("huggingface", self._route_from_huggingface, {
            "delegate": "openai",
            "complete": END,
        })
        graph.add_edge("openai", END)
        return graph.compile()

    def _huggingface_node(self, state: AgentState) -> AgentState:
        update = self.huggingface_agent.handle(state)
        return self._merge_state(state, update)

    def _openai_node(self, state: AgentState) -> AgentState:
        update = self.openai_agent.handle(state)
        return self._merge_state(state, update)

    @staticmethod
    def _merge_state(state: AgentState, update: Dict[str, Any]) -> AgentState:
        merged: AgentState = dict(state)
        for key, value in update.items():
            if key == "messages":
                merged.setdefault("messages", [])
                merged["messages"].extend(value)  # type: ignore[index]
            elif key == "context":
                base = merged.setdefault("context", {})
                base.update(value)  # type: ignore[assignment]
            else:
                merged[key] = value  # type: ignore[assignment]
        return merged

    def _route_from_huggingface(self, state: AgentState) -> str:
        context = state.get("context", {})
        return "delegate" if context.get("delegate_to_openai") else "complete"

    def invoke(self, user_input: str) -> AgentState:
        human_message = HumanMessage(content=user_input)
        initial_state: AgentState = {
            "messages": [human_message],
            "context": {},
        }
        result: AgentState = self._graph_app.invoke(initial_state)
        return result

    async def ainvoke(self, user_input: str) -> AgentState:
        human_message = HumanMessage(content=user_input)
        initial_state: AgentState = {
            "messages": [human_message],
            "context": {},
        }
        result: AgentState = await self._graph_app.ainvoke(initial_state)
        return result
