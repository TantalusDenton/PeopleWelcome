"""Multi-agent LangGraph workflow for collaborative AI discussions.

This module implements a state machine for orchestrating multiple AI agents
in collaborative discussions, with support for tool execution and agent spawning.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add App-logic to path for database imports
BASE_DIR = Path(__file__).resolve().parents[1]
APP_LOGIC_DIR = BASE_DIR / "App-logic"
sys.path.insert(0, str(APP_LOGIC_DIR))


@dataclass
class AIConfig:
    """Configuration for a participating AI."""
    id: str
    name: str
    owner_id: str
    system_prompt: Optional[str] = None
    model: str = "gpt-4o"


class MultiAgentState(TypedDict):
    """State for the multi-agent workflow."""
    # User's original prompt
    user_prompt: str

    # Configuration for participating AIs
    selected_ais: List[AIConfig]

    # Messages in the conversation (using LangGraph message reducer)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Individual AI responses
    responses: Dict[str, str]

    # Discussion log (for roundtable)
    discussion_log: List[Dict[str, Any]]

    # Final winning response(s)
    winning_response: Optional[str]

    # Tool execution results
    tool_results: List[Dict[str, Any]]

    # Spawned agents during this session
    spawned_agents: List[str]

    # Current phase of the workflow
    current_phase: str

    # User ID for the session
    user_id: str

    # Session ID
    session_id: str

    # Error tracking
    errors: List[str]


def create_llm(model: str = "gpt-4o", temperature: float = 0.7) -> ChatOpenAI:
    """Create a ChatOpenAI instance."""
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


async def collect_responses(state: MultiAgentState) -> MultiAgentState:
    """
    Node: Collect initial responses from all participating AIs.

    Each AI responds to the user's prompt independently.
    """
    logger.info("Collecting responses from all AIs...")

    responses = {}
    llm = create_llm()

    for ai in state["selected_ais"]:
        try:
            # Build messages for this AI
            messages = []

            # Add system prompt if available
            if ai.system_prompt:
                messages.append(SystemMessage(content=ai.system_prompt))
            else:
                messages.append(SystemMessage(
                    content=f"You are {ai.name}, a helpful AI assistant."
                ))

            # Add user prompt
            messages.append(HumanMessage(content=state["user_prompt"]))

            # Get response
            response = await llm.ainvoke(messages)
            responses[ai.id] = response.content

            logger.info(f"AI {ai.name} responded")

        except Exception as e:
            logger.error(f"Error getting response from {ai.name}: {e}")
            responses[ai.id] = f"[Error: Could not get response - {str(e)}]"

    return {
        **state,
        "responses": responses,
        "current_phase": "responses_collected"
    }


async def roundtable_discussion(state: MultiAgentState) -> MultiAgentState:
    """
    Node: Facilitate a roundtable discussion among AIs.

    AIs can see each other's responses and discuss the best answer.
    """
    logger.info("Starting roundtable discussion...")

    if len(state["selected_ais"]) < 2:
        # No discussion needed for single AI
        return {
            **state,
            "discussion_log": [],
            "current_phase": "discussion_complete"
        }

    discussion_log = []
    llm = create_llm(temperature=0.8)

    # Build context of all responses
    responses_context = "\n\n".join([
        f"{ai.name}'s response:\n{state['responses'].get(ai.id, 'No response')}"
        for ai in state["selected_ais"]
    ])

    # Each AI comments on the discussion
    for ai in state["selected_ais"]:
        try:
            messages = [
                SystemMessage(content=ai.system_prompt or f"You are {ai.name}."),
                HumanMessage(content=f"""The user asked: {state['user_prompt']}

Here are all the responses so far:
{responses_context}

Please provide your thoughts on the discussion. Do you agree with the other responses?
What would you add or modify? Keep your response concise.""")
            ]

            response = await llm.ainvoke(messages)

            discussion_log.append({
                "ai_id": ai.id,
                "ai_name": ai.name,
                "content": response.content,
                "type": "discussion"
            })

            logger.info(f"AI {ai.name} contributed to discussion")

        except Exception as e:
            logger.error(f"Error in discussion from {ai.name}: {e}")

    return {
        **state,
        "discussion_log": discussion_log,
        "current_phase": "discussion_complete"
    }


async def select_winner(state: MultiAgentState) -> MultiAgentState:
    """
    Node: Determine the winning/best response(s).

    Uses a meta-analysis to synthesize the best answer.
    """
    logger.info("Selecting best response...")

    if len(state["selected_ais"]) == 1:
        # Single AI - use its response directly
        ai = state["selected_ais"][0]
        return {
            **state,
            "winning_response": state["responses"].get(ai.id, ""),
            "current_phase": "winner_selected"
        }

    # Multi-AI: Synthesize best answer
    llm = create_llm(temperature=0.3)  # Lower temperature for synthesis

    responses_context = "\n\n".join([
        f"{ai.name}'s response:\n{state['responses'].get(ai.id, 'No response')}"
        for ai in state["selected_ais"]
    ])

    discussion_context = "\n\n".join([
        f"{entry['ai_name']}: {entry['content']}"
        for entry in state["discussion_log"]
    ])

    messages = [
        SystemMessage(content="""You are a synthesis AI. Your job is to analyze
multiple AI responses and their discussion to provide the best possible answer
to the user's question. Consider all perspectives and create a comprehensive response."""),
        HumanMessage(content=f"""User's question: {state['user_prompt']}

Initial responses:
{responses_context}

Discussion:
{discussion_context}

Please synthesize the best possible answer, incorporating the best elements from
all responses and addressing any disagreements constructively.""")
    ]

    response = await llm.ainvoke(messages)

    return {
        **state,
        "winning_response": response.content,
        "current_phase": "winner_selected"
    }


async def execute_tools(state: MultiAgentState) -> MultiAgentState:
    """
    Node: Execute any tools that were requested during the discussion.

    Detects tool requests in the winning response and executes them.
    """
    logger.info("Checking for tool execution...")

    from .dynamic_tools import get_tool_by_name
    import re

    tool_results = []
    winning_response = state.get("winning_response", "")

    # Tool request patterns to detect in the response
    tool_patterns = [
        (r"(?:search|web search|look up)[:\s]+[\"']?(.+?)[\"']?(?:\.|$|,)", "web_search"),
        (r"create (?:a )?workflow[:\s]+[\"']?(.+?)[\"']?(?:\.|$|,)", "workflow_provisioner"),
        (r"spawn (?:a )?(?:python )?server[:\s]+[\"']?(.+?)[\"']?(?:\.|$|,)", "python_server"),
        (r"build (?:an? )?integration[:\s]+[\"']?(.+?)[\"']?(?:\.|$|,)", "integration_builder"),
        (r"create (?:a )?rag[:\s]+[\"']?(.+?)[\"']?(?:\.|$|,)", "rag_builder"),
        (r"generate (?:code|script)[:\s]+[\"']?(.+?)[\"']?(?:\.|$|,)", "code_generator"),
    ]

    for pattern, tool_name in tool_patterns:
        matches = re.findall(pattern, winning_response, re.IGNORECASE)
        for match in matches:
            query = match.strip()
            if not query:
                continue

            tool = get_tool_by_name(tool_name)
            if tool:
                try:
                    logger.info(f"Executing tool '{tool_name}' with query: {query}")
                    result = tool._run(query=query)
                    tool_results.append({
                        "tool": tool_name,
                        "query": query,
                        "result": result,
                        "status": "success"
                    })
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' failed: {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "query": query,
                        "error": str(e),
                        "status": "failed"
                    })

    if tool_results:
        logger.info(f"Executed {len(tool_results)} tools")

    return {
        **state,
        "tool_results": tool_results,
        "current_phase": "tools_executed"
    }


async def spawn_agents(state: MultiAgentState) -> MultiAgentState:
    """
    Node: Handle agent spawning if requested during the discussion.

    Detects agent spawn requests in the winning response and creates
    persistent agents in the database.
    """
    logger.info("Checking for agent spawn requests...")

    import re
    import db_service

    spawned_agents = []
    winning_response = state.get("winning_response", "")

    # Patterns for detecting spawn requests
    spawn_patterns = [
        r"spawn (?:an? )?agent[:\s]+[\"']?(.+?)[\"']?\s+(?:for|to)\s+[\"']?(.+?)[\"']?(?:\.|$|,)",
        r"create (?:an? )?ai[:\s]+[\"']?(.+?)[\"']?\s+(?:for|to)\s+[\"']?(.+?)[\"']?(?:\.|$|,)",
        r"new agent[:\s]+[\"']?(.+?)[\"']?\s+(?:purpose|for)[:\s]+[\"']?(.+?)[\"']?(?:\.|$|,)",
    ]

    for pattern in spawn_patterns:
        matches = re.findall(pattern, winning_response, re.IGNORECASE)
        for agent_name, purpose in matches:
            agent_name = agent_name.strip()
            purpose = purpose.strip()

            if not agent_name or not purpose:
                continue

            try:
                # Get parent AI ID (use first selected AI)
                parent_ai_id = None
                if state["selected_ais"]:
                    parent_ai_id = state["selected_ais"][0].id

                # Create the spawned agent in the database
                agent = db_service.create_spawned_agent(
                    name=agent_name,
                    parent_ai_id=parent_ai_id,
                    config={"purpose": purpose, "spawned_by": "multi_agent_workflow"}
                )

                spawned_agents.append(agent["id"])
                logger.info(f"Spawned agent '{agent_name}' with ID {agent['id']}")

            except Exception as e:
                logger.error(f"Failed to spawn agent '{agent_name}': {e}")

    if spawned_agents:
        logger.info(f"Spawned {len(spawned_agents)} agents")

    return {
        **state,
        "spawned_agents": spawned_agents,
        "current_phase": "agents_spawned"
    }


def should_continue_discussion(state: MultiAgentState) -> str:
    """
    Edge condition: Determine if discussion should continue or proceed to selection.
    """
    # Simple heuristic: if there's significant disagreement, continue discussion
    # For now, always proceed to selection after one round
    return "select_winner"


def should_execute_tools(state: MultiAgentState) -> str:
    """
    Edge condition: Determine if tool execution is needed.

    Checks if the winning response contains keywords that suggest
    tool execution is requested.
    """
    response = state.get("winning_response", "").lower()

    # Keywords that indicate tool execution is needed
    tool_keywords = [
        "search", "look up", "web search",
        "create workflow", "spawn server", "python server",
        "build integration", "create rag",
        "generate code", "generate script"
    ]

    if any(kw in response for kw in tool_keywords):
        return "execute_tools"

    # Skip directly to spawn_agents if no tools needed
    return "spawn_agents"


def should_spawn_agents(state: MultiAgentState) -> str:
    """
    Edge condition: Determine if agent spawning is requested.

    Checks if the winning response contains keywords that suggest
    agent spawning is requested.
    """
    response = state.get("winning_response", "").lower()

    # Keywords that indicate agent spawning is requested
    spawn_keywords = [
        "spawn agent", "create ai", "new agent",
        "create agent", "spawn an agent", "create an ai"
    ]

    if any(kw in response for kw in spawn_keywords):
        return "spawn_agents"

    # End the workflow if no spawning needed
    return END


def build_multi_agent_graph() -> StateGraph:
    """
    Build the multi-agent workflow graph.

    Workflow:
    1. collect_responses - All AIs answer the user prompt
    2. roundtable_discussion - AIs debate and discuss
    3. select_winner - Determine best response
    4. execute_tools - Run any requested tools
    5. spawn_agents - Create new agents if requested
    """
    # Create the graph
    graph = StateGraph(MultiAgentState)

    # Add nodes
    graph.add_node("collect_responses", collect_responses)
    graph.add_node("roundtable_discussion", roundtable_discussion)
    graph.add_node("select_winner", select_winner)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("spawn_agents", spawn_agents)

    # Add edges
    graph.set_entry_point("collect_responses")
    graph.add_edge("collect_responses", "roundtable_discussion")
    graph.add_conditional_edges(
        "roundtable_discussion",
        should_continue_discussion,
        {
            "select_winner": "select_winner",
            "continue": "roundtable_discussion"
        }
    )
    graph.add_conditional_edges(
        "select_winner",
        should_execute_tools,
        {
            "execute_tools": "execute_tools",
            "spawn_agents": "spawn_agents"
        }
    )
    graph.add_conditional_edges(
        "execute_tools",
        should_spawn_agents,
        {
            "spawn_agents": "spawn_agents",
            END: END
        }
    )
    graph.add_edge("spawn_agents", END)

    return graph.compile()


class MultiAgentOrchestrator:
    """
    High-level orchestrator for multi-agent conversations.

    Provides a simple interface for running multi-AI discussions.
    """

    def __init__(self):
        self.graph = build_multi_agent_graph()

    async def run(
        self,
        user_prompt: str,
        selected_ais: List[Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a multi-agent discussion.

        Args:
            user_prompt: The user's question/prompt
            selected_ais: List of AI configurations
            user_id: User ID for the session
            session_id: Optional session ID (generated if not provided)

        Returns:
            Result dictionary with responses and discussion log
        """
        session_id = session_id or str(uuid.uuid4())

        # Convert AI configs
        ai_configs = [
            AIConfig(
                id=ai.get("id", str(uuid.uuid4())),
                name=ai.get("name", "AI"),
                owner_id=ai.get("owner_id", user_id),
                system_prompt=ai.get("system_prompt"),
                model=ai.get("model", "gpt-4o")
            )
            for ai in selected_ais
        ]

        # Initial state
        initial_state: MultiAgentState = {
            "user_prompt": user_prompt,
            "selected_ais": ai_configs,
            "messages": [HumanMessage(content=user_prompt)],
            "responses": {},
            "discussion_log": [],
            "winning_response": None,
            "tool_results": [],
            "spawned_agents": [],
            "current_phase": "initial",
            "user_id": user_id,
            "session_id": session_id,
            "errors": []
        }

        # Run the graph
        try:
            final_state = await self.graph.ainvoke(initial_state)

            return {
                "status": "success",
                "session_id": session_id,
                "user_prompt": user_prompt,
                "responses": final_state["responses"],
                "discussion_log": final_state["discussion_log"],
                "winning_response": final_state["winning_response"],
                "spawned_agents": final_state["spawned_agents"],
                "errors": final_state.get("errors", [])
            }

        except Exception as e:
            logger.error(f"Multi-agent workflow failed: {e}")
            return {
                "status": "error",
                "session_id": session_id,
                "error": str(e)
            }

    def run_sync(
        self,
        user_prompt: str,
        selected_ais: List[Dict[str, Any]],
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for run().
        """
        return asyncio.run(self.run(user_prompt, selected_ais, user_id, session_id))


# Convenience function
def create_orchestrator() -> MultiAgentOrchestrator:
    """Create a new MultiAgentOrchestrator instance."""
    return MultiAgentOrchestrator()
