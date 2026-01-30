"""Tests for the multi-agent LangGraph workflow."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent_graph import (
    MultiAgentState,
    AIConfig,
    collect_responses,
    roundtable_discussion,
    select_winner,
    execute_tools,
    spawn_agents,
    should_execute_tools,
    should_spawn_agents,
    build_multi_agent_graph,
    MultiAgentOrchestrator,
)


class TestAIConfig:
    """Tests for AIConfig dataclass."""

    def test_create_ai_config(self):
        """Test creating an AIConfig."""
        config = AIConfig(
            id="test-id",
            name="Test AI",
            owner_id="user-123",
            system_prompt="Be helpful.",
            model="gpt-4o"
        )
        assert config.id == "test-id"
        assert config.name == "Test AI"
        assert config.owner_id == "user-123"
        assert config.system_prompt == "Be helpful."
        assert config.model == "gpt-4o"

    def test_default_values(self):
        """Test AIConfig default values."""
        config = AIConfig(
            id="test-id",
            name="Test AI",
            owner_id="user-123"
        )
        assert config.system_prompt is None
        assert config.model == "gpt-4o"


class TestEdgeConditions:
    """Tests for workflow edge conditions."""

    def test_should_execute_tools_with_search(self):
        """Test that search keywords trigger tool execution."""
        state = {"winning_response": "I should search for the answer online."}
        assert should_execute_tools(state) == "execute_tools"

    def test_should_execute_tools_with_workflow(self):
        """Test that workflow keywords trigger tool execution."""
        state = {"winning_response": "Let me create a workflow for this task."}
        assert should_execute_tools(state) == "execute_tools"

    def test_should_execute_tools_no_keywords(self):
        """Test that absence of keywords skips tool execution."""
        state = {"winning_response": "Here is the answer to your question."}
        assert should_execute_tools(state) == "spawn_agents"

    def test_should_spawn_agents_with_spawn(self):
        """Test that spawn keywords trigger agent spawning."""
        state = {"winning_response": "I'll spawn an agent to handle this."}
        assert should_spawn_agents(state) == "spawn_agents"

    def test_should_spawn_agents_with_create_ai(self):
        """Test that create AI keywords trigger agent spawning."""
        state = {"winning_response": "Let me create an AI for this purpose."}
        assert should_spawn_agents(state) == "spawn_agents"

    def test_should_spawn_agents_no_keywords(self):
        """Test that absence of keywords ends workflow."""
        from langgraph.graph import END
        state = {"winning_response": "Task completed successfully."}
        assert should_spawn_agents(state) == END


class TestCollectResponses:
    """Tests for the collect_responses node."""

    @pytest.mark.asyncio
    @patch('multi_agent_graph.create_llm')
    async def test_collect_responses_single_ai(self, mock_create_llm):
        """Test collecting response from a single AI."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Test response"))
        mock_create_llm.return_value = mock_llm

        state = {
            "selected_ais": [
                AIConfig(id="ai-1", name="AI 1", owner_id="user-1")
            ],
            "user_prompt": "What is 2+2?",
            "responses": {},
            "current_phase": "initial"
        }

        result = await collect_responses(state)

        assert "ai-1" in result["responses"]
        assert result["responses"]["ai-1"] == "Test response"
        assert result["current_phase"] == "responses_collected"

    @pytest.mark.asyncio
    @patch('multi_agent_graph.create_llm')
    async def test_collect_responses_multiple_ais(self, mock_create_llm):
        """Test collecting responses from multiple AIs."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            MagicMock(content="Response 1"),
            MagicMock(content="Response 2"),
        ])
        mock_create_llm.return_value = mock_llm

        state = {
            "selected_ais": [
                AIConfig(id="ai-1", name="AI 1", owner_id="user-1"),
                AIConfig(id="ai-2", name="AI 2", owner_id="user-1"),
            ],
            "user_prompt": "What is 2+2?",
            "responses": {},
            "current_phase": "initial"
        }

        result = await collect_responses(state)

        assert len(result["responses"]) == 2
        assert "ai-1" in result["responses"]
        assert "ai-2" in result["responses"]


class TestSelectWinner:
    """Tests for the select_winner node."""

    @pytest.mark.asyncio
    async def test_select_winner_single_ai(self):
        """Test selecting winner with single AI (returns its response directly)."""
        state = {
            "selected_ais": [
                AIConfig(id="ai-1", name="AI 1", owner_id="user-1")
            ],
            "responses": {"ai-1": "The answer is 4."},
            "discussion_log": [],
            "user_prompt": "What is 2+2?",
        }

        result = await select_winner(state)

        assert result["winning_response"] == "The answer is 4."
        assert result["current_phase"] == "winner_selected"

    @pytest.mark.asyncio
    @patch('multi_agent_graph.create_llm')
    async def test_select_winner_multiple_ais(self, mock_create_llm):
        """Test selecting winner with multiple AIs (synthesis)."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Synthesized answer"))
        mock_create_llm.return_value = mock_llm

        state = {
            "selected_ais": [
                AIConfig(id="ai-1", name="AI 1", owner_id="user-1"),
                AIConfig(id="ai-2", name="AI 2", owner_id="user-1"),
            ],
            "responses": {
                "ai-1": "Answer from AI 1",
                "ai-2": "Answer from AI 2"
            },
            "discussion_log": [
                {"ai_name": "AI 1", "content": "I agree."}
            ],
            "user_prompt": "What is 2+2?",
        }

        result = await select_winner(state)

        assert result["winning_response"] == "Synthesized answer"
        assert result["current_phase"] == "winner_selected"


class TestExecuteTools:
    """Tests for the execute_tools node."""

    @pytest.mark.asyncio
    @patch('multi_agent_graph.get_tool_by_name')
    async def test_execute_tools_web_search(self, mock_get_tool):
        """Test executing web search tool."""
        mock_tool = MagicMock()
        mock_tool._run.return_value = "Search results"
        mock_get_tool.return_value = mock_tool

        state = {
            "winning_response": "Let me search for: Python tutorials",
            "tool_results": []
        }

        result = await execute_tools(state)

        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "web_search"
        assert result["tool_results"][0]["status"] == "success"
        assert result["current_phase"] == "tools_executed"

    @pytest.mark.asyncio
    @patch('multi_agent_graph.get_tool_by_name')
    async def test_execute_tools_no_tools_needed(self, mock_get_tool):
        """Test when no tools are needed."""
        state = {
            "winning_response": "Here is a simple answer.",
            "tool_results": []
        }

        result = await execute_tools(state)

        assert len(result["tool_results"]) == 0
        assert result["current_phase"] == "tools_executed"


class TestSpawnAgents:
    """Tests for the spawn_agents node."""

    @pytest.mark.asyncio
    @patch('multi_agent_graph.db_service')
    async def test_spawn_agents_creates_agent(self, mock_db_service):
        """Test spawning an agent from response."""
        mock_db_service.create_spawned_agent.return_value = {"id": "spawned-1"}

        state = {
            "winning_response": "I'll spawn an agent: DataAnalyzer for processing data.",
            "selected_ais": [
                AIConfig(id="ai-1", name="AI 1", owner_id="user-1")
            ],
            "spawned_agents": []
        }

        result = await spawn_agents(state)

        assert len(result["spawned_agents"]) == 1
        assert result["spawned_agents"][0] == "spawned-1"
        assert result["current_phase"] == "agents_spawned"

    @pytest.mark.asyncio
    async def test_spawn_agents_no_spawn_needed(self):
        """Test when no agents need to be spawned."""
        state = {
            "winning_response": "Task completed without needing new agents.",
            "selected_ais": [],
            "spawned_agents": []
        }

        result = await spawn_agents(state)

        assert len(result["spawned_agents"]) == 0
        assert result["current_phase"] == "agents_spawned"


class TestBuildMultiAgentGraph:
    """Tests for the graph building function."""

    def test_build_graph_returns_compiled_graph(self):
        """Test that build_multi_agent_graph returns a compiled graph."""
        graph = build_multi_agent_graph()
        assert graph is not None


class TestMultiAgentOrchestrator:
    """Tests for the MultiAgentOrchestrator class."""

    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly."""
        orchestrator = MultiAgentOrchestrator()
        assert orchestrator.graph is not None

    @pytest.mark.asyncio
    @patch('multi_agent_graph.build_multi_agent_graph')
    async def test_orchestrator_run(self, mock_build_graph):
        """Test running the orchestrator."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={
            "winning_response": "Test response",
            "responses": {"ai-1": "Response"},
            "discussion_log": [],
            "spawned_agents": [],
            "errors": []
        })
        mock_build_graph.return_value = mock_graph

        orchestrator = MultiAgentOrchestrator()
        orchestrator.graph = mock_graph

        result = await orchestrator.run(
            user_prompt="Test prompt",
            selected_ais=[{"id": "ai-1", "name": "AI 1", "owner_id": "user-1"}],
            user_id="user-1"
        )

        assert result["status"] == "success"
        assert "winning_response" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
