"""Tests for the user prompts module."""

import asyncio
import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from user_prompts import (
    UserPrompt,
    PromptType,
    UserPromptManager,
    get_prompt_manager,
    request_auth,
    request_config,
    resolve_prompt,
    get_pending_prompts,
)


class TestUserPrompt:
    """Tests for the UserPrompt dataclass."""

    def test_create_prompt(self):
        """Test creating a user prompt."""
        prompt = UserPrompt(
            id="test-id",
            prompt_type=PromptType.AUTH,
            message="Enter your API key"
        )
        assert prompt.id == "test-id"
        assert prompt.prompt_type == PromptType.AUTH
        assert prompt.message == "Enter your API key"
        assert not prompt.resolved
        assert prompt.response is None

    def test_is_expired_with_no_expiry(self):
        """Test is_expired when no expiry is set."""
        prompt = UserPrompt(
            id="test-id",
            prompt_type=PromptType.AUTH,
            message="Test"
        )
        assert not prompt.is_expired()

    def test_is_expired_future_expiry(self):
        """Test is_expired with future expiry."""
        prompt = UserPrompt(
            id="test-id",
            prompt_type=PromptType.AUTH,
            message="Test",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert not prompt.is_expired()

    def test_is_expired_past_expiry(self):
        """Test is_expired with past expiry."""
        prompt = UserPrompt(
            id="test-id",
            prompt_type=PromptType.AUTH,
            message="Test",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        assert prompt.is_expired()

    def test_to_dict(self):
        """Test converting prompt to dictionary."""
        prompt = UserPrompt(
            id="test-id",
            prompt_type=PromptType.CONFIG,
            message="Enter config",
            options=["option1", "option2"],
            required_fields=["field1"],
            session_id="session-123"
        )
        result = prompt.to_dict()

        assert result["id"] == "test-id"
        assert result["prompt_type"] == "config"
        assert result["message"] == "Enter config"
        assert result["options"] == ["option1", "option2"]
        assert result["required_fields"] == ["field1"]
        assert result["session_id"] == "session-123"
        assert result["resolved"] is False


class TestUserPromptManager:
    """Tests for the UserPromptManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = UserPromptManager(default_timeout_seconds=60)

    def test_request_auth(self):
        """Test requesting authentication."""
        prompt = self.manager.request_auth(
            message="Enter your API key",
            required_fields=["api_key"],
            session_id="session-1"
        )

        assert prompt.prompt_type == PromptType.AUTH
        assert prompt.message == "Enter your API key"
        assert prompt.required_fields == ["api_key"]
        assert prompt.session_id == "session-1"
        assert not prompt.resolved

    def test_request_config(self):
        """Test requesting configuration."""
        prompt = self.manager.request_config(
            message="Enter settings",
            required_fields=["setting1", "setting2"],
            metadata={"context": "test"}
        )

        assert prompt.prompt_type == PromptType.CONFIG
        assert prompt.required_fields == ["setting1", "setting2"]
        assert prompt.metadata == {"context": "test"}

    def test_request_approval(self):
        """Test requesting approval."""
        prompt = self.manager.request_approval(
            message="Approve this action?",
            workflow_id="workflow-1"
        )

        assert prompt.prompt_type == PromptType.APPROVAL
        assert prompt.options == ["approve", "deny"]
        assert prompt.workflow_id == "workflow-1"

    def test_request_choice(self):
        """Test requesting a choice."""
        options = ["Option A", "Option B", "Option C"]
        prompt = self.manager.request_choice(
            message="Select an option",
            options=options
        )

        assert prompt.prompt_type == PromptType.CHOICE
        assert prompt.options == options

    def test_resolve_prompt(self):
        """Test resolving a prompt."""
        prompt = self.manager.request_auth(
            message="Enter key",
            required_fields=["key"]
        )

        result = self.manager.resolve_prompt(
            prompt.id,
            {"key": "secret-key"}
        )

        assert result is True
        assert prompt.resolved
        assert prompt.response == {"key": "secret-key"}

    def test_resolve_nonexistent_prompt(self):
        """Test resolving a nonexistent prompt."""
        result = self.manager.resolve_prompt("fake-id", "response")
        assert result is False

    def test_resolve_already_resolved_prompt(self):
        """Test resolving an already resolved prompt."""
        prompt = self.manager.request_auth(
            message="Enter key",
            required_fields=["key"]
        )

        self.manager.resolve_prompt(prompt.id, "first-response")
        result = self.manager.resolve_prompt(prompt.id, "second-response")

        assert result is False
        assert prompt.response == "first-response"

    def test_get_prompt(self):
        """Test getting a prompt by ID."""
        prompt = self.manager.request_auth(
            message="Test",
            required_fields=["field"]
        )

        retrieved = self.manager.get_prompt(prompt.id)
        assert retrieved == prompt

    def test_get_prompt_nonexistent(self):
        """Test getting a nonexistent prompt."""
        result = self.manager.get_prompt("fake-id")
        assert result is None

    def test_get_pending_prompts(self):
        """Test getting pending prompts."""
        prompt1 = self.manager.request_auth(
            message="Auth 1",
            required_fields=["key"],
            session_id="session-1"
        )
        prompt2 = self.manager.request_config(
            message="Config 1",
            required_fields=["setting"],
            session_id="session-1"
        )
        prompt3 = self.manager.request_auth(
            message="Auth 2",
            required_fields=["key"],
            session_id="session-2"
        )

        # Resolve one prompt
        self.manager.resolve_prompt(prompt2.id, {"setting": "value"})

        # Get pending prompts for session-1
        pending = self.manager.get_pending_prompts(session_id="session-1")

        assert len(pending) == 1
        assert pending[0].id == prompt1.id

    def test_get_pending_prompts_by_workflow(self):
        """Test getting pending prompts by workflow ID."""
        prompt1 = self.manager.request_auth(
            message="Auth 1",
            required_fields=["key"],
            workflow_id="workflow-1"
        )
        prompt2 = self.manager.request_auth(
            message="Auth 2",
            required_fields=["key"],
            workflow_id="workflow-2"
        )

        pending = self.manager.get_pending_prompts(workflow_id="workflow-1")

        assert len(pending) == 1
        assert pending[0].id == prompt1.id

    def test_on_resolved_callback(self):
        """Test callback on prompt resolution."""
        callback_called = []

        prompt = self.manager.request_auth(
            message="Test",
            required_fields=["key"]
        )

        self.manager.on_resolved(
            prompt.id,
            lambda p: callback_called.append(p.response)
        )

        self.manager.resolve_prompt(prompt.id, "test-response")

        assert len(callback_called) == 1
        assert callback_called[0] == "test-response"

    @pytest.mark.asyncio
    async def test_wait_for_resolution(self):
        """Test async waiting for resolution."""
        prompt = self.manager.request_auth(
            message="Test",
            required_fields=["key"]
        )

        # Resolve in a separate task after a short delay
        async def resolve_later():
            await asyncio.sleep(0.1)
            self.manager.resolve_prompt(prompt.id, "async-response")

        asyncio.create_task(resolve_later())

        result = await self.manager.wait_for_resolution(prompt.id)

        assert result == "async-response"

    def test_cleanup_expired(self):
        """Test cleaning up expired prompts."""
        # Create a prompt that's already expired
        prompt = self.manager.request_auth(
            message="Test",
            required_fields=["key"],
            timeout_seconds=-1  # Already expired
        )

        # Create a resolved prompt
        prompt2 = self.manager.request_auth(
            message="Test 2",
            required_fields=["key"]
        )
        self.manager.resolve_prompt(prompt2.id, "response")

        # Create a pending prompt
        prompt3 = self.manager.request_auth(
            message="Test 3",
            required_fields=["key"]
        )

        cleaned = self.manager.cleanup_expired()

        assert cleaned == 2  # Expired + resolved
        assert self.manager.get_prompt(prompt3.id) is not None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_request_auth_convenience(self):
        """Test the request_auth convenience function."""
        prompt = request_auth(
            message="Enter key",
            required_fields=["api_key"]
        )

        assert prompt.prompt_type == PromptType.AUTH
        assert prompt.message == "Enter key"

    def test_request_config_convenience(self):
        """Test the request_config convenience function."""
        prompt = request_config(
            message="Enter settings",
            required_fields=["setting"]
        )

        assert prompt.prompt_type == PromptType.CONFIG

    def test_resolve_prompt_convenience(self):
        """Test the resolve_prompt convenience function."""
        prompt = request_auth(
            message="Test",
            required_fields=["key"]
        )

        result = resolve_prompt(prompt.id, "response")
        assert result is True

    def test_get_pending_prompts_convenience(self):
        """Test the get_pending_prompts convenience function."""
        request_auth(
            message="Test",
            required_fields=["key"],
            session_id="test-session"
        )

        pending = get_pending_prompts(session_id="test-session")
        assert len(pending) >= 1

    def test_get_prompt_manager_singleton(self):
        """Test that get_prompt_manager returns a singleton."""
        manager1 = get_prompt_manager()
        manager2 = get_prompt_manager()

        assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
