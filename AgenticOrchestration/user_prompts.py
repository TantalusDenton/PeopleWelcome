"""User prompt system for requesting authentication and configuration during workflows.

This module provides a mechanism for multi-agent workflows to pause and request
user input when authentication or configuration is needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts that can be requested from users."""
    AUTH = "auth"  # Authentication credentials
    CONFIG = "config"  # Configuration values
    APPROVAL = "approval"  # Approval for an action
    CHOICE = "choice"  # Selection from options


@dataclass
class UserPrompt:
    """Represents a prompt request to the user."""
    id: str
    prompt_type: PromptType
    message: str
    options: Optional[List[str]] = None
    required_fields: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    resolved: bool = False
    response: Optional[Any] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if the prompt has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert prompt to dictionary for API responses."""
        return {
            "id": self.id,
            "prompt_type": self.prompt_type.value,
            "message": self.message,
            "options": self.options,
            "required_fields": self.required_fields,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "resolved": self.resolved,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
        }


class UserPromptManager:
    """
    Manages user prompts for authentication and configuration requests.

    This class provides a thread-safe way to create, track, and resolve
    user prompts during multi-agent workflows.
    """

    def __init__(self, default_timeout_seconds: int = 300):
        """
        Initialize the prompt manager.

        Args:
            default_timeout_seconds: Default time before prompts expire (5 minutes)
        """
        self._prompts: Dict[str, UserPrompt] = {}
        self._lock = Lock()
        self._default_timeout = default_timeout_seconds
        self._callbacks: Dict[str, List[Callable[[UserPrompt], None]]] = {}

    def request_auth(
        self,
        message: str,
        required_fields: List[str],
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> UserPrompt:
        """
        Request authentication credentials from the user.

        Args:
            message: Message to display to the user
            required_fields: List of required authentication fields (e.g., ["api_key", "secret"])
            session_id: Optional session identifier
            workflow_id: Optional workflow identifier
            metadata: Optional additional metadata
            timeout_seconds: Time before prompt expires

        Returns:
            UserPrompt object representing the request
        """
        return self._create_prompt(
            prompt_type=PromptType.AUTH,
            message=message,
            required_fields=required_fields,
            session_id=session_id,
            workflow_id=workflow_id,
            metadata=metadata,
            timeout_seconds=timeout_seconds
        )

    def request_config(
        self,
        message: str,
        required_fields: List[str],
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> UserPrompt:
        """
        Request configuration values from the user.

        Args:
            message: Message to display to the user
            required_fields: List of required configuration fields
            session_id: Optional session identifier
            workflow_id: Optional workflow identifier
            metadata: Optional additional metadata
            timeout_seconds: Time before prompt expires

        Returns:
            UserPrompt object representing the request
        """
        return self._create_prompt(
            prompt_type=PromptType.CONFIG,
            message=message,
            required_fields=required_fields,
            session_id=session_id,
            workflow_id=workflow_id,
            metadata=metadata,
            timeout_seconds=timeout_seconds
        )

    def request_approval(
        self,
        message: str,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> UserPrompt:
        """
        Request approval for an action from the user.

        Args:
            message: Description of the action requiring approval
            session_id: Optional session identifier
            workflow_id: Optional workflow identifier
            metadata: Optional additional metadata
            timeout_seconds: Time before prompt expires

        Returns:
            UserPrompt object representing the request
        """
        return self._create_prompt(
            prompt_type=PromptType.APPROVAL,
            message=message,
            options=["approve", "deny"],
            session_id=session_id,
            workflow_id=workflow_id,
            metadata=metadata,
            timeout_seconds=timeout_seconds
        )

    def request_choice(
        self,
        message: str,
        options: List[str],
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> UserPrompt:
        """
        Request the user to choose from a list of options.

        Args:
            message: Message to display to the user
            options: List of options to choose from
            session_id: Optional session identifier
            workflow_id: Optional workflow identifier
            metadata: Optional additional metadata
            timeout_seconds: Time before prompt expires

        Returns:
            UserPrompt object representing the request
        """
        return self._create_prompt(
            prompt_type=PromptType.CHOICE,
            message=message,
            options=options,
            session_id=session_id,
            workflow_id=workflow_id,
            metadata=metadata,
            timeout_seconds=timeout_seconds
        )

    def _create_prompt(
        self,
        prompt_type: PromptType,
        message: str,
        required_fields: Optional[List[str]] = None,
        options: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None
    ) -> UserPrompt:
        """Create a new user prompt."""
        prompt_id = str(uuid.uuid4())
        timeout = timeout_seconds or self._default_timeout
        expires_at = datetime.utcnow() + timedelta(seconds=timeout)

        prompt = UserPrompt(
            id=prompt_id,
            prompt_type=prompt_type,
            message=message,
            options=options,
            required_fields=required_fields,
            metadata=metadata or {},
            expires_at=expires_at,
            session_id=session_id,
            workflow_id=workflow_id
        )

        with self._lock:
            self._prompts[prompt_id] = prompt

        logger.info(f"Created {prompt_type.value} prompt {prompt_id}")
        return prompt

    def resolve_prompt(
        self,
        prompt_id: str,
        response: Any
    ) -> bool:
        """
        Resolve a prompt with the user's response.

        Args:
            prompt_id: ID of the prompt to resolve
            response: User's response to the prompt

        Returns:
            True if prompt was resolved successfully
        """
        with self._lock:
            prompt = self._prompts.get(prompt_id)

            if not prompt:
                logger.warning(f"Prompt {prompt_id} not found")
                return False

            if prompt.resolved:
                logger.warning(f"Prompt {prompt_id} already resolved")
                return False

            if prompt.is_expired():
                logger.warning(f"Prompt {prompt_id} has expired")
                return False

            prompt.resolved = True
            prompt.response = response

            # Trigger callbacks
            callbacks = self._callbacks.get(prompt_id, [])
            for callback in callbacks:
                try:
                    callback(prompt)
                except Exception as e:
                    logger.error(f"Callback error for prompt {prompt_id}: {e}")

        logger.info(f"Resolved prompt {prompt_id}")
        return True

    def get_prompt(self, prompt_id: str) -> Optional[UserPrompt]:
        """Get a prompt by ID."""
        with self._lock:
            return self._prompts.get(prompt_id)

    def get_pending_prompts(
        self,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> List[UserPrompt]:
        """
        Get all pending (unresolved, unexpired) prompts.

        Args:
            session_id: Filter by session ID
            workflow_id: Filter by workflow ID

        Returns:
            List of pending prompts
        """
        with self._lock:
            prompts = []
            for prompt in self._prompts.values():
                if prompt.resolved or prompt.is_expired():
                    continue
                if session_id and prompt.session_id != session_id:
                    continue
                if workflow_id and prompt.workflow_id != workflow_id:
                    continue
                prompts.append(prompt)
            return prompts

    def on_resolved(
        self,
        prompt_id: str,
        callback: Callable[[UserPrompt], None]
    ) -> None:
        """
        Register a callback for when a prompt is resolved.

        Args:
            prompt_id: ID of the prompt
            callback: Function to call when resolved
        """
        with self._lock:
            if prompt_id not in self._callbacks:
                self._callbacks[prompt_id] = []
            self._callbacks[prompt_id].append(callback)

    async def wait_for_resolution(
        self,
        prompt_id: str,
        poll_interval: float = 0.5
    ) -> Optional[Any]:
        """
        Asynchronously wait for a prompt to be resolved.

        Args:
            prompt_id: ID of the prompt to wait for
            poll_interval: Seconds between checks

        Returns:
            The user's response, or None if expired
        """
        while True:
            prompt = self.get_prompt(prompt_id)

            if not prompt:
                return None

            if prompt.resolved:
                return prompt.response

            if prompt.is_expired():
                logger.warning(f"Prompt {prompt_id} expired while waiting")
                return None

            await asyncio.sleep(poll_interval)

    def cleanup_expired(self) -> int:
        """
        Remove expired prompts.

        Returns:
            Number of prompts removed
        """
        with self._lock:
            expired_ids = [
                pid for pid, prompt in self._prompts.items()
                if prompt.is_expired() or prompt.resolved
            ]
            for pid in expired_ids:
                del self._prompts[pid]
                if pid in self._callbacks:
                    del self._callbacks[pid]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired prompts")

        return len(expired_ids)


# Singleton instance
_prompt_manager: Optional[UserPromptManager] = None


def get_prompt_manager() -> UserPromptManager:
    """Get or create the singleton prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = UserPromptManager()
    return _prompt_manager


def request_auth(
    message: str,
    required_fields: List[str],
    **kwargs
) -> UserPrompt:
    """Convenience function to request authentication."""
    return get_prompt_manager().request_auth(message, required_fields, **kwargs)


def request_config(
    message: str,
    required_fields: List[str],
    **kwargs
) -> UserPrompt:
    """Convenience function to request configuration."""
    return get_prompt_manager().request_config(message, required_fields, **kwargs)


def resolve_prompt(prompt_id: str, response: Any) -> bool:
    """Convenience function to resolve a prompt."""
    return get_prompt_manager().resolve_prompt(prompt_id, response)


def get_pending_prompts(**kwargs) -> List[UserPrompt]:
    """Convenience function to get pending prompts."""
    return get_prompt_manager().get_pending_prompts(**kwargs)
