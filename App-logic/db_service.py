"""Database service layer with CRUD operations for all entities."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from database import get_connection, init_database


# ============================================================================
# User Operations
# ============================================================================

def create_user(user_id: str, username: str) -> Dict[str, Any]:
    """Create a new user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (id, username) VALUES (?, ?)",
            (user_id, username)
        )
        return {"id": user_id, "username": username}


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Get a user by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get a user by username."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_or_create_user(user_id: str, username: str) -> Dict[str, Any]:
    """Get existing user or create new one."""
    user = get_user(user_id)
    if user:
        return user
    return create_user(user_id, username)


# ============================================================================
# AI Operations
# ============================================================================

def create_ai(
    owner_id: str,
    name: str,
    is_public: bool = False,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new AI for a user."""
    ai_id = str(uuid.uuid4())
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO ais (id, owner_id, name, is_public, system_prompt)
               VALUES (?, ?, ?, ?, ?)""",
            (ai_id, owner_id, name, is_public, system_prompt)
        )
        return {
            "id": ai_id,
            "owner_id": owner_id,
            "name": name,
            "is_public": is_public,
            "system_prompt": system_prompt
        }


def get_ai(ai_id: str) -> Optional[Dict[str, Any]]:
    """Get an AI by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ais WHERE id = ?", (ai_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_ai_by_owner_and_name(owner_id: str, name: str) -> Optional[Dict[str, Any]]:
    """Get an AI by owner and name."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM ais WHERE owner_id = ? AND name = ?",
            (owner_id, name)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_ais_by_owner(owner_id: str) -> List[Dict[str, Any]]:
    """Get all AIs owned by a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM ais WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_public_ais() -> List[Dict[str, Any]]:
    """Get all public AIs."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT a.*, u.username as owner_username
               FROM ais a
               JOIN users u ON a.owner_id = u.id
               WHERE a.is_public = TRUE
               ORDER BY a.created_at DESC"""
        )
        return [dict(row) for row in cursor.fetchall()]


def update_ai_system_prompt(ai_id: str, system_prompt: str) -> bool:
    """Update an AI's system prompt."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE ais SET system_prompt = ? WHERE id = ?",
            (system_prompt, ai_id)
        )
        return cursor.rowcount > 0


def update_ai_model_path(ai_id: str, model_path: str) -> bool:
    """Update an AI's trained model path."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE ais SET model_path = ? WHERE id = ?",
            (model_path, ai_id)
        )
        return cursor.rowcount > 0


def update_ai_visibility(ai_id: str, is_public: bool) -> bool:
    """Update an AI's public visibility."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE ais SET is_public = ? WHERE id = ?",
            (is_public, ai_id)
        )
        return cursor.rowcount > 0


def delete_ai(ai_id: str) -> bool:
    """Delete an AI and all related data."""
    with get_connection() as conn:
        cursor = conn.cursor()
        # Delete related data first (foreign key constraints)
        cursor.execute("DELETE FROM ai_conversations WHERE ai_id = ?", (ai_id,))
        cursor.execute("DELETE FROM custom_integrations WHERE ai_id = ?", (ai_id,))
        cursor.execute("DELETE FROM spawned_agents WHERE parent_ai_id = ?", (ai_id,))
        cursor.execute("DELETE FROM training_jobs WHERE ai_id = ?", (ai_id,))
        cursor.execute("DELETE FROM tags WHERE ai_id = ?", (ai_id,))
        cursor.execute("DELETE FROM images WHERE ai_id = ?", (ai_id,))
        cursor.execute("DELETE FROM ais WHERE id = ?", (ai_id,))
        return cursor.rowcount > 0


# ============================================================================
# Image Operations
# ============================================================================

def create_image(
    owner_id: str,
    ai_id: str,
    s3_key: str,
    is_public: bool = False
) -> Dict[str, Any]:
    """Create a new image record."""
    image_id = str(uuid.uuid4())
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO images (id, owner_id, ai_id, s3_key, is_public)
               VALUES (?, ?, ?, ?, ?)""",
            (image_id, owner_id, ai_id, s3_key, is_public)
        )
        return {
            "id": image_id,
            "owner_id": owner_id,
            "ai_id": ai_id,
            "s3_key": s3_key,
            "is_public": is_public
        }


def get_image(image_id: str) -> Optional[Dict[str, Any]]:
    """Get an image by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_images_by_ai(ai_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Get paginated images for an AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM images
               WHERE ai_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (ai_id, limit, offset)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_images_by_owner(owner_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Get paginated images owned by a user."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM images
               WHERE owner_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (owner_id, limit, offset)
        )
        return [dict(row) for row in cursor.fetchall()]


def delete_image(image_id: str) -> bool:
    """Delete an image and its tags."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tags WHERE image_id = ?", (image_id,))
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        return cursor.rowcount > 0


# ============================================================================
# Tag Operations
# ============================================================================

def add_tag(image_id: str, ai_id: str, tag: str) -> Dict[str, Any]:
    """Add a tag to an image by an AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR IGNORE INTO tags (image_id, ai_id, tag)
               VALUES (?, ?, ?)""",
            (image_id, ai_id, tag)
        )
        cursor.execute(
            "SELECT * FROM tags WHERE image_id = ? AND ai_id = ? AND tag = ?",
            (image_id, ai_id, tag)
        )
        row = cursor.fetchone()
        return dict(row) if row else {"image_id": image_id, "ai_id": ai_id, "tag": tag}


def remove_tag(image_id: str, ai_id: str, tag: str) -> bool:
    """Remove a tag from an image."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM tags WHERE image_id = ? AND ai_id = ? AND tag = ?",
            (image_id, ai_id, tag)
        )
        return cursor.rowcount > 0


def get_tags_for_image(image_id: str) -> List[Dict[str, Any]]:
    """Get all tags for an image across all AIs."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT t.*, a.name as ai_name
               FROM tags t
               JOIN ais a ON t.ai_id = a.id
               WHERE t.image_id = ?""",
            (image_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_tags_for_image_by_ai(image_id: str, ai_id: str) -> List[str]:
    """Get tags for an image by a specific AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT tag FROM tags WHERE image_id = ? AND ai_id = ?",
            (image_id, ai_id)
        )
        return [row["tag"] for row in cursor.fetchall()]


def get_all_tags_by_ai(ai_id: str) -> List[Dict[str, Any]]:
    """Get all unique tags used by an AI with image counts."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT tag, COUNT(*) as count
               FROM tags
               WHERE ai_id = ?
               GROUP BY tag
               ORDER BY count DESC""",
            (ai_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_images_with_tag(ai_id: str, tag: str) -> List[Dict[str, Any]]:
    """Get all images that have a specific tag from an AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT i.*
               FROM images i
               JOIN tags t ON i.id = t.image_id
               WHERE t.ai_id = ? AND t.tag = ?""",
            (ai_id, tag)
        )
        return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# Conversation Operations
# ============================================================================

def add_conversation_message(
    ai_id: str,
    user_id: str,
    role: str,
    content: str
) -> Dict[str, Any]:
    """Add a message to an AI conversation."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO ai_conversations (ai_id, user_id, role, content)
               VALUES (?, ?, ?, ?)""",
            (ai_id, user_id, role, content)
        )
        return {
            "id": cursor.lastrowid,
            "ai_id": ai_id,
            "user_id": user_id,
            "role": role,
            "content": content
        }


def get_conversation_history(
    ai_id: str,
    user_id: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get conversation history between a user and AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM ai_conversations
               WHERE ai_id = ? AND user_id = ?
               ORDER BY created_at ASC
               LIMIT ?""",
            (ai_id, user_id, limit)
        )
        return [dict(row) for row in cursor.fetchall()]


def clear_conversation_history(ai_id: str, user_id: str) -> int:
    """Clear conversation history between a user and AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM ai_conversations WHERE ai_id = ? AND user_id = ?",
            (ai_id, user_id)
        )
        return cursor.rowcount


# ============================================================================
# Spawned Agent Operations
# ============================================================================

def create_spawned_agent(
    name: str,
    parent_ai_id: Optional[str] = None,
    tool_code: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a new spawned agent."""
    agent_id = str(uuid.uuid4())
    config_json = json.dumps(config) if config else None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO spawned_agents (id, parent_ai_id, name, tool_code, config)
               VALUES (?, ?, ?, ?, ?)""",
            (agent_id, parent_ai_id, name, tool_code, config_json)
        )
        return {
            "id": agent_id,
            "parent_ai_id": parent_ai_id,
            "name": name,
            "tool_code": tool_code,
            "config": config
        }


def get_spawned_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """Get a spawned agent by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM spawned_agents WHERE id = ?", (agent_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result.get("config"):
                result["config"] = json.loads(result["config"])
            return result
        return None


def get_all_spawned_agents() -> List[Dict[str, Any]]:
    """Get all spawned agents."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM spawned_agents ORDER BY created_at DESC")
        agents = []
        for row in cursor.fetchall():
            agent = dict(row)
            if agent.get("config"):
                agent["config"] = json.loads(agent["config"])
            agents.append(agent)
        return agents


def delete_spawned_agent(agent_id: str) -> bool:
    """Delete a spawned agent."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM spawned_agents WHERE id = ?", (agent_id,))
        return cursor.rowcount > 0


# ============================================================================
# Custom Integration Operations
# ============================================================================

def create_custom_integration(
    ai_id: str,
    integration_type: str,
    name: str,
    code: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a new custom integration."""
    integration_id = str(uuid.uuid4())
    config_json = json.dumps(config) if config else None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO custom_integrations
               (id, ai_id, integration_type, name, code, config)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (integration_id, ai_id, integration_type, name, code, config_json)
        )
        return {
            "id": integration_id,
            "ai_id": ai_id,
            "integration_type": integration_type,
            "name": name,
            "code": code,
            "config": config
        }


def get_integrations_by_ai(ai_id: str) -> List[Dict[str, Any]]:
    """Get all custom integrations for an AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM custom_integrations WHERE ai_id = ? ORDER BY created_at DESC",
            (ai_id,)
        )
        integrations = []
        for row in cursor.fetchall():
            integration = dict(row)
            if integration.get("config"):
                integration["config"] = json.loads(integration["config"])
            integrations.append(integration)
        return integrations


def delete_custom_integration(integration_id: str) -> bool:
    """Delete a custom integration."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM custom_integrations WHERE id = ?", (integration_id,))
        return cursor.rowcount > 0


# ============================================================================
# Training Job Operations
# ============================================================================

def create_training_job(ai_id: str) -> Dict[str, Any]:
    """Create a new training job."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_jobs (ai_id, status) VALUES (?, 'pending')",
            (ai_id,)
        )
        return {
            "id": cursor.lastrowid,
            "ai_id": ai_id,
            "status": "pending"
        }


def get_pending_training_jobs() -> List[Dict[str, Any]]:
    """Get all pending training jobs."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM training_jobs
               WHERE status = 'pending'
               ORDER BY created_at ASC"""
        )
        return [dict(row) for row in cursor.fetchall()]


def update_training_job_status(
    job_id: int,
    status: str,
    error_message: Optional[str] = None
) -> bool:
    """Update training job status."""
    with get_connection() as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        if status == "training":
            cursor.execute(
                "UPDATE training_jobs SET status = ?, started_at = ? WHERE id = ?",
                (status, now, job_id)
            )
        elif status in ("completed", "failed"):
            cursor.execute(
                """UPDATE training_jobs
                   SET status = ?, completed_at = ?, error_message = ?
                   WHERE id = ?""",
                (status, now, error_message, job_id)
            )
        else:
            cursor.execute(
                "UPDATE training_jobs SET status = ? WHERE id = ?",
                (status, job_id)
            )
        return cursor.rowcount > 0


def get_latest_training_job(ai_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent training job for an AI."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT * FROM training_jobs
               WHERE ai_id = ?
               ORDER BY created_at DESC
               LIMIT 1""",
            (ai_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


# Initialize database on import
init_database()
