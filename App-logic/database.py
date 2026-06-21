"""SQLite database schema and connection management for PeopleWelcome platform."""

import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

# Database path from environment or default
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "./data/peoplewelcome.db"))


def get_database_path() -> Path:
    """Get the configured database path, creating parent directories if needed."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DATABASE_PATH


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections with row factory."""
    conn = sqlite3.connect(str(get_database_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Initialize the database with all required tables."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # AIs table - stores AI personas with their configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ais (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                name TEXT NOT NULL,
                is_public BOOLEAN DEFAULT FALSE,
                system_prompt TEXT,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_id) REFERENCES users(id),
                UNIQUE(owner_id, name)
            )
        """)

        # Images table - stores image metadata with S3 references
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                ai_id TEXT NOT NULL,
                s3_key TEXT NOT NULL,
                is_public BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (owner_id) REFERENCES users(id),
                FOREIGN KEY (ai_id) REFERENCES ais(id)
            )
        """)

        # Tags table - stores image tags with AI associations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                ai_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id),
                FOREIGN KEY (ai_id) REFERENCES ais(id),
                UNIQUE(image_id, ai_id, tag)
            )
        """)

        # AI conversations table - stores chat history per AI
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ai_id) REFERENCES ais(id)
            )
        """)

        # Spawned agents table - persistent dynamically created agents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spawned_agents (
                id TEXT PRIMARY KEY,
                parent_ai_id TEXT,
                name TEXT NOT NULL,
                tool_code TEXT,
                config JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_ai_id) REFERENCES ais(id)
            )
        """)

        # Custom integrations table - RAG, tools, workflows
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS custom_integrations (
                id TEXT PRIMARY KEY,
                ai_id TEXT NOT NULL,
                integration_type TEXT NOT NULL,
                name TEXT NOT NULL,
                code TEXT,
                config JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ai_id) REFERENCES ais(id)
            )
        """)

        # Training jobs table - tracks classifier training status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ai_id) REFERENCES ais(id)
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ais_owner ON ais(owner_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ais_public ON ais(is_public)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_owner ON images(owner_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_ai ON images(ai_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_image ON tags(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_ai ON tags(ai_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_ai ON ai_conversations(ai_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_ai ON training_jobs(ai_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)")

        conn.commit()


def reset_database() -> None:
    """Drop all tables and reinitialize. USE WITH CAUTION."""
    with get_connection() as conn:
        cursor = conn.cursor()
        tables = [
            "custom_integrations",
            "spawned_agents",
            "ai_conversations",
            "tags",
            "images",
            "training_jobs",
            "ais",
            "users"
        ]
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
    init_database()


if __name__ == "__main__":
    init_database()
    print(f"Database initialized at {get_database_path()}")
