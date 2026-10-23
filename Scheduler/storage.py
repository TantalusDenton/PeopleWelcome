"""PostgreSQL persistence for PeopleWelcome."""
from __future__ import annotations

import sqlite3
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


ACTIVE_SUBSCRIPTION_STATUSES = {"active", "trialing"}
ADMIN_EMAILS = {
    email.strip().lower()
    for email in os.getenv("PEOPLEWELCOME_ADMIN_EMAILS", "").split(",")
    if email.strip()
}


class Store:
    def __init__(self, database_url: str):
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            open=False,
        )

    def initialize(self) -> None:
        self.pool.open(wait=True)
        with self.pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    email TEXT,
                    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                    stripe_customer_id TEXT UNIQUE,
                    stripe_subscription_id TEXT UNIQUE,
                    subscription_status TEXT NOT NULL DEFAULT 'inactive',
                    subscription_current_period_end TIMESTAMPTZ,
                    cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS ais (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    persona TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT 'openai' CHECK (model IN ('openai', 'unstoppable')),
                    is_public BOOLEAN NOT NULL DEFAULT FALSE,
                    original_avatar_url TEXT,
                    avatar_front_face_url TEXT,
                    avatar_side_face_url TEXT,
                    full_body_front_url TEXT,
                    full_body_side_url TEXT,
                    full_body_back_url TEXT,
                    profile_image_url TEXT,
                    image_generation_status TEXT NOT NULL DEFAULT 'not_requested',
                    image_generation_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(owner_id, name)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGSERIAL PRIMARY KEY,
                    ai_id TEXT NOT NULL REFERENCES ais(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_ais_owner_created ON ais(owner_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(ai_id, user_id, id);
                ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;
                """
            )

    def close(self) -> None:
        self.pool.close()

    @staticmethod
    def _premium(user: dict | None) -> dict | None:
        if not user:
            return None
        period_end = user.get("subscription_current_period_end")
        if isinstance(period_end, datetime) and period_end.tzinfo is None:
            period_end = period_end.replace(tzinfo=timezone.utc)
        current = period_end is None or period_end > datetime.now(timezone.utc)
        user["is_premium"] = bool(user.get("is_admin")) or (
            user.get("subscription_status") in ACTIVE_SUBSCRIPTION_STATUSES and current
        )
        return user

    def upsert_user(self, user_id: str, username: str, email: str | None = None) -> dict:
        is_admin = bool(email and email.lower() in ADMIN_EMAILS)
        with self.pool.connection() as connection:
            user = connection.execute(
                """
                INSERT INTO users(id, username, email, is_admin)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    username = EXCLUDED.username,
                    email = COALESCE(EXCLUDED.email, users.email),
                    is_admin = users.is_admin OR EXCLUDED.is_admin,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
                """,
                (user_id, username, email, is_admin),
            ).fetchone()
        return self._premium(user)

    def get_user(self, user_id: str) -> dict | None:
        with self.pool.connection() as connection:
            user = connection.execute("SELECT * FROM users WHERE id = %s", (user_id,)).fetchone()
        return self._premium(user)

    def get_user_by_customer(self, customer_id: str) -> dict | None:
        with self.pool.connection() as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE stripe_customer_id = %s", (customer_id,)
            ).fetchone()
        return self._premium(user)

    def set_stripe_customer(self, user_id: str, customer_id: str) -> dict | None:
        with self.pool.connection() as connection:
            user = connection.execute(
                """UPDATE users SET stripe_customer_id = %s, updated_at = CURRENT_TIMESTAMP
                   WHERE id = %s RETURNING *""",
                (customer_id, user_id),
            ).fetchone()
        return self._premium(user)

    def update_subscription(
        self,
        user_id: str,
        customer_id: str,
        subscription_id: str,
        status: str,
        current_period_end: datetime | None,
        cancel_at_period_end: bool,
    ) -> dict | None:
        with self.pool.connection() as connection:
            user = connection.execute(
                """
                UPDATE users SET
                    stripe_customer_id = %s,
                    stripe_subscription_id = %s,
                    subscription_status = %s,
                    subscription_current_period_end = %s,
                    cancel_at_period_end = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING *
                """,
                (customer_id, subscription_id, status, current_period_end, cancel_at_period_end, user_id),
            ).fetchone()
        return self._premium(user)

    def create_ai(
        self,
        owner_id: str,
        name: str,
        persona: str,
        model: str = "openai",
        is_public: bool = False,
        original_avatar_url: str | None = None,
        image_generation_status: str = "not_requested",
    ) -> dict:
        ai_id = str(uuid.uuid4())
        with self.pool.connection() as connection:
            ai = connection.execute(
                """
                INSERT INTO ais(
                    id, owner_id, name, persona, model, is_public,
                    original_avatar_url, image_generation_status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (ai_id, owner_id, name, persona, model, is_public, original_avatar_url, image_generation_status),
            ).fetchone()
        return ai

    def get_ai(self, ai_id: str) -> dict | None:
        with self.pool.connection() as connection:
            return connection.execute("SELECT * FROM ais WHERE id = %s", (ai_id,)).fetchone()

    def list_ais(self, owner_id: str | None = None, public_only: bool = False) -> list[dict]:
        if owner_id:
            query, parameters = "SELECT * FROM ais WHERE owner_id = %s ORDER BY created_at DESC", (owner_id,)
        elif public_only:
            query, parameters = "SELECT * FROM ais WHERE is_public = TRUE ORDER BY created_at DESC", ()
        else:
            query, parameters = "SELECT * FROM ais ORDER BY created_at DESC", ()
        with self.pool.connection() as connection:
            return list(connection.execute(query, parameters).fetchall())

    def update_persona(self, ai_id: str, persona: str) -> bool:
        with self.pool.connection() as connection:
            result = connection.execute("UPDATE ais SET persona = %s WHERE id = %s", (persona, ai_id))
            return result.rowcount > 0

    def update_generation(self, ai_id: str, status: str, error: str | None = None, **images: str) -> None:
        allowed = {
            "avatar_front_face_url", "avatar_side_face_url", "full_body_front_url",
            "full_body_side_url", "full_body_back_url", "profile_image_url",
        }
        fields = {key: value for key, value in images.items() if key in allowed}
        fields.update(image_generation_status=status, image_generation_error=error)
        assignments = ", ".join(f"{key} = %s" for key in fields)
        with self.pool.connection() as connection:
            connection.execute(
                f"UPDATE ais SET {assignments} WHERE id = %s",
                (*fields.values(), ai_id),
            )

    def add_message(self, ai_id: str, user_id: str, role: str, content: str) -> dict:
        with self.pool.connection() as connection:
            return connection.execute(
                """INSERT INTO messages(ai_id, user_id, role, content)
                   VALUES (%s, %s, %s, %s) RETURNING *""",
                (ai_id, user_id, role, content),
            ).fetchone()

    def history(self, ai_id: str, user_id: str) -> list[dict]:
        with self.pool.connection() as connection:
            return list(
                connection.execute(
                    """SELECT * FROM messages WHERE ai_id = %s AND user_id = %s
                       ORDER BY id ASC LIMIT 100""",
                    (ai_id, user_id),
                ).fetchall()
            )

    def import_legacy_owner_data(self, target_user_id: str, legacy_owner_ids: list[str]) -> dict:
        """Move explicitly configured pre-ZITADEL local records to their signed-in owner."""
        source_ids = [owner_id for owner_id in legacy_owner_ids if owner_id and owner_id != target_user_id]
        if not source_ids:
            return {"ais": 0, "messages": 0}
        with self.pool.connection() as connection:
            existing_names = {
                row["name"]
                for row in connection.execute("SELECT name FROM ais WHERE owner_id = %s", (target_user_id,))
            }
            legacy_ais = connection.execute(
                "SELECT id, name FROM ais WHERE owner_id = ANY(%s) ORDER BY created_at, id",
                (source_ids,),
            ).fetchall()
            ai_count = 0
            for legacy_ai in legacy_ais:
                imported_name = legacy_ai["name"]
                if imported_name in existing_names:
                    suffix = 1
                    while True:
                        candidate = f"{legacy_ai['name']} (imported{'' if suffix == 1 else f' {suffix}'})"
                        if candidate not in existing_names:
                            imported_name = candidate
                            break
                        suffix += 1
                connection.execute(
                    "UPDATE ais SET owner_id = %s, name = %s WHERE id = %s",
                    (target_user_id, imported_name, legacy_ai["id"]),
                )
                existing_names.add(imported_name)
                ai_count += 1
            message_count = connection.execute(
                "UPDATE messages SET user_id = %s WHERE user_id = ANY(%s)",
                (target_user_id, source_ids),
            ).rowcount
        return {"ais": ai_count, "messages": message_count}

    def migrate_sqlite(self, sqlite_path: Path) -> int:
        """Import the previous local database once; simulated Premium flags are not imported."""
        if not sqlite_path.is_file():
            return 0
        with self.pool.connection() as connection:
            if connection.execute("SELECT EXISTS(SELECT 1 FROM users)").fetchone()["exists"]:
                return 0
        legacy = sqlite3.connect(sqlite_path)
        legacy.row_factory = sqlite3.Row
        imported = 0
        try:
            users = list(legacy.execute("SELECT id, username, created_at FROM users"))
            ais = list(legacy.execute("SELECT * FROM ais"))
            messages = list(legacy.execute("SELECT * FROM messages"))
            known_users = {row["id"] for row in users}
            for row in ais:
                known_users.add(row["owner_id"])
            for row in messages:
                known_users.add(row["user_id"])
            user_names = {row["id"]: row["username"] for row in users}
            with self.pool.connection() as connection:
                for user_id in known_users:
                    connection.execute(
                        "INSERT INTO users(id, username) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (user_id, user_names.get(user_id, f"Legacy user {user_id}")),
                    )
                for row in ais:
                    data = dict(row)
                    columns = [
                        "id", "owner_id", "name", "persona", "model", "is_public",
                        "original_avatar_url", "avatar_front_face_url", "avatar_side_face_url",
                        "full_body_front_url", "full_body_side_url", "full_body_back_url",
                        "profile_image_url", "image_generation_status", "image_generation_error",
                    ]
                    values = [data.get(column) for column in columns]
                    values[columns.index("is_public")] = bool(data.get("is_public"))
                    connection.execute(
                        f"INSERT INTO ais({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))}) ON CONFLICT DO NOTHING",
                        tuple(values),
                    )
                for row in messages:
                    connection.execute(
                        """INSERT INTO messages(id, ai_id, user_id, role, content, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                        (row["id"], row["ai_id"], row["user_id"], row["role"], row["content"], row["created_at"]),
                    )
                connection.execute(
                    "SELECT setval(pg_get_serial_sequence('messages', 'id'), COALESCE((SELECT MAX(id) FROM messages), 1), TRUE)"
                )
            imported = len(ais) + len(messages)
        finally:
            legacy.close()
        return imported
