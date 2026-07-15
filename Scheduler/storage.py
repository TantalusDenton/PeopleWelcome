import sqlite3
import uuid
from pathlib import Path

class Store:
    def __init__(self, path: Path): self.path = path
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path); conn.row_factory = sqlite3.Row
        return conn
    def initialize(self):
        with self.connection() as c:
            c.executescript('''CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, username TEXT NOT NULL, is_premium INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS ais (id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, name TEXT NOT NULL, persona TEXT NOT NULL, model TEXT NOT NULL DEFAULT 'openai', is_public INTEGER NOT NULL DEFAULT 0, original_avatar_url TEXT, avatar_front_face_url TEXT, avatar_side_face_url TEXT, full_body_front_url TEXT, full_body_side_url TEXT, full_body_back_url TEXT, profile_image_url TEXT, image_generation_status TEXT NOT NULL DEFAULT 'not_requested', image_generation_error TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, UNIQUE(owner_id, name));
            CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, ai_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);''')
            existing = {row[1] for row in c.execute("PRAGMA table_info(ais)")}
            for column, definition in {
                "original_avatar_url": "TEXT", "avatar_front_face_url": "TEXT", "avatar_side_face_url": "TEXT",
                "full_body_front_url": "TEXT", "full_body_side_url": "TEXT", "full_body_back_url": "TEXT",
                "profile_image_url": "TEXT", "image_generation_status": "TEXT NOT NULL DEFAULT 'not_requested'",
                "image_generation_error": "TEXT",
            }.items():
                if column not in existing: c.execute(f"ALTER TABLE ais ADD COLUMN {column} {definition}")
    @staticmethod
    def row(row): return dict(row) if row else None
    def upsert_user(self, user_id, username):
        with self.connection() as c: c.execute("INSERT INTO users(id,username) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET username=excluded.username", (user_id, username))
        return self.get_user(user_id)
    def get_user(self, user_id):
        with self.connection() as c: return self.row(c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())
    def set_premium(self, user_id, premium):
        if not self.get_user(user_id): self.upsert_user(user_id, user_id)
        with self.connection() as c: c.execute("UPDATE users SET is_premium=? WHERE id=?", (premium, user_id))
        return self.get_user(user_id)
    def create_ai(self, owner_id, name, persona, model="openai", is_public=False, original_avatar_url=None, image_generation_status="not_requested"):
        ai = {"id": str(uuid.uuid4()), "owner_id": owner_id, "name": name, "persona": persona, "model": model, "is_public": is_public, "original_avatar_url": original_avatar_url, "image_generation_status": image_generation_status}
        with self.connection() as c: c.execute("INSERT INTO ais(id,owner_id,name,persona,model,is_public,original_avatar_url,image_generation_status) VALUES(:id,:owner_id,:name,:persona,:model,:is_public,:original_avatar_url,:image_generation_status)", ai)
        return self.get_ai(ai["id"])
    def get_ai(self, ai_id):
        with self.connection() as c: return self.row(c.execute("SELECT * FROM ais WHERE id=?", (ai_id,)).fetchone())
    def list_ais(self, owner_id=None, public_only=False):
        sql, args = "SELECT * FROM ais", []
        if owner_id: sql += " WHERE owner_id=?"; args.append(owner_id)
        elif public_only: sql += " WHERE is_public=1"
        sql += " ORDER BY created_at DESC"
        with self.connection() as c: return [dict(x) for x in c.execute(sql, args).fetchall()]
    def update_persona(self, ai_id, persona):
        with self.connection() as c: return c.execute("UPDATE ais SET persona=? WHERE id=?", (persona, ai_id)).rowcount > 0
    def update_generation(self, ai_id, status, error=None, **images):
        allowed = {"avatar_front_face_url", "avatar_side_face_url", "full_body_front_url", "full_body_side_url", "full_body_back_url", "profile_image_url"}
        fields = {key: value for key, value in images.items() if key in allowed}
        fields.update(image_generation_status=status, image_generation_error=error)
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self.connection() as c:
            c.execute(f"UPDATE ais SET {assignments} WHERE id=?", [*fields.values(), ai_id])
    def add_message(self, ai_id, user_id, role, content):
        with self.connection() as c:
            cursor=c.execute("INSERT INTO messages(ai_id,user_id,role,content) VALUES(?,?,?,?)", (ai_id,user_id,role,content)); return self.row(c.execute("SELECT * FROM messages WHERE id=?", (cursor.lastrowid,)).fetchone())
    def history(self, ai_id, user_id):
        with self.connection() as c: return [dict(x) for x in c.execute("SELECT * FROM messages WHERE ai_id=? AND user_id=? ORDER BY id ASC LIMIT 100", (ai_id,user_id)).fetchall()]
