const path = require("path");
const express = require("express");
const Database = require("better-sqlite3");

const dbPath = path.join(__dirname, "chat-ui.db");
const db = new Database(dbPath);

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT,
    name TEXT
  );

  CREATE TABLE IF NOT EXISTS connections (
    user_id TEXT NOT NULL,
    connected_user_id TEXT NOT NULL,
    PRIMARY KEY (user_id, connected_user_id)
  );
`);

const app = express();
app.use(express.json());

app.post("/register", (req, res) => {
  const { id, email, name } = req.body;
  if (!id) {
    res.status(400).json({ error: "User id is required" });
    return;
  }
  const statement = db.prepare(
    "INSERT OR REPLACE INTO users (id, email, name) VALUES (@id, @email, @name)"
  );
  statement.run({ id, email, name });
  res.json({ status: "ok" });
});

app.post("/connected-users", (req, res) => {
  const { userId, connectedUserIds } = req.body;
  if (!userId || !Array.isArray(connectedUserIds)) {
    res.status(400).json({ error: "userId and connectedUserIds are required" });
    return;
  }
  const deleteStatement = db.prepare("DELETE FROM connections WHERE user_id = ?");
  deleteStatement.run(userId);
  const insertStatement = db.prepare(
    "INSERT INTO connections (user_id, connected_user_id) VALUES (?, ?)"
  );
  const transaction = db.transaction((ids) => {
    ids.forEach((connectedId) => insertStatement.run(userId, connectedId));
  });
  transaction(connectedUserIds);
  res.json({ status: "ok" });
});

app.get("/connected-users/:userId", (req, res) => {
  const statement = db.prepare("SELECT connected_user_id FROM connections WHERE user_id = ?");
  const rows = statement.all(req.params.userId);
  res.json(rows.map((row) => row.connected_user_id));
});

const PORT = process.env.SQLITE_SERVER_PORT || 4300;

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`SQLite helper listening on port ${PORT}`);
  });
}

module.exports = { app, db };
