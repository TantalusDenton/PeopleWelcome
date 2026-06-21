const path = require('path');
const fs = require('fs');
const { v4: uuid } = require('uuid');
const Database = require('better-sqlite3');

const STORAGE_ROOT = path.join(__dirname, 'local_storage');
fs.mkdirSync(STORAGE_ROOT, { recursive: true });

const db = new Database(path.join(STORAGE_ROOT, 'app.db'));
db.pragma('journal_mode = WAL');

db.exec(`
CREATE TABLE IF NOT EXISTS ais (
  id TEXT PRIMARY KEY,
  username TEXT NOT NULL,
  name TEXT NOT NULL,
  public INTEGER DEFAULT 1,
  tagged_posts TEXT DEFAULT '[]',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(username, name)
);

CREATE TABLE IF NOT EXISTS ai_search (
  username TEXT NOT NULL,
  ai_name TEXT NOT NULL,
  UNIQUE(username, ai_name)
);

CREATE TABLE IF NOT EXISTS posts (
  image_id TEXT PRIMARY KEY,
  username TEXT NOT NULL,
  date TEXT NOT NULL,
  timestamp REAL,
  tag_instances INTEGER DEFAULT 0,
  taggers TEXT DEFAULT '[]',
  tags TEXT DEFAULT '[]',
  objects TEXT DEFAULT NULL,
  image_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_posts (
  ai_id TEXT NOT NULL,
  image_id TEXT NOT NULL,
  image_owner TEXT NOT NULL,
  PRIMARY KEY (ai_id, image_id)
);

CREATE TABLE IF NOT EXISTS tags (
  ai_id TEXT NOT NULL,
  image_id TEXT NOT NULL,
  tags TEXT NOT NULL,
  PRIMARY KEY (ai_id, image_id)
);

CREATE TABLE IF NOT EXISTS decision_trees (
  ai_id TEXT PRIMARY KEY,
  tree_data TEXT
);
`);

const getDateStamp = () => {
  const today = new Date();
  const dd = String(today.getDate()).padStart(2, '0');
  const mm = String(today.getMonth() + 1).padStart(2, '0');
  const yyyy = today.getFullYear();
  return `${yyyy}-${mm}-${dd}`;
};

const parseArray = (value, fallback = []) => {
  if (!value) {
    return [...fallback];
  }
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [...fallback];
  } catch {
    return [...fallback];
  }
};

const ensureAiRow = (username, aiName) => {
  const row = db
    .prepare('SELECT id, username, name FROM ais WHERE username = ? AND name = ?')
    .get(username, aiName);
  if (!row) {
    throw new Error(`AI ${aiName} for ${username} not found`);
  }
  return row;
};

const ensurePostRow = (imageId) => {
  const row = db
    .prepare('SELECT image_id, username, taggers, tags, tag_instances FROM posts WHERE image_id = ?')
    .get(imageId);
  if (!row) {
    throw new Error(`Post ${imageId} not found`);
  }
  return row;
};

const updatePostTags = (imageId, mutate) => {
  const post = ensurePostRow(imageId);
  const taggers = parseArray(post.taggers);
  const tags = parseArray(post.tags);
  const { taggers: nextTaggers, tags: nextTags } = mutate({
    taggers,
    tags,
  });

  db.prepare('UPDATE posts SET taggers = ?, tags = ?, tag_instances = ? WHERE image_id = ?').run(
    JSON.stringify(nextTaggers),
    JSON.stringify(nextTags),
    nextTags.length,
    imageId,
  );
};

const removeTagArtifacts = (aiId, imageId) => {
  db.prepare('DELETE FROM tags WHERE ai_id = ? AND image_id = ?').run(aiId, imageId);
  db.prepare('DELETE FROM ai_posts WHERE ai_id = ? AND image_id = ?').run(aiId, imageId);
};

const flattenTagValues = (rows) => {
  const unique = new Set();
  rows.forEach((row) => {
    parseArray(row.tags).forEach((tag) => unique.add(tag));
  });
  return Array.from(unique);
};

const dynamoRepo = {
  getAiIdByAiAndUser: async (user, ai) => {
    const row = ensureAiRow(user, ai);
    return { ai_id: row.id };
  },

  getAiAndUserByAiId: async (aiId) => {
    const row = db.prepare('SELECT username, name FROM ais WHERE id = ?').get(aiId);
    if (!row) {
      throw new Error(`AI ${aiId} not found`);
    }
    return { username: row.username, ai_name: row.name };
  },

  retrievePostsOfUser: async (user) => {
    return db
      .prepare('SELECT image_id, date, tag_instances FROM posts WHERE username = ? ORDER BY date DESC')
      .all(user);
  },

  retrievePostsOfAi: async (user, ai) => {
    const aiRow = ensureAiRow(user, ai);
    return db
      .prepare(
        `SELECT p.image_id, p.date, p.tag_instances
         FROM posts p
         JOIN ai_posts ap ON ap.image_id = p.image_id
         WHERE ap.ai_id = ?
         ORDER BY p.date DESC`,
      )
      .all(aiRow.id);
  },

  listPostsByPopularityAndDate: async () => {
    return db
      .prepare(
        'SELECT username, image_id, date, tag_instances FROM posts ORDER BY tag_instances DESC, date DESC LIMIT 250',
      )
      .all();
  },

  createAi: async (user, ai) => {
    const id = uuid();
    db.prepare('INSERT INTO ais (id, username, name) VALUES (?, ?, ?)').run(id, user, ai);
    return { ai_id: id };
  },

  deleteAi: async (user, ai) => {
    const row = ensureAiRow(user, ai);
    db.prepare('DELETE FROM ai_posts WHERE ai_id = ?').run(row.id);
    db.prepare('DELETE FROM tags WHERE ai_id = ?').run(row.id);
    db.prepare('DELETE FROM decision_trees WHERE ai_id = ?').run(row.id);
    db.prepare('DELETE FROM ais WHERE id = ?').run(row.id);
    db.prepare('DELETE FROM ai_search WHERE username = ? AND ai_name = ?').run(user, ai);
  },

  addAiToSearchTable: async (user, ai) => {
    db.prepare('INSERT OR IGNORE INTO ai_search (username, ai_name) VALUES (?, ?)').run(user, ai);
    return true;
  },

  removeAiFromSearchTable: async (user, ai) => {
    db.prepare('DELETE FROM ai_search WHERE username = ? AND ai_name = ?').run(user, ai);
  },

  createPostWithoutTags: async (user, imageId) => {
    const date = `${getDateStamp()}-0`;
    const timestamp = Date.now() / 1000 / 60 / 60 / 24;
    db.prepare(
      `INSERT INTO posts (image_id, username, date, timestamp, tag_instances, taggers, tags, image_path)
       VALUES (?, ?, ?, ?, 0, '[]', '[]', ?)`,
    ).run(imageId, user, date, timestamp, imageId);
  },

  deletePost: async (imageOwner, imageId) => {
    const post = db.prepare('SELECT taggers FROM posts WHERE username = ? AND image_id = ?').get(imageOwner, imageId);
    if (post) {
      parseArray(post.taggers).forEach((aiId) => {
        removeTagArtifacts(aiId, imageId);
      });
    }
    db.prepare('DELETE FROM posts WHERE username = ? AND image_id = ?').run(imageOwner, imageId);
  },

  insertTagInTagAndPostTables: async (user, ai, imageOwner, imageId, tag) => {
    const aiRow = ensureAiRow(user, ai);
    const existing = db
      .prepare('SELECT tags FROM tags WHERE ai_id = ? AND image_id = ?')
      .get(aiRow.id, imageId);
    const tags = existing ? parseArray(existing.tags) : [];
    if (tags.includes(tag)) {
      return -1;
    }
    tags.push(tag);
    db.prepare(
      `INSERT INTO tags (ai_id, image_id, tags)
       VALUES (?, ?, ?)
       ON CONFLICT(ai_id, image_id) DO UPDATE SET tags = excluded.tags`,
    ).run(aiRow.id, imageId, JSON.stringify(tags));

    db.prepare(
      `INSERT OR IGNORE INTO ai_posts (ai_id, image_id, image_owner)
       VALUES (?, ?, ?)`,
    ).run(aiRow.id, imageId, imageOwner);

    updatePostTags(imageId, ({ taggers, tags: postTags }) => {
      const nextTaggers = new Set(taggers);
      nextTaggers.add(aiRow.id);
      const nextTags = new Set(postTags);
      nextTags.add(tag);
      return {
        taggers: Array.from(nextTaggers),
        tags: Array.from(nextTags),
      };
    });

    return 0;
  },

  removeTagFromTagAndPostTables: async (user, ai, imageOwner, imageId, tag) => {
    const aiRow = ensureAiRow(user, ai);
    const existing = db
      .prepare('SELECT tags FROM tags WHERE ai_id = ? AND image_id = ?')
      .get(aiRow.id, imageId);
    if (!existing) {
      return -1;
    }
    const nextTags = parseArray(existing.tags).filter((value) => value !== tag);
    if (nextTags.length === 0) {
      removeTagArtifacts(aiRow.id, imageId);
    } else {
      db.prepare('UPDATE tags SET tags = ? WHERE ai_id = ? AND image_id = ?').run(
        JSON.stringify(nextTags),
        aiRow.id,
        imageId,
      );
    }

    updatePostTags(imageId, ({ taggers, tags: postTags }) => {
      const taggerSet = new Set(taggers);
      if (nextTags.length === 0) {
        taggerSet.delete(aiRow.id);
      }
      const nextPostTags = postTags.filter((value) => value !== tag);
      return {
        taggers: Array.from(taggerSet),
        tags: nextPostTags,
      };
    });

    return 0;
  },

  getAllTagsByAiAndUser: async (user, ai) => {
    const aiRow = ensureAiRow(user, ai);
    const rows = db.prepare('SELECT tags FROM tags WHERE ai_id = ?').all(aiRow.id);
    return flattenTagValues(rows);
  },

  getTagsByAiAndUserAndImageId: async (user, ai, imageId) => {
    const aiRow = ensureAiRow(user, ai);
    const row = db.prepare('SELECT tags FROM tags WHERE ai_id = ? AND image_id = ?').get(aiRow.id, imageId);
    return row ? parseArray(row.tags) : [];
  },

  getAllAisByUser: async (user) => {
    return db
      .prepare('SELECT id as ai_id, name as ai_name FROM ais WHERE username = ? ORDER BY created_at DESC')
      .all(user);
  },

  checkForDecisionTree: async (user, ai) => {
    const aiRow = ensureAiRow(user, ai);
    const row = db.prepare('SELECT tree_data FROM decision_trees WHERE ai_id = ?').get(aiRow.id);
    return Boolean(row && row.tree_data);
  },

  retrieveDecisionTree: async (user, ai) => {
    const aiRow = ensureAiRow(user, ai);
    const row = db.prepare('SELECT tree_data FROM decision_trees WHERE ai_id = ?').get(aiRow.id);
    return row?.tree_data ?? null;
  },

  retrieveGoogleVisionObjects: async (imageId) => {
    const row = db.prepare('SELECT objects FROM posts WHERE image_id = ?').get(imageId);
    if (!row || !row.objects) {
      return [];
    }
    return JSON.parse(row.objects);
  },

  checkForGoogleVisionObjects: async (imageId) => {
    const row = db.prepare('SELECT objects FROM posts WHERE image_id = ?').get(imageId);
    return Boolean(row && row.objects);
  },

  updateGoogleVisionObjects: async (imageId, objects) => {
    db.prepare('UPDATE posts SET objects = ? WHERE image_id = ?').run(JSON.stringify(objects ?? []), imageId);
  },

  updateDecisionTree: async (user, ai, treeData) => {
    const aiRow = ensureAiRow(user, ai);
    db.prepare(
      `INSERT INTO decision_trees (ai_id, tree_data)
       VALUES (?, ?)
       ON CONFLICT(ai_id) DO UPDATE SET tree_data = excluded.tree_data`,
    ).run(aiRow.id, typeof treeData === 'string' ? treeData : JSON.stringify(treeData ?? {}));
  },
};

exports.dynamo = dynamoRepo;
