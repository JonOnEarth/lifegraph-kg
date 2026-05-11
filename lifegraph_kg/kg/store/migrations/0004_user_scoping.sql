-- Phase 6: multi-user scoping for the SQLite backend.
--
-- Adds user_id to all data tables, drops the global (type, key) UNIQUE
-- on entities, replaces with (user_id, type, key). Matches the Supabase
-- migration applied to the lifegraph-kg project.
--
-- For existing single-user DBs we leave user_id nullable on the
-- migration step (SQLite can't ALTER COLUMN to NOT NULL after the
-- fact). Application code requires user_id on writes; reads coexist
-- with NULL rows by treating them as "legacy unowned".

ALTER TABLE episodes ADD COLUMN user_id TEXT;
ALTER TABLE entities ADD COLUMN user_id TEXT;
ALTER TABLE edges    ADD COLUMN user_id TEXT;
ALTER TABLE entity_episode_mention ADD COLUMN user_id TEXT;

-- SQLite doesn't support DROP CONSTRAINT directly; we re-create the
-- entities table to swap the UNIQUE definition. The migration runs
-- inside the SqliteStore's init_schema which already opens a tx, so
-- the rename is atomic from the caller's perspective.
CREATE TABLE entities_new (
  id              TEXT PRIMARY KEY,
  type            TEXT NOT NULL,
  kind            TEXT,
  key             TEXT NOT NULL,
  value           TEXT NOT NULL,
  attributes_json TEXT NOT NULL DEFAULT '{}',
  created_at      INTEGER NOT NULL,
  canonical_id    TEXT REFERENCES entities_new(id),
  user_id         TEXT,
  UNIQUE(user_id, type, key)
);
INSERT INTO entities_new (id, type, kind, key, value, attributes_json, created_at, canonical_id, user_id)
  SELECT id, type, kind, key, value, attributes_json, created_at, canonical_id, user_id FROM entities;
DROP TABLE entities;
ALTER TABLE entities_new RENAME TO entities;
CREATE INDEX IF NOT EXISTS idx_entities_type           ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_kind           ON entities(kind);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_id   ON entities(canonical_id);
CREATE INDEX IF NOT EXISTS idx_entities_user_id        ON entities(user_id);

-- Composite hot-path indexes.
CREATE INDEX IF NOT EXISTS idx_episodes_user_id        ON episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_user_occurred  ON episodes(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_edges_user_id           ON edges(user_id);
CREATE INDEX IF NOT EXISTS idx_edges_user_episode      ON edges(user_id, episode_id);
CREATE INDEX IF NOT EXISTS idx_eem_user_id             ON entity_episode_mention(user_id);
