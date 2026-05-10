-- SPDX-License-Identifier: Apache-2.0
-- L2 initial schema — episodes + entities + bi-temporal edges + grounding link.
--
-- The 4-class PKG ontology (Person/Place/Project/Topic) lives in `entities`,
-- discriminated by the `type` column. Topic uses `kind` for further refinement.
-- Affect/body-state are scalar columns on `episodes` (Conway's SMS principle).
--
-- Bi-temporal columns on `edges`:
--   t_event     — when the fact was true in reality
--   t_ingestion — when we learned about it
--   t_valid     — start of the validity window (= t_event by default)
--   t_invalid   — end of validity (NULL == still valid; supersede sets this)
-- This is the Graphiti / Zep model. Queries take an `as_of` parameter that
-- intersects the validity window, returning the fact that was true at that time.

CREATE TABLE IF NOT EXISTS episodes (
  id            TEXT PRIMARY KEY,
  text          TEXT NOT NULL,
  occurred_at   INTEGER NOT NULL,           -- unix-ms
  ingested_at   INTEGER NOT NULL,           -- unix-ms
  source        TEXT,                       -- "user" / "telegram" / "voice" / etc.
  predicates    TEXT NOT NULL DEFAULT '[]', -- JSON list of normalized verbs
  body_state    TEXT,                       -- e.g. "tired" / "累了"
  sentiment     TEXT,                       -- pos | neu | neg | NULL
  energy        TEXT                        -- high | medium | low | NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_occurred_at ON episodes(occurred_at);

CREATE TABLE IF NOT EXISTS entities (
  id              TEXT PRIMARY KEY,
  type            TEXT NOT NULL,            -- Person | Place | Project | Topic
  kind            TEXT,                     -- Topic kind discriminator (NULL for non-Topic)
  key             TEXT NOT NULL,            -- canonical lowercase
  value           TEXT NOT NULL,            -- surface form as first observed
  attributes_json TEXT NOT NULL DEFAULT '{}',
  created_at      INTEGER NOT NULL,
  -- Identity is (type, key) — entities with the same (type, key) are the same node.
  -- This is the dedup boundary for ingestion; the hygiene engine in L3 may merge
  -- entities with different keys into a single canonical_id.
  UNIQUE(type, key)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);

CREATE TABLE IF NOT EXISTS edges (
  id            TEXT PRIMARY KEY,
  -- from_entity NULL == "the user" / "me" — the implicit subject of life-log entries.
  -- This avoids creating a synthetic Person:me node and keeps queries simple.
  from_entity   TEXT,
  to_entity     TEXT NOT NULL,
  verb          TEXT NOT NULL,              -- normalized predicate ("ate", "met", ...)
  episode_id    TEXT NOT NULL,              -- back-reference (audit trail)
  t_event       INTEGER NOT NULL,
  t_ingestion   INTEGER NOT NULL,
  t_valid       INTEGER NOT NULL,
  t_invalid     INTEGER,                    -- NULL == currently valid
  attributes_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY (from_entity) REFERENCES entities(id),
  FOREIGN KEY (to_entity)   REFERENCES entities(id),
  FOREIGN KEY (episode_id)  REFERENCES episodes(id)
);

CREATE INDEX IF NOT EXISTS idx_edges_to_entity ON edges(to_entity);
CREATE INDEX IF NOT EXISTS idx_edges_from_entity ON edges(from_entity);
CREATE INDEX IF NOT EXISTS idx_edges_episode ON edges(episode_id);
CREATE INDEX IF NOT EXISTS idx_edges_t_valid ON edges(t_valid);
CREATE INDEX IF NOT EXISTS idx_edges_verb ON edges(verb);

-- Episode↔entity mention link. Cheap "this entity appeared in this episode"
-- lookup, used by `lg.episodes.mentioning(entity)`.
CREATE TABLE IF NOT EXISTS entity_episode_mention (
  entity_id   TEXT NOT NULL,
  episode_id  TEXT NOT NULL,
  PRIMARY KEY (entity_id, episode_id),
  FOREIGN KEY (entity_id)  REFERENCES entities(id),
  FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

-- Schema version tracking (for future migrations).
CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at INTEGER NOT NULL
);
