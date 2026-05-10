-- SPDX-License-Identifier: Apache-2.0
-- L3 hygiene engine — proposal/apply pipeline.
--
-- The `entities.canonical_id` column is added (nullable) to point a
-- merged entity at its winner. Queries that want only canonical
-- entities filter `canonical_id IS NULL`. This lets the audit trail
-- survive — merged entities are NOT deleted, they're aliased.

ALTER TABLE entities ADD COLUMN canonical_id TEXT REFERENCES entities(id);

CREATE INDEX IF NOT EXISTS idx_entities_canonical_id ON entities(canonical_id);

-- Proposed merges. The hygiene engine writes; the apply path reads.
-- `applied_at` is NULL while pending; set when the merge is applied.
CREATE TABLE IF NOT EXISTS merge_proposals (
  id            TEXT PRIMARY KEY,
  winner_id     TEXT NOT NULL,
  loser_id      TEXT NOT NULL,
  confidence    TEXT NOT NULL,           -- high | medium | low
  reason        TEXT NOT NULL,
  detail        TEXT NOT NULL DEFAULT '',
  proposed_at   INTEGER NOT NULL,        -- unix-ms
  applied_at    INTEGER,                 -- NULL while pending
  rejected_at   INTEGER,                 -- NULL while pending
  FOREIGN KEY (winner_id) REFERENCES entities(id),
  FOREIGN KEY (loser_id)  REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_merge_proposals_pending
  ON merge_proposals(applied_at, rejected_at);
