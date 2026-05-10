-- SPDX-License-Identifier: Apache-2.0
-- Task support — Episode gains a kind discriminator + lifecycle fields.
--
-- See lifegraph_kg/kg/episode.py for the design rationale. Logs vs tasks
-- share the same shape; the discriminator + a handful of nullable lifecycle
-- columns let "everything about Project X" stay a single query.
--
-- Existing rows get kind='log', status='active' by default — backward
-- compatible with all L2 + L3 functionality.

ALTER TABLE episodes ADD COLUMN kind         TEXT NOT NULL DEFAULT 'log';
ALTER TABLE episodes ADD COLUMN status       TEXT NOT NULL DEFAULT 'active';
ALTER TABLE episodes ADD COLUMN priority     TEXT;
ALTER TABLE episodes ADD COLUMN deadline     INTEGER;     -- unix-ms
ALTER TABLE episodes ADD COLUMN completed_at INTEGER;
ALTER TABLE episodes ADD COLUMN recurrence   TEXT;
ALTER TABLE episodes ADD COLUMN gtd_context  TEXT;
ALTER TABLE episodes ADD COLUMN action_verb  TEXT;

CREATE INDEX IF NOT EXISTS idx_episodes_kind_status ON episodes(kind, status);
CREATE INDEX IF NOT EXISTS idx_episodes_deadline ON episodes(deadline);
