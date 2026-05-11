-- Phase B Time-Spent fix: restore the duration field the legacy
-- LifeItem had. Stored as integer minutes; duration_inferred flags
-- AI-estimated values so the UI can render with a tilde.
ALTER TABLE episodes ADD COLUMN duration INTEGER;
ALTER TABLE episodes ADD COLUMN duration_inferred INTEGER;
