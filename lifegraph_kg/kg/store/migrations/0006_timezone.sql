-- Timezone handling. Mirrors the Supabase migration of the same
-- intent; see docs/timezone (the design doc) for the full model.
--
-- absolute vs floating only matters for tasks; logs leave time_mode
-- null. wall_clock_* fields are populated only for floating tasks.

ALTER TABLE episodes ADD COLUMN origin_tz TEXT;
ALTER TABLE episodes ADD COLUMN time_mode TEXT;
ALTER TABLE episodes ADD COLUMN wall_clock_hour INTEGER;
ALTER TABLE episodes ADD COLUMN wall_clock_minute INTEGER;
ALTER TABLE episodes ADD COLUMN wall_clock_date TEXT;
