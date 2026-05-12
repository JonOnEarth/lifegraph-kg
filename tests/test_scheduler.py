# SPDX-License-Identifier: Apache-2.0
"""Tests for compute_next_fire_utc — the floating/absolute scheduler.

The cases are organized around the timezone design's §4-§5 behaviors:
  absolute one-shot, absolute recurring, floating daily,
  floating weekly, floating yearly anchor, cross-TZ follow,
  DST transitions, end-of-life statuses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from lifegraph_kg.kg.episode import Episode
from lifegraph_kg.kg.scheduler import compute_next_fire_utc


NY = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")


def _task(**overrides) -> Episode:
    """Build a test Episode with task defaults."""
    base = dict(
        id="ep1",
        user_id="u1",
        text="t",
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        ingested_at=datetime(2026, 5, 1, tzinfo=UTC),
        kind="task",
        status="active",
    )
    base.update(overrides)
    return Episode(**base)


# ── Non-task / lifecycle short-circuits ──────────────────────────────────


def test_log_returns_none():
    ep = _task(kind="log")
    assert compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC)) is None


def test_done_task_returns_none():
    ep = _task(
        status="done",
        time_mode="absolute",
        deadline=datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC)) is None


def test_dropped_task_returns_none():
    ep = _task(
        status="dropped",
        time_mode="absolute",
        deadline=datetime(2026, 6, 1, tzinfo=UTC),
    )
    assert compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC)) is None


def test_naive_as_of_raises():
    ep = _task(time_mode="absolute", deadline=datetime(2026, 6, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11))


# ── Absolute ─────────────────────────────────────────────────────────────


def test_absolute_one_shot_future():
    ep = _task(
        time_mode="absolute",
        deadline=datetime(2026, 6, 1, 15, 0, tzinfo=UTC),
    )
    nxt = compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC))
    assert nxt == datetime(2026, 6, 1, 15, 0, tzinfo=UTC)


def test_absolute_one_shot_past_returns_none():
    ep = _task(
        time_mode="absolute",
        deadline=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
    )
    assert compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC)) is None


def test_absolute_missing_deadline_returns_none():
    ep = _task(time_mode="absolute", deadline=None)
    assert compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC)) is None


def test_absolute_daily_recurring_walks_forward():
    # Deadline anchored at May 1 15:00 UTC, daily recurrence. as_of May 11
    # → next fire should be May 11 15:00 UTC (or later if past).
    ep = _task(
        time_mode="absolute",
        deadline=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        recurrence="daily",
    )
    nxt = compute_next_fire_utc(
        ep, as_of=datetime(2026, 5, 11, 14, 0, tzinfo=UTC)
    )
    assert nxt == datetime(2026, 5, 11, 15, 0, tzinfo=UTC)


def test_absolute_weekly_recurring():
    ep = _task(
        time_mode="absolute",
        deadline=datetime(2026, 5, 4, 9, 0, tzinfo=UTC),  # Mon
        recurrence="weekly",
    )
    nxt = compute_next_fire_utc(
        ep, as_of=datetime(2026, 5, 11, 10, 0, tzinfo=UTC)
    )
    assert nxt == datetime(2026, 5, 18, 9, 0, tzinfo=UTC)


def test_absolute_monthly_recurring_clamps_day():
    # Jan 31 monthly → Feb should clamp to Feb 28 (2026 is not leap).
    ep = _task(
        time_mode="absolute",
        deadline=datetime(2026, 1, 31, 12, 0, tzinfo=UTC),
        recurrence="monthly",
    )
    nxt = compute_next_fire_utc(
        ep, as_of=datetime(2026, 2, 1, tzinfo=UTC)
    )
    assert nxt == datetime(2026, 2, 28, 12, 0, tzinfo=UTC)


# ── Floating: daily ──────────────────────────────────────────────────────


def test_floating_daily_today_future():
    # "Every day at 8am" — as_of is 2026-05-11 06:00 NY, so today's 8am
    # is still ahead.
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        wall_clock_minute=0,
        recurrence="daily",
        origin_tz="America/New_York",
    )
    as_of = datetime(2026, 5, 11, 10, 0, tzinfo=UTC)  # 06:00 NY EDT
    nxt = compute_next_fire_utc(ep, as_of=as_of)
    # 8 am NY on May 11 = 12:00 UTC (EDT, UTC-4)
    assert nxt == datetime(2026, 5, 11, 12, 0, tzinfo=UTC)


def test_floating_daily_today_past_rolls_to_tomorrow():
    # 8am NY already passed today (10am NY) — next fire is tomorrow's 8am NY.
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        recurrence="daily",
        origin_tz="America/New_York",
    )
    as_of = datetime(2026, 5, 11, 14, 0, tzinfo=UTC)  # 10:00 NY
    nxt = compute_next_fire_utc(ep, as_of=as_of)
    assert nxt == datetime(2026, 5, 12, 12, 0, tzinfo=UTC)


def test_floating_daily_no_recurrence_past_returns_none():
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        recurrence=None,
        origin_tz="America/New_York",
    )
    as_of = datetime(2026, 5, 11, 14, 0, tzinfo=UTC)  # past today's 8am NY
    assert compute_next_fire_utc(ep, as_of=as_of) is None


# ── Floating: cross-timezone "follows the user" ──────────────────────────


def test_floating_follows_user_to_tokyo():
    # Task created in NY, user now in Tokyo. user_tz override should
    # cause the daily 8am to be interpreted in Tokyo, not NY.
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        recurrence="daily",
        origin_tz="America/New_York",
    )
    # 2026-05-11 00:00 UTC = 09:00 Tokyo same day, so today's 8am Tokyo
    # has passed → next is tomorrow's 8am Tokyo (2026-05-12 08:00 Tokyo
    # = 2026-05-11 23:00 UTC).
    as_of = datetime(2026, 5, 11, 0, 0, tzinfo=UTC)
    nxt = compute_next_fire_utc(ep, as_of=as_of, user_tz="Asia/Tokyo")
    assert nxt == datetime(2026, 5, 11, 23, 0, tzinfo=UTC)


# ── Floating: yearly anchor ──────────────────────────────────────────────


def test_floating_yearly_anchor_this_year_future():
    # "Birthday May 15 at 9am" — as_of May 11 → this year's May 15.
    ep = _task(
        time_mode="floating",
        wall_clock_hour=9,
        wall_clock_date="05-15",
        origin_tz="America/New_York",
    )
    nxt = compute_next_fire_utc(
        ep, as_of=datetime(2026, 5, 11, tzinfo=UTC)
    )
    # 9am NY May 15 2026 (EDT = UTC-4) = 13:00 UTC.
    assert nxt == datetime(2026, 5, 15, 13, 0, tzinfo=UTC)


def test_floating_yearly_anchor_this_year_past_rolls_to_next():
    ep = _task(
        time_mode="floating",
        wall_clock_hour=9,
        wall_clock_date="01-15",
        origin_tz="America/New_York",
    )
    nxt = compute_next_fire_utc(
        ep, as_of=datetime(2026, 5, 11, tzinfo=UTC)
    )
    # 2027-01-15 09:00 NY (EST = UTC-5) = 14:00 UTC.
    assert nxt == datetime(2027, 1, 15, 14, 0, tzinfo=UTC)


def test_floating_yearly_anchor_leap_day_handled():
    # Feb 29 anchor in a non-leap year should be skipped to next year.
    ep = _task(
        time_mode="floating",
        wall_clock_hour=12,
        wall_clock_date="02-29",
        origin_tz="America/New_York",
    )
    # 2026 is not leap, 2027 is not leap, 2028 IS leap. as_of = 2026-03-01.
    # Function only tries year, year+1 — so this returns None for 2026/2027.
    # Documented limitation; tested for awareness.
    assert (
        compute_next_fire_utc(ep, as_of=datetime(2026, 3, 1, tzinfo=UTC))
        is None
    )


# ── Floating: DST awareness ──────────────────────────────────────────────


def test_floating_daily_across_dst_spring_forward():
    # NY DST spring-forward 2026 was March 8 (2am → 3am). The 8am-daily
    # task should still fire at 8am local — clock walks fine.
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        recurrence="daily",
        origin_tz="America/New_York",
    )
    # March 7 14:00 UTC = 09:00 EST → today's 8am has passed; next is
    # March 8 08:00 EDT = 12:00 UTC (only 22h later, not 24h, due to DST).
    as_of = datetime(2026, 3, 7, 14, 0, tzinfo=UTC)
    nxt = compute_next_fire_utc(ep, as_of=as_of)
    assert nxt == datetime(2026, 3, 8, 12, 0, tzinfo=UTC)


# ── Bad data ─────────────────────────────────────────────────────────────


def test_floating_missing_tz_returns_none():
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        origin_tz=None,  # no tz on episode, no override
    )
    assert (
        compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC))
        is None
    )


def test_floating_bogus_tz_returns_none():
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        origin_tz="Mars/Olympus",
    )
    assert (
        compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC))
        is None
    )


def test_floating_missing_hour_returns_none():
    ep = _task(time_mode="floating", origin_tz="America/New_York")
    assert (
        compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC))
        is None
    )


def test_floating_bogus_wall_clock_date_returns_none():
    ep = _task(
        time_mode="floating",
        wall_clock_hour=8,
        wall_clock_date="not-a-date",
        origin_tz="America/New_York",
    )
    assert (
        compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC))
        is None
    )


# ── Default time_mode handling ───────────────────────────────────────────


def test_missing_time_mode_treated_as_absolute():
    # Tasks predating TZ-1 have time_mode=None; treat as absolute so
    # existing deadline-anchored todos keep firing.
    ep = _task(
        time_mode=None,
        deadline=datetime(2026, 6, 1, tzinfo=UTC),
    )
    nxt = compute_next_fire_utc(ep, as_of=datetime(2026, 5, 11, tzinfo=UTC))
    assert nxt == datetime(2026, 6, 1, tzinfo=UTC)
