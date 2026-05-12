# SPDX-License-Identifier: Apache-2.0
"""compute_next_fire_utc — when should a task fire next?

Pure function over an Episode + an "as_of" reference moment. Returns the
next UTC datetime the scheduler should fire this task, or ``None`` when
the task has no future fires (one-shot in the past, completed, dropped).

The function honors the two ``time_mode`` modes from the timezone design:

  absolute  — task anchors a fixed UTC moment via ``deadline``. The user
              moving across timezones does NOT change when it fires.
              Recurring absolutes walk the recurrence forward from
              ``deadline`` in UTC.

  floating  — task carries a wall-clock spec (``wall_clock_hour`` +
              ``wall_clock_minute`` + optional ``wall_clock_date``) that
              gets interpreted in the user's CURRENT timezone at fire
              time. The same "8 am meditate" task fires at 8 am NY on
              Monday and 8 am Tokyo on Tuesday if the user travels.

The function is intentionally side-effect-free so the caller (Worker
cron, ai-service scheduler, or test) can drive it from any context.

Live-scheduler integration is deferred — preview env is currently
cron-disabled per the Phase-8 plan.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from lifegraph_kg.kg.episode import Episode

# Recurrence vocabulary — keep narrow for v0.1. Full RRULE support is a
# v0.2 expansion. Accepted forms:
#   "daily" / "weekly" / "monthly" / "yearly"
#   None — one-shot
_INTERVALS_DAYS: dict[str, int] = {
    "daily": 1,
    "weekly": 7,
}


def _advance(dt: datetime, recurrence: str) -> datetime:
    """Advance ``dt`` by one recurrence step. Caller guarantees recurrence
    is one of the supported forms; otherwise raises ValueError."""
    if recurrence in _INTERVALS_DAYS:
        return dt + timedelta(days=_INTERVALS_DAYS[recurrence])
    if recurrence == "monthly":
        # Calendar-month advance — clamp day-of-month if target month
        # is shorter (Jan-31 → Feb-28).
        month = dt.month + 1
        year = dt.year
        if month > 12:
            month -= 12
            year += 1
        # Find the last valid day in the target month.
        if month == 12:
            next_month_first = datetime(year + 1, 1, 1, tzinfo=dt.tzinfo)
        else:
            next_month_first = datetime(year, month + 1, 1, tzinfo=dt.tzinfo)
        last_day_target = (next_month_first - timedelta(days=1)).day
        day = min(dt.day, last_day_target)
        return dt.replace(year=year, month=month, day=day)
    if recurrence == "yearly":
        # Feb 29 on a non-leap year → Feb 28.
        try:
            return dt.replace(year=dt.year + 1)
        except ValueError:
            return dt.replace(year=dt.year + 1, day=28)
    raise ValueError(f"unsupported recurrence: {recurrence!r}")


def compute_next_fire_utc(
    episode: Episode,
    *,
    as_of: datetime,
    user_tz: str | None = None,
) -> datetime | None:
    """Return the next UTC datetime at which ``episode`` should fire.

    Returns ``None`` when:
      - the episode is not a task (logs never fire)
      - the task is done / dropped
      - the task is absolute one-shot in the past (no recurrence)
      - the task lacks the data needed to compute a fire time
        (absolute without deadline; floating without wall_clock_hour)

    ``user_tz`` overrides ``episode.origin_tz`` for floating-mode tasks —
    this is how "follow the user across timezones" works: pass the
    detected current TZ at scheduler time, not the one stored when the
    task was created.

    For absolute tasks, ``user_tz`` is ignored (deadline is UTC-anchored).
    """
    if episode.kind != "task":
        return None
    if episode.status != "active":
        return None
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware (datetime with tzinfo)")

    mode = episode.time_mode or "absolute"
    recurrence = episode.recurrence

    if mode == "absolute":
        return _next_absolute(
            deadline=episode.deadline,
            recurrence=recurrence,
            as_of=as_of,
        )

    # floating
    tz_name = user_tz or episode.origin_tz
    if tz_name is None:
        return None
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        return None
    if episode.wall_clock_hour is None:
        return None
    return _next_floating(
        hour=episode.wall_clock_hour,
        minute=episode.wall_clock_minute or 0,
        wall_clock_date=episode.wall_clock_date,
        recurrence=recurrence,
        as_of=as_of,
        user_tz=tz,
    )


def _next_absolute(
    *,
    deadline: datetime | None,
    recurrence: str | None,
    as_of: datetime,
) -> datetime | None:
    if deadline is None:
        return None
    deadline_utc = deadline.astimezone(UTC) if deadline.tzinfo else deadline.replace(tzinfo=UTC)
    if recurrence is None:
        return deadline_utc if deadline_utc > as_of else None
    # Recurring absolute — walk forward from the deadline anchor.
    cur = deadline_utc
    # Cap the walk so a degenerate recurrence can't infinite-loop.
    for _ in range(10_000):
        if cur > as_of:
            return cur
        cur = _advance(cur, recurrence)
    return None


def _next_floating(
    *,
    hour: int,
    minute: int,
    wall_clock_date: str | None,  # "MM-DD" or None
    recurrence: str | None,
    as_of: datetime,
    user_tz: ZoneInfo,
) -> datetime | None:
    user_now = as_of.astimezone(user_tz)

    # Yearly anchor (e.g. anniversary "01-15"): try this year, fall back
    # to next year if past. Build local datetime fresh each iteration so
    # the user_tz offset is correct across DST transitions.
    if wall_clock_date:
        try:
            month, day = (int(p) for p in wall_clock_date.split("-", 1))
        except (ValueError, AttributeError):
            return None
        for year in (user_now.year, user_now.year + 1):
            try:
                local = datetime(year, month, day, hour, minute, tzinfo=user_tz)
            except ValueError:
                continue  # e.g. Feb 29 on a non-leap year
            candidate = local.astimezone(UTC)
            if candidate > as_of:
                return candidate
        return None

    # Daily-anchored floating: combine HH:MM with today; if past, advance.
    base = datetime(
        user_now.year, user_now.month, user_now.day, hour, minute, tzinfo=user_tz,
    ).astimezone(UTC)
    if base > as_of:
        return base
    # Past today's HH:MM. If non-recurring, no future fire.
    if recurrence is None:
        return None
    # Otherwise advance per recurrence in user-local space.
    cur_local = base.astimezone(user_tz)
    for _ in range(10_000):
        cur_local = _advance(cur_local, recurrence)
        cur_utc = cur_local.astimezone(UTC)
        if cur_utc > as_of:
            return cur_utc
    return None


__all__ = ["compute_next_fire_utc"]
