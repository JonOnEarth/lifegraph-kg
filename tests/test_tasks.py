# SPDX-License-Identifier: Apache-2.0
"""Tests for task support — Episode.kind="task" + lifecycle + lg.tasks view.


Five groups:
  - schema: kind/status/priority/deadline columns persist + round-trip
  - lifecycle: complete_task / drop_task / reopen_task transitions
  - queries: pending / overdue / due_soon / by_context / by_priority
  - mixed: tasks and logs coexist; entity queries find both
  - migration parity: a "task" episode behaves identically to a "log"
                       episode for entity queries / mentions
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from lifegraph_kg import LifeGraph, Person, Topic
from lifegraph_kg.kg.episode import Episode

TEST_USER = "test-user"

T_NOW = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
T_TOMORROW = T_NOW + timedelta(days=1)
T_NEXT_WEEK = T_NOW + timedelta(days=7)
T_LAST_WEEK = T_NOW - timedelta(days=7)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _task_episode(
    *,
    text: str,
    deadline: datetime | None = None,
    priority: str | None = None,
    gtd_context: str | None = None,
    status: str = "active",
) -> Episode:
    """Build a task-kind Episode without going through extraction —
    keeps tests free of LLM dependency. The migration script uses this
    same path for legacy todos."""
    return Episode(
        user_id=TEST_USER, id=_new_id(),
        text=text,
        occurred_at=T_NOW,
        ingested_at=T_NOW,
        kind="task",
        status=status,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        deadline=deadline,
        gtd_context=gtd_context,
    )


# ----- schema persistence -----


def test_task_columns_round_trip() -> None:
    """All task fields persist + reload via _episode_from_row."""
    lg = LifeGraph()
    ep = _task_episode(
        text="Email Tao the report",
        deadline=T_TOMORROW,
        priority="high",
        gtd_context="@work",
    )
    lg._store.save_episode(ep, [], [])

    fetched = lg.episodes.get(ep.id)
    assert fetched is not None
    assert fetched.kind == "task"
    assert fetched.status == "active"
    assert fetched.priority == "high"
    assert fetched.deadline == T_TOMORROW
    assert fetched.gtd_context == "@work"
    assert fetched.completed_at is None


def test_existing_episodes_default_to_log_kind() -> None:
    """A vanilla Episode (no kind set) defaults to kind='log' — backward
    compat for L0-L2 callers."""
    lg = LifeGraph()
    ep = Episode(
        user_id=TEST_USER, id=_new_id(),
        text="Had ramen",
        occurred_at=T_NOW,
        ingested_at=T_NOW,
    )
    lg._store.save_episode(ep, [], [])

    fetched = lg.episodes.get(ep.id)
    assert fetched is not None
    assert fetched.kind == "log"
    assert fetched.status == "active"  # default


# ----- lifecycle -----


def test_complete_task_sets_completed_at() -> None:
    lg = LifeGraph()
    ep = _task_episode(text="thing")
    lg._store.save_episode(ep, [], [])

    completion_t = T_NOW + timedelta(hours=2)
    lg.complete_task(ep.id, at=completion_t)

    fetched = lg.episodes.get(ep.id)
    assert fetched is not None
    assert fetched.status == "done"
    assert fetched.completed_at == completion_t


def test_drop_task_sets_status_dropped() -> None:
    lg = LifeGraph()
    ep = _task_episode(text="thing")
    lg._store.save_episode(ep, [], [])

    lg.drop_task(ep.id)

    fetched = lg.episodes.get(ep.id)
    assert fetched is not None
    assert fetched.status == "dropped"
    assert fetched.completed_at is None  # dropped ≠ done


def test_reopen_task_clears_completion() -> None:
    """Reopening a done task clears completed_at — semantic invariant
    (an active task can't have a completion timestamp)."""
    lg = LifeGraph()
    ep = _task_episode(text="thing")
    lg._store.save_episode(ep, [], [])

    lg.complete_task(ep.id)
    fetched = lg.episodes.get(ep.id)
    assert fetched is not None and fetched.status == "done"

    lg.reopen_task(ep.id)
    fetched = lg.episodes.get(ep.id)
    assert fetched is not None
    assert fetched.status == "active"
    assert fetched.completed_at is None


# ----- task views -----


def test_pending_returns_only_active_tasks() -> None:
    lg = LifeGraph()
    ep_active = _task_episode(text="active thing")
    ep_done = _task_episode(text="done thing", status="done")
    lg._store.save_episode(ep_active, [], [])
    lg._store.save_episode(ep_done, [], [])

    pending = lg.tasks.pending(user_id=TEST_USER)
    assert len(pending) == 1
    assert pending[0].text == "active thing"


def test_overdue_filters_by_deadline() -> None:
    lg = LifeGraph()
    overdue = _task_episode(text="overdue", deadline=T_LAST_WEEK)
    future = _task_episode(text="future", deadline=T_NEXT_WEEK)
    lg._store.save_episode(overdue, [], [])
    lg._store.save_episode(future, [], [])

    od = lg.tasks.overdue(as_of=T_NOW, user_id=TEST_USER)
    assert {t.text for t in od} == {"overdue"}


def test_due_soon_window() -> None:
    lg = LifeGraph()
    very_soon = _task_episode(text="tomorrow", deadline=T_TOMORROW)
    next_week = _task_episode(text="next week", deadline=T_NEXT_WEEK)
    last_week = _task_episode(text="last week", deadline=T_LAST_WEEK)
    for ep in (very_soon, next_week, last_week):
        lg._store.save_episode(ep, [], [])

    soon = lg.tasks.due_soon(timedelta(days=3), as_of=T_NOW, user_id=TEST_USER)
    assert {t.text for t in soon} == {"tomorrow"}


def test_by_context_filters_active_only() -> None:
    lg = LifeGraph()
    a_work = _task_episode(text="work thing", gtd_context="@work")
    b_home = _task_episode(text="home thing", gtd_context="@home")
    c_done_work = _task_episode(text="done work thing", gtd_context="@work", status="done")
    for ep in (a_work, b_home, c_done_work):
        lg._store.save_episode(ep, [], [])

    work = lg.tasks.by_context("@work", user_id=TEST_USER)
    assert {t.text for t in work} == {"work thing"}  # done excluded


def test_by_priority_filters_active_only() -> None:
    lg = LifeGraph()
    high = _task_episode(text="high", priority="high")
    medium = _task_episode(text="medium", priority="medium")
    low_done = _task_episode(text="low done", priority="low", status="done")
    for ep in (high, medium, low_done):
        lg._store.save_episode(ep, [], [])

    h = lg.tasks.by_priority("high", user_id=TEST_USER)
    assert {t.text for t in h} == {"high"}


def test_completed_in_window() -> None:
    lg = LifeGraph()
    ep_a = _task_episode(text="a")
    ep_b = _task_episode(text="b")
    lg._store.save_episode(ep_a, [], [])
    lg._store.save_episode(ep_b, [], [])
    lg.complete_task(ep_a.id, at=T_NOW)
    lg.complete_task(ep_b.id, at=T_LAST_WEEK)

    recent = lg.tasks.completed_in(T_NOW - timedelta(days=2), T_NOW + timedelta(days=2), user_id=TEST_USER)
    assert {t.text for t in recent} == {"a"}


# ----- mixed log + task queries -----


def test_entity_query_finds_tasks_alongside_logs() -> None:
    """Tasks and logs share the entity-mention table — querying for a
    Person who appears in both returns both."""
    lg = LifeGraph()
    sara = Person(user_id=TEST_USER, value="Sara", key="sara")

    log_ep = Episode(
        user_id=TEST_USER, id=_new_id(),
        text="Met Sara today",
        occurred_at=T_NOW,
        ingested_at=T_NOW,
        kind="log",
    )
    task_ep = Episode(
        user_id=TEST_USER, id=_new_id(),
        text="Email Sara about the meeting",
        occurred_at=T_NOW,
        ingested_at=T_NOW,
        kind="task",
        status="active",
    )
    lg._store.save_episode(log_ep, [sara], [])
    lg._store.save_episode(task_ep, [sara], [])

    eps = lg.episodes.mentioning(sara)
    assert len(eps) == 2
    kinds = {e.kind for e in eps}
    assert kinds == {"log", "task"}


def test_topic_food_episodes_pivot_excludes_unrelated_tasks() -> None:
    """The food-pivot still works correctly with tasks in the mix."""
    lg = LifeGraph()
    ramen = Topic(user_id=TEST_USER, value="ramen", key="ramen", kind="food")
    laundry = Topic(user_id=TEST_USER, value="laundry", key="laundry", kind="general")

    log_ep = Episode(
        user_id=TEST_USER, id=_new_id(),
        text="Had ramen",
        occurred_at=T_NOW,
        ingested_at=T_NOW,
        kind="log",
    )
    task_ep = Episode(
        user_id=TEST_USER, id=_new_id(),
        text="Do laundry",
        occurred_at=T_NOW,
        ingested_at=T_NOW,
        kind="task",
    )
    lg._store.save_episode(log_ep, [ramen], [])
    lg._store.save_episode(task_ep, [laundry], [])

    food_eps = lg.query(Topic, kind="food", user_id=TEST_USER).episodes()
    assert len(food_eps) == 1
    assert food_eps[0].text == "Had ramen"


# ----- migration parity -----


def test_task_extraction_prompt_exists() -> None:
    """TASK_EXTRACTION_PROMPT exists and is a non-trivial string."""
    from lifegraph_kg.extract.prompt import TASK_EXTRACTION_PROMPT

    assert len(TASK_EXTRACTION_PROMPT) > 1000
    assert "task" in TASK_EXTRACTION_PROMPT.lower()
    assert "deadline_hint" in TASK_EXTRACTION_PROMPT
