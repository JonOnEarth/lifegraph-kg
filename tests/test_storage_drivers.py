# SPDX-License-Identifier: Apache-2.0
"""Tests for the L4 storage drivers (Postgres + Kuzu).

PostgresStore is full-fledged but the integration tests need a live
Postgres. They're gated on ``LIFEGRAPH_TEST_POSTGRES_URL`` env var —
skip in CI without secrets, run locally with::

    LIFEGRAPH_TEST_POSTGRES_URL=postgres://user@localhost/test_lifegraph \\
        uv run pytest tests/test_storage_drivers.py -v

The Postgres tests run the SAME assertions as test_kg_store.py — the
Store protocol is the contract; SQLite and Postgres should be
behaviorally identical.

KuzuStore is stubbed (L4.1) — only test the dispatch path.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from lifegraph_kg import LifeGraph, Person, Place, Topic
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode

POSTGRES_URL = os.environ.get("LIFEGRAPH_TEST_POSTGRES_URL")

T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 5, 5, 18, 0, 0, tzinfo=UTC)
T2 = datetime(2026, 5, 9, 9, 0, 0, tzinfo=UTC)


# ----- URI dispatch (no external deps) -----


def test_resolve_in_memory_returns_sqlite() -> None:
    """``:memory:`` is the default and routes to SqliteStore."""
    from lifegraph_kg.kg import _resolve_store
    from lifegraph_kg.kg.store.sqlite import SqliteStore

    store = _resolve_store(":memory:")
    assert isinstance(store, SqliteStore)


def test_resolve_sqlite_uri_strips_prefix() -> None:
    """``sqlite:///path`` and ``sqlite:path`` both reach SqliteStore."""
    from lifegraph_kg.kg import _resolve_store
    from lifegraph_kg.kg.store.sqlite import SqliteStore

    # Use :memory: as the path so the test doesn't touch disk.
    store_a = _resolve_store("sqlite:///:memory:")
    store_b = _resolve_store("sqlite::memory:")
    assert isinstance(store_a, SqliteStore)
    assert isinstance(store_b, SqliteStore)


def test_resolve_postgres_uri_imports_lazily() -> None:
    """``postgres://...`` dispatches to PostgresStore. Without psycopg
    installed the import fails — but the dispatch itself happens before
    the import, and the URL parsing is what we're testing here."""
    from lifegraph_kg.kg import _resolve_store

    if POSTGRES_URL is None:
        # We expect an ImportError or ConnectionError, not a
        # "wrong store type" mismatch. The dispatch logic is what's
        # under test.
        with pytest.raises((ImportError, Exception)):
            _resolve_store("postgres://user@nowhere:9999/db")
    else:
        from lifegraph_kg.kg.store.postgres import PostgresStore

        store = _resolve_store(POSTGRES_URL)
        assert isinstance(store, PostgresStore)
        store.close()


def test_resolve_kuzu_uri_dispatches_to_stub() -> None:
    """KuzuStore stub raises a clear NotImplementedError so users
    who try ``kuzu://`` in v0.1 get a helpful message."""
    from lifegraph_kg.kg import _resolve_store

    with pytest.raises(NotImplementedError, match=r"L4\.1"):
        _resolve_store("kuzu:///tmp/not-implemented")


# ----- Postgres integration tests (gated) -----

# These tests run only when LIFEGRAPH_TEST_POSTGRES_URL is set. They
# create a fresh isolated test DB on each run via a setup/teardown
# fixture that DROPs all our tables. Don't point this at a real DB.

requires_postgres = pytest.mark.skipif(
    POSTGRES_URL is None,
    reason="set LIFEGRAPH_TEST_POSTGRES_URL=postgres://... to run integration tests",
)


@pytest.fixture
def fresh_pg_store():  # type: ignore[no-untyped-def]
    """Fresh isolated PostgresStore. Drops + recreates schema."""
    from lifegraph_kg.kg.store.postgres import PostgresStore

    assert POSTGRES_URL is not None  # guarded by requires_postgres
    store = PostgresStore(POSTGRES_URL)
    # Drop everything we created — SAFE because we own these table names
    # and the user opted into a test DB.
    with store._conn.cursor() as cur:
        for table in (
            "merge_proposals",
            "entity_episode_mention",
            "edges",
            "entities",
            "episodes",
        ):
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    store._conn.commit()
    # Re-init schema fresh.
    store.init_schema()
    yield store
    store.close()


@requires_postgres
def test_postgres_save_and_get_episode(fresh_pg_store) -> None:  # type: ignore[no-untyped-def]
    s = fresh_pg_store
    ep = Episode(
        id="ep-1",
        text="Had ramen with Sara at Ippudo",
        occurred_at=T1,
        ingested_at=T1,
        source="user",
        predicates=["ate"],
    )
    sara = Person(value="Sara", key="sara")
    ramen = Topic(value="ramen", key="ramen", kind="food")
    s.save_episode(ep, entities=[sara, ramen], edges=[])

    fetched = s.get_episode("ep-1")
    assert fetched is not None
    assert fetched.text == ep.text
    assert fetched.predicates == ["ate"]


@requires_postgres
def test_postgres_entity_dedup_by_type_and_key(fresh_pg_store) -> None:  # type: ignore[no-untyped-def]
    s = fresh_pg_store
    sara = Person(value="Sara", key="sara")
    s.save_episode(
        Episode(id="ep-1", text="First", occurred_at=T0, ingested_at=T0),
        entities=[sara],
        edges=[],
    )
    s.save_episode(
        Episode(id="ep-2", text="Second", occurred_at=T1, ingested_at=T1),
        entities=[sara],
        edges=[],
    )
    rows = s.query_entities(type_="Person", key="sara")
    assert len(rows) == 1


@requires_postgres
def test_postgres_episodes_mentioning(fresh_pg_store) -> None:  # type: ignore[no-untyped-def]
    s = fresh_pg_store
    sara = Person(value="Sara", key="sara")
    s.save_episode(
        Episode(id="ep-1", text="Met Sara", occurred_at=T0, ingested_at=T0),
        entities=[sara],
        edges=[],
    )
    s.save_episode(
        Episode(id="ep-2", text="Saw Tao", occurred_at=T1, ingested_at=T1),
        entities=[Person(value="Tao", key="tao")],
        edges=[],
    )
    sara_id = s.find_entity_id("Person", "sara")
    assert sara_id is not None
    eps = s.episodes_mentioning(sara_id)
    assert {e.id for e in eps} == {"ep-1"}


@requires_postgres
def test_postgres_episodes_since_orders_by_time(fresh_pg_store) -> None:  # type: ignore[no-untyped-def]
    s = fresh_pg_store
    s.save_episode(
        Episode(id="old", text="old", occurred_at=T0, ingested_at=T0),
        entities=[],
        edges=[],
    )
    s.save_episode(
        Episode(id="new", text="new", occurred_at=T2, ingested_at=T2),
        entities=[],
        edges=[],
    )
    eps = s.episodes_since(T0 - timedelta(days=1))
    assert [e.id for e in eps] == ["new", "old"]


@requires_postgres
def test_postgres_invalidate_edge_supersede(fresh_pg_store) -> None:  # type: ignore[no-untyped-def]
    """Sara-moves-to-Tokyo on Postgres. Same semantics as SQLite test."""
    s = fresh_pg_store
    sara = Person(value="Sara", key="sara")
    berlin = Place(value="Berlin", key="berlin")
    tokyo = Place(value="Tokyo", key="tokyo")

    ep1 = Episode(
        id="ep-berlin",
        text="Sara lives in Berlin",
        occurred_at=datetime(2025, 6, 1, tzinfo=UTC),
        ingested_at=datetime(2025, 6, 1, tzinfo=UTC),
        predicates=["lives_in"],
    )
    s.save_episode(ep1, entities=[sara, berlin], edges=[])

    sara_id = s.find_entity_id("Person", "sara")
    berlin_id = s.find_entity_id("Place", "berlin")
    assert sara_id and berlin_id

    edge = Edge(
        id="edge-1",
        from_entity=sara_id,
        to_entity=berlin_id,
        verb="lives_in",
        episode_id=ep1.id,
        t_event=ep1.occurred_at,
        t_ingestion=ep1.ingested_at,
        t_valid=ep1.occurred_at,
    )
    s.add_edges([edge])

    move_t = datetime(2026, 1, 15, tzinfo=UTC)
    s.invalidate_edge("edge-1", move_t)
    s.save_episode(
        Episode(
            id="ep-tokyo",
            text="Sara moved to Tokyo",
            occurred_at=move_t,
            ingested_at=move_t,
            predicates=["lives_in"],
        ),
        entities=[tokyo],
        edges=[],
    )

    # Pre-move query
    facts_dec = s.edges_as_of(datetime(2025, 12, 1, tzinfo=UTC), verb="lives_in")
    assert len(facts_dec) == 1
    assert facts_dec[0].to_entity == berlin_id

    # Audit trail preserved
    edges_for_ep = s.edges_for_episode("ep-berlin")
    assert len(edges_for_ep) == 1
    assert edges_for_ep[0].t_invalid == move_t


@requires_postgres
def test_postgres_lifegraph_end_to_end() -> None:
    """LifeGraph(store=postgres://...) works just like the SQLite path."""
    from tests.test_extraction import FakeClient

    extraction = json.dumps(
        {
            "predicates": ["ate"],
            "body_state": None,
            "sentiment": None,
            "energy": None,
            "entities": [
                {"type": "Person", "value": "Sara", "key": "sara"},
                {"type": "Topic", "kind": "food", "value": "ramen", "key": "ramen"},
            ],
        }
    )
    fake = FakeClient(extraction_response=extraction)
    assert POSTGRES_URL is not None  # guarded by mark
    lg = LifeGraph(store=POSTGRES_URL, llm=fake)

    # Reset DB before test
    from lifegraph_kg.kg.store.postgres import PostgresStore

    assert isinstance(lg._store, PostgresStore)
    with lg._store._conn.cursor() as cur:
        for t in (
            "merge_proposals",
            "entity_episode_mention",
            "edges",
            "entities",
            "episodes",
        ):
            cur.execute(f"DELETE FROM {t}")
    lg._store._conn.commit()

    ep = lg.log("Had ramen with Sara at Ippudo", occurred_at=T1)
    assert ep.predicates == ["ate"]
    sara = lg.query(Person, key="sara").one()
    assert sara.value == "Sara"
    eps = lg.episodes.mentioning(sara)
    assert len(eps) == 1
    lg._store.close()
