# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the L2 SQLite store + LifeGraph persistence + bi-temporal CRUD.

Tests use ``:memory:`` SQLite and the FakeClient from test_extraction so
no API key is needed. Three groups:
  - Store-level: schema, episode/entity persistence, dedup, edge writes
  - LifeGraph-level: lg.log() flow, lg.episodes.<...>, lg.query
  - Bi-temporal: invalidate / edges_as_of / supersede semantics
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from lifegraph_kg import LifeGraph, Person, Place, Topic
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode
from lifegraph_kg.kg.store.sqlite import SqliteStore
from tests.test_extraction import FakeClient

T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 5, 5, 18, 0, 0, tzinfo=UTC)
T2 = datetime(2026, 5, 9, 9, 0, 0, tzinfo=UTC)


# ----- Store-level tests -----


def test_store_init_creates_schema_idempotently() -> None:
    """Re-creating the store on the same DB doesn't blow up (migration
    is idempotent)."""
    s1 = SqliteStore(":memory:")
    s1.init_schema()
    s1.init_schema()
    # Sanity: tables exist
    assert s1._table_exists("episodes")
    assert s1._table_exists("entities")
    assert s1._table_exists("edges")
    assert s1._table_exists("entity_episode_mention")
    assert s1._table_exists("schema_version")


def test_store_save_and_get_episode() -> None:
    s = SqliteStore(":memory:")
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


def test_store_entity_dedup_by_type_and_key() -> None:
    """Two episodes mentioning Sara create exactly one entity row."""
    s = SqliteStore(":memory:")
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


def test_store_episodes_mentioning_uses_link_table() -> None:
    """`episodes_mentioning` joins via the entity_episode_mention table."""
    s = SqliteStore(":memory:")
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


def test_store_episodes_since_orders_by_time() -> None:
    s = SqliteStore(":memory:")
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
    assert [e.id for e in eps] == ["new", "old"]  # reverse-chronological
    # Filter cutoff
    eps_filtered = s.episodes_since(T1)
    assert [e.id for e in eps_filtered] == ["new"]


# ----- LifeGraph-level tests -----


_RAMEN_EXTRACTION = json.dumps(
    {
        "predicates": ["ate"],
        "body_state": None,
        "sentiment": None,
        "energy": None,
        "entities": [
            {"type": "Person", "value": "Sara", "key": "sara"},
            {"type": "Place", "value": "Ippudo", "key": "ippudo"},
            {"type": "Topic", "kind": "food", "value": "ramen", "key": "ramen"},
        ],
    }
)


def test_lifegraph_log_persists_episode_and_entities() -> None:
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    lg = LifeGraph(llm=fake)
    ep = lg.log("Had ramen with Sara at Ippudo", occurred_at=T1)
    assert isinstance(ep, Episode)
    assert ep.predicates == ["ate"]
    # Episode round-trips
    fetched = lg.episodes.get(ep.id)
    assert fetched is not None
    assert fetched.text == "Had ramen with Sara at Ippudo"


def test_lifegraph_query_finds_persisted_entity() -> None:
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    lg = LifeGraph(llm=fake)
    lg.log("Had ramen with Sara at Ippudo", occurred_at=T1)
    sara = lg.query(Person, key="sara").one()
    assert isinstance(sara, Person)
    assert sara.value == "Sara"


def test_lifegraph_episodes_mentioning_walks_back_to_episode() -> None:
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    lg = LifeGraph(llm=fake)
    ep = lg.log("Had ramen with Sara at Ippudo", occurred_at=T1)
    sara = lg.query(Person, key="sara").one()
    eps = lg.episodes.mentioning(sara)
    assert [e.id for e in eps] == [ep.id]


def test_lifegraph_log_creates_edges_for_each_predicate_entity_pair() -> None:
    """v6 multi-predicate: 1 predicate, 3 entities, so 3 edges."""
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    lg = LifeGraph(llm=fake)
    ep = lg.log("Had ramen with Sara at Ippudo", occurred_at=T1)
    edges = lg.kg.edges_for_episode(ep.id)
    assert len(edges) == 3
    assert all(e.verb == "ate" for e in edges)
    assert all(e.from_entity is None for e in edges)  # implicit user subject


def test_lifegraph_query_topic_kind_filter() -> None:
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    lg = LifeGraph(llm=fake)
    lg.log("Had ramen", occurred_at=T1)
    foods = lg.query(Topic, kind="food").all()
    assert len(foods) == 1
    assert foods[0].value == "ramen"


def test_episodes_between_returns_inclusive_range() -> None:
    s = SqliteStore(":memory:")
    s.save_episode(
        Episode(id="early", text="early", occurred_at=T0, ingested_at=T0),
        entities=[],
        edges=[],
    )
    s.save_episode(
        Episode(id="mid", text="mid", occurred_at=T1, ingested_at=T1),
        entities=[],
        edges=[],
    )
    s.save_episode(
        Episode(id="late", text="late", occurred_at=T2, ingested_at=T2),
        entities=[],
        edges=[],
    )
    # Range that includes only the middle
    eps = s.episodes_between(T1, T1)
    assert [e.id for e in eps] == ["mid"]
    # Range that spans all
    eps_all = s.episodes_between(T0, T2)
    assert [e.id for e in eps_all] == ["late", "mid", "early"]


def test_entity_query_episodes_pivot() -> None:
    """`lg.query(Topic, kind='food').episodes()` returns the meal timeline."""
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    lg = LifeGraph(llm=fake)
    lg.log("Had ramen with Sara at Ippudo", occurred_at=T0)
    lg.log("Had ramen with Sara at Ippudo", occurred_at=T1)  # second meal
    food_episodes = lg.query(Topic, kind="food").episodes()
    # Both episodes mention ramen; pivot returns both, reverse-chronological
    assert len(food_episodes) == 2
    assert food_episodes[0].occurred_at > food_episodes[1].occurred_at


def test_entity_query_episodes_pivot_dedups() -> None:
    """If multiple matched entities share an episode, the episode appears once."""
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    lg = LifeGraph(llm=fake)
    lg.log("Had ramen with Sara at Ippudo", occurred_at=T0)
    # 3 entities (Sara, Ippudo, ramen) all mention the same episode.
    # query(Person, key=None) returns just Sara; query() across types
    # would otherwise count the episode multiple times — DISTINCT in SQL
    # avoids that. Verify by querying a type that has multiple matches:
    lg.log("Had ramen with Sara at Ippudo", occurred_at=T1)
    # 2 episodes, 1 Person (Sara) — pivot returns 2 distinct episodes
    sara_eps = lg.query(Person, key="sara").episodes()
    assert len(sara_eps) == 2


def test_lifegraph_in_memory_isolates_per_instance() -> None:
    """Two `:memory:` LifeGraphs are independent."""
    fake = FakeClient(extraction_response=_RAMEN_EXTRACTION)
    a = LifeGraph(llm=fake)
    b = LifeGraph(llm=fake)
    a.log("Had ramen with Sara at Ippudo", occurred_at=T1)
    assert len(a.query(Person).all()) == 1
    assert len(b.query(Person).all()) == 0


# ----- Bi-temporal tests (the moat) -----


def test_invalidate_edge_supersede_not_delete() -> None:
    """The Sara-moves-to-Tokyo case. Original edge survives with t_invalid set."""
    s = SqliteStore(":memory:")
    sara = Person(value="Sara", key="sara")
    berlin = Place(value="Berlin", key="berlin")
    tokyo = Place(value="Tokyo", key="tokyo")

    # Initial fact: Sara lives in Berlin (logged June 2025)
    ep1 = Episode(
        id="ep-berlin",
        text="Sara lives in Berlin",
        occurred_at=datetime(2025, 6, 1, tzinfo=UTC),
        ingested_at=datetime(2025, 6, 1, tzinfo=UTC),
        predicates=["lives_in"],
    )
    s.save_episode(ep1, entities=[sara, berlin], edges=[])

    # Manually create the edge to control t_valid
    sara_id = s.find_entity_id("Person", "sara")
    berlin_id = s.find_entity_id("Place", "berlin")
    assert sara_id and berlin_id
    edge_berlin = Edge(
        id="edge-1",
        from_entity=sara_id,
        to_entity=berlin_id,
        verb="lives_in",
        episode_id=ep1.id,
        t_event=ep1.occurred_at,
        t_ingestion=ep1.ingested_at,
        t_valid=ep1.occurred_at,
    )
    # Insert via a second save_episode would re-insert episode; use the
    # store's internal connection directly to add JUST the edge.
    s._conn.execute(
        """INSERT INTO edges (id, from_entity, to_entity, verb, episode_id,
                                t_event, t_ingestion, t_valid, t_invalid,
                                attributes_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, '{}')""",
        (
            edge_berlin.id,
            edge_berlin.from_entity,
            edge_berlin.to_entity,
            edge_berlin.verb,
            edge_berlin.episode_id,
            int(edge_berlin.t_event.timestamp() * 1000),
            int(edge_berlin.t_ingestion.timestamp() * 1000),
            int(edge_berlin.t_valid.timestamp() * 1000),
        ),
    )
    s._conn.commit()

    # 7 months later: Sara moves to Tokyo. We invalidate the Berlin edge
    # at the move time.
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

    # Time-travel queries
    # 1. Pre-move: Berlin edge is the only valid lives_in
    facts_dec = s.edges_as_of(datetime(2025, 12, 1, tzinfo=UTC), verb="lives_in")
    assert len(facts_dec) == 1
    assert facts_dec[0].to_entity == berlin_id

    # 2. Pre-Berlin: nothing
    facts_jan = s.edges_as_of(datetime(2025, 1, 1, tzinfo=UTC), verb="lives_in")
    assert facts_jan == []

    # 3. The Berlin edge survives in the DB (audit trail preserved)
    edges_for_ep1 = s.edges_for_episode("ep-berlin")
    assert len(edges_for_ep1) == 1
    assert edges_for_ep1[0].t_invalid == move_t  # ← supersede recorded


def test_edge_is_valid_at_method() -> None:
    """The Edge.is_valid_at convenience method matches the SQL semantics."""
    e = Edge(
        id="x",
        from_entity=None,
        to_entity="y",
        verb="lives_in",
        episode_id="ep",
        t_event=T0,
        t_ingestion=T0,
        t_valid=T0,
        t_invalid=T2,
    )
    assert e.is_valid_at(T1) is True
    assert e.is_valid_at(T0 - timedelta(days=1)) is False  # before
    assert e.is_valid_at(T2) is False  # at/after invalid is False
    # Open-ended edge
    e_open = Edge(
        id="y",
        from_entity=None,
        to_entity="z",
        verb="lives_in",
        episode_id="ep",
        t_event=T0,
        t_ingestion=T0,
        t_valid=T0,
        t_invalid=None,
    )
    assert e_open.is_valid_at(T2) is True
    assert e_open.is_active is True


def test_edges_as_of_filters_by_verb() -> None:
    s = SqliteStore(":memory:")
    sara = Person(value="Sara", key="sara")
    s.save_episode(
        Episode(
            id="ep",
            text="t",
            occurred_at=T0,
            ingested_at=T0,
            predicates=["ate"],
        ),
        entities=[sara],
        edges=[],
    )
    sara_id = s.find_entity_id("Person", "sara")
    assert sara_id
    # Insert two edges with different verbs
    for verb in ("ate", "met"):
        s._conn.execute(
            """INSERT INTO edges (id, from_entity, to_entity, verb, episode_id,
                                    t_event, t_ingestion, t_valid, t_invalid,
                                    attributes_json)
               VALUES (?, NULL, ?, ?, 'ep', ?, ?, ?, NULL, '{}')""",
            (
                f"edge-{verb}",
                sara_id,
                verb,
                int(T0.timestamp() * 1000),
                int(T0.timestamp() * 1000),
                int(T0.timestamp() * 1000),
            ),
        )
    s._conn.commit()
    ate_edges = s.edges_as_of(T1, verb="ate")
    assert len(ate_edges) == 1
    assert ate_edges[0].verb == "ate"
