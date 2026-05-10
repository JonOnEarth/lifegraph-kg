# SPDX-License-Identifier: Apache-2.0
"""PostgresStore — opt-in Postgres backend.

For deployments that need shared/multi-user access. Same Store protocol
as SqliteStore — `LifeGraph(store="postgres://user:pass@host/db")` is
the only change. PGlite (the WASM Postgres in the existing LifeGraph
frontend) speaks the PG protocol, so the same DSN works against PGlite
when you tunnel it.

Install with: ``pip install 'lifegraph-kg[postgres]'`` (psycopg[binary]).

Differences vs SQLite:
  - INTEGER → BIGINT for unix-ms timestamps (Y2038 safety)
  - placeholders: %s (psycopg) instead of ?
  - executescript: semicolon-split statements
  - ON CONFLICT syntax is the same (PG 9.5+)

The schema is embedded here as Python strings rather than .sql files
so that backend-specific tweaks (BIGINT, JSONB if we want it later)
don't bleed across backends.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from lifegraph_kg.classes import Person, Place, Project, Topic
from lifegraph_kg.extract.schema import EntityT
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.rows import DictRow


SCHEMA_SQL = [
    """\
CREATE TABLE IF NOT EXISTS episodes (
  id            TEXT PRIMARY KEY,
  text          TEXT NOT NULL,
  occurred_at   BIGINT NOT NULL,
  ingested_at   BIGINT NOT NULL,
  source        TEXT,
  predicates    TEXT NOT NULL DEFAULT '[]',
  body_state    TEXT,
  sentiment     TEXT,
  energy        TEXT
)""",
    "CREATE INDEX IF NOT EXISTS idx_episodes_occurred_at ON episodes(occurred_at)",
    """\
CREATE TABLE IF NOT EXISTS entities (
  id              TEXT PRIMARY KEY,
  type            TEXT NOT NULL,
  kind            TEXT,
  key             TEXT NOT NULL,
  value           TEXT NOT NULL,
  attributes_json TEXT NOT NULL DEFAULT '{}',
  created_at      BIGINT NOT NULL,
  canonical_id    TEXT REFERENCES entities(id),
  UNIQUE(type, key)
)""",
    "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
    "CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind)",
    "CREATE INDEX IF NOT EXISTS idx_entities_canonical_id ON entities(canonical_id)",
    """\
CREATE TABLE IF NOT EXISTS edges (
  id            TEXT PRIMARY KEY,
  from_entity   TEXT REFERENCES entities(id),
  to_entity     TEXT NOT NULL REFERENCES entities(id),
  verb          TEXT NOT NULL,
  episode_id    TEXT NOT NULL REFERENCES episodes(id),
  t_event       BIGINT NOT NULL,
  t_ingestion   BIGINT NOT NULL,
  t_valid       BIGINT NOT NULL,
  t_invalid     BIGINT,
  attributes_json TEXT NOT NULL DEFAULT '{}'
)""",
    "CREATE INDEX IF NOT EXISTS idx_edges_to_entity ON edges(to_entity)",
    "CREATE INDEX IF NOT EXISTS idx_edges_from_entity ON edges(from_entity)",
    "CREATE INDEX IF NOT EXISTS idx_edges_episode ON edges(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_t_valid ON edges(t_valid)",
    "CREATE INDEX IF NOT EXISTS idx_edges_verb ON edges(verb)",
    """\
CREATE TABLE IF NOT EXISTS entity_episode_mention (
  entity_id   TEXT NOT NULL REFERENCES entities(id),
  episode_id  TEXT NOT NULL REFERENCES episodes(id),
  PRIMARY KEY (entity_id, episode_id)
)""",
    """\
CREATE TABLE IF NOT EXISTS merge_proposals (
  id            TEXT PRIMARY KEY,
  winner_id     TEXT NOT NULL REFERENCES entities(id),
  loser_id      TEXT NOT NULL REFERENCES entities(id),
  confidence    TEXT NOT NULL,
  reason        TEXT NOT NULL,
  detail        TEXT NOT NULL DEFAULT '',
  proposed_at   BIGINT NOT NULL,
  applied_at    BIGINT,
  rejected_at   BIGINT
)""",
    "CREATE INDEX IF NOT EXISTS idx_merge_proposals_pending "
    "ON merge_proposals(applied_at, rejected_at)",
]


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _entity_from_row(row: dict[str, Any]) -> EntityT:
    type_ = row["type"]
    common = {
        "key": row["key"],
        "value": row["value"],
        "attributes": json.loads(row["attributes_json"] or "{}"),
    }
    if type_ == "Person":
        return Person(**common)
    if type_ == "Place":
        return Place(**common)
    if type_ == "Project":
        return Project(**common)
    if type_ == "Topic":
        return Topic(kind=row["kind"] or "general", **common)
    raise ValueError(f"Unknown entity type in DB: {type_!r}")


def _episode_from_row(row: dict[str, Any]) -> Episode:
    occurred_at = _from_ms(row["occurred_at"])
    ingested_at = _from_ms(row["ingested_at"])
    assert occurred_at is not None
    assert ingested_at is not None
    return Episode(
        id=row["id"],
        text=row["text"],
        occurred_at=occurred_at,
        ingested_at=ingested_at,
        source=row["source"] or "user",
        predicates=json.loads(row["predicates"] or "[]"),
        body_state=row["body_state"],
        sentiment=row["sentiment"],
        energy=row["energy"],
    )


def _edge_from_row(row: dict[str, Any]) -> Edge:
    t_event = _from_ms(row["t_event"])
    t_ingestion = _from_ms(row["t_ingestion"])
    t_valid = _from_ms(row["t_valid"])
    assert t_event is not None
    assert t_ingestion is not None
    assert t_valid is not None
    return Edge(
        id=row["id"],
        from_entity=row["from_entity"],
        to_entity=row["to_entity"],
        verb=row["verb"],
        episode_id=row["episode_id"],
        t_event=t_event,
        t_ingestion=t_ingestion,
        t_valid=t_valid,
        t_invalid=_from_ms(row["t_invalid"]),
    )


class PostgresStore:
    """Postgres backend implementing the Store protocol.

    Args:
        dsn: a Postgres DSN like ``postgres://user:pass@host:5432/dbname``.
              The ``postgresql://`` scheme is also accepted (psycopg
              normalizes both).
    """

    def __init__(self, dsn: str) -> None:
        # Lazy import — psycopg is an optional dependency.
        import psycopg
        from psycopg.rows import dict_row

        self._dsn = dsn
        self._conn: Connection[DictRow] = psycopg.connect(dsn, row_factory=dict_row)
        self.init_schema()

    # --- schema ---

    def init_schema(self) -> None:
        """Apply the schema. Each CREATE statement is idempotent
        (IF NOT EXISTS), so re-running on an existing DB is safe."""
        with self._conn.cursor() as cur:
            for stmt in SCHEMA_SQL:
                cur.execute(stmt)
        self._conn.commit()

    # --- save (atomic write) ---

    def save_episode(
        self,
        episode: Episode,
        entities: list[Any],
        edges: list[Edge],
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO episodes
                     (id, text, occurred_at, ingested_at, source,
                      predicates, body_state, sentiment, energy)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    episode.id,
                    episode.text,
                    _to_ms(episode.occurred_at),
                    _to_ms(episode.ingested_at),
                    episode.source,
                    json.dumps(episode.predicates, ensure_ascii=False),
                    episode.body_state,
                    episode.sentiment,
                    episode.energy,
                ),
            )
            for entity in entities:
                cur.execute(
                    """INSERT INTO entities
                         (id, type, kind, key, value, attributes_json, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (type, key) DO NOTHING""",
                    (
                        _new_id(),
                        entity.type,
                        getattr(entity, "kind", None),
                        entity.key,
                        entity.value,
                        json.dumps(entity.attributes, ensure_ascii=False),
                        _to_ms(datetime.now(UTC)),
                    ),
                )
                cur.execute(
                    "SELECT id FROM entities WHERE type = %s AND key = %s",
                    (entity.type, entity.key),
                )
                row = cur.fetchone()
                if row is not None:
                    cur.execute(
                        """INSERT INTO entity_episode_mention (entity_id, episode_id)
                           VALUES (%s, %s) ON CONFLICT DO NOTHING""",
                        (row["id"], episode.id),
                    )

            for edge in edges:
                self._insert_edge(cur, edge)
        self._conn.commit()

    def _insert_edge(self, cur: Any, edge: Edge) -> None:
        cur.execute(
            """INSERT INTO edges
                 (id, from_entity, to_entity, verb, episode_id,
                  t_event, t_ingestion, t_valid, t_invalid, attributes_json)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, '{}')""",
            (
                edge.id,
                edge.from_entity,
                edge.to_entity,
                edge.verb,
                edge.episode_id,
                _to_ms(edge.t_event),
                _to_ms(edge.t_ingestion),
                _to_ms(edge.t_valid),
                _to_ms(edge.t_invalid) if edge.t_invalid else None,
            ),
        )

    def add_edges(self, edges: list[Edge]) -> None:
        if not edges:
            return
        with self._conn.cursor() as cur:
            for edge in edges:
                self._insert_edge(cur, edge)
        self._conn.commit()

    # --- episode reads ---

    def get_episode(self, episode_id: str) -> Episode | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM episodes WHERE id = %s", (episode_id,))
            row = cur.fetchone()
        return _episode_from_row(row) if row else None

    def episodes_since(self, t: datetime, limit: int | None = None) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE occurred_at >= %s ORDER BY occurred_at DESC"
        params: list[Any] = [_to_ms(t)]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    def episodes_between(
        self, start: datetime, end: datetime, limit: int | None = None
    ) -> list[Episode]:
        sql = (
            "SELECT * FROM episodes "
            "WHERE occurred_at >= %s AND occurred_at <= %s "
            "ORDER BY occurred_at DESC"
        )
        params: list[Any] = [_to_ms(start), _to_ms(end)]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    def episodes_mentioning(self, entity_id: str, limit: int | None = None) -> list[Episode]:
        sql = (
            "SELECT e.* FROM episodes e "
            "JOIN entity_episode_mention m ON m.episode_id = e.id "
            "WHERE m.entity_id = %s "
            "ORDER BY e.occurred_at DESC"
        )
        params: list[Any] = [entity_id]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    def episodes_mentioning_any(
        self, entity_ids: list[str], limit: int | None = None
    ) -> list[Episode]:
        if not entity_ids:
            return []
        sql = (
            "SELECT DISTINCT e.* FROM episodes e "
            "JOIN entity_episode_mention m ON m.episode_id = e.id "
            "WHERE m.entity_id = ANY(%s) "
            "ORDER BY e.occurred_at DESC"
        )
        params: list[Any] = [entity_ids]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    # --- entity reads ---

    def find_entity(self, type_: str, key: str) -> EntityT | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE type = %s AND key = %s",
                (type_, key),
            )
            row = cur.fetchone()
        return _entity_from_row(row) if row else None

    def find_entity_id(self, type_: str, key: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE type = %s AND key = %s",
                (type_, key),
            )
            row = cur.fetchone()
        return row["id"] if row else None

    def query_entities(
        self,
        type_: str | None = None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[EntityT]:
        sql = "SELECT * FROM entities WHERE 1=1"
        params: list[Any] = []
        if type_ is not None:
            sql += " AND type = %s"
            params.append(type_)
        if kind is not None:
            sql += " AND kind = %s"
            params.append(kind)
        if key is not None:
            sql += " AND key = %s"
            params.append(key)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_entity_from_row(r) for r in cur.fetchall()]

    # --- bi-temporal CRUD ---

    def invalidate_edge(self, edge_id: str, t_invalid: datetime) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE edges SET t_invalid = %s WHERE id = %s AND t_invalid IS NULL",
                (_to_ms(t_invalid), edge_id),
            )
        self._conn.commit()

    def edges_as_of(self, t: datetime, *, verb: str | None = None) -> list[Edge]:
        ms = _to_ms(t)
        sql = "SELECT * FROM edges WHERE t_valid <= %s AND (t_invalid IS NULL OR %s < t_invalid)"
        params: list[Any] = [ms, ms]
        if verb is not None:
            sql += " AND verb = %s"
            params.append(verb)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_edge_from_row(r) for r in cur.fetchall()]

    def edges_for_episode(self, episode_id: str) -> list[Edge]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM edges WHERE episode_id = %s ORDER BY id",
                (episode_id,),
            )
            return [_edge_from_row(r) for r in cur.fetchall()]

    # --- hygiene ---

    def record_proposal(
        self,
        proposal_id: str,
        winner_id: str,
        loser_id: str,
        confidence: str,
        reason: str,
        detail: str = "",
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO merge_proposals
                     (id, winner_id, loser_id, confidence, reason, detail, proposed_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    proposal_id,
                    winner_id,
                    loser_id,
                    confidence,
                    reason,
                    detail,
                    _to_ms(datetime.now(UTC)),
                ),
            )
        self._conn.commit()

    def apply_merge(self, winner_id: str, loser_id: str) -> None:
        if winner_id == loser_id:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE edges SET to_entity = %s WHERE to_entity = %s",
                (winner_id, loser_id),
            )
            cur.execute(
                "UPDATE edges SET from_entity = %s WHERE from_entity = %s",
                (winner_id, loser_id),
            )
            # Postgres equivalent of "UPDATE OR IGNORE": we DELETE
            # conflicting rows first, then UPDATE the rest.
            cur.execute(
                """DELETE FROM entity_episode_mention
                   WHERE entity_id = %s
                     AND episode_id IN (
                       SELECT episode_id FROM entity_episode_mention WHERE entity_id = %s
                     )""",
                (loser_id, winner_id),
            )
            cur.execute(
                "UPDATE entity_episode_mention SET entity_id = %s WHERE entity_id = %s",
                (winner_id, loser_id),
            )
            cur.execute(
                "UPDATE entities SET canonical_id = %s WHERE id = %s",
                (winner_id, loser_id),
            )
        self._conn.commit()

    def mark_proposal_applied(self, proposal_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE merge_proposals SET applied_at = %s WHERE id = %s",
                (_to_ms(datetime.now(UTC)), proposal_id),
            )
        self._conn.commit()

    def mark_proposal_rejected(self, proposal_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE merge_proposals SET rejected_at = %s WHERE id = %s",
                (_to_ms(datetime.now(UTC)), proposal_id),
            )
        self._conn.commit()

    # --- introspection ---

    def close(self) -> None:
        self._conn.close()
