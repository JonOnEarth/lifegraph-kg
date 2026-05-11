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
  id                  TEXT PRIMARY KEY,
  user_id             TEXT NOT NULL,
  text                TEXT NOT NULL,
  occurred_at         BIGINT NOT NULL,
  ingested_at         BIGINT NOT NULL,
  source              TEXT,
  predicates          TEXT NOT NULL DEFAULT '[]',
  body_state          TEXT,
  sentiment           TEXT,
  energy              TEXT,
  duration            INTEGER,
  duration_inferred   BOOLEAN,
  origin_tz           TEXT,
  time_mode           TEXT,
  wall_clock_hour     INTEGER,
  wall_clock_minute   INTEGER,
  wall_clock_date     TEXT,
  kind                TEXT NOT NULL DEFAULT 'log',
  status              TEXT NOT NULL DEFAULT 'active',
  priority            TEXT,
  deadline            BIGINT,
  completed_at        BIGINT,
  recurrence          TEXT,
  gtd_context         TEXT,
  action_verb         TEXT
)""",
    "CREATE INDEX IF NOT EXISTS idx_episodes_occurred_at ON episodes(occurred_at)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_kind_status ON episodes(kind, status)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_deadline ON episodes(deadline)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_user_occurred ON episodes(user_id, occurred_at DESC)",
    """\
CREATE TABLE IF NOT EXISTS entities (
  id              TEXT PRIMARY KEY,
  user_id         TEXT NOT NULL,
  type            TEXT NOT NULL,
  kind            TEXT,
  key             TEXT NOT NULL,
  value           TEXT NOT NULL,
  attributes_json TEXT NOT NULL DEFAULT '{}',
  created_at      BIGINT NOT NULL,
  canonical_id    TEXT REFERENCES entities(id),
  UNIQUE(user_id, type, key)
)""",
    "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
    "CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind)",
    "CREATE INDEX IF NOT EXISTS idx_entities_canonical_id ON entities(canonical_id)",
    "CREATE INDEX IF NOT EXISTS idx_entities_user_id ON entities(user_id)",
    """\
CREATE TABLE IF NOT EXISTS edges (
  id            TEXT PRIMARY KEY,
  user_id       TEXT NOT NULL,
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
    "CREATE INDEX IF NOT EXISTS idx_edges_user_id ON edges(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_user_episode ON edges(user_id, episode_id)",
    """\
CREATE TABLE IF NOT EXISTS entity_episode_mention (
  entity_id   TEXT NOT NULL REFERENCES entities(id),
  episode_id  TEXT NOT NULL REFERENCES episodes(id),
  user_id     TEXT NOT NULL,
  PRIMARY KEY (entity_id, episode_id)
)""",
    "CREATE INDEX IF NOT EXISTS idx_eem_user_id ON entity_episode_mention(user_id)",
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


_LEGACY_USER = ""


def _entity_from_row(row: dict[str, Any]) -> EntityT:
    type_ = row["type"]
    common = {
        "id": row.get("id"),
        "user_id": row.get("user_id") or _LEGACY_USER,
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
        user_id=row.get("user_id") or _LEGACY_USER,
        text=row["text"],
        occurred_at=occurred_at,
        ingested_at=ingested_at,
        source=row["source"] or "user",
        predicates=json.loads(row["predicates"] or "[]"),
        body_state=row["body_state"],
        sentiment=row["sentiment"],
        energy=row["energy"],
        duration=row.get("duration"),
        duration_inferred=row.get("duration_inferred"),
        origin_tz=row.get("origin_tz"),
        time_mode=row.get("time_mode"),
        wall_clock_hour=row.get("wall_clock_hour"),
        wall_clock_minute=row.get("wall_clock_minute"),
        wall_clock_date=row.get("wall_clock_date"),
        kind=row.get("kind") or "log",
        status=row.get("status") or "active",
        priority=row.get("priority"),
        deadline=_from_ms(row.get("deadline")),
        completed_at=_from_ms(row.get("completed_at")),
        recurrence=row.get("recurrence"),
        gtd_context=row.get("gtd_context"),
        action_verb=row.get("action_verb"),
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
        user_id=row.get("user_id") or _LEGACY_USER,
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
        uid = episode.user_id
        with self._conn.cursor() as cur:
            cur.execute(
                """INSERT INTO episodes
                     (id, user_id, text, occurred_at, ingested_at, source,
                      predicates, body_state, sentiment, energy,
                      duration, duration_inferred,
                      origin_tz, time_mode,
                      wall_clock_hour, wall_clock_minute, wall_clock_date,
                      kind, status, priority, deadline, completed_at,
                      recurrence, gtd_context, action_verb)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    episode.id,
                    uid,
                    episode.text,
                    _to_ms(episode.occurred_at),
                    _to_ms(episode.ingested_at),
                    episode.source,
                    json.dumps(episode.predicates, ensure_ascii=False),
                    episode.body_state,
                    episode.sentiment,
                    episode.energy,
                    episode.duration,
                    episode.duration_inferred,
                    episode.origin_tz,
                    episode.time_mode,
                    episode.wall_clock_hour,
                    episode.wall_clock_minute,
                    episode.wall_clock_date,
                    episode.kind,
                    episode.status,
                    episode.priority,
                    _to_ms(episode.deadline) if episode.deadline else None,
                    _to_ms(episode.completed_at) if episode.completed_at else None,
                    episode.recurrence,
                    episode.gtd_context,
                    episode.action_verb,
                ),
            )
            for entity in entities:
                cur.execute(
                    """INSERT INTO entities
                         (id, user_id, type, kind, key, value, attributes_json, created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (user_id, type, key) DO NOTHING""",
                    (
                        _new_id(),
                        uid,
                        entity.type,
                        getattr(entity, "kind", None),
                        entity.key,
                        entity.value,
                        json.dumps(entity.attributes, ensure_ascii=False),
                        _to_ms(datetime.now(UTC)),
                    ),
                )
                cur.execute(
                    "SELECT id FROM entities WHERE user_id = %s AND type = %s AND key = %s",
                    (uid, entity.type, entity.key),
                )
                row = cur.fetchone()
                if row is not None:
                    cur.execute(
                        """INSERT INTO entity_episode_mention (entity_id, episode_id, user_id)
                           VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
                        (row["id"], episode.id, uid),
                    )

            for edge in edges:
                self._insert_edge(cur, edge)
        self._conn.commit()

    def _insert_edge(self, cur: Any, edge: Edge) -> None:
        cur.execute(
            """INSERT INTO edges
                 (id, user_id, from_entity, to_entity, verb, episode_id,
                  t_event, t_ingestion, t_valid, t_invalid, attributes_json)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, '{}')""",
            (
                edge.id,
                edge.user_id,
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

    def episodes_since(
        self, t: datetime, *, user_id: str, limit: int | None = None
    ) -> list[Episode]:
        sql = (
            "SELECT * FROM episodes WHERE user_id = %s AND occurred_at >= %s "
            "ORDER BY occurred_at DESC"
        )
        params: list[Any] = [user_id, _to_ms(t)]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    def episodes_between(
        self,
        start: datetime,
        end: datetime,
        *,
        user_id: str,
        limit: int | None = None,
    ) -> list[Episode]:
        sql = (
            "SELECT * FROM episodes "
            "WHERE user_id = %s AND occurred_at >= %s AND occurred_at <= %s "
            "ORDER BY occurred_at DESC"
        )
        params: list[Any] = [user_id, _to_ms(start), _to_ms(end)]
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    def episodes_mentioning(
        self, entity_id: str, *, limit: int | None = None
    ) -> list[Episode]:
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
        self, entity_ids: list[str], *, limit: int | None = None
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

    def get_entity(self, entity_id: str) -> EntityT | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
            row = cur.fetchone()
        return _entity_from_row(row) if row else None

    def find_entity(self, type_: str, key: str, *, user_id: str) -> EntityT | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE user_id = %s AND type = %s AND key = %s",
                (user_id, type_, key),
            )
            row = cur.fetchone()
        return _entity_from_row(row) if row else None

    def find_entity_id(self, type_: str, key: str, *, user_id: str) -> str | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM entities WHERE user_id = %s AND type = %s AND key = %s",
                (user_id, type_, key),
            )
            row = cur.fetchone()
        return row["id"] if row else None

    def query_entities(
        self,
        *,
        user_id: str,
        type_: str | None = None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[EntityT]:
        sql = "SELECT * FROM entities WHERE user_id = %s"
        params: list[Any] = [user_id]
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

    def edges_as_of(
        self, t: datetime, *, user_id: str, verb: str | None = None
    ) -> list[Edge]:
        ms = _to_ms(t)
        sql = (
            "SELECT * FROM edges WHERE user_id = %s AND t_valid <= %s "
            "AND (t_invalid IS NULL OR %s < t_invalid)"
        )
        params: list[Any] = [user_id, ms, ms]
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

    # --- task lifecycle ---

    def update_task_status(
        self,
        episode_id: str,
        status: str,
        completed_at: datetime | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            if completed_at is not None:
                cur.execute(
                    "UPDATE episodes SET status = %s, completed_at = %s WHERE id = %s",
                    (status, _to_ms(completed_at), episode_id),
                )
            else:
                cur.execute(
                    "UPDATE episodes SET status = %s, completed_at = NULL WHERE id = %s",
                    (status, episode_id),
                )
        self._conn.commit()

    def query_tasks(
        self,
        *,
        user_id: str,
        status: str | None = None,
        priority: str | None = None,
        gtd_context: str | None = None,
        deadline_before: datetime | None = None,
        deadline_after: datetime | None = None,
    ) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE user_id = %s AND kind = 'task'"
        params: list[Any] = [user_id]
        if status is not None:
            sql += " AND status = %s"
            params.append(status)
        if priority is not None:
            sql += " AND priority = %s"
            params.append(priority)
        if gtd_context is not None:
            sql += " AND gtd_context = %s"
            params.append(gtd_context)
        if deadline_before is not None:
            sql += " AND deadline IS NOT NULL AND deadline < %s"
            params.append(_to_ms(deadline_before))
        if deadline_after is not None:
            sql += " AND deadline IS NOT NULL AND deadline > %s"
            params.append(_to_ms(deadline_after))
        # Postgres has native NULLS LAST
        sql += " ORDER BY deadline ASC NULLS LAST, occurred_at DESC"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    # --- mutation + bulk-list (Phase 7) ---

    def update_episode(
        self,
        episode_id: str,
        *,
        text: str | None = None,
        sentiment: str | None = None,
        energy: str | None = None,
        body_state: str | None = None,
        priority: str | None = None,
        deadline: datetime | None = None,
        recurrence: str | None = None,
        gtd_context: str | None = None,
        action_verb: str | None = None,
        source: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if text is not None:
            sets.append("text = %s")
            params.append(text)
        if sentiment is not None:
            sets.append("sentiment = %s")
            params.append(sentiment)
        if energy is not None:
            sets.append("energy = %s")
            params.append(energy)
        if body_state is not None:
            sets.append("body_state = %s")
            params.append(body_state)
        if priority is not None:
            sets.append("priority = %s")
            params.append(priority)
        if deadline is not None:
            sets.append("deadline = %s")
            params.append(_to_ms(deadline))
        if recurrence is not None:
            sets.append("recurrence = %s")
            params.append(recurrence)
        if gtd_context is not None:
            sets.append("gtd_context = %s")
            params.append(gtd_context)
        if action_verb is not None:
            sets.append("action_verb = %s")
            params.append(action_verb)
        if source is not None:
            sets.append("source = %s")
            params.append(source)
        if not sets:
            return
        params.append(episode_id)
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE episodes SET {', '.join(sets)} WHERE id = %s", params
            )
        self._conn.commit()

    def delete_episode(self, episode_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM entity_episode_mention WHERE episode_id = %s",
                (episode_id,),
            )
            cur.execute(
                "DELETE FROM edges WHERE episode_id = %s", (episode_id,)
            )
            cur.execute(
                "DELETE FROM episodes WHERE id = %s", (episode_id,)
            )
        self._conn.commit()

    def list_episodes(
        self,
        *,
        user_id: str,
        kind: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE user_id = %s"
        params: list[Any] = [user_id]
        if kind is not None:
            sql += " AND kind = %s"
            params.append(kind)
        if status is not None:
            sql += " AND status = %s"
            params.append(status)
        if since is not None:
            sql += " AND occurred_at >= %s"
            params.append(_to_ms(since))
        sql += " ORDER BY occurred_at DESC"
        if limit is not None:
            sql += " LIMIT %s OFFSET %s"
            params.append(limit)
            params.append(offset)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [_episode_from_row(r) for r in cur.fetchall()]

    def mentions_for_user(self, user_id: str) -> list[tuple[str, str]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT entity_id, episode_id FROM entity_episode_mention WHERE user_id = %s",
                (user_id,),
            )
            return [(r["entity_id"], r["episode_id"]) for r in cur.fetchall()]

    # --- introspection ---

    def close(self) -> None:
        self._conn.close()
