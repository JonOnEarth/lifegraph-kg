# SPDX-License-Identifier: Apache-2.0
"""SQLite store — the default backend for v0.1.

Choices:
- Standard-library `sqlite3` (no third-party SQL deps).
- Datetimes stored as unix-millisecond INTEGERs (timezone-agnostic at
  the storage layer; the Python side handles tz-aware datetimes).
- All mutations run inside a single transaction so save_episode is atomic.
- Schema applied via the SQL files in `migrations/`.

In-memory DB is supported via `path=":memory:"` — used by tests.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from lifegraph_kg.classes import Entity, Person, Place, Project, Topic
from lifegraph_kg.extract.schema import EntityT
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
SCHEMA_VERSION = 1


def _to_ms(dt: datetime) -> int:
    """Datetime → unix-ms. Naïve datetimes are assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _new_id() -> str:
    """Short, URL-safe identifier. UUID4 hex truncated to 16 chars —
    plenty of entropy for personal-scale graphs."""
    return uuid.uuid4().hex[:16]


def _entity_from_row(row: sqlite3.Row) -> EntityT:
    """Reconstruct a typed entity from a DB row."""
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


def _episode_from_row(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"],
        text=row["text"],
        occurred_at=_from_ms(row["occurred_at"]),  # type: ignore[arg-type]
        ingested_at=_from_ms(row["ingested_at"]),  # type: ignore[arg-type]
        source=row["source"] or "user",
        predicates=json.loads(row["predicates"] or "[]"),
        body_state=row["body_state"],
        sentiment=row["sentiment"],
        energy=row["energy"],
    )


def _edge_from_row(row: sqlite3.Row) -> Edge:
    return Edge(
        id=row["id"],
        from_entity=row["from_entity"],
        to_entity=row["to_entity"],
        verb=row["verb"],
        episode_id=row["episode_id"],
        t_event=_from_ms(row["t_event"]),  # type: ignore[arg-type]
        t_ingestion=_from_ms(row["t_ingestion"]),  # type: ignore[arg-type]
        t_valid=_from_ms(row["t_valid"]),  # type: ignore[arg-type]
        t_invalid=_from_ms(row["t_invalid"]),
    )


class SqliteStore:
    """SQLite implementation of the Store protocol.

    Use ``SqliteStore(":memory:")`` for tests, or ``SqliteStore("path.db")``
    for a file-backed store.
    """

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        # `check_same_thread=False` so a LifeGraph created on one thread
        # can be used from another (common in async code paths). The
        # library doesn't currently use threads, but we don't want to
        # accidentally pin users.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self.init_schema()

    # --- schema ---

    def init_schema(self) -> None:
        cur = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            if self._table_exists("schema_version")
            else "SELECT 0"
        )
        current = cur.fetchone()
        current_version = current[0] if current else 0
        if current_version >= SCHEMA_VERSION:
            return
        # Apply each migration in order.
        for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
            sql = migration.read_text()
            self._conn.executescript(sql)
        # Stamp the version.
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, _to_ms(datetime.now(UTC))),
        )
        self._conn.commit()

    def _table_exists(self, name: str) -> bool:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cur.fetchone() is not None

    # --- save (the atomic write path) ---

    def save_episode(
        self,
        episode: Episode,
        entities: list[Entity],
        edges: list[Edge],
    ) -> None:
        """Atomic write: episode row + dedup entities + edges + mention links."""
        with self._conn:  # transaction
            self._conn.execute(
                """INSERT INTO episodes (id, text, occurred_at, ingested_at, source,
                                          predicates, body_state, sentiment, energy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                self._conn.execute(
                    """INSERT INTO entities
                         (id, type, kind, key, value, attributes_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(type, key) DO NOTHING""",
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
                # Record mention link — fetch the (possibly pre-existing) entity ID.
                row = self._conn.execute(
                    "SELECT id FROM entities WHERE type=? AND key=?",
                    (entity.type, entity.key),
                ).fetchone()
                if row is not None:
                    self._conn.execute(
                        """INSERT INTO entity_episode_mention (entity_id, episode_id)
                           VALUES (?, ?) ON CONFLICT DO NOTHING""",
                        (row["id"], episode.id),
                    )

            for edge in edges:
                self._insert_edge(edge)

    def _insert_edge(self, edge: Edge) -> None:
        """Internal helper — insert one edge row. Caller manages the
        transaction (used both by save_episode under one transaction
        and by add_edges below)."""
        self._conn.execute(
            """INSERT INTO edges (id, from_entity, to_entity, verb, episode_id,
                                   t_event, t_ingestion, t_valid, t_invalid,
                                   attributes_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                "{}",
            ),
        )

    def add_edges(self, edges: list[Edge]) -> None:
        """Atomically insert a batch of edges. Used by the LifeGraph
        facade's two-phase save (write episode + entities first to get
        IDs, then resolve and write edges)."""
        if not edges:
            return
        with self._conn:
            for edge in edges:
                self._insert_edge(edge)

    # --- episode reads ---

    def get_episode(self, episode_id: str) -> Episode | None:
        row = self._conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return _episode_from_row(row) if row else None

    def episodes_since(self, t: datetime, limit: int | None = None) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE occurred_at >= ? ORDER BY occurred_at DESC"
        params: list[object] = [_to_ms(t)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_episode_from_row(r) for r in self._conn.execute(sql, params)]

    def episodes_mentioning(self, entity_id: str, limit: int | None = None) -> list[Episode]:
        sql = (
            "SELECT e.* FROM episodes e "
            "JOIN entity_episode_mention m ON m.episode_id = e.id "
            "WHERE m.entity_id = ? "
            "ORDER BY e.occurred_at DESC"
        )
        params: list[object] = [entity_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_episode_from_row(r) for r in self._conn.execute(sql, params)]

    # --- entity reads ---

    def find_entity(self, type_: str, key: str) -> EntityT | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE type = ? AND key = ?", (type_, key)
        ).fetchone()
        return _entity_from_row(row) if row else None

    def find_entity_id(self, type_: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM entities WHERE type = ? AND key = ?", (type_, key)
        ).fetchone()
        return row["id"] if row else None

    def query_entities(
        self,
        type_: str | None = None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[EntityT]:
        sql = "SELECT * FROM entities WHERE 1=1"
        params: list[object] = []
        if type_ is not None:
            sql += " AND type = ?"
            params.append(type_)
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        if key is not None:
            sql += " AND key = ?"
            params.append(key)
        return [_entity_from_row(r) for r in self._conn.execute(sql, params)]

    # --- bi-temporal CRUD ---

    def invalidate_edge(self, edge_id: str, t_invalid: datetime) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE edges SET t_invalid = ? WHERE id = ? AND t_invalid IS NULL",
                (_to_ms(t_invalid), edge_id),
            )

    def edges_as_of(self, t: datetime, *, verb: str | None = None) -> list[Edge]:
        ms = _to_ms(t)
        sql = "SELECT * FROM edges WHERE t_valid <= ? AND (t_invalid IS NULL OR ? < t_invalid)"
        params: list[object] = [ms, ms]
        if verb is not None:
            sql += " AND verb = ?"
            params.append(verb)
        return [_edge_from_row(r) for r in self._conn.execute(sql, params)]

    def edges_for_episode(self, episode_id: str) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE episode_id = ? ORDER BY id",
            (episode_id,),
        )
        return [_edge_from_row(r) for r in rows]

    # --- introspection ---

    def close(self) -> None:
        self._conn.close()
