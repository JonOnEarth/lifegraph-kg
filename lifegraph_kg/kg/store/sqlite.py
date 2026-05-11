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
# The latest schema version. Bump in lockstep with adding a new
# `00NN_name.sql` migration file. The runner applies only migrations
# with version > current_version, so old DBs migrate forward.
SCHEMA_VERSION = 4


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


# Phase-6 single-user fallback: legacy rows from an unscoped DB have
# NULL user_id. We keep them readable for forensic value but loading
# them via _from_row uses this sentinel; new writes must specify a
# real user_id.
_LEGACY_USER = ""


def _entity_from_row(row: sqlite3.Row) -> EntityT:
    """Reconstruct a typed entity from a DB row."""
    type_ = row["type"]
    common = {
        "user_id": row["user_id"] or _LEGACY_USER,
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
    # The task-support columns are added by migration 0003 — they're
    # accessed by name and SQLite returns NULL for missing columns on
    # older DBs that haven't migrated. We tolerate both.
    keys = row.keys() if hasattr(row, "keys") else []

    def _get(field: str, default: object = None) -> object:
        return row[field] if field in keys else default

    return Episode(
        id=row["id"],
        user_id=_get("user_id") or _LEGACY_USER,  # type: ignore[arg-type]
        text=row["text"],
        occurred_at=_from_ms(row["occurred_at"]),  # type: ignore[arg-type]
        ingested_at=_from_ms(row["ingested_at"]),  # type: ignore[arg-type]
        source=row["source"] or "user",
        predicates=json.loads(row["predicates"] or "[]"),
        body_state=row["body_state"],
        sentiment=row["sentiment"],
        energy=row["energy"],
        kind=_get("kind", "log") or "log",  # type: ignore[arg-type]
        status=_get("status", "active") or "active",  # type: ignore[arg-type]
        priority=_get("priority"),  # type: ignore[arg-type]
        deadline=_from_ms(_get("deadline")),  # type: ignore[arg-type]
        completed_at=_from_ms(_get("completed_at")),  # type: ignore[arg-type]
        recurrence=_get("recurrence"),  # type: ignore[arg-type]
        gtd_context=_get("gtd_context"),  # type: ignore[arg-type]
        action_verb=_get("action_verb"),  # type: ignore[arg-type]
    )


def _edge_from_row(row: sqlite3.Row) -> Edge:
    keys = row.keys() if hasattr(row, "keys") else []
    return Edge(
        id=row["id"],
        user_id=(row["user_id"] if "user_id" in keys else None) or _LEGACY_USER,
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
        """Apply pending migrations. Idempotent — only migrations whose
        version > current_version are applied. Each migration's leading
        digits in the filename are its version number (`0001_initial.sql`
        is version 1, `0002_hygiene.sql` is version 2)."""
        if self._table_exists("schema_version"):
            row = self._conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            current_version = row[0] if row else 0
        else:
            current_version = 0
        if current_version >= SCHEMA_VERSION:
            return

        for migration in sorted(MIGRATIONS_DIR.glob("*.sql")):
            # Filename like "0001_initial.sql" → version 1
            try:
                version = int(migration.name.split("_", 1)[0])
            except ValueError:
                continue
            if version <= current_version:
                continue  # already applied
            sql = migration.read_text()
            self._conn.executescript(sql)
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, _to_ms(datetime.now(UTC))),
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
        """Atomic write: episode row + dedup entities + edges + mention links.
        Every entity inherits ``episode.user_id`` for dedup boundary
        purposes — caller-supplied ``entity.user_id`` is ignored if it
        disagrees, matching the invariant on the wire (the request was
        authenticated as a single user)."""
        uid = episode.user_id
        with self._conn:  # transaction
            self._conn.execute(
                """INSERT INTO episodes (id, user_id, text, occurred_at, ingested_at, source,
                                          predicates, body_state, sentiment, energy,
                                          kind, status, priority, deadline,
                                          completed_at, recurrence, gtd_context,
                                          action_verb)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                self._conn.execute(
                    """INSERT INTO entities
                         (id, user_id, type, kind, key, value, attributes_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(user_id, type, key) DO NOTHING""",
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
                # Record mention link — fetch the (possibly pre-existing) entity ID.
                row = self._conn.execute(
                    "SELECT id FROM entities WHERE user_id=? AND type=? AND key=?",
                    (uid, entity.type, entity.key),
                ).fetchone()
                if row is not None:
                    self._conn.execute(
                        """INSERT INTO entity_episode_mention (entity_id, episode_id, user_id)
                           VALUES (?, ?, ?) ON CONFLICT DO NOTHING""",
                        (row["id"], episode.id, uid),
                    )

            for edge in edges:
                self._insert_edge(edge)

    def _insert_edge(self, edge: Edge) -> None:
        """Internal helper — insert one edge row. Caller manages the
        transaction (used both by save_episode under one transaction
        and by add_edges below)."""
        self._conn.execute(
            """INSERT INTO edges (id, user_id, from_entity, to_entity, verb, episode_id,
                                   t_event, t_ingestion, t_valid, t_invalid,
                                   attributes_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

    def episodes_since(
        self, t: datetime, *, user_id: str, limit: int | None = None
    ) -> list[Episode]:
        sql = (
            "SELECT * FROM episodes WHERE user_id = ? AND occurred_at >= ? "
            "ORDER BY occurred_at DESC"
        )
        params: list[object] = [user_id, _to_ms(t)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_episode_from_row(r) for r in self._conn.execute(sql, params)]

    def episodes_between(
        self,
        start: datetime,
        end: datetime,
        *,
        user_id: str,
        limit: int | None = None,
    ) -> list[Episode]:
        """Episodes with `occurred_at` in [start, end]. Inclusive on both ends."""
        sql = (
            "SELECT * FROM episodes WHERE user_id = ? AND occurred_at >= ? "
            "AND occurred_at <= ? ORDER BY occurred_at DESC"
        )
        params: list[object] = [user_id, _to_ms(start), _to_ms(end)]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_episode_from_row(r) for r in self._conn.execute(sql, params)]

    def episodes_mentioning(
        self, entity_id: str, *, limit: int | None = None
    ) -> list[Episode]:
        # entity_id encodes user_id transitively (entities are user-scoped),
        # so we don't need an explicit user_id filter here.
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

    def episodes_mentioning_any(
        self, entity_ids: list[str], *, limit: int | None = None
    ) -> list[Episode]:
        """Episodes that mention ANY of the given entities. Used by the
        ``_EntityQuery.episodes()`` pivot — given a set of matched entities,
        return episodes that mention any one of them, deduplicated and
        reverse-chronological. user_id is implicit via the entity rows."""
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        sql = (
            f"SELECT DISTINCT e.* FROM episodes e "
            f"JOIN entity_episode_mention m ON m.episode_id = e.id "
            f"WHERE m.entity_id IN ({placeholders}) "
            f"ORDER BY e.occurred_at DESC"
        )
        params: list[object] = list(entity_ids)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_episode_from_row(r) for r in self._conn.execute(sql, params)]

    # --- entity reads ---

    def get_entity(self, entity_id: str) -> EntityT | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return _entity_from_row(row) if row else None

    def find_entity(self, type_: str, key: str, *, user_id: str) -> EntityT | None:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE user_id = ? AND type = ? AND key = ?",
            (user_id, type_, key),
        ).fetchone()
        return _entity_from_row(row) if row else None

    def find_entity_id(self, type_: str, key: str, *, user_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM entities WHERE user_id = ? AND type = ? AND key = ?",
            (user_id, type_, key),
        ).fetchone()
        return row["id"] if row else None

    def query_entities(
        self,
        *,
        user_id: str,
        type_: str | None = None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[EntityT]:
        sql = "SELECT * FROM entities WHERE user_id = ?"
        params: list[object] = [user_id]
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

    def edges_as_of(
        self, t: datetime, *, user_id: str, verb: str | None = None
    ) -> list[Edge]:
        ms = _to_ms(t)
        sql = (
            "SELECT * FROM edges WHERE user_id = ? AND t_valid <= ? "
            "AND (t_invalid IS NULL OR ? < t_invalid)"
        )
        params: list[object] = [user_id, ms, ms]
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

    # --- hygiene (L3) ---

    def record_proposal(
        self,
        proposal_id: str,
        winner_id: str,
        loser_id: str,
        confidence: str,
        reason: str,
        detail: str = "",
    ) -> None:
        """Persist a MergeProposal so it can be reviewed + applied later."""
        with self._conn:
            self._conn.execute(
                """INSERT INTO merge_proposals
                     (id, winner_id, loser_id, confidence, reason, detail, proposed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
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

    def apply_merge(self, winner_id: str, loser_id: str) -> None:
        """Apply a merge: redirect loser's edges + mentions to winner,
        and set loser.canonical_id = winner. The loser entity row
        survives — it's an alias, not a deletion. Audit-trail-preserving."""
        if winner_id == loser_id:
            return
        with self._conn:
            # Redirect edges
            self._conn.execute(
                "UPDATE edges SET to_entity = ? WHERE to_entity = ?",
                (winner_id, loser_id),
            )
            self._conn.execute(
                "UPDATE edges SET from_entity = ? WHERE from_entity = ?",
                (winner_id, loser_id),
            )
            # Redirect mentions (UPSERT-like: drop conflicts that already
            # link winner to the same episode)
            self._conn.execute(
                """UPDATE OR IGNORE entity_episode_mention
                   SET entity_id = ? WHERE entity_id = ?""",
                (winner_id, loser_id),
            )
            # Drop now-orphaned mention rows for the loser
            self._conn.execute(
                "DELETE FROM entity_episode_mention WHERE entity_id = ?",
                (loser_id,),
            )
            # Mark the loser as merged
            self._conn.execute(
                "UPDATE entities SET canonical_id = ? WHERE id = ?",
                (winner_id, loser_id),
            )

    def mark_proposal_applied(self, proposal_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE merge_proposals SET applied_at = ? WHERE id = ?",
                (_to_ms(datetime.now(UTC)), proposal_id),
            )

    def mark_proposal_rejected(self, proposal_id: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE merge_proposals SET rejected_at = ? WHERE id = ?",
                (_to_ms(datetime.now(UTC)), proposal_id),
            )

    # --- task lifecycle (L3.1) ---

    def update_task_status(
        self,
        episode_id: str,
        status: str,
        completed_at: datetime | None = None,
    ) -> None:
        """Transition a task's status. ``completed_at`` is set only when
        moving to ``"done"`` — caller decides timing (test harness clocks,
        backfills, etc.)."""
        with self._conn:
            if completed_at is not None:
                self._conn.execute(
                    "UPDATE episodes SET status = ?, completed_at = ? WHERE id = ?",
                    (status, _to_ms(completed_at), episode_id),
                )
            else:
                self._conn.execute(
                    "UPDATE episodes SET status = ?, completed_at = NULL WHERE id = ?",
                    (status, episode_id),
                )

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
        """Filter task-kind episodes by lifecycle attributes."""
        sql = "SELECT * FROM episodes WHERE user_id = ? AND kind = 'task'"
        params: list[object] = [user_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        if priority is not None:
            sql += " AND priority = ?"
            params.append(priority)
        if gtd_context is not None:
            sql += " AND gtd_context = ?"
            params.append(gtd_context)
        if deadline_before is not None:
            sql += " AND deadline IS NOT NULL AND deadline < ?"
            params.append(_to_ms(deadline_before))
        if deadline_after is not None:
            sql += " AND deadline IS NOT NULL AND deadline > ?"
            params.append(_to_ms(deadline_after))
        sql += " ORDER BY deadline ASC NULLS LAST, occurred_at DESC"
        # SQLite doesn't support NULLS LAST natively without ORDER BY tricks;
        # use a CASE to push NULL deadlines to the end.
        sql = sql.replace(
            "ORDER BY deadline ASC NULLS LAST",
            "ORDER BY CASE WHEN deadline IS NULL THEN 1 ELSE 0 END, deadline ASC",
        )
        return [_episode_from_row(r) for r in self._conn.execute(sql, params)]

    # --- introspection ---

    def close(self) -> None:
        self._conn.close()
