# SPDX-License-Identifier: Apache-2.0
"""Storage drivers — same SQL across backends, swappable transport.

The `Store` Protocol defines the minimal operation set every backend
must implement. The library never touches a backend directly — all
persistence flows through this protocol so swapping SQLite for
Postgres / PGlite / PostgREST is one config change.

**Multi-tenant (Phase 6+)** — every write carries ``user_id`` on the
domain model (Episode.user_id / Entity.user_id / Edge.user_id) and
every query takes a ``user_id`` kwarg. The dedup boundary for
entities is ``(user_id, type, key)`` so two users with a friend
named "Sara" get distinct Person rows.

Methods that operate on a specific row by primary key (``get_episode``,
``edges_for_episode``, ``invalidate_edge``, ``update_task_status``)
take just the ID — the ID itself encodes the user. Methods that
filter/scan (``episodes_since``, ``query_entities``, ``query_tasks``,
``edges_as_of``) require explicit ``user_id``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from lifegraph_kg.classes import Entity
from lifegraph_kg.extract.schema import EntityT
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode


class Store(Protocol):
    """Minimal contract every storage driver implements."""

    def init_schema(self) -> None:
        """Apply migrations (idempotent)."""

    def save_episode(
        self,
        episode: Episode,
        entities: list[Entity],
        edges: list[Edge],
    ) -> None:
        """Atomically persist an episode + dedup entities + write edges
        + record entity↔episode mentions. user_id comes from
        episode.user_id (and must match entities/edges)."""

    def add_edges(self, edges: list[Edge]) -> None:
        """Atomically insert a batch of edges (each carries its own
        user_id). Used by the LifeGraph facade's two-phase save."""

    def find_entity_id(self, type_: str, key: str, *, user_id: str) -> str | None:
        """Resolve `(user_id, type, key)` to the stored entity's ID, or None."""

    # --- Episode reads ---

    def get_episode(self, episode_id: str) -> Episode | None: ...

    def episodes_since(
        self, t: datetime, *, user_id: str, limit: int | None = None
    ) -> list[Episode]: ...

    def episodes_between(
        self, start: datetime, end: datetime, *, user_id: str, limit: int | None = None
    ) -> list[Episode]: ...

    def episodes_mentioning(
        self, entity_id: str, *, limit: int | None = None
    ) -> list[Episode]:
        """Entity IDs are globally unique and encode their user_id
        implicitly via the entity row; no separate user_id arg needed."""

    def episodes_mentioning_any(
        self, entity_ids: list[str], *, limit: int | None = None
    ) -> list[Episode]: ...

    # --- Entity reads ---

    def find_entity(self, type_: str, key: str, *, user_id: str) -> EntityT | None: ...

    def get_entity(self, entity_id: str) -> EntityT | None:
        """Look up an entity by its primary key (global, encodes user_id).
        Convenience for callers that have an edge.to_entity in hand and
        need the typed object without N round-trips through query_entities."""

    def query_entities(
        self,
        *,
        user_id: str,
        type_: str | None = None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[EntityT]: ...

    # --- Bi-temporal CRUD ---

    def invalidate_edge(self, edge_id: str, t_invalid: datetime) -> None: ...

    def edges_as_of(
        self, t: datetime, *, user_id: str, verb: str | None = None
    ) -> list[Edge]: ...

    def edges_for_episode(self, episode_id: str) -> list[Edge]: ...

    # --- Task lifecycle (L3.1) ---

    def update_task_status(
        self,
        episode_id: str,
        status: str,
        completed_at: datetime | None = None,
    ) -> None: ...

    def query_tasks(
        self,
        *,
        user_id: str,
        status: str | None = None,
        priority: str | None = None,
        gtd_context: str | None = None,
        deadline_before: datetime | None = None,
        deadline_after: datetime | None = None,
    ) -> list[Episode]: ...

    # --- Mutation + bulk-list (Phase 7) ---

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
        """Patch an existing episode in place. Only fields explicitly
        provided (non-None) are updated. ID encodes user_id via the row."""

    def delete_episode(self, episode_id: str) -> None:
        """Delete an episode + its edges + mentions. Entities survive
        (they may be referenced by other episodes). ID encodes user_id."""

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
        """Bulk-list episodes for a user with optional kind/status/since
        filters. Used by the frontend sync route. Most-recent first."""

    def mentions_for_user(self, user_id: str) -> list[tuple[str, str]]:
        """Return all (entity_id, episode_id) pairs for ``user_id``.
        Used by sync/list paths to build a per-episode entities map in
        a single round-trip instead of N find_entity_id calls."""
