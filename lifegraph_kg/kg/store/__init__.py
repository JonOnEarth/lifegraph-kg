# SPDX-License-Identifier: Apache-2.0
"""Storage drivers — same SQL across backends, swappable transport.

The `Store` Protocol defines the minimal operation set every backend
must implement. The library never touches a backend directly — all
persistence flows through this protocol so swapping SQLite for
Postgres / Kuzu / PGlite is one config change.

L2 ships SQLite. Postgres and Kuzu are L4.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from lifegraph_kg.classes import Entity
from lifegraph_kg.extract.schema import EntityT
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode


class Store(Protocol):
    """Minimal contract every storage driver implements.

    The methods are deliberately low-level — `save_episode` takes an
    Episode + entities + edges as a unit, and the higher-level
    `LifeGraph` API composes calls into these primitives.
    """

    def init_schema(self) -> None:
        """Apply migrations (idempotent)."""

    def save_episode(
        self,
        episode: Episode,
        entities: list[Entity],
        edges: list[Edge],
    ) -> None:
        """Atomically persist an episode + dedup entities + write edges
        + record entity↔episode mentions."""

    def add_edges(self, edges: list[Edge]) -> None:
        """Atomically insert a batch of edges. Used by the LifeGraph
        facade's two-phase save (episode + entities first, then edges
        once entity IDs are known)."""

    def find_entity_id(self, type_: str, key: str) -> str | None:
        """Resolve `(type, key)` to the stored entity's ID, or None."""

    # --- Episode reads ---

    def get_episode(self, episode_id: str) -> Episode | None: ...

    def episodes_since(self, t: datetime, limit: int | None = None) -> list[Episode]: ...

    def episodes_between(
        self, start: datetime, end: datetime, limit: int | None = None
    ) -> list[Episode]: ...

    def episodes_mentioning(self, entity_id: str, limit: int | None = None) -> list[Episode]: ...

    def episodes_mentioning_any(
        self, entity_ids: list[str], limit: int | None = None
    ) -> list[Episode]: ...

    # --- Entity reads ---

    def find_entity(self, type_: str, key: str) -> EntityT | None: ...

    def query_entities(
        self,
        type_: str | None = None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[EntityT]: ...

    # --- Bi-temporal CRUD ---

    def invalidate_edge(self, edge_id: str, t_invalid: datetime) -> None:
        """Mark an edge as no longer valid as of `t_invalid`. The edge
        survives in the DB; only `t_invalid` is set. This preserves the
        audit trail."""

    def edges_as_of(self, t: datetime, *, verb: str | None = None) -> list[Edge]:
        """Return edges that were valid at time `t`. Used for
        time-travel queries: 'what did I think was true on 2025-12-01?'"""

    def edges_for_episode(self, episode_id: str) -> list[Edge]: ...
