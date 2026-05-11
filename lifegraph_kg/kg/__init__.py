# SPDX-License-Identifier: Apache-2.0
"""LifeGraph facade — extraction + persistent autobiographical KG.

L1 was extraction-only; L2 adds the SQLite-backed bi-temporal store.
The Episode-as-node + verb-as-edge model is now real.

Usage:
    from lifegraph_kg import LifeGraph
    lg = LifeGraph(store="sqlite:///me.db")     # or :memory: by default
    lg.log("Had ramen with Sara at Ippudo")     # extracts + persists

    sara = lg.query(Person, key="sara").one()
    for ep in lg.episodes.mentioning(sara):
        print(ep.text, ep.occurred_at)

    # Bi-temporal: time-travel queries
    facts_now = lg.kg.edges_as_of(datetime.now())
    facts_then = lg.kg.edges_as_of(datetime(2025, 12, 1))

    # Bi-temporal: supersede instead of delete
    lg.kg.invalidate_edge(edge_id, datetime.now())
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from lifegraph_kg.classes import Entity
from lifegraph_kg.extract import extract
from lifegraph_kg.extract.schema import EntityT, ExtractionResult
from lifegraph_kg.hygiene.dedup import propose_merges as _propose_merges
from lifegraph_kg.hygiene.proposals import MergeProposal
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode

if TYPE_CHECKING:
    from lifegraph_kg.kg.store import Store
    from lifegraph_kg.llm.client import LlmClient


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _resolve_store(spec: str) -> Store:
    """Parse a ``store=`` URI string and return a Store instance.

    Recognized:
      - ``":memory:"``                 — in-memory SQLite (good for tests)
      - ``"sqlite:///path/to.db"``     — file-backed SQLite (default)
      - ``"sqlite:path/to.db"``        — same, less-strict form
      - ``"postgres://user@host/db"``  — Postgres (requires ``[postgres]`` extra)
      - ``"postgresql://..."``         — same, alternate scheme
      - bare path                      — assumed SQLite file

    For native-graph traversal (Cypher path queries) on top of Postgres,
    install the `Apache AGE <https://age.apache.org>`_ extension and use
    a Postgres URI — AGE adds property-graph capabilities without
    changing the connection string.
    """
    if spec == ":memory:":
        from lifegraph_kg.kg.store.sqlite import SqliteStore

        return SqliteStore(":memory:")
    if spec.startswith("sqlite:///"):
        from lifegraph_kg.kg.store.sqlite import SqliteStore

        return SqliteStore(spec.removeprefix("sqlite:///"))
    if spec.startswith("sqlite:"):
        from lifegraph_kg.kg.store.sqlite import SqliteStore

        return SqliteStore(spec.removeprefix("sqlite:"))
    if spec.startswith(("postgres://", "postgresql://")):
        from lifegraph_kg.kg.store.postgres import PostgresStore

        return PostgresStore(spec)
    # Bare path — assume SQLite file.
    from lifegraph_kg.kg.store.sqlite import SqliteStore

    return SqliteStore(spec)


class _EpisodeView:
    """`lg.episodes.<...>` — episode-level recall.

    Separated from LifeGraph so the call sites read like a query DSL:
    `lg.episodes.since(t)`, `lg.episodes.mentioning(sara)`.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def get(self, episode_id: str) -> Episode | None:
        return self._store.get_episode(episode_id)

    def since(
        self, t: datetime, *, user_id: str, limit: int | None = None
    ) -> list[Episode]:
        return self._store.episodes_since(t, user_id=user_id, limit=limit)

    def between(
        self,
        start: datetime,
        end: datetime,
        *,
        user_id: str,
        limit: int | None = None,
    ) -> list[Episode]:
        """Episodes with occurred_at in [start, end]. Inclusive both ends."""
        return self._store.episodes_between(start, end, user_id=user_id, limit=limit)

    def mentioning(self, entity: Entity | str, *, limit: int | None = None) -> list[Episode]:
        """Episodes that mention `entity` (by Entity object or by id).
        entity.user_id (or the entity_id's owning user) defines the scope."""
        if isinstance(entity, Entity):
            ent_id = self._store.find_entity_id(
                entity.type, entity.key, user_id=entity.user_id
            )
            if ent_id is None:
                return []
            return self._store.episodes_mentioning(ent_id, limit=limit)
        return self._store.episodes_mentioning(entity, limit=limit)


class _EntityQuery:
    """Result of `lg.query(Type, **filters)`. Supports `.one()` and `.all()`.

    Designed to grow into a fluent traversal DSL in L2.1+:
        lg.query(Person, key="sara").related(Activity).since(t)

    For v0.1 it's a small wrapper around the store's `query_entities`.
    """

    def __init__(self, store: Store, entities: list[EntityT]) -> None:
        self._store = store
        self._entities = entities

    def all(self) -> list[EntityT]:
        return list(self._entities)

    def one(self) -> EntityT:
        """Return the single match. Raises if 0 or >1 results."""
        if len(self._entities) == 0:
            raise LookupError("No entity matched the query.")
        if len(self._entities) > 1:
            raise LookupError(f"Expected exactly one entity, got {len(self._entities)}.")
        return self._entities[0]

    def first(self) -> EntityT | None:
        """Return the first match, or None if empty."""
        return self._entities[0] if self._entities else None

    def episodes(self, *, limit: int | None = None) -> list[Episode]:
        """Pivot from entities to episodes — return all episodes that
        mention any of the matched entities, deduplicated and ordered
        most-recent first.

        Examples:
            sara = lg.query(Person, user_id=u, key="sara").one()
            lg.query(Person, user_id=u, key="sara").episodes()
            lg.query(Topic, user_id=u, kind="food").episodes()
        """
        ids: list[str] = []
        for e in self._entities:
            ent_id = self._store.find_entity_id(e.type, e.key, user_id=e.user_id)
            if ent_id is not None:
                ids.append(ent_id)
        return self._store.episodes_mentioning_any(ids, limit=limit)

    def __iter__(self) -> Iterator[EntityT]:
        return iter(self._entities)

    def __len__(self) -> int:
        return len(self._entities)


class _HygieneOps:
    """`lg.hygiene.<...>` — dedup + canonicalization.

    L3 v0.1: pure-string heuristics (NFKC + casefold + substring +
    Levenshtein). The differentiator between this package and
    Graphiti / Mem0 / Letta — explicit, reviewable, audit-preserving
    dedup instead of "trust the LLM + graph merge".
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def propose(
        self,
        *,
        user_id: str,
        type_: str | None = None,
        kind: str | None = None,
        record: bool = False,
    ) -> list[MergeProposal]:
        """Run heuristic dedup over a user's entity set."""
        entities = self._store.query_entities(user_id=user_id, type_=type_, kind=kind)
        proposals = _propose_merges(entities)
        if record:
            from lifegraph_kg.kg.store.sqlite import SqliteStore

            if isinstance(self._store, SqliteStore):
                for p in proposals:
                    win_id = self._store.find_entity_id(
                        p.winner.type, p.winner.key, user_id=user_id
                    )
                    los_id = self._store.find_entity_id(
                        p.loser.type, p.loser.key, user_id=user_id
                    )
                    if win_id and los_id:
                        self._store.record_proposal(
                            proposal_id=_new_id(),
                            winner_id=win_id,
                            loser_id=los_id,
                            confidence=p.confidence,
                            reason=p.reason,
                            detail=p.detail,
                        )
        return proposals

    def apply(self, proposal: MergeProposal, *, user_id: str) -> None:
        """Apply a merge for ``user_id``: redirect loser's edges + mentions to winner."""
        from lifegraph_kg.kg.store.sqlite import SqliteStore

        if not isinstance(self._store, SqliteStore):
            raise NotImplementedError("apply() requires SqliteStore; other backends in L4.")
        winner_id = self._store.find_entity_id(
            proposal.winner.type, proposal.winner.key, user_id=user_id
        )
        loser_id = self._store.find_entity_id(
            proposal.loser.type, proposal.loser.key, user_id=user_id
        )
        if winner_id is None or loser_id is None:
            return  # one side was already merged away
        self._store.apply_merge(winner_id, loser_id)

    def auto_apply(
        self,
        *,
        user_id: str,
        type_: str | None = None,
        kind: str | None = None,
    ) -> list[MergeProposal]:
        """Run propose() + apply every safe proposal for this user."""
        proposals = self.propose(user_id=user_id, type_=type_, kind=kind)
        applied: list[MergeProposal] = []
        for p in proposals:
            if p.is_safe_to_auto_apply:
                self.apply(p, user_id=user_id)
                applied.append(p)
        return applied


class _TaskView:
    """`lg.tasks.<...>` — task-flavored episode queries.

    Tasks are episodes with ``kind="task"``. The view filters by
    lifecycle attributes (status, priority, deadline, gtd_context)
    so call sites read like a query DSL: ``lg.tasks.pending()``,
    ``lg.tasks.due_soon(within=...)``, ``lg.tasks.by_context("@work")``.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    def pending(self, *, user_id: str) -> list[Episode]:
        """Tasks that are still active (not done, not dropped)."""
        return self._store.query_tasks(user_id=user_id, status="active")

    def overdue(
        self, *, user_id: str, as_of: datetime | None = None
    ) -> list[Episode]:
        """Active tasks past their deadline."""
        cutoff = as_of if as_of is not None else datetime.now(UTC)
        return self._store.query_tasks(
            user_id=user_id, status="active", deadline_before=cutoff
        )

    def due_soon(
        self,
        within: timedelta,
        *,
        user_id: str,
        as_of: datetime | None = None,
    ) -> list[Episode]:
        """Active tasks whose deadline falls within ``[as_of, as_of + within]``."""
        now = as_of if as_of is not None else datetime.now(UTC)
        return self._store.query_tasks(
            user_id=user_id,
            status="active",
            deadline_after=now,
            deadline_before=now + within,
        )

    def completed_in(
        self, start: datetime, end: datetime, *, user_id: str
    ) -> list[Episode]:
        """Tasks completed in [start, end]."""
        all_done = self._store.query_tasks(user_id=user_id, status="done")
        return [
            t for t in all_done if t.completed_at is not None and start <= t.completed_at <= end
        ]

    def by_context(self, context: str, *, user_id: str) -> list[Episode]:
        """Tasks matching a GTD context (e.g. ``@work``)."""
        return self._store.query_tasks(
            user_id=user_id, status="active", gtd_context=context
        )

    def by_priority(self, priority: str, *, user_id: str) -> list[Episode]:
        """Tasks at a given priority level (high / medium / low)."""
        return self._store.query_tasks(
            user_id=user_id, status="active", priority=priority
        )


class _KgOps:
    """`lg.kg.<...>` — graph-level operations (bi-temporal CRUD)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def invalidate_edge(self, edge_id: str, t_invalid: datetime) -> None:
        """Mark an edge as no longer valid as of `t_invalid` — supersede,
        not delete. The edge_id encodes user_id transitively."""
        self._store.invalidate_edge(edge_id, t_invalid)

    def edges_as_of(
        self, t: datetime, *, user_id: str, verb: str | None = None
    ) -> list[Edge]:
        """Edges that were valid at time `t` for the given user."""
        return self._store.edges_as_of(t, user_id=user_id, verb=verb)

    def edges_for_episode(self, episode_id: str) -> list[Edge]:
        return self._store.edges_for_episode(episode_id)


class LifeGraph:
    """The user-facing facade for the personal knowledge graph.

    Constructor:
        - `store=":memory:"`         in-memory SQLite (default — testing)
        - `store="sqlite:///path"`   file-backed SQLite
        - `store="postgres://..."`   Postgres (needs ``[postgres]`` extra)
        - `store=<Store instance>`   inject a pre-constructed backend
                                     (e.g. RestStore for auth-only deploys)
        - `llm=...`                  inject a custom LLM client (mock for tests)
    """

    def __init__(
        self,
        store: str | Store = ":memory:",
        *,
        llm: LlmClient | None = None,
    ) -> None:
        self._llm = llm
        self._store = _resolve_store(store) if isinstance(store, str) else store
        self.episodes = _EpisodeView(self._store)
        self.tasks = _TaskView(self._store)
        self.kg = _KgOps(self._store)
        self.hygiene = _HygieneOps(self._store)

    # --- ingestion ---

    def log(
        self,
        text: str,
        *,
        user_id: str,
        occurred_at: datetime | None = None,
        source: str = "user",
    ) -> Episode:
        """Extract entities + persist as an Episode scoped to ``user_id``.

        Edges are created automatically: for each (predicate, entity) pair
        in the extraction, an edge is written with from_entity=NULL,
        verb=predicate, to_entity=entity.id, all bi-temporal anchors set
        from ``occurred_at``.
        """
        result = extract(text, llm=self._llm)
        return self.persist(
            text, result, user_id=user_id, occurred_at=occurred_at, source=source
        )

    def persist(
        self,
        text: str,
        extraction: ExtractionResult,
        *,
        user_id: str,
        occurred_at: datetime | None = None,
        source: str = "user",
    ) -> Episode:
        """Persist a pre-extracted result as an Episode scoped to ``user_id``."""
        now = datetime.now(UTC)
        # Re-stamp entities with the request's user_id — the extractor
        # may default to a placeholder; the trust boundary is the caller.
        scoped_entities = [e.model_copy(update={"user_id": user_id}) for e in extraction.entities]
        ep = Episode(
            id=_new_id(),
            user_id=user_id,
            text=text,
            occurred_at=occurred_at if occurred_at is not None else now,
            ingested_at=now,
            source=source,
            predicates=extraction.predicates,
            body_state=extraction.body_state,
            sentiment=extraction.sentiment,
            energy=extraction.energy,
        )

        # Two-phase save: episode + entities first (so entity IDs are
        # known via dedup); then resolve IDs and write edges.
        self._store.save_episode(ep, scoped_entities, edges=[])

        edges: list[Edge] = []
        for predicate in extraction.predicates:
            for entity in scoped_entities:
                ent_id = self._store.find_entity_id(
                    entity.type, entity.key, user_id=user_id
                )
                if ent_id is None:
                    continue
                edges.append(
                    Edge(
                        id=_new_id(),
                        user_id=user_id,
                        from_entity=None,
                        to_entity=ent_id,
                        verb=predicate,
                        episode_id=ep.id,
                        t_event=ep.occurred_at,
                        t_ingestion=ep.ingested_at,
                        t_valid=ep.occurred_at,
                        t_invalid=None,
                    )
                )
        self._store.add_edges(edges)
        return ep

    # --- task ingestion + lifecycle ---

    def task(
        self,
        text: str,
        *,
        user_id: str,
        deadline: datetime | None = None,
        priority: str | None = None,
        gtd_context: str | None = None,
        recurrence: str | None = None,
        action_verb: str | None = None,
        occurred_at: datetime | None = None,
        source: str = "user",
    ) -> Episode:
        """Create a task scoped to ``user_id``."""
        result = extract(text, llm=self._llm)
        return self._persist_task(
            text,
            result,
            user_id=user_id,
            occurred_at=occurred_at,
            source=source,
            deadline=deadline,
            priority=priority,
            gtd_context=gtd_context,
            recurrence=recurrence,
            action_verb=action_verb,
        )

    def _persist_task(
        self,
        text: str,
        extraction: ExtractionResult,
        *,
        user_id: str,
        occurred_at: datetime | None = None,
        source: str = "user",
        deadline: datetime | None = None,
        priority: str | None = None,
        gtd_context: str | None = None,
        recurrence: str | None = None,
        action_verb: str | None = None,
    ) -> Episode:
        """Internal — persist a task with the same two-phase save as
        ``persist()`` but with task-flavored metadata + kind='task'."""
        now = datetime.now(UTC)
        scoped_entities = [e.model_copy(update={"user_id": user_id}) for e in extraction.entities]
        ep = Episode(
            id=_new_id(),
            user_id=user_id,
            text=text,
            occurred_at=occurred_at if occurred_at is not None else now,
            ingested_at=now,
            source=source,
            predicates=extraction.predicates,
            body_state=extraction.body_state,
            sentiment=extraction.sentiment,
            energy=extraction.energy,
            kind="task",
            status="active",
            priority=priority,  # type: ignore[arg-type]
            deadline=deadline,
            recurrence=recurrence,
            gtd_context=gtd_context,
            action_verb=action_verb,
        )
        self._store.save_episode(ep, scoped_entities, edges=[])

        edges: list[Edge] = []
        for predicate in extraction.predicates:
            for entity in scoped_entities:
                ent_id = self._store.find_entity_id(
                    entity.type, entity.key, user_id=user_id
                )
                if ent_id is None:
                    continue
                edges.append(
                    Edge(
                        id=_new_id(),
                        user_id=user_id,
                        from_entity=None,
                        to_entity=ent_id,
                        verb=predicate,
                        episode_id=ep.id,
                        t_event=ep.occurred_at,
                        t_ingestion=ep.ingested_at,
                        t_valid=ep.occurred_at,
                        t_invalid=None,
                    )
                )
        self._store.add_edges(edges)
        return ep

    def complete_task(self, episode_id: str, *, at: datetime | None = None) -> None:
        """Mark a task as done. ``at`` defaults to now."""
        completed_at = at if at is not None else datetime.now(UTC)
        self._store.update_task_status(episode_id, "done", completed_at)

    def drop_task(self, episode_id: str) -> None:
        """Mark a task as dropped (won't do; not done)."""
        self._store.update_task_status(episode_id, "dropped")

    def reopen_task(self, episode_id: str) -> None:
        """Reopen a completed/dropped task back to active."""
        self._store.update_task_status(episode_id, "active")

    # --- querying ---

    def query(
        self,
        type_: type[Entity],
        *,
        user_id: str,
        kind: str | None = None,
        key: str | None = None,
    ) -> _EntityQuery:
        """Query entities of a given type for a specific user.

        Example:
            lg.query(Person, user_id=u, key="sara").one()
            lg.query(Topic, user_id=u, kind="food").all()
        """
        type_name = str(type_.model_fields["type"].default)
        results = self._store.query_entities(
            user_id=user_id, type_=type_name, kind=kind, key=key
        )
        return _EntityQuery(self._store, results)


__all__ = ["LifeGraph"]
