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
from datetime import UTC, datetime
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
    """Parse a `store=` URI string and return a Store instance.

    Recognized:
      - ":memory:"             — in-memory SQLite (good for tests)
      - "sqlite:///path/to.db" — file-backed SQLite (default)
      - "sqlite:path/to.db"    — same, less-strict form
    """
    from lifegraph_kg.kg.store.sqlite import SqliteStore

    if spec == ":memory:":
        return SqliteStore(":memory:")
    if spec.startswith("sqlite:///"):
        return SqliteStore(spec.removeprefix("sqlite:///"))
    if spec.startswith("sqlite:"):
        return SqliteStore(spec.removeprefix("sqlite:"))
    # Bare path — assume SQLite file.
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

    def since(self, t: datetime, *, limit: int | None = None) -> list[Episode]:
        return self._store.episodes_since(t, limit=limit)

    def between(self, start: datetime, end: datetime, *, limit: int | None = None) -> list[Episode]:
        """Episodes with occurred_at in [start, end]. Inclusive both ends."""
        return self._store.episodes_between(start, end, limit=limit)

    def mentioning(self, entity: Entity | str, *, limit: int | None = None) -> list[Episode]:
        """Episodes that mention `entity` (by Entity object or by id)."""
        if isinstance(entity, Entity):
            ent_id = self._store.find_entity_id(entity.type, entity.key)
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
            # All episodes mentioning a specific person
            sara = lg.query(Person, key="sara").one()
            lg.query(Person, key="sara").episodes()

            # All episodes that mention any food (a personal eating timeline)
            lg.query(Topic, kind="food").episodes()
        """
        ids: list[str] = []
        for e in self._entities:
            ent_id = self._store.find_entity_id(e.type, e.key)
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
        type_: str | None = None,
        kind: str | None = None,
        record: bool = False,
    ) -> list[MergeProposal]:
        """Run heuristic dedup over the entity set.

        Args:
            type_:  restrict to a specific entity type (Person / Place / Project / Topic)
            kind:   restrict to a specific Topic kind (food / media / etc.)
            record: if True, persist each proposal to the merge_proposals table
                    so they survive across sessions.

        Returns the list of proposed merges. Each MergeProposal has a
        `confidence` field and an `is_safe_to_auto_apply` property —
        callers decide what to do with each.
        """
        entities = self._store.query_entities(type_=type_, kind=kind)
        proposals = _propose_merges(entities)
        if record:
            from lifegraph_kg.kg.store.sqlite import SqliteStore

            if isinstance(self._store, SqliteStore):
                for p in proposals:
                    win_id = self._store.find_entity_id(p.winner.type, p.winner.key)
                    los_id = self._store.find_entity_id(p.loser.type, p.loser.key)
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

    def apply(self, proposal: MergeProposal) -> None:
        """Apply a merge: redirect loser's edges + mentions to winner.

        Idempotent: applying the same merge twice is a no-op (the second
        call finds nothing to redirect). Audit-preserving: the loser
        entity row stays in the DB with `canonical_id` pointing at the
        winner — it's an alias, not a deletion.
        """
        from lifegraph_kg.kg.store.sqlite import SqliteStore

        if not isinstance(self._store, SqliteStore):
            raise NotImplementedError("apply() requires SqliteStore; other backends in L4.")
        winner_id = self._store.find_entity_id(proposal.winner.type, proposal.winner.key)
        loser_id = self._store.find_entity_id(proposal.loser.type, proposal.loser.key)
        if winner_id is None or loser_id is None:
            return  # one side was already merged away
        self._store.apply_merge(winner_id, loser_id)

    def auto_apply(
        self, *, type_: str | None = None, kind: str | None = None
    ) -> list[MergeProposal]:
        """Run propose() and apply every proposal that is_safe_to_auto_apply.

        Returns the list of proposals that were applied. The default
        is_safe rule is `confidence == "high"` AND `reason ==
        "exact_normalized"` — only the truly unambiguous cases.
        """
        proposals = self.propose(type_=type_, kind=kind)
        applied: list[MergeProposal] = []
        for p in proposals:
            if p.is_safe_to_auto_apply:
                self.apply(p)
                applied.append(p)
        return applied


class _KgOps:
    """`lg.kg.<...>` — graph-level operations (bi-temporal CRUD)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def invalidate_edge(self, edge_id: str, t_invalid: datetime) -> None:
        """Mark an edge as no longer valid as of `t_invalid` — supersede,
        not delete. The edge survives in the DB; only `t_invalid` is set.
        Subsequent `edges_as_of(t)` queries with `t < t_invalid` will
        still return it.
        """
        self._store.invalidate_edge(edge_id, t_invalid)

    def edges_as_of(self, t: datetime, *, verb: str | None = None) -> list[Edge]:
        """Edges that were valid at time `t`. Optionally filter by verb."""
        return self._store.edges_as_of(t, verb=verb)

    def edges_for_episode(self, episode_id: str) -> list[Edge]:
        return self._store.edges_for_episode(episode_id)


class LifeGraph:
    """The user-facing facade for the personal knowledge graph.

    Constructor:
        - `store=":memory:"`         in-memory SQLite (default — testing)
        - `store="sqlite:///path"`   file-backed SQLite
        - `llm=...`                  inject a custom LLM client (mock for tests)
    """

    def __init__(
        self,
        store: str = ":memory:",
        *,
        llm: LlmClient | None = None,
    ) -> None:
        self._llm = llm
        self._store = _resolve_store(store)
        self.episodes = _EpisodeView(self._store)
        self.kg = _KgOps(self._store)
        self.hygiene = _HygieneOps(self._store)

    # --- ingestion ---

    def log(
        self,
        text: str,
        *,
        occurred_at: datetime | None = None,
        source: str = "user",
    ) -> Episode:
        """Extract entities + persist as an Episode.

        Returns the persisted Episode. Edges are created automatically:
        for each (predicate, entity) pair in the extraction, an edge is
        written with `from_entity=NULL` (the user is the implicit
        subject), `verb=predicate`, `to_entity=entity.id`,
        `t_event=occurred_at`, `t_valid=occurred_at`, `t_invalid=NULL`.

        The L2 default emits one edge per (predicate, entity) pair. This is
        coarser than L3's role-aware extraction (which we'll add when
        we have richer signals) but matches Graphiti's bi-temporal
        edge model and is queryable.
        """
        result = extract(text, llm=self._llm)
        return self.persist(text, result, occurred_at=occurred_at, source=source)

    def persist(
        self,
        text: str,
        extraction: ExtractionResult,
        *,
        occurred_at: datetime | None = None,
        source: str = "user",
    ) -> Episode:
        """Persist a pre-extracted result as an Episode.

        Useful for cases where extraction was done elsewhere (batch
        import, mock for tests, custom extractor).
        """
        now = datetime.now(UTC)
        ep = Episode(
            id=_new_id(),
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
        # known via dedup); then resolve IDs and write edges. This
        # keeps the dedup logic inside the store while letting the
        # facade build edges that reference the deduplicated entities.
        self._store.save_episode(ep, list(extraction.entities), edges=[])

        edges: list[Edge] = []
        for predicate in extraction.predicates:
            for entity in extraction.entities:
                ent_id = self._store.find_entity_id(entity.type, entity.key)
                if ent_id is None:
                    continue
                edges.append(
                    Edge(
                        id=_new_id(),
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

    # --- querying ---

    def query(
        self,
        type_: type[Entity],
        *,
        kind: str | None = None,
        key: str | None = None,
    ) -> _EntityQuery:
        """Query entities of a given type with optional filters.

        Example:
            lg.query(Person, key="sara").one()
            lg.query(Topic, kind="food").all()
        """
        type_name = str(type_.model_fields["type"].default)
        results = self._store.query_entities(type_=type_name, kind=kind, key=key)
        return _EntityQuery(self._store, results)


__all__ = ["LifeGraph"]
