# SPDX-License-Identifier: Apache-2.0
"""Eval-suite types — Pydantic schemas for fixture validation and scoring.

These mirror what ``lifegraph_kg`` will define in L1+; once the library
ships its real types, the scoring functions and fixture loaders will
import from there. Until then, eval owns its own copies so the suite is
testable before the library's data layer exists.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Core data model (mirrors L1+ lifegraph_kg.types) ---


class Span(BaseModel):
    """A char interval into source episode text."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    def overlap(self, other: Span) -> int:
        """Length of overlap with another span (0 if disjoint)."""
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def iou(self, other: Span) -> float:
        """Intersection-over-union with another span."""
        inter = self.overlap(other)
        union = (self.end - self.start) + (other.end - other.start) - inter
        return inter / union if union > 0 else 0.0


class Entity(BaseModel):
    """An extracted typed node."""

    type: str  # e.g. "Person", "Place", "Activity"
    key: str  # canonical key (lowercase form, etc.)
    value: str  # surface form as extracted
    attributes: dict[str, Any] = Field(default_factory=dict)


class Grounding(BaseModel):
    """Link from an entity back to a span of source text."""

    entity_key: str  # identifies the entity by (type, key)
    entity_type: str
    span: Span


class Edge(BaseModel):
    """A bi-temporal edge between two entities."""

    type: str  # e.g. "lives_in", "ate"
    from_entity: tuple[str, str]  # (type, key)
    to_entity: tuple[str, str]
    t_event: datetime  # when the fact was true (event time)
    t_ingestion: datetime  # when we learned it
    t_valid: datetime
    t_invalid: datetime | None = None  # None == still valid


class Episode(BaseModel):
    """A raw entry — original text + timestamp + source."""

    id: str
    text: str
    occurred_at: datetime
    ingested_at: datetime
    source: str = "synthetic"
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Fixture schemas (one per eval category) ---


class FixtureMetadata(BaseModel):
    source: Literal["synthetic", "curated", "private"] = "synthetic"
    license: str = "Apache-2.0"
    notes: str = ""


class ExtractionFixture(BaseModel):
    """Input: a piece of text. Expected: entities + groundings."""

    id: str
    description: str
    category: Literal["extraction"] = "extraction"
    input_text: str
    expected_entities: list[Entity]
    expected_groundings: list[Grounding]
    metadata: FixtureMetadata = Field(default_factory=FixtureMetadata)


class TemporalEvent(BaseModel):
    """An event in a temporal fixture's history."""

    at: datetime
    text: str  # natural-language input that should produce one or more edges
    expected_facts: list[Edge] = Field(default_factory=list)


class TemporalQuery(BaseModel):
    """A time-travel query against the replayed history."""

    as_of: datetime
    subject: tuple[str, str]  # (type, key)
    predicate: str
    expected_object: tuple[str, str] | None  # None = no fact at this time


class TemporalFixture(BaseModel):
    """Sequence of events with supersede semantics, plus time-travel queries."""

    id: str
    description: str
    category: Literal["temporal"] = "temporal"
    history: list[TemporalEvent]
    queries: list[TemporalQuery]
    metadata: FixtureMetadata = Field(default_factory=FixtureMetadata)


class HygienePair(BaseModel):
    """A pair of entities that should (or should not) be merged."""

    a: Entity
    b: Entity
    should_merge: bool


class HygieneFixture(BaseModel):
    id: str
    description: str
    category: Literal["hygiene"] = "hygiene"
    pairs: list[HygienePair]
    metadata: FixtureMetadata = Field(default_factory=FixtureMetadata)


class RecallQuery(BaseModel):
    """A TPP (time/place/person) query with expected episode hits."""

    description: str
    time_range: tuple[datetime, datetime] | None = None
    place_key: str | None = None  # entity key for a Place
    person_key: str | None = None  # entity key for a Person
    expected_episode_ids: list[str]  # ranked, most-relevant first


class RecallFixture(BaseModel):
    id: str
    description: str
    category: Literal["recall"] = "recall"
    corpus: list[Episode]
    queries: list[RecallQuery]
    metadata: FixtureMetadata = Field(default_factory=FixtureMetadata)


# --- Eval report ---


class CategoryResult(BaseModel):
    """Per-category metrics + status."""

    status: Literal["pass", "fail", "skipped", "not_yet_implemented"] = "not_yet_implemented"
    phase_target: str  # "L1", "L2", etc.
    fixtures_run: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class EvalReport(BaseModel):
    """Top-level report emitted by run_all."""

    lifegraph_kg_version: str
    timestamp: datetime
    categories: dict[str, CategoryResult]
    ci_gate_pass: bool = False
    llm_judge: dict[str, str] | None = None  # model, temperature, etc.
