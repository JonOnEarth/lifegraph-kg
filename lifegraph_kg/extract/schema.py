# SPDX-License-Identifier: Apache-2.0
"""Pydantic schema for extraction output — what `lg.log(text)` returns.

Mirrors the v6 prompt's output shape. Every field corresponds to a
canonical decision from the personal-KG ontology research:

- `predicates`  list[str] — multi-action support (Conway: episodes
  often contain compound actions). Each becomes an edge label in L2.
- `body_state`  Conway's affective summary feature of the episode.
  NOT an entity (no persistent identity).
- `sentiment`   Episode-level affect, default null (don't infer
  neutral from absence — that was the v5 hallucination bug).
- `energy`      Same as sentiment — explicit-only.
- `entities`    The 4-class typed nodes from `lifegraph_kg.classes`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lifegraph_kg.classes import Person, Place, Project, Topic

# Discriminated-union over the 4 entity classes. Pydantic uses the
# `type` Literal field on each subclass to pick the right one when
# parsing JSON.
EntityT = Person | Place | Project | Topic

Sentiment = Literal["pos", "neu", "neg"]
Energy = Literal["high", "medium", "low"]


class ExtractionResult(BaseModel):
    """The full output of one `extract(text)` call.

    Designed to be Pydantic-validated against the LLM's JSON output —
    invalid responses raise pydantic.ValidationError, which the caller
    can choose to handle (retry, fall back to empty, etc).
    """

    model_config = ConfigDict(frozen=True)

    # Episode metadata — scalar fields on the future Episode node.
    predicates: list[str] = Field(default_factory=list)
    body_state: str | None = None
    sentiment: Sentiment | None = None
    energy: Energy | None = None

    # The 4-class typed entities. Discriminated union by `type`.
    entities: list[EntityT] = Field(default_factory=list)

    # Optional metadata about the extraction itself (filled by the
    # extract() pipeline, not by the LLM):
    substring_violations: list[str] = Field(default_factory=list)
    critic_issues: list[dict[str, str]] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    """Output of the critic pass — flags issues with an extraction."""

    model_config = ConfigDict(frozen=True)

    valid: bool = True
    issues: list[dict[str, str]] = Field(default_factory=list)
