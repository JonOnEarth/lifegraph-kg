# SPDX-License-Identifier: Apache-2.0
"""Episode — the unit-of-record for autobiographical memory.

An Episode is a Perdurant in DOLCE terms (it unfolds over time) and
maps to CIDOC-CRM's E5 Event and Conway's Event-Specific Knowledge.
It IS the activity instance — there is no separate Activity node.

`predicates`, `body_state`, `sentiment`, `energy` are scalar metadata
on the episode itself, not separate node-classes. This matches Conway's
Self-Memory System: affect and bodily-state are sensory-perceptual-
conceptual-affective summary features of the episode, not participants.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["pos", "neu", "neg"]
Energy = Literal["high", "medium", "low"]


class Episode(BaseModel):
    """A persisted life-log entry.

    Fields:
      - `id`           — unique identifier (ULID-ish, generated on save)
      - `text`         — the original entry, verbatim
      - `occurred_at`  — when the event happened in the user's life
      - `ingested_at`  — when this record was created in the store
      - `source`       — provenance: "user", "telegram", "voice", etc.
      - `predicates`   — list of normalized verbs (the v6 multi-action design)
      - `body_state`   — bodily state if explicitly mentioned ("tired", "累了")
      - `sentiment`    — affective valence, only when explicit
      - `energy`       — energy level, only when explicit
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    occurred_at: datetime
    ingested_at: datetime
    source: str = "user"
    predicates: list[str] = Field(default_factory=list)
    body_state: str | None = None
    sentiment: Sentiment | None = None
    energy: Energy | None = None
