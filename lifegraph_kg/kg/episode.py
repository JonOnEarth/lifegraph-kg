# SPDX-License-Identifier: Apache-2.0
"""Episode — the unit-of-record for autobiographical memory + tasks.

In DOLCE terms an Episode is a Perdurant (it unfolds over time); maps to
CIDOC-CRM E5 Event and Conway's Event-Specific Knowledge.

`predicates`, `body_state`, `sentiment`, `energy` are scalar metadata on
the episode itself, not separate node-classes (Conway's SMS: affect is
a feature of the episode, not a participant).

## Logs vs. tasks

Episodes have a ``kind`` discriminator: ``"log"`` (default — events that
happened, autobiographical memory) or ``"task"`` (intents — prospective
memory). They share most of the schema; tasks add lifecycle fields:

  - ``status``       active | done | dropped (default active)
  - ``priority``     high | medium | low | None
  - ``deadline``     when the task is due
  - ``completed_at`` when status moved to "done"
  - ``recurrence``   "daily" / "weekly" / RRULE-ish
  - ``gtd_context``  "@home", "@work", etc.
  - ``action_verb``  primary verb (often == predicates[0])

A completed task and a log are functionally equivalent for queries
("what did I do today" returns both).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["pos", "neu", "neg"]
Energy = Literal["high", "medium", "low"]
EpisodeKind = Literal["log", "task"]
EpisodeStatus = Literal["active", "done", "dropped"]
Priority = Literal["high", "medium", "low"]


class Episode(BaseModel):
    """A persisted life-log entry or task.

    Common fields apply to both kinds. Task-only fields (``status``,
    ``priority``, ``deadline``, ``completed_at``, ``recurrence``,
    ``gtd_context``, ``action_verb``) are present on logs too with
    defaults — the cost is one nullable column per field, the win is
    a uniform query surface.
    """

    model_config = ConfigDict(frozen=True)

    # Common to logs + tasks
    id: str
    text: str
    occurred_at: datetime
    ingested_at: datetime
    source: str = "user"
    predicates: list[str] = Field(default_factory=list)
    body_state: str | None = None
    sentiment: Sentiment | None = None
    energy: Energy | None = None

    # Discriminator — defaults to "log" so existing call sites are unchanged
    kind: EpisodeKind = "log"

    # Task lifecycle (only meaningful when kind == "task", but present on
    # all rows for schema simplicity). Defaults preserve log semantics:
    # active+nothing-pending matches existing log behavior.
    status: EpisodeStatus = "active"
    priority: Priority | None = None
    deadline: datetime | None = None
    completed_at: datetime | None = None
    recurrence: str | None = None
    gtd_context: str | None = None
    action_verb: str | None = None
