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

# Timezone mode for tasks. Logs are always implicitly "absolute" (the
# event happened at a fixed physical moment) and leave this null.
#
#   absolute → "May 15, 3pm meeting with Tom" — UTC-fixed; reschedule
#              doesn't follow the user across timezones.
#   floating → "Every day at 8am meditate" — interpreted in the user's
#              CURRENT timezone at fire time; follows them around.
#
# See docs: §3.2 / §3.3 + §5.3 dispatcher.
TimeMode = Literal["absolute", "floating"]


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
    user_id: str
    text: str
    occurred_at: datetime
    ingested_at: datetime
    source: str = "user"
    predicates: list[str] = Field(default_factory=list)
    body_state: str | None = None
    sentiment: Sentiment | None = None
    energy: Energy | None = None

    # Duration (minutes). Either user-stated or AI-inferred from a
    # conventional activity (meal=30, workout=45, meeting=30, etc.).
    # ``duration_inferred`` flags the second case so the UI can show
    # "~30 min" with a tilde, distinguishing from ground-truth.
    duration: int | None = None
    duration_inferred: bool | None = None

    # Timezone handling. Per the docs §3 model: store IANA names never
    # offsets so we cover DST + historical rule changes. ``origin_tz``
    # is audit-truth for "where the user was" — useful even on logs
    # ("oh right I was in Tokyo when I had that ramen").
    #
    # Tasks additionally carry ``time_mode``. floating tasks defer
    # their fire time to the user's CURRENT tz at scheduler time and
    # store the wall-clock spec instead of a UTC instant.
    origin_tz: str | None = None
    time_mode: TimeMode | None = None
    wall_clock_hour: int | None = None     # 0-23
    wall_clock_minute: int | None = None   # 0-59
    wall_clock_date: str | None = None     # "MM-DD" (yearly) or None

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
