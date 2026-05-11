# SPDX-License-Identifier: Apache-2.0
"""Edge — bi-temporal verb-as-edge between entities.

The edge model is the core of `lifegraph-kg`'s differentiation from
agent-memory frameworks. Every edge carries:

  - `verb`         — normalized lowercase predicate (the v6 multi-action design)
  - `from_entity`  — subject (NULL == "the user / me", an implicit subject)
  - `to_entity`    — object (always a real entity)
  - `episode_id`   — provenance back-reference (audit trail)
  - `t_event`      — when the fact was true (event time)
  - `t_ingestion`  — when we learned about it
  - `t_valid`      — start of validity window (= t_event by default)
  - `t_invalid`    — end of validity (NULL == currently valid)

This is the Graphiti / Zep bi-temporal model. New contradicting facts
**invalidate** prior edges (set `t_invalid`); they don't delete them.
That preserves the audit trail and makes time-travel queries possible:
"what did I think Sara's job was on 2025-12-01?".

NULL `from_entity` is a deliberate design choice: life-log entries have
the user as the implicit subject. Creating a synthetic "Person:me" node
for every edge would be noise. Queries that walk from the user start
with "from_entity IS NULL".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Edge(BaseModel):
    """A bi-temporal edge between (optional) `from_entity` and `to_entity`.

    ``user_id`` is denormalized onto every edge so user-scoped queries
    don't need to JOIN through entities / episodes. It must always
    equal the user_id of the referenced episode + entities; the store
    layer enforces that invariant on write.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    from_entity: str | None  # NULL == the user / me
    to_entity: str
    verb: str
    episode_id: str
    t_event: datetime
    t_ingestion: datetime
    t_valid: datetime
    t_invalid: datetime | None = None

    @property
    def is_active(self) -> bool:
        """True if the edge is currently valid (t_invalid is NULL)."""
        return self.t_invalid is None

    def is_valid_at(self, t: datetime) -> bool:
        """True if the edge was valid at time `t`."""
        if t < self.t_valid:
            return False
        if self.t_invalid is not None and t >= self.t_invalid:
            return False
        return True
