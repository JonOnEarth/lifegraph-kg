# SPDX-License-Identifier: Apache-2.0
"""MergeProposal — the hygiene engine's output unit.

Every merge proposal is a `(entity_a, entity_b, confidence, reason)`
tuple. The engine produces proposals; the user (or auto-apply policy)
decides which to apply. Apply-time, the loser entity's edges + mentions
get redirected to the winner, and the loser is marked merged via its
canonical_id pointer.

This is the same proposal/apply pipeline the legacy LifeGraph used —
explicitly modeled so dedup is reviewable rather than silent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lifegraph_kg.classes import Entity

# Confidence buckets — discrete instead of continuous because the
# heuristics produce score plateaus and "0.83 vs 0.85" rarely matters.
# Auto-apply policy in v0.1 only triggers on `high`.
ConfidenceLevel = Literal["high", "medium", "low"]

ProposalReason = Literal[
    "exact_normalized",  # canonical_form(a) == canonical_form(b)
    "substring_qualifier",  # one is a qualified form of the other (Ippudo / Ippudo NYC)
    "edit_distance",  # Levenshtein distance ≤ threshold
    "role_term_pair",  # 老板 / boss in the same context — needs context to confirm
]


class MergeProposal(BaseModel):
    """A proposal to merge `loser` into `winner` (loser becomes an alias).

    `winner` is chosen by simple policy: the entity with the more
    canonical / longer / earlier-created form. Apply-time we redirect
    all of loser's edges + mentions to winner and set loser's
    canonical_id to point at winner.
    """

    model_config = ConfigDict(frozen=True)

    winner: Entity
    loser: Entity
    confidence: ConfidenceLevel
    reason: ProposalReason
    detail: str = Field(default="")

    @property
    def is_safe_to_auto_apply(self) -> bool:
        """v0.1 policy: only `exact_normalized` with `high` confidence
        is auto-apply-safe. Everything else needs human review."""
        return self.confidence == "high" and self.reason == "exact_normalized"

    def __str__(self) -> str:
        return (
            f"merge {self.loser.type}:{self.loser.value!r} → "
            f"{self.winner.value!r}  [{self.confidence}, {self.reason}]"
        )
