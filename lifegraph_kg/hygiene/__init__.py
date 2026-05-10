# SPDX-License-Identifier: Apache-2.0
"""Hygiene engine — opt-in dedup + canonicalization layer.

The differentiator between `lifegraph-kg` and Graphiti / Mem0 / Letta:
all of them rely on the LLM + graph-merge heuristics for dedup. We
make the dedup pipeline an explicit, reviewable step.

L3 v0.1 ships pure-string heuristics (NFKC, casefold, substring
qualification, Damerau-Levenshtein) — no embeddings, no LLM calls,
deterministic. Embedding-based fuzzy dedup lands as L3.1, opt-in via
`pip install lifegraph-kg[hygiene-embeddings]`.

Public API (via the LifeGraph facade):
    proposals = lg.hygiene.propose()             # scan all entities, return MergeProposals
    proposals = lg.hygiene.propose(type=Person)  # restrict to one type
    lg.hygiene.apply(proposal)                   # apply one merge
    lg.hygiene.auto_apply()                       # apply all is_safe_to_auto_apply ones
"""

from __future__ import annotations

from lifegraph_kg.hygiene.dedup import propose_merges
from lifegraph_kg.hygiene.normalize import canonical_form, normalized_value_eq
from lifegraph_kg.hygiene.proposals import (
    ConfidenceLevel,
    MergeProposal,
    ProposalReason,
)

__all__ = [
    "ConfidenceLevel",
    "MergeProposal",
    "ProposalReason",
    "canonical_form",
    "normalized_value_eq",
    "propose_merges",
]
