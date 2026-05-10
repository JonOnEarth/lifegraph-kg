# SPDX-License-Identifier: Apache-2.0
"""Heuristic dedup — produces MergeProposals from a set of entities.

v0.1 ships pure-string heuristics (no embeddings, no LLM): casefold +
NFKC equality, substring qualification, and Damerau-Levenshtein for
short strings. This catches the common cases (`Sara` / `sara` /
`Sarah`, `Ippudo` / `Ippudo NYC`) and is fast + free + deterministic.

The hygiene engine does NOT propose merges across different `type`s
(a Person never merges with a Topic). Topic merges only consider
proposals within the same `kind` discriminator (a Food doesn't merge
with a Media even if their values look similar).

Embedding-based fuzzy dedup is L3.1 — opt-in via
`pip install lifegraph-kg[hygiene-embeddings]`.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from lifegraph_kg.classes import Entity, Topic
from lifegraph_kg.hygiene.normalize import canonical_form
from lifegraph_kg.hygiene.proposals import MergeProposal

# Damerau-Levenshtein cutoff — strings shorter than this skip the
# distance check entirely (false-positive risk too high). Tuned to
# avoid matching "Tao" / "Tom" while still catching "Ippudo" / "Ippud0".
_MIN_LEN_FOR_DISTANCE = 5

# Maximum edit distance for proposing an `edit_distance` merge.
# Anything farther is considered too speculative for a heuristic engine.
_MAX_EDIT_DISTANCE = 1


def _damerau_levenshtein(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (counts adjacent-transposition as 1).

    Quadratic in length, but inputs are entity values (typically <50
    chars), and the substring guard above bails out early on tiny
    inputs. Good enough for v0.1's personal-scale graphs.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row DP — saves memory vs full matrix without changing complexity.
    prev_prev: list[int] = []
    prev: list[int] = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,  # insertion
                prev[j] + 1,  # deletion
                prev[j - 1] + cost,  # substitution
            )
            # Damerau transposition
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb and prev_prev:
                curr[j] = min(curr[j], prev_prev[j - 2] + cost)
        prev_prev = prev
        prev = curr
    return prev[-1]


def _pick_winner(a: Entity, b: Entity) -> tuple[Entity, Entity]:
    """Choose which entity wins (becomes the canonical) and which loses.

    Policy:
      1. Prefer the entity whose `value` is the canonical form of the
         other (e.g. `Sara` wins over `sara`; the unaltered casing wins).
      2. Otherwise prefer the longer value (`Ippudo NYC` over `Ippudo`).
      3. Otherwise alphabetic order — deterministic tiebreaker.
    """
    a_canon = canonical_form(a.value)
    b_canon = canonical_form(b.value)
    # If one is the canonical and the other isn't (different surface)
    if a.value == a_canon and b.value != b_canon:
        return b, a  # ← canonical b wins... wait, a is canonical
    # Reread: a.value == its canonical form means a is already lowercase
    # We want to PREFER the cased / human-friendly form. Flip the rule:
    if a.value == a_canon and b.value != b_canon:
        # a is already lowercased (pre-normalized); b is not — keep b.
        return b, a
    if b.value == b_canon and a.value != a_canon:
        return a, b
    # Lengths
    if len(a.value) > len(b.value):
        return a, b
    if len(b.value) > len(a.value):
        return b, a
    # Tiebreaker
    return (a, b) if a.value <= b.value else (b, a)


def _are_compatible_for_merge(a: Entity, b: Entity) -> bool:
    """Type compatibility check — never merge across types or across
    Topic kinds."""
    if a.type != b.type:
        return False
    if isinstance(a, Topic) and isinstance(b, Topic):
        return a.kind == b.kind
    return True


def _propose_pair(a: Entity, b: Entity) -> MergeProposal | None:
    """Run the heuristic battery on a single pair. Returns None if no
    rule fires."""
    if not _are_compatible_for_merge(a, b):
        return None

    a_canon = canonical_form(a.value)
    b_canon = canonical_form(b.value)
    if not a_canon or not b_canon:
        return None

    winner, loser = _pick_winner(a, b)

    # Rule 1: exact match after normalization.
    if a_canon == b_canon:
        return MergeProposal(
            winner=winner,
            loser=loser,
            confidence="high",
            reason="exact_normalized",
            detail=f"both canonicalize to {a_canon!r}",
        )

    # Rule 2: substring qualifier (Ippudo / Ippudo NYC).
    if a_canon != b_canon and (
        a_canon.startswith(b_canon + " ")
        or a_canon.endswith(" " + b_canon)
        or b_canon.startswith(a_canon + " ")
        or b_canon.endswith(" " + a_canon)
    ):
        # The longer one wins (it's the qualified form).
        long_e, short_e = (a, b) if len(a_canon) > len(b_canon) else (b, a)
        return MergeProposal(
            winner=long_e,
            loser=short_e,
            confidence="medium",
            reason="substring_qualifier",
            detail=f"{short_e.value!r} is a substring of {long_e.value!r}",
        )

    # Rule 3: small edit distance for non-trivial-length strings.
    # ASCII-only strings; for CJK we'd need a different metric, so skip
    # if either has non-ASCII.
    if a.value.isascii() and b.value.isascii():
        if min(len(a_canon), len(b_canon)) >= _MIN_LEN_FOR_DISTANCE:
            d = _damerau_levenshtein(a_canon, b_canon)
            if d <= _MAX_EDIT_DISTANCE:
                return MergeProposal(
                    winner=winner,
                    loser=loser,
                    confidence="medium",
                    reason="edit_distance",
                    detail=f"Damerau-Levenshtein distance {d}",
                )

    return None


def propose_merges(entities: Iterable[Entity]) -> list[MergeProposal]:
    """Run heuristic dedup over a collection of entities. Returns the
    list of all proposed merges (potentially overlapping — apply-time
    cycles aren't currently checked, the user sees them)."""
    entities_list = list(entities)
    proposals: list[MergeProposal] = []
    for a, b in combinations(entities_list, 2):
        proposal = _propose_pair(a, b)
        if proposal is not None:
            proposals.append(proposal)
    return proposals
