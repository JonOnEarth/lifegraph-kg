# SPDX-License-Identifier: Apache-2.0
"""Hygiene engine metrics.

Three things to measure:

1. Dedup precision/recall over a labelled set of (a, b, should_merge) pairs.
2. False-merge rate — how often do we merge entities that shouldn't be?
3. Grounding survival — after a merge, both source groundings remain
   attached to the canonical entity.
"""

from __future__ import annotations

from tests.eval.types import Entity, Grounding


def dedup_precision_recall(
    proposed_merges: list[tuple[Entity, Entity]],
    golden_pairs: list[tuple[Entity, Entity, bool]],
) -> dict[str, float]:
    """Score the hygiene engine's dedup proposals against a labelled set.

    proposed_merges: pairs the engine says should be merged.
    golden_pairs: (a, b, should_merge) labelled ground truth.

    Returns: precision, recall, f1, false_merge_rate, plus raw counts.
    """
    proposed_set = {frozenset({(a.type, a.key), (b.type, b.key)}) for a, b in proposed_merges}
    golden_positive = {
        frozenset({(a.type, a.key), (b.type, b.key)}) for a, b, should in golden_pairs if should
    }
    golden_negative = {
        frozenset({(a.type, a.key), (b.type, b.key)}) for a, b, should in golden_pairs if not should
    }

    tp = len(proposed_set & golden_positive)
    fp_against_negatives = len(proposed_set & golden_negative)
    fp_unknown = len(proposed_set - golden_positive - golden_negative)
    fp = fp_against_negatives + fp_unknown
    fn = len(golden_positive - proposed_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_merge_rate = fp_against_negatives / len(golden_negative) if golden_negative else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_merge_rate": false_merge_rate,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def grounding_survival(
    groundings_before_merge: list[Grounding],
    groundings_after_merge: list[Grounding],
    canonical_entity: Entity,
) -> dict[str, float]:
    """All groundings that targeted any merged-in entity must still exist on
    the canonical entity after the merge.

    The exact spans must be preserved (start, end identical) since they're
    pointers into immutable episode text.
    """
    if not groundings_before_merge:
        return {"grounding_survival": 1.0, "lost": 0.0, "checked": 0.0}

    after_spans = {
        (g.span.start, g.span.end)
        for g in groundings_after_merge
        if (g.entity_type, g.entity_key) == (canonical_entity.type, canonical_entity.key)
    }
    lost = 0
    for g in groundings_before_merge:
        if (g.span.start, g.span.end) not in after_spans:
            lost += 1
    survival = (len(groundings_before_merge) - lost) / len(groundings_before_merge)
    return {
        "grounding_survival": survival,
        "lost": float(lost),
        "checked": float(len(groundings_before_merge)),
    }
