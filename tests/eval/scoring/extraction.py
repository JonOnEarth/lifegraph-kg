# SPDX-License-Identifier: Apache-2.0
"""Extraction quality metrics.

Three things to measure for ``lg.log(text)``:

1. Entity F1 — micro-averaged precision/recall/F1 of (type, key) tuples.
2. Type accuracy — when an entity is matched by key, is its type right?
3. Grounding IoU — for matched entities, how well do char-intervals align?
"""

from __future__ import annotations

from collections import defaultdict

from tests.eval.types import Entity, Grounding


def _entity_id(e: Entity) -> tuple[str, str]:
    """Identity of an entity for matching: (type, key)."""
    return (e.type, e.key)


def entity_f1(predicted: list[Entity], golden: list[Entity]) -> dict[str, float]:
    """Micro precision/recall/F1 on (type, key) tuples.

    Also reports per-class F1 so regressions in a single class don't hide.
    """
    pred_set = {_entity_id(e) for e in predicted}
    gold_set = {_entity_id(e) for e in golden}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-class F1 — useful to spot a single class regressing
    per_class: dict[str, dict[str, float]] = {}
    classes = {t for t, _ in pred_set | gold_set}
    for cls in classes:
        cls_pred = {(t, k) for t, k in pred_set if t == cls}
        cls_gold = {(t, k) for t, k in gold_set if t == cls}
        cls_tp = len(cls_pred & cls_gold)
        cls_fp = len(cls_pred - cls_gold)
        cls_fn = len(cls_gold - cls_pred)
        cls_p = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        cls_r = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        cls_f = 2 * cls_p * cls_r / (cls_p + cls_r) if (cls_p + cls_r) > 0 else 0.0
        per_class[cls] = {"precision": cls_p, "recall": cls_r, "f1": cls_f}

    out: dict[str, float] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
    for cls, m in per_class.items():
        for metric, val in m.items():
            out[f"{cls}.{metric}"] = val
    return out


def type_accuracy(predicted: list[Entity], golden: list[Entity]) -> float:
    """For entities matched by key, what fraction have the correct type?

    This catches "we found Sara but called her a Place" failures that
    entity_f1 misses (since F1 is over (type, key) — a wrong type just
    looks like a miss, not a type confusion).
    """
    gold_by_key: dict[str, str] = {e.key: e.type for e in golden}
    matches = [(e, gold_by_key[e.key]) for e in predicted if e.key in gold_by_key]
    if not matches:
        return 0.0
    correct = sum(1 for pred, gold_type in matches if pred.type == gold_type)
    return correct / len(matches)


def grounding_iou(
    predicted: list[Grounding],
    golden: list[Grounding],
) -> dict[str, float]:
    """Macro-average IoU on char-intervals for matched entities.

    For each (entity_type, entity_key) in golden, find the best-overlapping
    predicted grounding and take the IoU. Average across all golden groundings.
    """
    if not golden:
        return {"grounding_iou": 0.0, "matched": 0.0, "missed": 0.0}

    pred_by_entity: dict[tuple[str, str], list[Grounding]] = defaultdict(list)
    for g in predicted:
        pred_by_entity[(g.entity_type, g.entity_key)].append(g)

    ious: list[float] = []
    missed = 0
    for g in golden:
        candidates = pred_by_entity.get((g.entity_type, g.entity_key), [])
        if not candidates:
            missed += 1
            ious.append(0.0)
            continue
        best = max(candidates, key=lambda p: p.span.iou(g.span))
        ious.append(best.span.iou(g.span))

    return {
        "grounding_iou": sum(ious) / len(ious) if ious else 0.0,
        "matched": float(len(ious) - missed),
        "missed": float(missed),
    }
