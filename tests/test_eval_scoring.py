# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the eval scoring functions themselves.


These tests run without the library — they exercise the pure-logic
scoring functions on hand-built inputs. They guard against eval
regressions when scoring code changes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.eval.scoring.extraction import (
    entity_f1,
    grounding_iou,
    type_accuracy,
)
from tests.eval.scoring.hygiene import (
    dedup_precision_recall,
    grounding_survival,
)
from tests.eval.scoring.perf import (
    latency_summary,
    memory_growth,
    throughput,
)
from tests.eval.scoring.recall import (
    precision_at_k,
    recall_at_k,
    recall_summary,
    reciprocal_rank,
)
from tests.eval.scoring.temporal import (
    as_of_accuracy,
    fact_at,
    invalidate_not_delete,
)
from tests.eval.types import (
    Edge,
    Entity,
    Grounding,
    Span,
    TemporalQuery,
)

TEST_USER = "test-user"

# ----- extraction -----


def test_entity_f1_perfect() -> None:
    pred = [Entity(type="Person", key="sara", value="Sara")]
    gold = [Entity(type="Person", key="sara", value="Sara")]
    m = entity_f1(pred, gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_entity_f1_partial() -> None:
    pred = [
        Entity(type="Person", key="sara", value="Sara"),
        Entity(type="Place", key="ippudo", value="Ippudo"),
        Entity(type="Food", key="udon", value="udon"),  # spurious
    ]
    gold = [
        Entity(type="Person", key="sara", value="Sara"),
        Entity(type="Place", key="ippudo", value="Ippudo"),
        Entity(type="Food", key="ramen", value="ramen"),  # missed
    ]
    m = entity_f1(pred, gold)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_type_accuracy_catches_type_confusion() -> None:
    pred = [Entity(type="Place", key="sara", value="Sara")]  # wrong type
    gold = [Entity(type="Person", key="sara", value="Sara")]
    assert type_accuracy(pred, gold) == 0.0


def test_grounding_iou_perfect_alignment() -> None:
    pred = [Grounding(entity_type="Person", entity_key="sara", span=Span(start=15, end=19))]
    gold = [Grounding(entity_type="Person", entity_key="sara", span=Span(start=15, end=19))]
    m = grounding_iou(pred, gold)
    assert m["grounding_iou"] == 1.0
    assert m["missed"] == 0


def test_grounding_iou_partial_overlap() -> None:
    pred = [Grounding(entity_type="Person", entity_key="sara", span=Span(start=14, end=20))]
    gold = [Grounding(entity_type="Person", entity_key="sara", span=Span(start=15, end=19))]
    m = grounding_iou(pred, gold)
    # Predicted: [14, 20) length 6; Golden: [15, 19) length 4; overlap 4; union 6.
    assert m["grounding_iou"] == pytest.approx(4 / 6)


def test_grounding_iou_missing_entity() -> None:
    m = grounding_iou(
        predicted=[],
        golden=[Grounding(entity_type="Person", entity_key="sara", span=Span(start=15, end=19))],
    )
    assert m["grounding_iou"] == 0.0
    assert m["missed"] == 1


# ----- temporal -----


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _edge(
    pred: str,
    obj: str,
    t_valid: str,
    t_invalid: str | None = None,
) -> Edge:
    return Edge(
        user_id=TEST_USER, type=pred,
        from_entity=("Person", "sara"),
        to_entity=("Place", obj),
        t_event=_ts(t_valid),
        t_ingestion=_ts(t_valid),
        t_valid=_ts(t_valid),
        t_invalid=_ts(t_invalid) if t_invalid else None,
    )


def test_fact_at_returns_active_edge() -> None:
    edges = [
        _edge("lives_in", "berlin", "2025-06-01T09:00", "2026-01-15T18:00"),
        _edge("lives_in", "tokyo", "2026-01-15T18:00"),
    ]
    assert fact_at(edges, ("Person", "sara"), "lives_in", _ts("2025-12-01T00:00")) == (
        "Place",
        "berlin",
    )
    assert fact_at(edges, ("Person", "sara"), "lives_in", _ts("2026-03-01T00:00")) == (
        "Place",
        "tokyo",
    )
    assert fact_at(edges, ("Person", "sara"), "lives_in", _ts("2025-01-01T00:00")) is None


def test_as_of_accuracy_perfect() -> None:
    edges = [
        _edge("lives_in", "berlin", "2025-06-01T09:00", "2026-01-15T18:00"),
        _edge("lives_in", "tokyo", "2026-01-15T18:00"),
    ]
    queries = [
        TemporalQuery(
            as_of=_ts("2025-12-01T00:00"),
            subject=("Person", "sara"),
            predicate="lives_in",
            expected_object=("Place", "berlin"),
        ),
        TemporalQuery(
            as_of=_ts("2026-03-01T00:00"),
            subject=("Person", "sara"),
            predicate="lives_in",
            expected_object=("Place", "tokyo"),
        ),
    ]
    m = as_of_accuracy(edges, queries)
    assert m["as_of_accuracy"] == 1.0


def test_invalidate_not_delete_passes_when_history_preserved() -> None:
    before = [_edge("lives_in", "berlin", "2025-06-01T09:00")]
    after = [
        _edge("lives_in", "berlin", "2025-06-01T09:00", "2026-01-15T18:00"),
        _edge("lives_in", "tokyo", "2026-01-15T18:00"),
    ]
    m = invalidate_not_delete(before, after)
    assert m["invalidate_not_delete"] == 1.0
    assert m["missing_count"] == 0


def test_invalidate_not_delete_fails_when_edge_lost() -> None:
    before = [_edge("lives_in", "berlin", "2025-06-01T09:00")]
    after = [_edge("lives_in", "tokyo", "2026-01-15T18:00")]
    m = invalidate_not_delete(before, after)
    assert m["invalidate_not_delete"] == 0.0
    assert m["missing_count"] == 1


# ----- hygiene -----


def test_dedup_precision_recall_perfect() -> None:
    sara1 = Entity(type="Person", key="sara", value="Sara")
    sara2 = Entity(type="Person", key="sarah", value="Sarah")
    alex1 = Entity(type="Person", key="alex-smith", value="Alex Smith")
    alex2 = Entity(type="Person", key="alex-johnson", value="Alex Johnson")

    proposed = [(sara1, sara2)]
    labelled = [(sara1, sara2, True), (alex1, alex2, False)]

    m = dedup_precision_recall(proposed, labelled)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["false_merge_rate"] == 0.0


def test_dedup_false_merge_caught() -> None:
    sara = Entity(type="Person", key="sara", value="Sara")
    sarah = Entity(type="Person", key="sarah", value="Sarah")
    alex1 = Entity(type="Person", key="alex-smith", value="Alex Smith")
    alex2 = Entity(type="Person", key="alex-johnson", value="Alex Johnson")

    proposed = [(sara, sarah), (alex1, alex2)]  # second is wrong
    labelled = [(sara, sarah, True), (alex1, alex2, False)]

    m = dedup_precision_recall(proposed, labelled)
    assert m["precision"] == 0.5
    assert m["false_merge_rate"] == 1.0


def test_grounding_survival_perfect() -> None:
    sara = Entity(type="Person", key="sara", value="Sara")
    before = [
        Grounding(entity_type="Person", entity_key="sara", span=Span(start=0, end=4)),
        Grounding(entity_type="Person", entity_key="sarah", span=Span(start=10, end=15)),
    ]
    after = [
        Grounding(entity_type="Person", entity_key="sara", span=Span(start=0, end=4)),
        Grounding(entity_type="Person", entity_key="sara", span=Span(start=10, end=15)),
    ]
    m = grounding_survival(before, after, sara)
    assert m["grounding_survival"] == 1.0


def test_grounding_survival_lost() -> None:
    sara = Entity(type="Person", key="sara", value="Sara")
    before = [
        Grounding(entity_type="Person", entity_key="sara", span=Span(start=0, end=4)),
        Grounding(entity_type="Person", entity_key="sarah", span=Span(start=10, end=15)),
    ]
    after = [
        Grounding(entity_type="Person", entity_key="sara", span=Span(start=0, end=4)),
    ]
    m = grounding_survival(before, after, sara)
    assert m["grounding_survival"] == 0.5
    assert m["lost"] == 1


# ----- recall -----


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=2) == 0.5
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 0.5
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b"], {"a", "c"}, k=10) == 0.5
    assert recall_at_k(["a", "c"], {"a", "c"}, k=2) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"b"}) == 0.0


def test_recall_summary_aggregates() -> None:
    queries = [
        (["a", "b"], ["a"]),
        (["x", "y"], ["y"]),
    ]
    m = recall_summary(queries, k_values=(1, 5))
    assert m["queries"] == 2
    assert m["mrr"] == pytest.approx((1.0 + 0.5) / 2)


# ----- perf -----


def test_latency_summary_basic() -> None:
    m = latency_summary([10.0, 20.0, 30.0, 40.0, 50.0])
    assert m["samples"] == 5
    assert m["min"] == 10.0
    assert m["max"] == 50.0
    assert m["p50"] == 30.0
    assert m["mean"] == 30.0


def test_throughput() -> None:
    assert throughput(100, duration_s=1.0) == 100.0
    assert throughput(1000, duration_s=2.0) == 500.0
    assert throughput(10, duration_s=0) == 0.0


def test_memory_growth_linear() -> None:
    # Linear: mb = 0.5 * n
    measurements = [(100, 50.0), (1000, 500.0), (10000, 5000.0)]
    m = memory_growth(measurements)
    assert m["alpha"] == pytest.approx(1.0, abs=1e-6)
    assert m["linear_or_better"] == 1.0
