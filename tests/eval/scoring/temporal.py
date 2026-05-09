# SPDX-License-Identifier: Apache-2.0
"""Temporal CRUD metrics.

Three things to measure for the bi-temporal store:

1. Supersede correctness — when a contradicting fact arrives, is the prior
   edge invalidated (not deleted) and the new edge inserted?
2. as_of accuracy — for time-travel queries, do we return the fact that was
   true at the queried time?
3. Invalidate-not-delete — every historical edge survives, with t_invalid set.
"""

from __future__ import annotations

from datetime import datetime

from tests.eval.types import Edge, TemporalQuery


def fact_at(
    edges: list[Edge],
    subject: tuple[str, str],
    predicate: str,
    as_of: datetime,
) -> tuple[str, str] | None:
    """Find the (object_type, object_key) for a fact at a given time.

    Returns None if no edge is valid at that time.
    """
    matching = [
        e
        for e in edges
        if e.from_entity == subject
        and e.type == predicate
        and e.t_valid <= as_of
        and (e.t_invalid is None or as_of < e.t_invalid)
    ]
    if not matching:
        return None
    # If the model is correct there should be exactly one — pick the most
    # recently ingested if multiple (shouldn't happen, but be defensive).
    most_recent = max(matching, key=lambda e: e.t_ingestion)
    return most_recent.to_entity


def as_of_accuracy(
    edges: list[Edge],
    queries: list[TemporalQuery],
) -> dict[str, float]:
    """For each query, does ``fact_at`` return the expected object?"""
    if not queries:
        return {"as_of_accuracy": 0.0, "queries_run": 0.0}

    correct = 0
    for q in queries:
        actual = fact_at(edges, q.subject, q.predicate, q.as_of)
        if actual == q.expected_object:
            correct += 1
    return {
        "as_of_accuracy": correct / len(queries),
        "queries_run": float(len(queries)),
        "correct": float(correct),
    }


def supersede_correctness(
    edges_after_replay: list[Edge],
    expected_supersessions: list[tuple[Edge, Edge]],
) -> dict[str, float]:
    """For each (old_edge, new_edge) pair, check the supersede invariant.

    Required:
      - old_edge is in edges_after_replay with t_invalid set
      - new_edge is in edges_after_replay with t_invalid == None
      - old_edge.t_invalid == new_edge.t_valid (no gap)
    """
    if not expected_supersessions:
        return {"supersede_correctness": 0.0, "pairs_checked": 0.0}

    by_id: dict[
        tuple[tuple[str, str], str, tuple[str, str]],
        list[Edge],
    ] = {}
    for e in edges_after_replay:
        by_id.setdefault((e.from_entity, e.type, e.to_entity), []).append(e)

    correct = 0
    for old_edge, new_edge in expected_supersessions:
        old_match = next(
            (
                e
                for e in by_id.get((old_edge.from_entity, old_edge.type, old_edge.to_entity), [])
                if e.t_valid == old_edge.t_valid
            ),
            None,
        )
        new_match = next(
            (
                e
                for e in by_id.get((new_edge.from_entity, new_edge.type, new_edge.to_entity), [])
                if e.t_valid == new_edge.t_valid
            ),
            None,
        )
        if (
            old_match is not None
            and new_match is not None
            and old_match.t_invalid is not None
            and new_match.t_invalid is None
            and old_match.t_invalid == new_match.t_valid
        ):
            correct += 1

    return {
        "supersede_correctness": correct / len(expected_supersessions),
        "pairs_checked": float(len(expected_supersessions)),
        "correct": float(correct),
    }


def invalidate_not_delete(
    edges_before: list[Edge],
    edges_after: list[Edge],
) -> dict[str, float]:
    """Every edge in ``before`` must still exist in ``after``.

    Edges may be invalidated (t_invalid set) but never deleted.
    """
    if not edges_before:
        return {"invalidate_not_delete": 1.0, "missing_count": 0.0}

    after_keys = {(e.from_entity, e.type, e.to_entity, e.t_valid) for e in edges_after}
    missing = 0
    for e in edges_before:
        if (e.from_entity, e.type, e.to_entity, e.t_valid) not in after_keys:
            missing += 1
    survival = (len(edges_before) - missing) / len(edges_before)
    return {
        "invalidate_not_delete": survival,
        "missing_count": float(missing),
        "checked": float(len(edges_before)),
    }
