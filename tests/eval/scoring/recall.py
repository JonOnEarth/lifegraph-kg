# SPDX-License-Identifier: Apache-2.0
"""Recall metrics for time/place/person (TPP) queries.

Three things to measure:

1. Precision@K — of the top-K episodes returned, how many are relevant?
2. Recall@K — of all relevant episodes, how many appear in the top-K?
3. MRR (mean reciprocal rank) — for each query, 1 / rank-of-first-relevant.

Latency belongs in scoring/perf.py; here we score answer quality.
"""

from __future__ import annotations


def precision_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Of the first k predictions, what fraction are relevant?"""
    if k <= 0:
        return 0.0
    top_k = predicted_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Of all relevant items, what fraction appear in the top k?"""
    if not relevant_ids:
        return 0.0
    top_k_set = set(predicted_ids[:k])
    return len(top_k_set & relevant_ids) / len(relevant_ids)


def reciprocal_rank(predicted_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / (rank of first relevant prediction). 0 if none found."""
    for i, pid in enumerate(predicted_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / i
    return 0.0


def recall_summary(
    queries: list[tuple[list[str], list[str]]],
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Aggregate metrics across many queries.

    queries: list of (predicted_ids_ranked, relevant_ids_unranked) tuples.
    Returns macro-averaged P@K, R@K for each K, plus MRR.
    """
    if not queries:
        return {"queries": 0.0}

    out: dict[str, float] = {"queries": float(len(queries))}
    rrs: list[float] = []
    for k in k_values:
        ps: list[float] = []
        rs: list[float] = []
        for predicted, relevant in queries:
            relevant_set = set(relevant)
            ps.append(precision_at_k(predicted, relevant_set, k))
            rs.append(recall_at_k(predicted, relevant_set, k))
        out[f"precision@{k}"] = sum(ps) / len(ps)
        out[f"recall@{k}"] = sum(rs) / len(rs)

    for predicted, relevant in queries:
        rrs.append(reciprocal_rank(predicted, set(relevant)))
    out["mrr"] = sum(rrs) / len(rrs)
    return out
