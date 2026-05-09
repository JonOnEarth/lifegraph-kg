# SPDX-License-Identifier: Apache-2.0
"""Performance metrics — latency, throughput, scaling.

Most metrics here are summary statistics over arrays of measurements.
The runners (``runners/run_perf.py``) are responsible for actually
collecting measurements; these are the pure aggregators.
"""

from __future__ import annotations

import math
from statistics import median


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy default)."""
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    rank = (pct / 100) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    fraction = rank - lo
    return sorted_vals[lo] + fraction * (sorted_vals[hi] - sorted_vals[lo])


def latency_summary(durations_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99 + min/max/mean for a list of latency measurements."""
    if not durations_ms:
        return {"samples": 0.0}
    return {
        "samples": float(len(durations_ms)),
        "p50": median(durations_ms),
        "p95": _percentile(durations_ms, 95),
        "p99": _percentile(durations_ms, 99),
        "min": min(durations_ms),
        "max": max(durations_ms),
        "mean": sum(durations_ms) / len(durations_ms),
    }


def throughput(operations: int, duration_s: float) -> float:
    """Operations per second."""
    if duration_s <= 0:
        return 0.0
    return operations / duration_s


def memory_growth(measurements: list[tuple[int, float]]) -> dict[str, float]:
    """Given (n_episodes, mb) pairs, fit a power-law and report exponent.

    For an autobiographical memory engine, the goal is sub-linear (or at
    worst linear) memory growth as the corpus scales. Exponent > 1.2 is a
    smell test failure.
    """
    if len(measurements) < 2:
        return {"data_points": float(len(measurements))}

    # Power-law: mb = c * n^alpha → log(mb) = log(c) + alpha * log(n)
    # Solve via least squares on log-log.
    log_n = [math.log(n) for n, _ in measurements]
    log_mb = [math.log(mb) for _, mb in measurements]
    n = len(measurements)
    sum_x = sum(log_n)
    sum_y = sum(log_mb)
    sum_xy = sum(x * y for x, y in zip(log_n, log_mb, strict=True))
    sum_xx = sum(x * x for x in log_n)
    denom = n * sum_xx - sum_x**2
    alpha = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else float("nan")
    log_c = (sum_y - alpha * sum_x) / n if not math.isnan(alpha) else float("nan")
    return {
        "data_points": float(n),
        "alpha": alpha,
        "linear_or_better": float(alpha <= 1.0),
        "constant_factor_mb": math.exp(log_c) if not math.isnan(log_c) else float("nan"),
        "max_observed_mb": max(mb for _, mb in measurements),
        "max_observed_n": float(max(n_ for n_, _ in measurements)),
    }
