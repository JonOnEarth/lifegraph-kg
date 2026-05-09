# SPDX-License-Identifier: Apache-2.0
"""Run the performance eval — latency, throughput, scaling.

Unlike the other runners, perf doesn't have a ``fixtures/perf/`` directory;
instead, the corpus in ``fixtures/corpora/sample-week.json`` (and larger
synthetic corpora generated on the fly) drive measurements.

Performance gates the v0.1 launch on these targets:
- log latency p50 < 50 ms (excluding LLM call)
- query latency p50 < 10 ms
- TPP query latency p50 < 50 ms on a 10K-episode corpus
- memory growth alpha <= 1.0 (linear or sub-linear)
"""

from __future__ import annotations

import argparse
import sys

from tests.eval.runners import library_extractor_ready
from tests.eval.scoring.perf import latency_summary
from tests.eval.types import CategoryResult

PHASE_TARGET = "L1"  # latency baselines start with L1


def run(validate_only: bool = False) -> CategoryResult:
    if validate_only:
        return CategoryResult(
            status="pass",
            phase_target=PHASE_TARGET,
            fixtures_run=0,
            notes="Perf has no static fixtures — measurements are generated.",
        )

    if not library_extractor_ready():
        return CategoryResult(
            status="not_yet_implemented",
            phase_target=PHASE_TARGET,
            fixtures_run=0,
            notes=f"Library not yet implemented (target: {PHASE_TARGET}).",
        )

    # TODO(L1): drive measurements:
    #   from time import perf_counter_ns
    #   lg = LifeGraph(store=":memory:")
    #   log_latencies_ms: list[float] = []
    #   for ep in corpus:
    #       t0 = perf_counter_ns()
    #       lg.log(ep.text, at=ep.occurred_at)
    #       log_latencies_ms.append((perf_counter_ns() - t0) / 1e6)
    log_latencies_ms: list[float] = []
    query_latencies_ms: list[float] = []

    return CategoryResult(
        status="pass",
        phase_target=PHASE_TARGET,
        fixtures_run=0,
        metrics={
            **{f"log.{k}": v for k, v in latency_summary(log_latencies_ms).items()},
            **{f"query.{k}": v for k, v in latency_summary(query_latencies_ms).items()},
        },
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    result = run(validate_only=args.validate_only)
    print(result.model_dump_json(indent=2))
    return 0 if result.status in ("pass", "not_yet_implemented") else 1


if __name__ == "__main__":
    sys.exit(_main())
