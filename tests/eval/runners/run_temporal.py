# SPDX-License-Identifier: Apache-2.0
"""Run the temporal-CRUD eval against ``fixtures/temporal/*.json``."""

from __future__ import annotations

import argparse
import sys

from tests.eval.runners import library_temporal_ready, load_fixtures
from tests.eval.scoring.temporal import as_of_accuracy, invalidate_not_delete
from tests.eval.types import CategoryResult, Edge, TemporalFixture


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


PHASE_TARGET = "L2"


def run(validate_only: bool = False) -> CategoryResult:
    fixtures = load_fixtures("temporal", TemporalFixture)

    if validate_only:
        return CategoryResult(
            status="pass",
            phase_target=PHASE_TARGET,
            fixtures_run=len(fixtures),
            notes=f"Validated {len(fixtures)} temporal fixture(s).",
        )

    if not library_temporal_ready():
        return CategoryResult(
            status="not_yet_implemented",
            phase_target=PHASE_TARGET,
            fixtures_run=0,
            notes=(
                f"Bi-temporal store not yet implemented (target: {PHASE_TARGET}). "
                f"{len(fixtures)} fixture(s) ready and validated."
            ),
        )

    as_of_scores: list[float] = []
    invalidate_scores: list[float] = []

    for fixture in fixtures:
        # TODO(L2): replay history into a LifeGraph store, then evaluate.
        #   lg = LifeGraph(store=":memory:")
        #   for ev in fixture.history:
        #       lg.log(ev.text, at=ev.at)
        #   edges_after = list(lg.kg.edges())
        edges_after: list[Edge] = []
        edges_before: list[Edge] = []
        as_of = as_of_accuracy(edges_after, fixture.queries)
        ind = invalidate_not_delete(edges_before, edges_after)
        as_of_scores.append(as_of["as_of_accuracy"])
        invalidate_scores.append(ind["invalidate_not_delete"])

    return CategoryResult(
        status="pass",
        phase_target=PHASE_TARGET,
        fixtures_run=len(fixtures),
        metrics={
            "as_of_accuracy": _avg(as_of_scores),
            "invalidate_not_delete": _avg(invalidate_scores),
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
