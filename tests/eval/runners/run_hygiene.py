# SPDX-License-Identifier: Apache-2.0
"""Run the hygiene eval against ``fixtures/hygiene/*.json``."""

from __future__ import annotations

import argparse
import sys

from tests.eval.runners import library_hygiene_ready, load_fixtures
from tests.eval.scoring.hygiene import dedup_precision_recall
from tests.eval.types import CategoryResult, Entity, HygieneFixture


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


PHASE_TARGET = "L3"


def run(validate_only: bool = False) -> CategoryResult:
    fixtures = load_fixtures("hygiene", HygieneFixture)

    if validate_only:
        return CategoryResult(
            status="pass",
            phase_target=PHASE_TARGET,
            fixtures_run=len(fixtures),
            notes=f"Validated {len(fixtures)} hygiene fixture(s).",
        )

    if not library_hygiene_ready():
        return CategoryResult(
            status="not_yet_implemented",
            phase_target=PHASE_TARGET,
            fixtures_run=0,
            notes=(
                f"Hygiene engine not yet implemented (target: {PHASE_TARGET}). "
                f"{len(fixtures)} fixture(s) ready and validated."
            ),
        )

    f1_scores: list[float] = []
    fmr_scores: list[float] = []

    for fixture in fixtures:
        # TODO(L3): feed all entities into the hygiene engine, get proposals.
        #   from lifegraph_kg.hygiene.dedup import propose_merges
        #   entities = [p.a for p in fixture.pairs] + [p.b for p in fixture.pairs]
        #   proposed = propose_merges(entities)
        proposed: list[tuple[Entity, Entity]] = []
        labelled = [(p.a, p.b, p.should_merge) for p in fixture.pairs]
        scores = dedup_precision_recall(proposed, labelled)
        f1_scores.append(scores["f1"])
        fmr_scores.append(scores["false_merge_rate"])

    return CategoryResult(
        status="pass",
        phase_target=PHASE_TARGET,
        fixtures_run=len(fixtures),
        metrics={
            "dedup_f1": _avg(f1_scores),
            "false_merge_rate": _avg(fmr_scores),
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
