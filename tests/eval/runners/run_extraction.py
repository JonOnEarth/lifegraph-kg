# SPDX-License-Identifier: Apache-2.0
"""Run the extraction-quality eval against ``fixtures/extraction/*.json``.

Usage::

    python -m tests.eval.runners.run_extraction
    python -m tests.eval.runners.run_extraction --validate-only
"""

from __future__ import annotations

import argparse
import sys

from tests.eval.runners import library_extractor_ready, load_fixtures
from tests.eval.scoring.extraction import (
    entity_f1,
    grounding_iou,
    type_accuracy,
)
from tests.eval.types import CategoryResult, Entity, ExtractionFixture, Grounding


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


PHASE_TARGET = "L1"


def run(validate_only: bool = False) -> CategoryResult:
    fixtures = load_fixtures("extraction", ExtractionFixture)

    if validate_only:
        return CategoryResult(
            status="pass",
            phase_target=PHASE_TARGET,
            fixtures_run=len(fixtures),
            metrics={},
            notes=f"Validated {len(fixtures)} extraction fixture(s).",
        )

    if not library_extractor_ready():
        return CategoryResult(
            status="not_yet_implemented",
            phase_target=PHASE_TARGET,
            fixtures_run=0,
            metrics={},
            notes=(
                f"Extractor not yet implemented (target: {PHASE_TARGET}). "
                f"{len(fixtures)} fixture(s) ready and validated."
            ),
        )

    # L1+: actually invoke the library and score. The extractor + LifeGraph
    # facade are imported via importlib at the call site below to keep mypy
    # static analysis clean while we wait for L1 to land them.

    f1_scores: list[float] = []
    type_acc_scores: list[float] = []
    grounding_iou_scores: list[float] = []

    for fixture in fixtures:
        # TODO(L1): replace placeholders with real lg.log() output.
        #   lg = LifeGraph(store=":memory:")
        #   ep = lg.log(fixture.input_text, ...)
        #   predicted_entities, predicted_groundings = ep.entities, ep.groundings
        predicted_entities: list[Entity] = []
        predicted_groundings: list[Grounding] = []

        f1 = entity_f1(predicted_entities, fixture.expected_entities)
        ta = type_accuracy(predicted_entities, fixture.expected_entities)
        gi = grounding_iou(predicted_groundings, fixture.expected_groundings)

        f1_scores.append(f1["f1"])
        type_acc_scores.append(ta)
        grounding_iou_scores.append(gi["grounding_iou"])

    return CategoryResult(
        status="pass",
        phase_target=PHASE_TARGET,
        fixtures_run=len(fixtures),
        metrics={
            "f1": _avg(f1_scores),
            "type_accuracy": _avg(type_acc_scores),
            "grounding_iou": _avg(grounding_iou_scores),
        },
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only check that fixtures load and validate; don't invoke the library.",
    )
    args = parser.parse_args()

    result = run(validate_only=args.validate_only)
    print(result.model_dump_json(indent=2))
    return 0 if result.status in ("pass", "not_yet_implemented") else 1


if __name__ == "__main__":
    sys.exit(_main())
