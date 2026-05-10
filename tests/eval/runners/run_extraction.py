# SPDX-License-Identifier: Apache-2.0
"""Run the extraction-quality eval against ``fixtures/extraction/*.json``.

Now wired to L1's actual extractor. Three modes:
  - ``--validate-only`` — just load + Pydantic-validate fixtures, no API calls.
  - default mode        — call the library on each fixture, score, return real metrics.
  - no API key          — skip live calls, return ``not_yet_implemented`` with a
                         note so CI passes without secrets.

Usage::

    python -m tests.eval.runners.run_extraction
    python -m tests.eval.runners.run_extraction --validate-only
"""

from __future__ import annotations

import argparse
import os
import sys

from tests.eval.runners import library_extractor_ready, load_fixtures
from tests.eval.scoring.extraction import (
    entity_f1,
    episode_metadata_accuracy,
    grounding_iou,
    predicate_f1,
    type_accuracy,
)
from tests.eval.types import CategoryResult, Entity, ExtractionFixture

PHASE_TARGET = "L1"


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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

    if "ANTHROPIC_API_KEY" not in os.environ:
        return CategoryResult(
            status="not_yet_implemented",
            phase_target=PHASE_TARGET,
            fixtures_run=0,
            metrics={},
            notes=(
                "Library is built but ANTHROPIC_API_KEY is not set. "
                f"{len(fixtures)} fixture(s) validated; live scoring requires key."
            ),
        )

    # Exercise the library's extract() directly (we score the extraction
    # itself, not persistence — those have their own runners).
    from lifegraph_kg import extract

    f1_scores: list[float] = []
    type_acc_scores: list[float] = []
    pred_f1_scores: list[float] = []
    metadata_avg_scores: list[float] = []
    grounding_iou_scores: list[float] = []

    for fixture in fixtures:
        result = extract(fixture.input_text)

        # Convert library Entity objects to eval Entity for scoring (same shape).
        predicted: list[Entity] = [
            Entity(type=e.type, key=e.key, value=e.value, attributes=e.attributes)
            for e in result.entities
        ]

        f1 = entity_f1(predicted, fixture.expected_entities)
        ta = type_accuracy(predicted, fixture.expected_entities)
        pf1 = predicate_f1(result.predicates, fixture.expected_predicates)
        meta = episode_metadata_accuracy(
            result.body_state,
            result.sentiment,
            result.energy,
            fixture.expected_body_state,
            fixture.expected_sentiment,
            fixture.expected_energy,
        )
        # Grounding only scored when the fixture provides expected groundings
        # (synthetic fixtures don't ship them yet).
        if fixture.expected_groundings:
            gi = grounding_iou([], fixture.expected_groundings)  # L1 doesn't emit groundings yet
            grounding_iou_scores.append(gi["grounding_iou"])

        f1_scores.append(f1["f1"])
        type_acc_scores.append(ta)
        pred_f1_scores.append(pf1["f1"])
        metadata_avg_scores.append(meta["metadata_avg"])

    metrics = {
        "entity_f1": _avg(f1_scores),
        "type_accuracy": _avg(type_acc_scores),
        "predicate_f1": _avg(pred_f1_scores),
        "metadata_accuracy": _avg(metadata_avg_scores),
    }
    if grounding_iou_scores:
        metrics["grounding_iou"] = _avg(grounding_iou_scores)

    return CategoryResult(
        status="pass",
        phase_target=PHASE_TARGET,
        fixtures_run=len(fixtures),
        metrics=metrics,
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
