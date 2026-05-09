# SPDX-License-Identifier: Apache-2.0
"""Run the TPP-recall eval against ``fixtures/recall/*.json``."""

from __future__ import annotations

import argparse
import sys

from tests.eval.runners import library_extractor_ready, load_fixtures
from tests.eval.scoring.recall import recall_summary
from tests.eval.types import CategoryResult, RecallFixture

# Recall depends on episodes being ingested + an episode-level query API.
# That ships with L2 (episode store), but in practice we also need the
# extractor (L1) for entities that the queries filter on. Gate on L1 to
# keep things simple; L2 is implicitly required.
PHASE_TARGET = "L1+L2"


def run(validate_only: bool = False) -> CategoryResult:
    fixtures = load_fixtures("recall", RecallFixture)

    if validate_only:
        return CategoryResult(
            status="pass",
            phase_target=PHASE_TARGET,
            fixtures_run=len(fixtures),
            notes=f"Validated {len(fixtures)} recall fixture(s).",
        )

    if not library_extractor_ready():
        return CategoryResult(
            status="not_yet_implemented",
            phase_target=PHASE_TARGET,
            fixtures_run=0,
            notes=(
                f"Extractor + episode store not yet implemented (target: {PHASE_TARGET}). "
                f"{len(fixtures)} fixture(s) ready and validated."
            ),
        )

    all_query_results: list[tuple[list[str], list[str]]] = []
    for fixture in fixtures:
        # TODO(L2): ingest the corpus, then run each TPP query and collect
        # ranked episode IDs.
        for query in fixture.queries:
            predicted_ids: list[str] = []  # placeholder
            all_query_results.append((predicted_ids, query.expected_episode_ids))

    metrics = recall_summary(all_query_results)
    return CategoryResult(
        status="pass",
        phase_target=PHASE_TARGET,
        fixtures_run=len(fixtures),
        metrics=metrics,
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
