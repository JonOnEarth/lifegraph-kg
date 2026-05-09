# SPDX-License-Identifier: Apache-2.0
"""Orchestrate all eval categories and emit a JSON report.

Usage::

    python -m tests.eval.runners.run_all
    python -m tests.eval.runners.run_all --output reports/$(date +%F).json
    python -m tests.eval.runners.run_all --validate-only

CI-gate semantics: run_all returns exit code 0 when every category is
either ``pass`` or ``not_yet_implemented`` (the latter is expected for
phases that haven't shipped yet). Any ``fail`` returns 1, which CI uses
to block a PR that regresses metrics.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from lifegraph_kg import __version__
from tests.eval.runners import (
    run_extraction,
    run_hygiene,
    run_perf,
    run_recall,
    run_temporal,
)
from tests.eval.scoring.llm_judge import is_enabled as llm_judge_enabled
from tests.eval.scoring.llm_judge import judge_config
from tests.eval.types import EvalReport

CATEGORY_RUNNERS = [
    ("extraction", run_extraction.run),
    ("temporal", run_temporal.run),
    ("hygiene", run_hygiene.run),
    ("recall", run_recall.run),
    ("perf", run_perf.run),
]


def assemble(validate_only: bool = False) -> EvalReport:
    categories = {name: runner(validate_only=validate_only) for name, runner in CATEGORY_RUNNERS}
    ci_gate_pass = all(c.status in ("pass", "not_yet_implemented") for c in categories.values())
    return EvalReport(
        lifegraph_kg_version=__version__,
        timestamp=datetime.now(UTC),
        categories=categories,
        ci_gate_pass=ci_gate_pass,
        llm_judge=judge_config() if llm_judge_enabled() else None,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the report JSON to this path (also prints to stdout).",
    )
    args = parser.parse_args()

    report = assemble(validate_only=args.validate_only)
    body = report.model_dump_json(indent=2)
    print(body)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body)
        print(f"\nWrote report → {args.output}", file=sys.stderr)

    return 0 if report.ci_gate_pass else 1


if __name__ == "__main__":
    sys.exit(_main())
