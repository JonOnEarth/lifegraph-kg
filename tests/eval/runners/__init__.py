# SPDX-License-Identifier: Apache-2.0
"""Runners — orchestrate fixture load → library call → scoring → report.

Each category has its own runner; ``run_all.py`` aggregates them. Runners
must error gracefully when their target phase hasn't shipped — they
emit ``CategoryResult(status="not_yet_implemented", phase_target=...)``
instead of crashing.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES_ROOT = REPO_ROOT / "tests" / "eval" / "fixtures"

T = TypeVar("T", bound=BaseModel)


def load_fixtures(category: str, model: type[T]) -> list[T]:
    """Load and validate every JSON fixture in ``fixtures/<category>/``."""
    fixture_dir = FIXTURES_ROOT / category
    if not fixture_dir.exists():
        return []
    out: list[T] = []
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as f:
            out.append(model.model_validate(json.load(f)))
    return out


def _module_has_attr(module_path: str, attr: str) -> bool:
    """Probe whether a (future) library module exists and exports an attr.

    Used to gate runners on whether their target phase has shipped.
    Going through importlib + getattr keeps this static-analysis-clean
    while still being a runtime feature check (so mypy doesn't try to
    resolve L1+ modules that don't exist yet).
    """
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return False
    return hasattr(mod, attr)


def library_extractor_ready() -> bool:
    """L1 flips this to True when the extractor lands."""
    return _module_has_attr("lifegraph_kg.kg", "LifeGraph")


def library_temporal_ready() -> bool:
    """L2 flips this to True when bi-temporal CRUD lands."""
    return _module_has_attr("lifegraph_kg.kg.temporal", "invalidate")


def library_hygiene_ready() -> bool:
    """L3 flips this to True when the hygiene engine lands."""
    return _module_has_attr("lifegraph_kg.hygiene.dedup", "propose_merges")


def library_storage_drivers_ready() -> bool:
    """L4 flips this to True when Postgres + Kuzu drivers land."""
    return _module_has_attr("lifegraph_kg.kg.store.postgres", "PostgresStore") and _module_has_attr(
        "lifegraph_kg.kg.store.kuzu", "KuzuStore"
    )
