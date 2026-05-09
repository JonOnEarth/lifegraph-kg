# SPDX-License-Identifier: Apache-2.0
"""Skeleton tests — confirm the package imports and submodules exist.

Real tests land in L1+ (default-schema smoke, bi-temporal supersede,
hygiene dedup, etc.). These tests just verify the L0 layout.
"""

from importlib import import_module

import lifegraph_kg


def test_version_string() -> None:
    assert isinstance(lifegraph_kg.__version__, str)
    assert lifegraph_kg.__version__


def test_submodules_importable() -> None:
    for name in ("kg", "kg.store", "extract", "hygiene", "llm"):
        import_module(f"lifegraph_kg.{name}")
