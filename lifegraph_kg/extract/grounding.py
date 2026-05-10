# SPDX-License-Identifier: Apache-2.0
"""Substring assertion — the no-translate enforcement layer.

Every entity `value` MUST appear as a verbatim substring of the source
episode text after Unicode-NFKC normalization + casefolding + whitespace
collapse. This is the cheap deterministic defense against the
translation-drift failure mode the existing system has (e.g. Chinese
`完成了一篇 paper 的 review` extracted as English `"review paper"`).

Note: this only applies to entity *values*. Predicates are intentionally
normalized to lowercase English verbs for cross-language queryability —
`修复` and `fixed` and `修复了` all collapse to predicate `"fixed"`.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import TypeVar

from lifegraph_kg.classes import Entity

E = TypeVar("E", bound=Entity)


def normalize_for_substring(s: str) -> str:
    """Unicode-NFKC + casefold + whitespace collapse."""
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.casefold().split())


def violates_substring(entity_value: str, source_text: str) -> bool:
    """True if `entity_value` is NOT a substring of `source_text` after
    normalization. Empty values pass (treated as no-op)."""
    v = normalize_for_substring(entity_value)
    if not v:
        return False
    src = normalize_for_substring(source_text)
    return v not in src


def filter_substring_violations(
    entities: Iterable[E], source_text: str
) -> tuple[list[E], list[str]]:
    """Drop entities whose `value` isn't in `source_text`. Return
    (kept_entities, violating_values). The pipeline auto-rejects
    rather than re-prompting — simpler and lower-latency than a
    correction loop.

    Generic over Entity subtypes so the discriminated-union
    (Person|Place|Project|Topic) type is preserved through the call.
    """
    kept: list[E] = []
    violations: list[str] = []
    for e in entities:
        if violates_substring(e.value, source_text):
            violations.append(e.value)
        else:
            kept.append(e)
    return kept, violations
