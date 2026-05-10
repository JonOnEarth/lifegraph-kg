# SPDX-License-Identifier: Apache-2.0
"""LifeGraph facade — the user-facing entry point.

Usage:
    from lifegraph_kg import LifeGraph
    lg = LifeGraph()
    result = lg.log("Had ramen with Sara at Ippudo. Felt tired.")
    # result.predicates  → ["ate"]
    # result.body_state  → "tired"
    # result.entities    → [Person(value="Sara"), Place(value="Ippudo"), Topic(kind="food", value="ramen")]

L1 surface: extraction only (no persistence). L2 will add the SQLite
episode store and bi-temporal CRUD; the `log()` method becomes a real
write at that point. For L1, `log()` returns the extraction without
storing it — same shape, future-compatible.
"""

from __future__ import annotations

from lifegraph_kg.extract import extract
from lifegraph_kg.extract.schema import ExtractionResult
from lifegraph_kg.llm.client import LlmClient


class LifeGraph:
    """The user-facing facade for the personal knowledge graph.

    L1: thin wrapper around `extract()`. L2 adds the bi-temporal
    episode store; `log()` becomes the persistence entry point.

    Pass `llm=` to inject a custom LLM client (or a mock for tests).
    """

    def __init__(self, *, llm: LlmClient | None = None) -> None:
        self._llm = llm

    def log(self, text: str) -> ExtractionResult:
        """Ingest a natural-language entry. Returns the extraction.

        In L1 this is equivalent to `lifegraph_kg.extract(text)`. In L2,
        this method also persists the episode + entities + edges to the
        bi-temporal store.
        """
        return extract(text, llm=self._llm)


__all__ = ["LifeGraph"]
