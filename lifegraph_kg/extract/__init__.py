# SPDX-License-Identifier: Apache-2.0
"""Extraction — natural language to typed PKG entities.

Public API: `extract(text, *, llm=None) -> ExtractionResult`.

The pipeline (3 techniques layered, in order):
  1. Extractor LLM call — v6 prompt with 6 inline few-shot examples
  2. Substring assertion — drop entities whose `value` isn't in source
  3. Critic LLM call    — Haiku validates, flags translation /
                         hallucination / missing-action issues

Each step records its output on the returned ExtractionResult so the
caller can inspect what was rejected and why.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from lifegraph_kg.extract.grounding import filter_substring_violations
from lifegraph_kg.extract.prompt import CRITIC_PROMPT, EXTRACTION_PROMPT
from lifegraph_kg.extract.schema import CriticVerdict, EntityT, ExtractionResult
from lifegraph_kg.llm.client import LlmClient, default_client

# Pinned models — bumping these is a minor version change since results
# are model-dependent. Critic uses Haiku for cost; extractor uses Sonnet
# for quality.
EXTRACTOR_MODEL = "claude-sonnet-4-6"
CRITIC_MODEL = "claude-haiku-4-5-20251001"


def _parse_json_dict(body: str) -> dict[str, Any]:
    """Parse the first {...} block from `body`. Tolerates preamble
    (Sonnet sometimes adds a sentence even when told not to) and
    stripped fences."""
    start = body.find("{")
    end = body.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        parsed = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_entities(raw: list[Any]) -> list[EntityT]:
    """Pydantic-validate each raw entity dict against the discriminated
    union. Skip entries that fail validation rather than failing the
    whole extraction — the substring + critic passes will catch any
    quality issues with what's left."""
    out: list[EntityT] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Topic with kind=null is invalid; default to "general".
        if item.get("type") == "Topic" and item.get("kind") is None:
            item = {**item, "kind": "general"}
        try:
            # Parse a single-element ExtractionResult to get Pydantic to
            # dispatch the discriminated union by `type`.
            wrapper = ExtractionResult.model_validate({"entities": [item]})
            out.extend(wrapper.entities)
        except ValidationError:
            continue
    return out


def extract(text: str, *, llm: LlmClient | None = None) -> ExtractionResult:
    """Extract a personal-KG ExtractionResult from `text`.

    Returns an ExtractionResult with episode metadata + 4-class entities,
    plus diagnostic fields (`substring_violations`, `critic_issues`)
    showing what was rejected and why.

    Pass `llm=` to inject a mock for testing.
    """
    client = llm or default_client()

    # 1. Extractor pass
    body = client.chat(
        EXTRACTION_PROMPT.format(text=text),
        model=EXTRACTOR_MODEL,
        max_tokens=2048,
        temperature=0.0,
    )
    parsed = _parse_json_dict(body)
    entities = _coerce_entities(parsed.get("entities", []) or [])

    # 2. Substring assertion — drop entities whose value isn't in source.
    entities, violations = filter_substring_violations(entities, text)

    # 3. Critic pass — flag any remaining issues for the caller.
    critic_input = {
        "predicates": parsed.get("predicates", []),
        "body_state": parsed.get("body_state"),
        "sentiment": parsed.get("sentiment"),
        "energy": parsed.get("energy"),
        "entities": [e.model_dump() for e in entities],
    }
    critic_body = client.chat(
        CRITIC_PROMPT.format(text=text, extraction=json.dumps(critic_input, ensure_ascii=False)),
        model=CRITIC_MODEL,
        max_tokens=512,
        temperature=0.0,
    )
    critic_parsed = _parse_json_dict(critic_body)
    try:
        verdict = CriticVerdict.model_validate(critic_parsed)
    except ValidationError:
        verdict = CriticVerdict()

    # 4. Assemble the final result.
    result = ExtractionResult(
        predicates=list(parsed.get("predicates", []) or []),
        body_state=parsed.get("body_state"),
        sentiment=parsed.get("sentiment"),
        energy=parsed.get("energy"),
        entities=entities,
        substring_violations=violations,
        critic_issues=verdict.issues,
    )
    return result


__all__ = ["CRITIC_MODEL", "EXTRACTOR_MODEL", "extract"]
