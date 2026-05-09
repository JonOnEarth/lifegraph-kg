# SPDX-License-Identifier: Apache-2.0
"""LLM-as-judge for open-ended categories.

Used only where rule-based metrics can't score:
- Is this canonicalized name reasonable for the original?
- Is this extraction's surface form acceptable (handles paraphrases)?

Rule-based scoring is preferred everywhere it applies. This module is
opt-in: requires ``LIFEGRAPH_EVAL_LLM_JUDGE=1`` and ``ANTHROPIC_API_KEY``.

Pinned configuration (model + temperature) is part of the eval report so
results are reproducible across runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

# Pinned for reproducibility. Bump these together when the judge changes,
# and bump the eval report's `llm_judge` section in lockstep.
JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_TEMPERATURE = 0.0
JUDGE_VERSION = "1"  # bump on prompt changes


@dataclass(frozen=True)
class JudgmentResult:
    """A single LLM-judge verdict.

    score: 0.0-1.0 quality score the judge assigned.
    rationale: the judge's free-text reasoning (kept for audit).
    """

    score: float
    rationale: str


def is_enabled() -> bool:
    """LLM judge is opt-in via env var."""
    return os.environ.get("LIFEGRAPH_EVAL_LLM_JUDGE") == "1"


def judge_config() -> dict[str, str]:
    """Pinned config to embed in the eval report."""
    return {
        "model": JUDGE_MODEL,
        "temperature": str(JUDGE_TEMPERATURE),
        "version": JUDGE_VERSION,
    }


_CANONICAL_PROMPT = """\
You are scoring an entity-canonicalization decision in a personal-memory
knowledge graph.

Original surface form: {original}
Proposed canonical form: {canonical}

Is the canonical form a reasonable, lossless normalization of the original?
A good normalization preserves identity (same person/place/etc.) and
removes only superficial variation (capitalization, spacing, common
nicknames already in the user's graph).

Respond ONLY in this JSON format:
{{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}}
"""


_PARAPHRASE_PROMPT = """\
You are scoring whether an extracted entity captures the right meaning
from a sentence — even if the surface form isn't a verbatim substring.

Sentence: {text}
Extracted entity (type={type_}, value={value}):

Is this a reasonable extraction? Allow paraphrases (e.g. "Mum" → Person
"mother") only when the meaning is unambiguous from the sentence.

Respond ONLY in this JSON format:
{{"score": <float 0.0-1.0>, "rationale": "<one sentence>"}}
"""


def _call_judge(prompt: str) -> JudgmentResult:
    """Invoke Anthropic; never called if is_enabled() is False."""
    # Lazy import: anthropic is a runtime dep already, but only spin up
    # the client if the judge actually runs.
    from anthropic import Anthropic
    from anthropic.types import TextBlock

    client = Anthropic()
    msg = client.messages.create(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [b for b in msg.content if isinstance(b, TextBlock)]
    text = text_blocks[0].text if text_blocks else "{}"
    parsed: dict[str, Any] = json.loads(_extract_json(text))
    return JudgmentResult(
        score=float(parsed.get("score", 0.0)),
        rationale=str(parsed.get("rationale", "")),
    )


def _extract_json(text: str) -> str:
    """Pull out the first {...} block — defensive against preamble."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return "{}"
    return text[start : end + 1]


def judge_canonicalization(original: str, canonical: str) -> JudgmentResult:
    """Score whether ``canonical`` is a reasonable normalization of ``original``."""
    if not is_enabled():
        return JudgmentResult(score=0.0, rationale="llm_judge disabled")
    return _call_judge(_CANONICAL_PROMPT.format(original=original, canonical=canonical))


def judge_paraphrase_extraction(text: str, type_: str, value: str) -> JudgmentResult:
    """Score whether an entity captures the right meaning, allowing paraphrase."""
    if not is_enabled():
        return JudgmentResult(score=0.0, rationale="llm_judge disabled")
    return _call_judge(_PARAPHRASE_PROMPT.format(text=text, type_=type_, value=value))
