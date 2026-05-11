# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the extraction pipeline.


Tests use a fake LLM client so they run in CI without API keys. Live
API calls happen only via the eval runners under `tests/eval/runners/`,
which are gated on `library_extractor_ready()` AND `ANTHROPIC_API_KEY`.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from lifegraph_kg import LifeGraph, extract
from lifegraph_kg.classes import Person, Place, Project, Topic
from lifegraph_kg.extract.grounding import (
    filter_substring_violations,
    normalize_for_substring,
    violates_substring,
)

TEST_USER = "test-user"


class FakeClient:
    """LlmClient stub. Routes by which prompt it sees:
    - extraction prompts (long, with few-shot) → returns canned extraction
    - critic prompts (short, structured)        → returns valid verdict.

    Tests can override `extraction_response` per-test to inject specific
    fake LLM outputs.
    """

    def __init__(
        self,
        *,
        extraction_response: str | Callable[[str], str] = "{}",
        critic_response: str = '{"valid": true, "issues": []}',
    ) -> None:
        self.extraction_response = extraction_response
        self.critic_response = critic_response
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append({"prompt": prompt, "model": model})
        # Critic prompts have a known marker; everything else is extraction.
        if "validating a personal-knowledge-graph" in prompt:
            return self.critic_response
        if callable(self.extraction_response):
            return self.extraction_response(prompt)
        return self.extraction_response


# ----- substring assertion (grounding) -----


def test_normalize_collapses_whitespace_and_case() -> None:
    assert normalize_for_substring("  Hello\tWorld\n") == "hello world"


def test_violates_substring_chinese_translation_caught() -> None:
    # Source is Chinese; "review paper" was the v2 failure mode.
    src = "完成了一篇 paper 的 review"
    assert violates_substring("review paper", src) is True


def test_violates_substring_verbatim_passes() -> None:
    src = "Had ramen with Sara at Ippudo"
    assert violates_substring("Sara", src) is False
    assert violates_substring("ramen", src) is False
    # Case-insensitive
    assert violates_substring("SARA", src) is False


def test_filter_drops_violators() -> None:
    src = "Had ramen with Sara at Ippudo"
    entities = [
        Person(user_id=TEST_USER, value="Sara", key="sara"),
        Topic(user_id=TEST_USER, value="ramen", key="ramen", kind="food"),
        Topic(user_id=TEST_USER, value="not-in-source", key="not-in-source", kind="general"),
    ]
    kept, violations = filter_substring_violations(entities, src)
    assert len(kept) == 2
    assert violations == ["not-in-source"]


# ----- extract() pipeline -----


_VALID_EXTRACTION = json.dumps(
    {
        "predicates": ["ate"],
        "body_state": None,
        "sentiment": None,
        "energy": None,
        "entities": [
            {"type": "Person", "value": "Sara", "key": "sara"},
            {"type": "Place", "value": "Ippudo", "key": "ippudo"},
            {"type": "Topic", "kind": "food", "value": "ramen", "key": "ramen"},
        ],
    }
)


def test_extract_clean_path() -> None:
    """Happy path: LLM returns valid JSON, all entities are in source."""
    fake = FakeClient(extraction_response=_VALID_EXTRACTION)
    r = extract("Had ramen with Sara at Ippudo", llm=fake)
    assert r.predicates == ["ate"]
    assert len(r.entities) == 3
    assert isinstance(r.entities[0], Person)
    assert isinstance(r.entities[2], Topic)
    assert r.entities[2].kind == "food"
    assert r.substring_violations == []


def test_extract_rejects_translated_value() -> None:
    """Substring assertion drops translated values (the original sin)."""
    extraction = json.dumps(
        {
            "predicates": ["reviewed"],
            "entities": [
                # value is a translation, not a substring of source.
                {
                    "type": "Topic",
                    "kind": "general",
                    "value": "review paper",
                    "key": "review-paper",
                },
            ],
        }
    )
    fake = FakeClient(extraction_response=extraction)
    r = extract("完成了一篇 paper 的 review", llm=fake)
    assert r.entities == []
    assert r.substring_violations == ["review paper"]


def test_extract_handles_preamble_in_llm_output() -> None:
    """Sonnet sometimes adds a preamble despite being told not to —
    parser strips to the first {...} block."""
    fake = FakeClient(extraction_response="Sure, here's the JSON:\n" + _VALID_EXTRACTION)
    r = extract("Had ramen with Sara at Ippudo", llm=fake)
    assert len(r.entities) == 3


def test_extract_handles_invalid_json_gracefully() -> None:
    """Garbled LLM output → empty result, not a crash."""
    fake = FakeClient(extraction_response="this is not JSON at all")
    r = extract("anything", llm=fake)
    assert r.predicates == []
    assert r.entities == []
    assert r.substring_violations == []


def test_extract_skips_invalid_entities() -> None:
    """An entity with a bad `type` doesn't fail the whole extraction —
    just that entity is dropped."""
    extraction = json.dumps(
        {
            "predicates": [],
            "entities": [
                {"type": "Person", "value": "Sara", "key": "sara"},
                {"type": "NotARealType", "value": "x", "key": "x"},
                {"type": "Place", "value": "Ippudo", "key": "ippudo"},
            ],
        }
    )
    fake = FakeClient(extraction_response=extraction)
    r = extract("Sara Ippudo", llm=fake)
    assert len(r.entities) == 2
    assert isinstance(r.entities[0], Person)
    assert isinstance(r.entities[1], Place)


def test_extract_topic_defaults_kind_to_general() -> None:
    """Topic without explicit `kind` defaults to general — required by
    schema, but the LLM might omit it on `general` cases."""
    extraction = json.dumps(
        {
            "predicates": [],
            "entities": [{"type": "Topic", "value": "thing", "key": "thing"}],
        }
    )
    fake = FakeClient(extraction_response=extraction)
    r = extract("thing", llm=fake)
    assert len(r.entities) == 1
    assert isinstance(r.entities[0], Topic)
    assert r.entities[0].kind == "general"


def test_extract_passes_critic_issues_through() -> None:
    """Critic issues end up on the result for the caller to inspect."""
    fake = FakeClient(
        extraction_response=_VALID_EXTRACTION,
        critic_response=json.dumps(
            {"valid": False, "issues": [{"kind": "missing_action", "detail": "..."}]}
        ),
    )
    r = extract("Had ramen with Sara at Ippudo", llm=fake)
    assert r.critic_issues == [{"kind": "missing_action", "detail": "..."}]


def test_extract_calls_extractor_then_critic() -> None:
    """Pipeline order: extractor model, then critic model."""
    fake = FakeClient(extraction_response=_VALID_EXTRACTION)
    extract("Had ramen with Sara at Ippudo", llm=fake)
    assert len(fake.calls) == 2
    # First call uses extractor model, second uses critic model.
    from lifegraph_kg.extract import CRITIC_MODEL, EXTRACTOR_MODEL

    assert fake.calls[0]["model"] == EXTRACTOR_MODEL
    assert fake.calls[1]["model"] == CRITIC_MODEL


# ----- LifeGraph facade -----


def test_lifegraph_log_returns_episode() -> None:
    """LifeGraph.log(user_id=TEST_USER) returns the persisted Episode in L2."""
    from lifegraph_kg.kg.episode import Episode

    fake = FakeClient(extraction_response=_VALID_EXTRACTION)
    lg = LifeGraph(llm=fake)
    ep = lg.log("Had ramen with Sara at Ippudo", user_id=TEST_USER)
    assert isinstance(ep, Episode)
    assert ep.predicates == ["ate"]
    # Entities are persisted, queryable via lg.query(user_id=TEST_USER)
    assert len(lg.query(Person, user_id=TEST_USER).all()) == 1
    assert len(lg.query(Place, user_id=TEST_USER).all()) == 1
    assert len(lg.query(Topic, kind="food", user_id=TEST_USER).all()) == 1


def test_lifegraph_log_handles_chinese() -> None:
    """End-to-end with Chinese source + the canonical fixture entities."""
    extraction = json.dumps(
        {
            "predicates": ["fixed"],
            "body_state": None,
            "sentiment": None,
            "energy": None,
            "entities": [
                {"type": "Project", "value": "TimeWises", "key": "timewises"},
                {"type": "Topic", "kind": "general", "value": "UI bug", "key": "ui-bug"},
            ],
        }
    )
    fake = FakeClient(extraction_response=extraction)
    lg = LifeGraph(llm=fake)
    ep = lg.log("晚上修复了 TimeWises 的几个 UI bug", user_id=TEST_USER)
    assert ep.predicates == ["fixed"]
    timewises = lg.query(Project, key="timewises", user_id=TEST_USER).one()
    assert timewises.value == "TimeWises"
    bugs = lg.query(Topic, key="ui-bug", user_id=TEST_USER).one()
    assert bugs.value == "UI bug"


# ----- ontology classes -----


def test_topic_kind_required_validation() -> None:
    """Topic.kind has a default but only valid values are accepted."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Topic(user_id=TEST_USER, value="x", key="x", kind="not-a-valid-kind")  # type: ignore[arg-type]


def test_entities_are_frozen() -> None:
    """All entity classes are frozen — accidentally mutating an entity
    raises ValidationError at runtime."""
    from pydantic import ValidationError

    sara = Person(user_id=TEST_USER, value="Sara", key="sara")
    with pytest.raises(ValidationError):
        sara.value = "Other"
