# SPDX-License-Identifier: Apache-2.0
"""Tests for the L3 hygiene engine.


Three groups:
  - normalize: canonical_form on EN, CN, mixed; punctuation; whitespace
  - dedup:     propose_merges across compatibility / heuristics
  - apply:     end-to-end through LifeGraph — propose, apply, verify
                edges + mentions are redirected, audit-trail preserved
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifegraph_kg import LifeGraph, Person, Place, Topic
from lifegraph_kg.hygiene import canonical_form, propose_merges
from lifegraph_kg.hygiene.dedup import _damerau_levenshtein
from lifegraph_kg.hygiene.proposals import MergeProposal
from tests.test_extraction import FakeClient

TEST_USER = "test-user"

T1 = datetime(2026, 5, 1, 19, 0, tzinfo=UTC)
T2 = datetime(2026, 5, 5, 17, 0, tzinfo=UTC)


# ----- normalize -----


def test_canonical_form_basic() -> None:
    assert canonical_form("Sara") == "sara"
    assert canonical_form("  Sara  ") == "sara"
    assert canonical_form("Sara.") == "sara"
    assert canonical_form("SARA") == "sara"


def test_canonical_form_chinese_punctuation_stripped() -> None:
    assert canonical_form("吃了。") == "吃了"
    assert canonical_form("修复！") == "修复"


def test_canonical_form_nfkc_fullwidth() -> None:
    """NFKC normalizes fullwidth → halfwidth digits/letters."""
    assert canonical_form("ＡＢＣ") == "abc"


def test_canonical_form_empty() -> None:
    assert canonical_form("") == ""
    assert canonical_form("   ") == ""


# ----- damerau-levenshtein -----


def test_damerau_levenshtein_zero() -> None:
    assert _damerau_levenshtein("ippudo", "ippudo") == 0


def test_damerau_levenshtein_substitution() -> None:
    assert _damerau_levenshtein("ippudo", "ippud0") == 1


def test_damerau_levenshtein_transposition() -> None:
    """Damerau counts adjacent transposition as 1, not 2."""
    assert _damerau_levenshtein("abcd", "abdc") == 1


def test_damerau_levenshtein_empty() -> None:
    assert _damerau_levenshtein("", "abc") == 3
    assert _damerau_levenshtein("abc", "") == 3
    assert _damerau_levenshtein("", "") == 0


# ----- dedup proposals -----


def test_propose_exact_normalized_match() -> None:
    """`Sara` and `sara` collapse to the same canonical form."""
    proposals = propose_merges(
        [Person(user_id=TEST_USER, value="Sara", key="sara1"), Person(user_id=TEST_USER, value="sara", key="sara2")]
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.confidence == "high"
    assert p.reason == "exact_normalized"
    assert p.is_safe_to_auto_apply is True


def test_propose_substring_qualifier() -> None:
    """`Ippudo NYC` is a qualified form of `Ippudo` — propose merge."""
    proposals = propose_merges(
        [Place(user_id=TEST_USER, value="Ippudo", key="ippudo"), Place(user_id=TEST_USER, value="Ippudo NYC", key="ippudo-nyc")]
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.reason == "substring_qualifier"
    assert p.winner.value == "Ippudo NYC"  # longer wins
    assert p.loser.value == "Ippudo"
    assert p.confidence == "medium"
    # Not auto-apply-safe — substring matching can produce false positives.
    assert p.is_safe_to_auto_apply is False


def test_propose_edit_distance_short_strings_skipped() -> None:
    """Edit distance is suppressed for short strings — `Tao` / `Tom`
    must NOT be proposed as a merge despite Levenshtein = 1."""
    proposals = propose_merges([Person(user_id=TEST_USER, value="Tao", key="tao"), Person(user_id=TEST_USER, value="Tom", key="tom")])
    assert proposals == []


def test_propose_edit_distance_long_strings() -> None:
    """For non-trivial-length strings, distance ≤ 1 produces a proposal."""
    proposals = propose_merges(
        [Place(user_id=TEST_USER, value="Ippudo", key="ippudo"), Place(user_id=TEST_USER, value="Ippud0", key="ippud0")]
    )
    assert len(proposals) == 1
    assert proposals[0].reason == "edit_distance"


def test_propose_does_not_cross_types() -> None:
    """A Person never merges with a Place even if values match."""
    proposals = propose_merges(
        [Person(user_id=TEST_USER, value="Mercury", key="mercury"), Place(user_id=TEST_USER, value="Mercury", key="mercury")]
    )
    assert proposals == []


def test_propose_does_not_cross_topic_kinds() -> None:
    """A Topic{food} never merges with a Topic{media} even with same value."""
    proposals = propose_merges(
        [
            Topic(user_id=TEST_USER, value="apple", key="apple", kind="food"),
            Topic(user_id=TEST_USER, value="apple", key="apple", kind="media"),  # the company
        ]
    )
    assert proposals == []


def test_propose_does_not_cross_distinct_people_with_similar_names() -> None:
    """Adversarial case: `Alex Smith` and `Alex Johnson` must NOT merge."""
    proposals = propose_merges(
        [
            Person(user_id=TEST_USER, value="Alex Smith", key="alex-smith"),
            Person(user_id=TEST_USER, value="Alex Johnson", key="alex-johnson"),
        ]
    )
    assert proposals == []


# ----- apply pipeline (end-to-end through LifeGraph) -----


_SARA_EXTRACTION_1 = """{
  "predicates": ["met"],
  "body_state": null, "sentiment": null, "energy": null,
  "entities": [{"type": "Person", "value": "Sara", "key": "sara1"}]
}"""

_SARA_EXTRACTION_2 = """{
  "predicates": ["called"],
  "body_state": null, "sentiment": null, "energy": null,
  "entities": [{"type": "Person", "value": "sara", "key": "sara2"}]
}"""


def test_lifegraph_hygiene_propose_returns_proposals() -> None:
    """End-to-end: log two episodes mentioning the same person under
    different keys, propose detects the merge."""
    fake = FakeClient(extraction_response=_SARA_EXTRACTION_1)
    lg = LifeGraph(llm=fake)
    lg.log("Met Sara today", occurred_at=T1, user_id=TEST_USER)

    fake.extraction_response = _SARA_EXTRACTION_2
    lg.log("Called sara again", occurred_at=T2, user_id=TEST_USER)

    # Two Person entities exist (different keys).
    assert len(lg.query(Person, user_id=TEST_USER).all()) == 2

    proposals = lg.hygiene.propose(type_="Person", user_id=TEST_USER)
    assert len(proposals) == 1
    assert proposals[0].confidence == "high"


def test_lifegraph_hygiene_apply_redirects_edges() -> None:
    """Apply path: after merging, edges that pointed at the loser now
    point at the winner; loser entity row stays (audit trail)."""
    fake = FakeClient(extraction_response=_SARA_EXTRACTION_1)
    lg = LifeGraph(llm=fake)
    ep1 = lg.log("Met Sara today", occurred_at=T1, user_id=TEST_USER)

    fake.extraction_response = _SARA_EXTRACTION_2
    ep2 = lg.log("Called sara again", occurred_at=T2, user_id=TEST_USER)

    proposals = lg.hygiene.propose(type_="Person", user_id=TEST_USER)
    lg.hygiene.apply(proposals[0], user_id=TEST_USER)

    # Entity rows count is unchanged (audit trail preserved).
    assert len(lg.query(Person, user_id=TEST_USER).all()) == 2
    # But the canonical view: only one Person has canonical_id IS NULL
    # (verified via the underlying store).
    from lifegraph_kg.kg.store.sqlite import SqliteStore

    assert isinstance(lg._store, SqliteStore)
    rows = lg._store._conn.execute(
        "SELECT COUNT(*) FROM entities WHERE type='Person' AND canonical_id IS NULL"
    ).fetchone()
    assert rows[0] == 1

    # Edges from BOTH episodes now point at the same canonical entity.
    edges_ep1 = lg.kg.edges_for_episode(ep1.id)
    edges_ep2 = lg.kg.edges_for_episode(ep2.id)
    assert len(edges_ep1) == 1
    assert len(edges_ep2) == 1
    assert edges_ep1[0].to_entity == edges_ep2[0].to_entity  # ← merged target


def test_lifegraph_hygiene_auto_apply_only_high_confidence() -> None:
    """auto_apply only fires for `is_safe_to_auto_apply` proposals."""
    fake = FakeClient(extraction_response=_SARA_EXTRACTION_1)
    lg = LifeGraph(llm=fake)
    lg.log("Met Sara today", occurred_at=T1, user_id=TEST_USER)

    fake.extraction_response = _SARA_EXTRACTION_2
    lg.log("Called sara again", occurred_at=T2, user_id=TEST_USER)

    # Add a substring-qualifier case (medium confidence — should NOT auto-apply)
    place_extraction = """{
      "predicates": ["went"], "body_state": null, "sentiment": null, "energy": null,
      "entities": [
        {"type": "Place", "value": "Ippudo", "key": "ippudo"}
      ]}"""
    fake.extraction_response = place_extraction
    lg.log("Went to Ippudo", occurred_at=T1, user_id=TEST_USER)
    place_extraction_2 = """{
      "predicates": ["went"], "body_state": null, "sentiment": null, "energy": null,
      "entities": [
        {"type": "Place", "value": "Ippudo NYC", "key": "ippudo-nyc"}
      ]}"""
    fake.extraction_response = place_extraction_2
    lg.log("Went to Ippudo NYC", occurred_at=T2, user_id=TEST_USER)

    applied = lg.hygiene.auto_apply(user_id=TEST_USER)
    # Only the Person merge (high confidence + exact_normalized) should apply.
    assert len(applied) == 1
    assert applied[0].reason == "exact_normalized"


def test_proposal_record_persists() -> None:
    """propose(record=True) writes proposals to the merge_proposals table."""
    fake = FakeClient(extraction_response=_SARA_EXTRACTION_1)
    lg = LifeGraph(llm=fake)
    lg.log("Met Sara today", occurred_at=T1, user_id=TEST_USER)
    fake.extraction_response = _SARA_EXTRACTION_2
    lg.log("Called sara again", occurred_at=T2, user_id=TEST_USER)

    proposals = lg.hygiene.propose(type_="Person", record=True, user_id=TEST_USER)
    assert len(proposals) == 1

    from lifegraph_kg.kg.store.sqlite import SqliteStore

    assert isinstance(lg._store, SqliteStore)
    rows = lg._store._conn.execute("SELECT * FROM merge_proposals").fetchall()
    assert len(rows) == 1
    assert rows[0]["confidence"] == "high"
    assert rows[0]["applied_at"] is None


def test_merge_proposal_str_repr() -> None:
    """MergeProposal has a readable string repr for review UIs."""
    sara = Person(user_id=TEST_USER, value="Sara", key="sara1")
    sara2 = Person(user_id=TEST_USER, value="sara", key="sara2")
    p = MergeProposal(
        winner=sara,
        loser=sara2,
        confidence="high",
        reason="exact_normalized",
        detail="both canonicalize to 'sara'",
    )
    s = str(p)
    assert "merge" in s
    assert "Sara" in s
    assert "high" in s
