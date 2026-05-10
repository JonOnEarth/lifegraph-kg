# SPDX-License-Identifier: Apache-2.0
"""Example 4 — hygiene engine: dedup with audit trail.

Real life-logs accumulate noise: the same person under different
spellings ("Sara" / "sarah"), the same place under qualifications
("Ippudo" / "Ippudo NYC"), occasional typos. The hygiene engine
proposes merges; you (or auto-apply policy) decide what to keep.

Demonstrates:
  - lg.hygiene.propose() — heuristic dedup with confidence scores
  - lg.hygiene.apply()    — redirect edges + mentions, set canonical_id
  - audit-preserving      — loser entity rows survive as aliases
  - lg.hygiene.auto_apply() — only high-confidence exact matches

Note: this example uses the FakeClient pattern from tests/test_extraction.py
to seed entities with deliberately divergent keys (Sara / sara / Sarah).
In real usage, the extractor's normalization usually handles the easy
cases, but legacy / imported data is exactly where hygiene shines.
"""

from __future__ import annotations

import json

from lifegraph_kg import LifeGraph, Person, Place
from lifegraph_kg.hygiene import propose_merges


def main() -> None:
    # Direct entity construction (skipping the LLM for example clarity).
    # Imagine these came from a legacy system import where the same person
    # was logged under three different keys.
    sara_a = Person(value="Sara", key="sara1")
    sara_b = Person(value="sara", key="sara2")
    sarah = Person(value="Sarah", key="sara3")
    alex_smith = Person(value="Alex Smith", key="alex-smith")
    alex_johnson = Person(value="Alex Johnson", key="alex-johnson")
    ippudo_a = Place(value="Ippudo", key="ippudo")
    ippudo_b = Place(value="Ippudo NYC", key="ippudo-nyc")
    ichiran = Place(value="Ichiran", key="ichiran")

    print("=== Heuristic dedup proposals (no DB needed) ===\n")
    proposals = propose_merges(
        [sara_a, sara_b, sarah, alex_smith, alex_johnson, ippudo_a, ippudo_b, ichiran]
    )
    for p in proposals:
        flag = "✅ auto-apply-safe" if p.is_safe_to_auto_apply else "❓ needs review"
        print(f"  {p}  ({flag})")
    if not proposals:
        print("  (none)")

    print("\nWhat the engine did NOT propose:")
    print("  - Alex Smith ↔ Alex Johnson  (distinct people, name overlap is incidental)")
    print("  - Ippudo ↔ Ichiran           (different ramen shops)")

    print("\n=== End-to-end through LifeGraph: propose + apply ===\n")

    # Use a tiny fake LLM client so the example doesn't need an API key.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tests.test_extraction import FakeClient

    fake = FakeClient(
        extraction_response=json.dumps(
            {
                "predicates": ["met"],
                "body_state": None,
                "sentiment": None,
                "energy": None,
                "entities": [{"type": "Person", "value": "Sara", "key": "sara1"}],
            }
        )
    )
    lg = LifeGraph(llm=fake)
    lg.log("Met Sara today")

    # Second log with a different key for the same person (simulates legacy drift).
    fake.extraction_response = json.dumps(
        {
            "predicates": ["called"],
            "body_state": None,
            "sentiment": None,
            "energy": None,
            "entities": [{"type": "Person", "value": "sara", "key": "sara2"}],
        }
    )
    lg.log("Called sara about the deadline")

    print(f"Persons before hygiene: {len(lg.query(Person).all())}")
    for p in lg.query(Person).all():
        print(f"  - {p.value!r}  key={p.key}")

    # Auto-apply only fires for high-confidence exact_normalized merges.
    applied = lg.hygiene.auto_apply()
    print(f"\nAuto-applied: {len(applied)} merge(s)")
    for p in applied:
        print(f"  {p}")

    # Loser entity row survives in the DB (aliased via canonical_id).
    print(f"\nPersons after hygiene: {len(lg.query(Person).all())}  (audit trail preserved)")
    sqlstore = lg._store  # type: ignore[attr-defined]
    canon_count = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM entities WHERE type='Person' AND canonical_id IS NULL"
    ).fetchone()[0]
    print(f"Canonical Persons (canonical_id IS NULL): {canon_count}")


if __name__ == "__main__":
    main()
