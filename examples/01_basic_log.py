# SPDX-License-Identifier: Apache-2.0
"""Example 1 — basic log + query.

The minimum viable usage. Set ``ANTHROPIC_API_KEY`` and run:

    uv run python examples/01_basic_log.py
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifegraph_kg import LifeGraph, Person, Place, Topic


def main() -> None:
    # In-memory store. Pass store="sqlite:///me.db" to persist across runs.
    lg = LifeGraph()

    # Log a single entry. The extractor produces:
    #   predicates: ["ate"]
    #   entities:   Person:Sara, Place:Ippudo, Topic:ramen{kind=food}
    #   sentiment:  None  (no explicit affect cue)
    #   body_state: None
    ep = lg.log(
        "Had ramen with Sara at Ippudo",
        occurred_at=datetime(2026, 5, 9, 19, 0, tzinfo=UTC),
    )

    print(f"Episode {ep.id}: {ep.text}")
    print(f"  predicates: {ep.predicates}")
    print(f"  occurred at: {ep.occurred_at}")

    # Query each entity class.
    print(f"\nPersons: {[p.value for p in lg.query(Person).all()]}")
    print(f"Places:  {[p.value for p in lg.query(Place).all()]}")
    print(f"Foods:   {[t.value for t in lg.query(Topic, kind='food').all()]}")

    # Walk back from a person to their episodes.
    sara = lg.query(Person, key="sara").one()
    print(f"\nEpisodes mentioning {sara.value}:")
    for e in lg.episodes.mentioning(sara):
        print(f"  {e.occurred_at.date()}: {e.text}")


if __name__ == "__main__":
    main()
