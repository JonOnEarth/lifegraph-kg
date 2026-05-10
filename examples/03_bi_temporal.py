# SPDX-License-Identifier: Apache-2.0
"""Example 3 — bi-temporal supersede + time-travel queries.

The Sara-moves-to-Tokyo case. Demonstrates:
  - facts that change over time (Sara lives in X, X changes)
  - supersede via lg.kg.invalidate_edge() — original edge survives
    in the DB with `t_invalid` set; not deleted
  - time-travel: lg.kg.edges_as_of(t) returns what was true at time t

This is the differentiating feature over flat-CRUD memory frameworks
(Mem0 default mode). Same model as Graphiti / Zep.
"""

from __future__ import annotations

from datetime import UTC, datetime

from lifegraph_kg import LifeGraph
from lifegraph_kg.kg.edge import Edge


def main() -> None:
    lg = LifeGraph()

    # June 2025: Sara lives in Berlin.
    june_2025 = datetime(2025, 6, 1, 9, 0, tzinfo=UTC)
    ep1 = lg.log("Sara lives in Berlin", occurred_at=june_2025)

    sara_id = lg._store.find_entity_id("Person", "sara")  # type: ignore[attr-defined]
    berlin_id = lg._store.find_entity_id("Place", "berlin")  # type: ignore[attr-defined]
    assert sara_id and berlin_id

    # The L1 extractor produces edges per (predicate, entity) pair,
    # but they're tagged with the episode's occurred_at as t_valid.
    # For this example we want a clean "lives_in" edge with predictable
    # IDs, so create one directly via the store. (Real apps mostly let
    # the extractor do this, but you CAN write edges manually for
    # imports / migrations.)
    edge_berlin = Edge(
        id="edge-berlin",
        from_entity=sara_id,
        to_entity=berlin_id,
        verb="lives_in",
        episode_id=ep1.id,
        t_event=june_2025,
        t_ingestion=june_2025,
        t_valid=june_2025,
    )
    lg._store.add_edges([edge_berlin])  # type: ignore[attr-defined]

    # January 2026: Sara moves to Tokyo. We invalidate the Berlin edge.
    move_t = datetime(2026, 1, 15, 18, 0, tzinfo=UTC)
    lg.kg.invalidate_edge("edge-berlin", move_t)

    # And log the new fact:
    ep2 = lg.log("Sara moved to Tokyo", occurred_at=move_t)
    tokyo_id = lg._store.find_entity_id("Place", "tokyo")  # type: ignore[attr-defined]
    if tokyo_id:
        lg._store.add_edges(  # type: ignore[attr-defined]
            [
                Edge(
                    id="edge-tokyo",
                    from_entity=sara_id,
                    to_entity=tokyo_id,
                    verb="lives_in",
                    episode_id=ep2.id,
                    t_event=move_t,
                    t_ingestion=move_t,
                    t_valid=move_t,
                )
            ]
        )

    # Time-travel queries
    def show_lives_in(at: datetime) -> None:
        edges = lg.kg.edges_as_of(at, verb="lives_in")
        if not edges:
            print(f"  {at.date()}: (no fact recorded)")
            return
        for e in edges:
            label = "Berlin" if e.to_entity == berlin_id else "Tokyo"
            print(f"  {at.date()}: Sara lives in {label}")

    print("=== Time-travel queries ===")
    show_lives_in(datetime(2025, 12, 1, tzinfo=UTC))  # mid-Berlin period
    show_lives_in(datetime(2026, 3, 1, tzinfo=UTC))  # post-move
    show_lives_in(datetime(2025, 1, 1, tzinfo=UTC))  # before any fact known

    # Audit trail — the Berlin edge survives in the DB (invalidated, not deleted).
    print("\n=== Audit trail for Sara's lives_in history ===")
    for e in lg.kg.edges_for_episode(ep1.id):
        if e.verb == "lives_in":
            label = "Berlin" if e.to_entity == berlin_id else "Tokyo"
            print(
                f"  edge {e.id}: {label}  t_valid={e.t_valid.date()}  "
                f"t_invalid={e.t_invalid.date() if e.t_invalid else 'NULL'}"
            )


if __name__ == "__main__":
    main()
