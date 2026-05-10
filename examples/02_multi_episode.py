# SPDX-License-Identifier: Apache-2.0
"""Example 2 — multi-episode timeline + the food/people pivot.

Show what the graph looks like after a week of logs. Demonstrates:
  - persistent SQLite store (file-backed, survives restart)
  - multi-action predicates (one episode → multiple verbs)
  - lg.episodes.{since, between} for time-range queries
  - lg.query(Topic, kind="food").episodes() — the personal eating timeline
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lifegraph_kg import LifeGraph, Person, Topic


def main() -> None:
    # File-backed: re-run the script and the data is still there.
    # Delete examples/example.db to start fresh.
    lg = LifeGraph(store="sqlite:///examples/example.db")

    # Five days of life-log entries.
    entries = [
        ("Mon 9am", "Standup with Priya. Reviewed the migration PR."),
        ("Mon 7pm", "Had ramen with Sara at Ippudo. Tired after a long day."),
        ("Tue 10am", "Coffee with Alex at Blue Bottle. Felt energized."),
        ("Wed 1pm", "Lunch at Ichiran alone. Drafted the Q3 plan."),
        ("Thu 7pm", "Met Sara again for tea. Talked about Tokyo."),
    ]
    base = datetime(2026, 5, 4, tzinfo=UTC)
    for i, (label, text) in enumerate(entries):
        when = base + timedelta(days=i, hours=9 + (i % 3) * 4)
        ep = lg.log(text, occurred_at=when)
        print(f"  {label}: predicates={ep.predicates}  body={ep.body_state}  sent={ep.sentiment}")

    # Time-range queries
    print("\n=== Episodes in the second half of the week ===")
    for e in lg.episodes.between(base + timedelta(days=2), base + timedelta(days=7)):
        print(f"  {e.occurred_at.date()}: {e.text}")

    # Per-entity recall
    print("\n=== Sara mentions ===")
    sara = lg.query(Person, key="sara").first()
    if sara is not None:
        for e in lg.episodes.mentioning(sara):
            print(f"  {e.occurred_at.date()}: {e.text}")

    # The pivot: query foods, walk to their episodes — your eating timeline.
    print("\n=== Eating timeline ===")
    for e in lg.query(Topic, kind="food").episodes():
        print(f"  {e.occurred_at.date()}: {e.text}")


if __name__ == "__main__":
    main()
