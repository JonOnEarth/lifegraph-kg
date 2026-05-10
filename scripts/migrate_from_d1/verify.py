# SPDX-License-Identifier: Apache-2.0
"""Phase 2: verify the migration.

Reads the migrated SQLite + the original D1 dump and reports:
  - Counts: episodes vs original logs (should match)
  - Type distribution: Person / Place / Project / Topic by kind
  - Predicate vocabulary: most common verbs across all episodes
  - Body-state coverage: how many episodes captured an explicit body_state
  - Sentiment / energy distribution
  - Sample-by-sample comparison: 5 random old → new pairs

Usage:
    uv run python verify.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from lifegraph_kg import LifeGraph, Person, Place, Project, Topic

HERE = Path(__file__).parent
DEFAULT_DB = HERE / "data" / "migrated.db"
DEFAULT_RAW = HERE / "data" / "raw_d1_full.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"ERROR: {args.db} not found. Run migrate_to_lifegraph_kg.py first.")
        return 1

    lg = LifeGraph(store=f"sqlite:///{args.db}")
    rng = random.Random(args.seed)

    # --- counts ---
    print("=" * 70)
    print("COUNTS")
    print("=" * 70)
    raw_rows = json.loads(args.raw.read_text())
    raw_logs = [r for r in raw_rows if r.get("item_type") == "log"]
    raw_todos = [r for r in raw_rows if r.get("item_type") == "todo"]
    print(f"Original D1 logs:       {len(raw_logs)}")
    print(f"Original D1 todos:      {len(raw_todos)}")

    sqlstore = lg._store  # type: ignore[attr-defined]
    n_episodes = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes"
    ).fetchone()[0]
    n_logs = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes WHERE kind = 'log'"
    ).fetchone()[0]
    n_tasks = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes WHERE kind = 'task'"
    ).fetchone()[0]
    n_tasks_done = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes WHERE kind = 'task' AND status = 'done'"
    ).fetchone()[0]
    n_tasks_active = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes WHERE kind = 'task' AND status = 'active'"
    ).fetchone()[0]
    n_entities = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM entities"
    ).fetchone()[0]
    n_edges = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM edges"
    ).fetchone()[0]
    n_mentions = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM entity_episode_mention"
    ).fetchone()[0]
    print(f"Migrated episodes:      {n_episodes}  (logs: {n_logs}, tasks: {n_tasks})")
    if n_tasks > 0:
        print(f"  task status:          active: {n_tasks_active}, done: {n_tasks_done}")
    print(f"Migrated entities:      {n_entities}")
    print(f"Migrated edges:         {n_edges}")
    print(f"Mention links:          {n_mentions}")

    log_delta = len(raw_logs) - n_logs
    todo_delta = len(raw_todos) - n_tasks
    if log_delta == 0:
        print(f"✅ Log count matches D1 ({n_logs}).")
    else:
        print(f"⚠️  {log_delta} logs not migrated.")
    if todo_delta == 0 and n_tasks > 0:
        print(f"✅ Task count matches D1 todos ({n_tasks}).")
    elif n_tasks == 0:
        print("(i) No tasks migrated yet — run migrate_todos.py.")
    else:
        print(f"⚠️  {todo_delta} todos not migrated.")

    # --- type / kind distribution ---
    print()
    print("=" * 70)
    print("ENTITY TYPE DISTRIBUTION")
    print("=" * 70)
    for cls in (Person, Place, Project):
        n = len(lg.query(cls).all())
        print(f"  {cls.__name__:10s}  {n}")

    topics = lg.query(Topic).all()
    print(f"  {'Topic':10s}  {len(topics)}")
    kind_counts: Counter[str] = Counter(t.kind for t in topics)
    for kind, n in kind_counts.most_common():
        print(f"    └─ kind={kind:10s} {n}")

    # --- predicate vocabulary ---
    print()
    print("=" * 70)
    print("TOP PREDICATES (top 20)")
    print("=" * 70)
    rows = sqlstore._conn.execute("SELECT predicates FROM episodes").fetchall()  # type: ignore[attr-defined]
    pred_counts: Counter[str] = Counter()
    for row in rows:
        for p in json.loads(row["predicates"] or "[]"):
            pred_counts[p] += 1
    for verb, n in pred_counts.most_common(20):
        print(f"  {verb:25s} {n}")

    n_no_predicate = sum(1 for r in rows if json.loads(r["predicates"] or "[]") == [])
    n_multi = sum(1 for r in rows if len(json.loads(r["predicates"] or "[]")) >= 2)
    print()
    print(f"Episodes with 0 predicates: {n_no_predicate}")
    print(f"Episodes with ≥2 predicates (multi-action): {n_multi}")

    # --- affect/body coverage ---
    print()
    print("=" * 70)
    print("EPISODE METADATA COVERAGE")
    print("=" * 70)
    n_body = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes WHERE body_state IS NOT NULL"
    ).fetchone()[0]
    n_sent = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes WHERE sentiment IS NOT NULL"
    ).fetchone()[0]
    n_energy = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM episodes WHERE energy IS NOT NULL"
    ).fetchone()[0]
    print(f"  body_state present: {n_body} ({100 * n_body / n_episodes:.1f}%)")
    print(f"  sentiment present:  {n_sent} ({100 * n_sent / n_episodes:.1f}%)")
    print(f"  energy present:     {n_energy} ({100 * n_energy / n_episodes:.1f}%)")

    sent_dist = sqlstore._conn.execute(  # type: ignore[attr-defined]
        "SELECT sentiment, COUNT(*) as n FROM episodes GROUP BY sentiment"
    ).fetchall()
    sent_summary = {r["sentiment"] or "NULL": r["n"] for r in sent_dist}
    print(f"  sentiment distribution: {sent_summary}")

    # --- sample comparison ---
    print()
    print("=" * 70)
    print(f"SAMPLE-BY-SAMPLE: {args.samples} random old → new pairs")
    print("=" * 70)
    sample_logs = rng.sample(raw_logs, min(args.samples, len(raw_logs)))
    for i, raw in enumerate(sample_logs, 1):
        print(f"\n--- Sample {i} ---")
        text = raw.get("text", "")
        print(f"Text: {text[:120]}")
        print()

        # Old: parse legacy nodes
        old_nodes = json.loads(raw.get("nodes") or "[]")
        print("OLD (legacy):")
        if old_nodes:
            for n in old_nodes:
                print(f"  {n.get('type', '?'):8s}  {n.get('value', '?')}")
        else:
            print("  (no nodes)")
        print(f"  sentiment={raw.get('sentiment')}, energy={raw.get('energy')}")

        # New: find the migrated episode by approximate text match
        # (we don't have a direct ID mapping; use text similarity)
        cur = sqlstore._conn.execute(  # type: ignore[attr-defined]
            "SELECT * FROM episodes WHERE text = ? LIMIT 1", (text,)
        )
        ep_row = cur.fetchone()
        print()
        print("NEW (lifegraph-kg):")
        if ep_row is not None:
            preds = json.loads(ep_row["predicates"] or "[]")
            print(f"  predicates: {preds}")
            print(f"  body_state: {ep_row['body_state']}")
            print(f"  sentiment:  {ep_row['sentiment']}, energy: {ep_row['energy']}")
            ents = sqlstore._conn.execute(  # type: ignore[attr-defined]
                "SELECT e.* FROM entities e "
                "JOIN entity_episode_mention m ON m.entity_id = e.id "
                "WHERE m.episode_id = ?",
                (ep_row["id"],),
            ).fetchall()
            for ent in ents:
                kind = f"{{kind:{ent['kind']}}}" if ent["kind"] else ""
                print(f"  {ent['type']}{kind:13s}  {ent['value']}")
        else:
            print("  ❌ no matching episode found")

    print()
    print("=" * 70)
    print("Migration verification complete.")
    print(f"DB: {args.db}")
    print(f"Open it: sqlite3 {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
