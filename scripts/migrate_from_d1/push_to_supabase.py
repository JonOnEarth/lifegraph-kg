# SPDX-License-Identifier: Apache-2.0
"""Phase 3: push migrated SQLite → Supabase Postgres.

Copies all 5 tables (episodes, entities, edges, entity_episode_mention,
merge_proposals) from the local ``migrated.db`` to Supabase, preserving
all primary-key IDs so frontends/services already referencing them stay
valid.

The schema is created by ``PostgresStore.init_schema()`` (idempotent),
then rows are bulk-inserted with ``ON CONFLICT DO NOTHING`` so the
script is **safe to re-run** — interrupted runs resume cleanly.

Usage:
    export SUPABASE_DB_URL='postgresql://postgres:pass@db.xxx.supabase.co:5432/postgres'
    uv run python push_to_supabase.py
    # or:
    uv run python push_to_supabase.py --dsn "$SUPABASE_DB_URL"

Notes on the DSN:
  - Use the **Session pooler** or direct connection string from the
    Supabase dashboard → Project Settings → Database → Connection String.
  - The **Transaction pooler** (port 6543) will also work, but doesn't
    allow ``LISTEN/NOTIFY`` or prepared statements > 1 statement; the
    Session pooler (port 5432) is safer for bulk inserts.
  - psycopg accepts both ``postgres://`` and ``postgresql://``.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DB = HERE / "data" / "migrated.db"


# Tuples of (table_name, sqlite_select, postgres_insert).
# Order matters: episodes & entities first (referenced by FKs), then
# edges + mentions + proposals which reference them.
TABLES = [
    (
        "episodes",
        """SELECT id, text, occurred_at, ingested_at, source,
                  predicates, body_state, sentiment, energy,
                  kind, status, priority, deadline, completed_at,
                  recurrence, gtd_context, action_verb
             FROM episodes""",
        """INSERT INTO episodes
             (id, text, occurred_at, ingested_at, source,
              predicates, body_state, sentiment, energy,
              kind, status, priority, deadline, completed_at,
              recurrence, gtd_context, action_verb)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO NOTHING""",
    ),
    (
        "entities",
        """SELECT id, type, kind, key, value, attributes_json,
                  created_at, canonical_id
             FROM entities""",
        """INSERT INTO entities
             (id, type, kind, key, value, attributes_json,
              created_at, canonical_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO NOTHING""",
    ),
    (
        "edges",
        """SELECT id, from_entity, to_entity, verb, episode_id,
                  t_event, t_ingestion, t_valid, t_invalid,
                  attributes_json
             FROM edges""",
        """INSERT INTO edges
             (id, from_entity, to_entity, verb, episode_id,
              t_event, t_ingestion, t_valid, t_invalid,
              attributes_json)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO NOTHING""",
    ),
    (
        "entity_episode_mention",
        "SELECT entity_id, episode_id FROM entity_episode_mention",
        """INSERT INTO entity_episode_mention (entity_id, episode_id)
           VALUES (%s, %s)
           ON CONFLICT DO NOTHING""",
    ),
    (
        "merge_proposals",
        """SELECT id, winner_id, loser_id, confidence, reason,
                  detail, proposed_at, applied_at, rejected_at
             FROM merge_proposals""",
        """INSERT INTO merge_proposals
             (id, winner_id, loser_id, confidence, reason,
              detail, proposed_at, applied_at, rejected_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (id) DO NOTHING""",
    ),
]


def _resolve_dsn(arg_dsn: str | None) -> str:
    if arg_dsn:
        return arg_dsn
    for env_var in ("SUPABASE_DB_URL", "POSTGRES_URL", "DATABASE_URL"):
        if env_var in os.environ:
            print(f"Using DSN from ${env_var}")
            return os.environ[env_var]
    print(
        "ERROR: no DSN. Provide via --dsn or one of "
        "SUPABASE_DB_URL / POSTGRES_URL / DATABASE_URL env vars.",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dsn", default=None, help="Supabase Postgres DSN")
    ap.add_argument("--batch", type=int, default=200, help="Rows per executemany batch")
    ap.add_argument(
        "--drop-existing",
        action="store_true",
        help="DANGER: TRUNCATE all 5 tables before copying. Use only for clean re-pushes.",
    )
    args = ap.parse_args()

    if not args.db.exists():
        print(f"ERROR: {args.db} not found.", file=sys.stderr)
        return 1

    dsn = _resolve_dsn(args.dsn)

    # Lazy imports so the help text works without psycopg installed.
    try:
        import psycopg
    except ImportError:
        print(
            "ERROR: psycopg not installed. Install with: "
            "uv sync --extra postgres",
            file=sys.stderr,
        )
        return 2

    from lifegraph_kg.kg.store.postgres import PostgresStore

    # Open both ends.
    src = sqlite3.connect(args.db)
    src.row_factory = sqlite3.Row

    print(f"Connecting to Supabase…")
    store = PostgresStore(dsn)  # runs init_schema() automatically
    dst = store._conn

    print("Schema ready.")

    if args.drop_existing:
        print("⚠️  --drop-existing: truncating destination tables…")
        with dst.cursor() as cur:
            cur.execute(
                "TRUNCATE merge_proposals, entity_episode_mention, edges, "
                "entities, episodes RESTART IDENTITY CASCADE"
            )
        dst.commit()

    t_start = time.time()
    totals: dict[str, tuple[int, int]] = {}  # table -> (src_count, copied)

    for table, select_sql, insert_sql in TABLES:
        src_n = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"\n[{table}] {src_n} rows in source")
        if src_n == 0:
            totals[table] = (0, 0)
            continue

        # Pre-count destination so we can report deltas.
        with dst.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
            dst_before = row["count"] if isinstance(row, dict) else row[0]
        print(f"          {dst_before} rows already in destination")

        rows = src.execute(select_sql).fetchall()
        batch: list[tuple] = []
        copied = 0
        with dst.cursor() as cur:
            for r in rows:
                batch.append(tuple(r))
                if len(batch) >= args.batch:
                    cur.executemany(insert_sql, batch)
                    copied += len(batch)
                    batch = []
                    print(
                        f"  {copied:5d}/{src_n} …",
                        end="\r",
                        flush=True,
                    )
            if batch:
                cur.executemany(insert_sql, batch)
                copied += len(batch)
        dst.commit()

        with dst.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
            dst_after = row["count"] if isinstance(row, dict) else row[0]
        new_rows = dst_after - dst_before
        totals[table] = (src_n, new_rows)
        print(f"  → {dst_after} rows in destination (+{new_rows} new)")

    src.close()
    store.close()

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Push complete in {elapsed:.1f}s")
    for table, (src_n, new_n) in totals.items():
        marker = "✅" if new_n == src_n or new_n == 0 else "⚠️ "
        print(f"  {marker} {table:25s} source={src_n:5d}  new={new_n:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
