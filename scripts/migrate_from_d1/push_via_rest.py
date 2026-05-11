# SPDX-License-Identifier: Apache-2.0
"""Phase 3 (alt): push migrated.db → Supabase via PostgREST.

For users who logged into Supabase with OAuth (no DB password). Uses
the project's ``service_role`` API key to bypass RLS and POST batches
directly to ``/rest/v1/{table}``.

Why this exists alongside ``push_to_supabase.py``:
  - ``push_to_supabase.py`` requires a Postgres DSN (password)
  - This script needs only the project URL + service_role key

Both write the same rows; choose whichever fits the credentials you
already have.

Usage:
    export SUPABASE_PROJECT_URL='https://<ref>.supabase.co'
    export SUPABASE_SERVICE_ROLE_KEY='eyJ...'
    uv run python push_via_rest.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).parent
DEFAULT_DB = HERE / "data" / "migrated.db"


# (table, columns, on_conflict_columns)
TABLES = [
    (
        "episodes",
        [
            "id",
            "user_id",
            "text",
            "occurred_at",
            "ingested_at",
            "source",
            "predicates",
            "body_state",
            "sentiment",
            "energy",
            "kind",
            "status",
            "priority",
            "deadline",
            "completed_at",
            "recurrence",
            "gtd_context",
            "action_verb",
        ],
        "id",
    ),
    (
        "entities",
        [
            "id",
            "user_id",
            "type",
            "kind",
            "key",
            "value",
            "attributes_json",
            "created_at",
            "canonical_id",
        ],
        "id",
    ),
    (
        "edges",
        [
            "id",
            "user_id",
            "from_entity",
            "to_entity",
            "verb",
            "episode_id",
            "t_event",
            "t_ingestion",
            "t_valid",
            "t_invalid",
            "attributes_json",
        ],
        "id",
    ),
    (
        "entity_episode_mention",
        ["entity_id", "episode_id", "user_id"],
        "entity_id,episode_id",
    ),
    (
        "merge_proposals",
        [
            "id",
            "winner_id",
            "loser_id",
            "confidence",
            "reason",
            "detail",
            "proposed_at",
            "applied_at",
            "rejected_at",
        ],
        "id",
    ),
]


def _row_to_dict(row: sqlite3.Row, cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in cols:
        out[c] = row[c]
    return out


def _post_batch(
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    table: str,
    on_conflict: str,
    rows: list[dict[str, Any]],
) -> int:
    """POST a batch of rows; returns number actually inserted."""
    url = f"{base_url}/rest/v1/{table}?on_conflict={on_conflict}"
    h = dict(headers)
    h["Prefer"] = "resolution=ignore-duplicates,return=representation"
    r = client.post(url, headers=h, content=json.dumps(rows, ensure_ascii=False))
    if r.status_code >= 300:
        raise RuntimeError(f"{table} POST failed {r.status_code}: {r.text[:300]}")
    inserted = r.json()
    return len(inserted) if isinstance(inserted, list) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--url", default=os.environ.get("SUPABASE_PROJECT_URL"))
    ap.add_argument("--key", default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    if not args.url or not args.key:
        print(
            "ERROR: set SUPABASE_PROJECT_URL and SUPABASE_SERVICE_ROLE_KEY "
            "(or pass --url and --key).",
            file=sys.stderr,
        )
        return 2
    if not args.db.exists():
        print(f"ERROR: {args.db} not found.", file=sys.stderr)
        return 1

    base_url = args.url.rstrip("/")
    headers = {
        "apikey": args.key,
        "Authorization": f"Bearer {args.key}",
        "Content-Type": "application/json",
    }

    src = sqlite3.connect(args.db)
    src.row_factory = sqlite3.Row

    t_start = time.time()
    totals: dict[str, tuple[int, int]] = {}

    with httpx.Client(timeout=60.0) as client:
        for table, cols, on_conflict in TABLES:
            src_n = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"\n[{table}] {src_n} rows to push")
            if src_n == 0:
                totals[table] = (0, 0)
                continue

            rows_iter = src.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
            batch: list[dict[str, Any]] = []
            inserted_total = 0
            for r in rows_iter:
                batch.append(_row_to_dict(r, cols))
                if len(batch) >= args.batch:
                    inserted_total += _post_batch(
                        client, base_url, headers, table, on_conflict, batch
                    )
                    batch = []
                    print(f"  {inserted_total:5d}/{src_n} …", end="\r", flush=True)
            if batch:
                inserted_total += _post_batch(
                    client, base_url, headers, table, on_conflict, batch
                )
            totals[table] = (src_n, inserted_total)
            print(f"  → {inserted_total} inserted (skipped duplicates: {src_n - inserted_total})")

    src.close()

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"REST push complete in {elapsed:.1f}s")
    for table, (src_n, ins_n) in totals.items():
        print(f"  {table:25s} source={src_n:5d}  inserted={ins_n:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
