# SPDX-License-Identifier: Apache-2.0
"""Dump migrated.db into batched INSERT SQL files for MCP execute_sql.

Why this exists: the Supabase MCP plugin doesn't expose the database
password, so we can't run ``push_to_supabase.py`` directly. Instead we
emit one SQL file per (table, batch) pair, and the orchestrator
(Claude) reads each file and calls ``mcp__supabase__execute_sql``.

Usage:
    uv run python dump_for_mcp.py --batch 200

Produces files in ``data/mcp_dump/``:
    01_episodes_000.sql
    01_episodes_001.sql
    ...
    02_entities_000.sql
    03_edges_000.sql
    04_mentions_000.sql
    05_proposals_000.sql
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_DB = HERE / "data" / "migrated.db"
DEFAULT_OUT = HERE / "data" / "mcp_dump"


def _sql_quote(v: object) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("'", "''")
    return f"'{s}'"


# (table, columns, sqlite_select, on_conflict)
TABLES = [
    (
        "episodes",
        [
            "id",
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
        """SELECT id, text, occurred_at, ingested_at, source,
                  predicates, body_state, sentiment, energy,
                  kind, status, priority, deadline, completed_at,
                  recurrence, gtd_context, action_verb FROM episodes ORDER BY id""",
        "(id) DO NOTHING",
    ),
    (
        "entities",
        ["id", "type", "kind", "key", "value", "attributes_json", "created_at", "canonical_id"],
        """SELECT id, type, kind, key, value, attributes_json,
                  created_at, canonical_id FROM entities ORDER BY id""",
        "(id) DO NOTHING",
    ),
    (
        "edges",
        [
            "id",
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
        """SELECT id, from_entity, to_entity, verb, episode_id,
                  t_event, t_ingestion, t_valid, t_invalid,
                  attributes_json FROM edges ORDER BY id""",
        "(id) DO NOTHING",
    ),
    (
        "entity_episode_mention",
        ["entity_id", "episode_id"],
        "SELECT entity_id, episode_id FROM entity_episode_mention ORDER BY entity_id, episode_id",
        "(entity_id, episode_id) DO NOTHING",
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
        """SELECT id, winner_id, loser_id, confidence, reason,
                  detail, proposed_at, applied_at, rejected_at FROM merge_proposals""",
        "(id) DO NOTHING",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # Wipe stale files from previous runs.
    for f in args.out.glob("*.sql"):
        f.unlink()

    src = sqlite3.connect(args.db)
    src.row_factory = sqlite3.Row

    manifest_lines: list[str] = []
    for order_i, (table, cols, select_sql, on_conflict) in enumerate(TABLES, 1):
        rows = src.execute(select_sql).fetchall()
        if not rows:
            print(f"  [{table}] 0 rows — skipping")
            continue
        col_list = ", ".join(cols)
        batches = [rows[i : i + args.batch] for i in range(0, len(rows), args.batch)]
        for batch_i, batch in enumerate(batches):
            values_lines = []
            for r in batch:
                quoted = [_sql_quote(r[c]) for c in cols]
                values_lines.append(f"({', '.join(quoted)})")
            # Single-line form: easier to read back into execute_sql without
            # the Read-tool line-number prefixes getting in the way.
            values = ", ".join(values_lines)
            sql = (
                f"INSERT INTO {table} ({col_list}) VALUES {values} "
                f"ON CONFLICT {on_conflict};"
            )
            fname = f"{order_i:02d}_{table}_{batch_i:03d}.sql"
            (args.out / fname).write_text(sql)
            manifest_lines.append(f"{fname}\t{len(batch)} rows")
        print(f"  [{table}] {len(rows)} rows → {len(batches)} files")

    (args.out / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n")
    src.close()
    print(f"\nWrote {len(manifest_lines)} SQL files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
