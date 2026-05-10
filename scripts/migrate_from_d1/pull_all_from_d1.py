# SPDX-License-Identifier: Apache-2.0
"""Phase 0: pull every life_item for a given user from the old D1.

Uses ``wrangler d1 execute --remote --json`` against the lifegraph-db
in the existing LifeGraph project. Output is gitignored.

Usage:
    python pull_all_from_d1.py --user-id <uuid> [--out data/raw_d1_full.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_USER_ID = "e4a81cd7-d276-41ce-a5b6-f45856ac431b"
DEFAULT_OUT = Path(__file__).parent / "data" / "raw_d1_full.json"
WRANGLER_CWD = Path("/Users/wuxu/Documents/GitHub/new-time/lifegraph/worker")
DB_NAME = "lifegraph-db"


def pull(user_id: str, out_path: Path) -> int:
    """Pull every life_item for `user_id` via wrangler. Returns count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pull both logs and todos so the migration is complete. The migration
    # script knows how to handle item_type when re-ingesting.
    sql = (
        "SELECT id, text, item_type, status, created_at, occurred_at, updated_at, "
        "       nodes, types, sentiment, energy, duration, duration_inferred, "
        "       context, priority, deadline, action_verb, recurrence, gtd_context, "
        "       completed_at, source, timezone, image_url, cost, currency "
        f"FROM life_items WHERE user_id = '{user_id}' "
        "ORDER BY occurred_at ASC"
    )
    print(f"Querying D1 for user {user_id[:8]}…", flush=True)
    result = subprocess.run(
        [
            "npx",
            "wrangler",
            "d1",
            "execute",
            DB_NAME,
            "--remote",
            "--json",
            "--command",
            sql,
        ],
        cwd=WRANGLER_CWD,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or not payload:
        print("ERROR: unexpected wrangler output shape.", file=sys.stderr)
        return 0
    rows = payload[0].get("results", [])

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Pulled {len(rows)} rows → {out_path}")

    # Quick stats for confidence
    by_type: dict[str, int] = {}
    for row in rows:
        t = row.get("item_type") or "?"
        by_type[t] = by_type.get(t, 0) + 1
    print(f"  item_type distribution: {by_type}")
    n_with_nodes = sum(1 for r in rows if r.get("nodes") and r["nodes"] != "[]")
    print(f"  rows with non-empty nodes: {n_with_nodes}")

    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user-id", default=DEFAULT_USER_ID)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    n = pull(args.user_id, args.out)
    if n == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
