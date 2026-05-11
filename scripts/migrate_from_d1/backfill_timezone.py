# SPDX-License-Identifier: Apache-2.0
"""Backfill ``origin_tz`` (and ``time_mode='absolute'`` for tasks) on
the 770 migrated episodes.

The D1 dump's ``timezone`` field is present on a minority of rows (36
of 712 for jon). For the rest we default to ``--default-tz`` (jon's
main timezone, America/New_York). Every existing task gets
``time_mode='absolute'`` per the design doc §10 Phase-1 fallback —
tasks with legacy deadline dates were always "fixed physical moment"
in the original system; floating recurrence only makes sense for new
entries extracted under the updated v6 prompt.

Usage:
    export SUPABASE_PROJECT_URL='https://<ref>.supabase.co'
    export SUPABASE_SERVICE_ROLE_KEY='eyJ...'
    uv run python scripts/migrate_from_d1/backfill_timezone.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
DEFAULT_DUMP = HERE / "data" / "raw_d1_full.json"
JON_ID = "e4a81cd7-d276-41ce-a5b6-f45856ac431b"
XURAN_ID = "172b8b6a-f9fe-4626-8087-325246ceb71d"


def _to_ms(s: object) -> int | None:
    if s is None or s == "":
        return None
    try:
        ts = int(s)
    except (TypeError, ValueError):
        return None
    return ts if ts > 1e12 else ts * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument(
        "--user-id",
        default=JON_ID,
        help="user_id whose Supabase episodes to update from this dump",
    )
    ap.add_argument(
        "--default-tz",
        default="America/New_York",
        help="IANA tz to use when the dump row has no timezone (most rows)",
    )
    ap.add_argument(
        "--url",
        default=os.environ.get("SUPABASE_PROJECT_URL"),
    )
    ap.add_argument(
        "--key",
        default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.url or not args.key:
        print(
            "ERROR: SUPABASE_PROJECT_URL + SUPABASE_SERVICE_ROLE_KEY required.",
            file=sys.stderr,
        )
        return 2
    if not args.dump.exists():
        print(f"ERROR: {args.dump} not found.", file=sys.stderr)
        return 1

    rows = json.loads(args.dump.read_text())
    print(f"Loaded {len(rows)} legacy rows from {args.dump.name}.")
    print(f"  default origin_tz when row.timezone is null: {args.default_tz}")

    headers = {
        "apikey": args.key,
        "Authorization": f"Bearer {args.key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    base = args.url.rstrip("/") + "/rest/v1"

    matched = 0
    updated_tz = 0
    updated_mode = 0
    skipped_no_match = 0
    started = time.time()

    with httpx.Client(timeout=30.0) as client:
        for i, row in enumerate(rows):
            text = (row.get("text") or "").strip()
            ms = _to_ms(row.get("occurred_at") or row.get("created_at"))
            if not text or ms is None:
                skipped_no_match += 1
                continue

            # Find the matching Supabase episode by (user, occurred_at, text).
            params = {
                "user_id": f"eq.{args.user_id}",
                "occurred_at": f"eq.{ms}",
                "select": "id,kind,origin_tz,time_mode",
                "limit": "5",
                "text": f"eq.{text}",
            }
            r = client.get(base + "/episodes", headers=headers, params=params)
            hits = r.json() if r.status_code < 300 else []
            if not hits:
                # Fallback: prefix-match on text (handles minor quote escaping).
                params["text"] = f"ilike.{text[:40]}*"
                r = client.get(base + "/episodes", headers=headers, params=params)
                hits = r.json() if r.status_code < 300 else []
            if not hits:
                skipped_no_match += 1
                continue
            matched += 1

            ep = hits[0]
            updates: dict = {}
            if not ep.get("origin_tz"):
                tz = (row.get("timezone") or "").strip() or args.default_tz
                updates["origin_tz"] = tz
            # Tasks default to 'absolute' (their legacy deadline date
            # was a fixed UTC moment). Logs leave time_mode null.
            if ep.get("kind") == "task" and not ep.get("time_mode"):
                updates["time_mode"] = "absolute"

            if not updates:
                continue

            if args.dry_run:
                print(f"  [{i:4d}] would PATCH {ep['id']}: {updates}")
                continue

            r = client.patch(
                base + "/episodes",
                headers=headers,
                params={"id": f"eq.{ep['id']}"},
                content=json.dumps(updates, ensure_ascii=False),
            )
            if r.status_code >= 300:
                print(f"  [{i}] patch failed {r.status_code}: {r.text[:120]}", file=sys.stderr)
                continue
            if "origin_tz" in updates:
                updated_tz += 1
            if "time_mode" in updates:
                updated_mode += 1
            if (updated_tz + updated_mode) % 100 == 0:
                print(f"  …{updated_tz} origin_tz + {updated_mode} time_mode")

    elapsed = time.time() - started
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  legacy rows seen:       {len(rows)}")
    print(f"  matched in Supabase:    {matched}")
    print(f"  PATCHed origin_tz:      {updated_tz}")
    print(f"  PATCHed time_mode:      {updated_mode}")
    print(f"  skipped (no match):     {skipped_no_match}")
    if args.dry_run:
        print("  (--dry-run: nothing actually written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
