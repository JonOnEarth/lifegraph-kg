# SPDX-License-Identifier: Apache-2.0
"""Backfill ``duration`` and ``duration_inferred`` for migrated episodes.

The Phase-6 D1 → lifegraph-kg migration re-extracted every row through
the v6 prompt, which (in its original form) didn't capture duration.
The legacy D1 dump (``raw_d1_full.json``) DID have duration. This
script reads the dump, matches each row to a Supabase episode by
``(user_id, text, occurred_at)``, and PATCHes the duration values via
PostgREST.

Usage:
    export SUPABASE_PROJECT_URL='https://<ref>.supabase.co'
    export SUPABASE_SERVICE_ROLE_KEY='eyJ...'
    uv run python scripts/migrate_from_d1/backfill_duration.py
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

# legacy → jon's user_id; xuran227's data uses a different dump file.
USER_ID_MAP: dict[str | None, str] = {
    # The pull script filtered by user_id when it created the dump but
    # didn't include user_id in the SELECT, so most rows here came up
    # with user_id == None. Treat them as jon's data — the only user
    # who's been migrated to Supabase via this dump path so far.
    None: "e4a81cd7-d276-41ce-a5b6-f45856ac431b",
}


def _to_ms(s: object) -> int | None:
    """Legacy D1 stored occurred_at as either unix-ms or unix-seconds.
    Tolerate both."""
    if s is None or s == "":
        return None
    try:
        ts = int(s)
    except (TypeError, ValueError):
        return None
    if ts > 1e12:
        return ts
    return ts * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument(
        "--user-id",
        default="e4a81cd7-d276-41ce-a5b6-f45856ac431b",
        help="user_id to attribute rows in the dump to (the pull script omitted "
        "user_id from the SELECT; this fills it in)",
    )
    ap.add_argument(
        "--url",
        default=os.environ.get("SUPABASE_PROJECT_URL"),
        help="Supabase project URL",
    )
    ap.add_argument(
        "--key",
        default=os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
        help="service_role key",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.url or not args.key:
        print("ERROR: SUPABASE_PROJECT_URL + SUPABASE_SERVICE_ROLE_KEY required.", file=sys.stderr)
        return 2
    if not args.dump.exists():
        print(f"ERROR: {args.dump} not found.", file=sys.stderr)
        return 1

    rows = json.loads(args.dump.read_text())
    print(f"Loaded {len(rows)} legacy rows from {args.dump.name}.")
    with_dur = [r for r in rows if r.get("duration") not in (None, "")]
    print(f"  {len(with_dur)} have non-null duration.")

    headers = {
        "apikey": args.key,
        "Authorization": f"Bearer {args.key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    base = args.url.rstrip("/") + "/rest/v1"

    matched = 0
    updated = 0
    skipped_no_match = 0
    started = time.time()

    with httpx.Client(timeout=30.0) as client:
        for i, row in enumerate(with_dur):
            text = (row.get("text") or "").strip()
            ms = _to_ms(row.get("occurred_at") or row.get("created_at"))
            if not text or ms is None:
                skipped_no_match += 1
                continue

            # Look up the Supabase episode by (user_id, text, occurred_at).
            # PostgREST: GET /episodes?user_id=eq.X&text=eq.Y&occurred_at=eq.Z&select=id
            params = {
                "user_id": f"eq.{args.user_id}",
                "occurred_at": f"eq.{ms}",
                "select": "id",
                "limit": "5",
            }
            # text can contain special chars; use POST RPC-style param via JSON
            # body would be cleaner but PostgREST GET with `text=eq.<exact>` works
            # for our exact-match needs once we URL-encode.
            r = client.get(
                base + "/episodes",
                headers=headers,
                params={**params, "text": f"eq.{text}"},
            )
            if r.status_code >= 300:
                print(f"  [{i}] lookup failed {r.status_code}: {r.text[:120]}", file=sys.stderr)
                continue
            hits = r.json()
            if not hits:
                # Some texts in the dump have escaped quotes that got
                # different post-migration. Try a fallback: match
                # occurred_at + first 40 chars of text.
                fallback_text = text[:40]
                r = client.get(
                    base + "/episodes",
                    headers=headers,
                    params={
                        **params,
                        "text": f"ilike.{fallback_text}*",
                    },
                )
                hits = r.json() if r.status_code < 300 else []
            if not hits:
                skipped_no_match += 1
                continue

            matched += 1
            episode_id = hits[0]["id"]
            # Coerce duration / duration_inferred to native types.
            try:
                duration = int(row.get("duration") or 0) or None
            except (ValueError, TypeError):
                duration = None
            di_raw = row.get("duration_inferred")
            if isinstance(di_raw, str):
                duration_inferred = di_raw.lower() in ("1", "true", "yes")
            elif isinstance(di_raw, (int, bool)):
                duration_inferred = bool(di_raw)
            else:
                duration_inferred = None

            if duration is None:
                continue  # nothing to write

            if args.dry_run:
                print(
                    f"  [{i:4d}] would PATCH {episode_id} "
                    f"duration={duration} inferred={duration_inferred} "
                    f"(text={text[:50]!r})"
                )
                continue

            r = client.patch(
                base + "/episodes",
                headers=headers,
                params={"id": f"eq.{episode_id}"},
                content=json.dumps(
                    {"duration": duration, "duration_inferred": duration_inferred},
                    ensure_ascii=False,
                ),
            )
            if r.status_code >= 300:
                print(f"  [{i}] patch failed {r.status_code}: {r.text[:120]}", file=sys.stderr)
                continue
            updated += 1
            if updated % 50 == 0:
                print(f"  …{updated} patched")

    elapsed = time.time() - started
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print(f"  legacy rows with duration: {len(with_dur)}")
    print(f"  matched in Supabase:        {matched}")
    print(f"  successfully PATCHed:       {updated}")
    print(f"  skipped (no match):         {skipped_no_match}")
    if args.dry_run:
        print("  (--dry-run: nothing actually written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
