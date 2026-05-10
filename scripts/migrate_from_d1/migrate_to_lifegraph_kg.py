# SPDX-License-Identifier: Apache-2.0
"""Phase 1: re-extract every log via v6 prompt + persist to lifegraph-kg SQLite.

Reads the JSON dump from Phase 0, calls ``lg.log()`` for each LOG entry
(todos are skipped — they have GTD fields lifegraph-kg doesn't model),
and writes to a local SQLite DB. Resume-safe: tracks completed item IDs
in a sidecar file so re-runs skip already-done work.

Cost: ~$0.005/entry, ~$2-3 for 437 logs.
Time: ~30-45 min wall clock with Anthropic rate limits.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python migrate_to_lifegraph_kg.py
    # or with a custom DB path:
    uv run python migrate_to_lifegraph_kg.py --out data/migrated.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lifegraph_kg import LifeGraph

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "data" / "raw_d1_full.json"
DEFAULT_OUT = HERE / "data" / "migrated.db"
DEFAULT_PROGRESS = HERE / "data" / "migration_progress.json"


def load_progress(path: Path) -> dict[str, dict]:
    """Resume support: load already-processed item IDs (and their results)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_progress(path: Path, progress: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    ap.add_argument("--limit", type=int, default=None, help="Stop after N entries (testing)")
    ap.add_argument(
        "--rate-limit-s",
        type=float,
        default=1.0,
        help="Min seconds between LLM calls (1.0 ≈ 60/min, well under Anthropic limit)",
    )
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run pull_all_from_d1.py first.")
        return 1

    rows = json.loads(args.input.read_text())
    logs = [r for r in rows if r.get("item_type") == "log"]
    print(f"Loaded {len(rows)} total rows; {len(logs)} are logs (todos skipped).")

    progress = load_progress(args.progress)
    print(
        f"Resume state: {len(progress)} already processed; {len(logs) - len(progress)} remaining."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lg = LifeGraph(store=f"sqlite:///{args.out}")

    processed = 0
    failures: list[dict] = []
    t_start = time.time()

    for row in logs:
        if args.limit is not None and processed >= args.limit:
            break

        item_id = str(row["id"])
        if item_id in progress and progress[item_id].get("status") == "ok":
            continue  # already done

        text = (row.get("text") or "").strip()
        if not text:
            progress[item_id] = {"status": "skipped", "reason": "empty text"}
            continue

        # occurred_at is unix-seconds in legacy schema; convert to datetime.
        occurred_at_raw = row.get("occurred_at") or row.get("created_at")
        try:
            # Legacy stored as unix-seconds (some entries unix-ms, accept both).
            ts = int(occurred_at_raw)
            if ts > 1e12:  # already ms
                occurred_at = datetime.fromtimestamp(ts / 1000, tz=UTC)
            else:
                occurred_at = datetime.fromtimestamp(ts, tz=UTC)
        except (TypeError, ValueError):
            occurred_at = datetime.now(UTC)

        source = row.get("source") or "user"

        try:
            ep = lg.log(text, occurred_at=occurred_at, source=source)
            progress[item_id] = {
                "status": "ok",
                "episode_id": ep.id,
                "predicates": ep.predicates,
                "body_state": ep.body_state,
                "sentiment": ep.sentiment,
                "energy": ep.energy,
            }
            processed += 1

            if processed % 10 == 0:
                save_progress(args.progress, progress)
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = len(logs) - len(progress)
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"  [{processed:3d}/{len(logs)}] {elapsed:.1f}s elapsed, "
                    f"{rate:.2f}/s, ETA {eta / 60:.1f} min  "
                    f"(predicates={ep.predicates}, body={ep.body_state})",
                    flush=True,
                )
        except Exception as e:
            failures.append({"id": item_id, "text": text[:80], "error": repr(e)})
            progress[item_id] = {"status": "error", "error": repr(e)}
            print(f"  [error] item {item_id}: {e}", file=sys.stderr)

        time.sleep(args.rate_limit_s)

    # Final save
    save_progress(args.progress, progress)

    elapsed = time.time() - t_start
    n_ok = sum(1 for v in progress.values() if v.get("status") == "ok")
    n_err = sum(1 for v in progress.values() if v.get("status") == "error")
    n_skip = sum(1 for v in progress.values() if v.get("status") == "skipped")
    print()
    print("=" * 60)
    print(f"Done. {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  ok:      {n_ok}")
    print(f"  error:   {n_err}")
    print(f"  skipped: {n_skip}")
    print(f"  output:  {args.out}")
    if failures:
        print()
        print("Sample failures (first 5):")
        for f in failures[:5]:
            print(f"  {f['id']}: {f['error'][:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
