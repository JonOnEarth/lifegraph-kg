# SPDX-License-Identifier: Apache-2.0
"""Migrate todos from old D1 → lifegraph-kg as task-kind episodes.

Unlike the log migration (which re-extracts via v6 prompt), todos
already have the structured fields we want (priority, deadline,
recurrence, gtd_context, action_verb), so we DO re-extract for entities
+ predicates but PRESERVE the structured fields directly.

The migrated DB is the same one logs went into — todos and logs land
side-by-side as Episode rows discriminated by ``kind``.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python migrate_todos.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from lifegraph_kg import LifeGraph, extract

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "data" / "raw_d1_full.json"
DEFAULT_OUT = HERE / "data" / "migrated.db"
DEFAULT_PROGRESS = HERE / "data" / "todo_migration_progress.json"

# Map legacy status → lifegraph-kg status. Legacy uses "active" + "completed";
# our model uses "active" + "done" + "dropped". No "dropped" in legacy data.
_STATUS_MAP = {
    "active": "active",
    "completed": "done",
}


def _ms_to_dt(value: object) -> datetime | None:
    """Legacy ``deadline`` and ``completed_at`` came in as either ISO
    strings, unix-seconds, or unix-ms. Tolerate all three; return UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        ts = int(value)
        if ts > 1e12:
            return datetime.fromtimestamp(ts / 1000, tz=UTC)
        return datetime.fromtimestamp(ts, tz=UTC)
    return None


def load_progress(path: Path) -> dict[str, dict]:
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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rate-limit-s", type=float, default=1.0)
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"ERROR: {args.input} not found.")
        return 1

    rows = json.loads(args.input.read_text())
    todos = [r for r in rows if r.get("item_type") == "todo"]
    print(f"Loaded {len(rows)} total rows; {len(todos)} are todos.")

    progress = load_progress(args.progress)
    print(
        f"Resume state: {len(progress)} already processed; {len(todos) - len(progress)} remaining."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lg = LifeGraph(store=f"sqlite:///{args.out}")

    processed = 0
    failures: list[dict] = []
    t_start = time.time()

    for row in todos:
        if args.limit is not None and processed >= args.limit:
            break

        item_id = str(row["id"])
        if item_id in progress and progress[item_id].get("status") == "ok":
            continue

        text = (row.get("text") or "").strip()
        if not text:
            progress[item_id] = {"status": "skipped", "reason": "empty text"}
            continue

        # Original timestamp (creation/occurred)
        occurred_raw = row.get("occurred_at") or row.get("created_at")
        try:
            ts = int(occurred_raw)
            if ts > 1e12:
                occurred_at = datetime.fromtimestamp(ts / 1000, tz=UTC)
            else:
                occurred_at = datetime.fromtimestamp(ts, tz=UTC)
        except (TypeError, ValueError):
            occurred_at = datetime.now(UTC)

        legacy_status = row.get("status") or "active"
        new_status = _STATUS_MAP.get(legacy_status, "active")
        deadline = _ms_to_dt(row.get("deadline"))
        completed_at = _ms_to_dt(row.get("completed_at"))
        priority = row.get("priority")
        recurrence = row.get("recurrence")
        gtd_context = row.get("gtd_context")
        action_verb = row.get("action_verb")
        source = row.get("source") or "user"

        try:
            # Re-extract entities + predicates via v6 (same as logs).
            # We don't use lg.task() because it would default-set occurred_at
            # to now; we need the legacy timestamp + the structured lifecycle
            # fields. Use the lower-level _persist_task path.
            result = extract(text)
            ep = lg._persist_task(  # type: ignore[attr-defined]
                text,
                result,
                occurred_at=occurred_at,
                source=source,
                deadline=deadline,
                priority=priority,
                gtd_context=gtd_context,
                recurrence=recurrence,
                action_verb=action_verb,
            )
            # Apply terminal status if not active.
            if new_status == "done":
                lg.complete_task(ep.id, at=completed_at or occurred_at)
            progress[item_id] = {
                "status": "ok",
                "episode_id": ep.id,
                "kind": "task",
                "task_status": new_status,
                "predicates": ep.predicates,
                "deadline": deadline.isoformat() if deadline else None,
            }
            processed += 1

            if processed % 10 == 0:
                save_progress(args.progress, progress)
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = len(todos) - len(progress)
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"  [{processed:3d}/{len(todos)}] {elapsed:.1f}s elapsed, "
                    f"{rate:.2f}/s, ETA {eta / 60:.1f} min  "
                    f"[{new_status}] {ep.predicates}",
                    flush=True,
                )
        except Exception as e:
            failures.append({"id": item_id, "text": text[:80], "error": repr(e)})
            progress[item_id] = {"status": "error", "error": repr(e)}
            print(f"  [error] item {item_id}: {e}", file=sys.stderr)

        time.sleep(args.rate_limit_s)

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
