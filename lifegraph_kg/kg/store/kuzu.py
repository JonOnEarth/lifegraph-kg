# SPDX-License-Identifier: Apache-2.0
"""KuzuStore — embedded property-graph backend (planned for L4.1).

Status: **stub**. The class exists so `_resolve_store("kuzu:///...")`
can dispatch correctly and the eval framework's
`library_storage_drivers_ready()` can detect it. The actual
implementation is L4.1.

## Why Kuzu's a meaningful add (vs just SQLite + Postgres)

Kuzu is a single-file embedded property-graph DB (Apache 2.0). Unlike
SQLite, native graph queries — Cypher path traversals, variable-length
hops, MATCH patterns — run as graph algorithms instead of recursive CTEs.
For an autobiographical KG that grows into thousands of edges, that
matters once we add features like "two-hop friend network" or
"connected by any verb in any number of steps".

## Why it's not in v0.1

Kuzu's data model is fundamentally different from the relational schema
we're using:
  - Nodes and rels declared as schemas (Cypher DDL), not CREATE TABLE
  - Bi-temporal columns become rel properties; queries express validity
    as a path predicate rather than a WHERE clause
  - Mention links become typed rels rather than a separate table
  - No `executescript`-style multi-statement DDL — Kuzu has its own
    `create_node_table` / `create_rel_table` API

A clean port is a few days of work — and would benefit from us first
having a richer test suite to compare semantics against (especially
for the bi-temporal supersede + as_of cases). For v0.1 we ship
SqliteStore (default) + PostgresStore (multi-user/shared), and Kuzu
follows in v0.2 along with embedding-based fuzzy hygiene.

## Design notes for L4.1

  - Schema: `Person`, `Place`, `Project`, `Topic` are node tables.
    `Episode` is a node table (the perdurant). `MENTIONS`, plus one rel
    table per verb (or a single `RELATED` rel with `verb` property —
    the latter is simpler but loses Kuzu's per-rel-type indexing).
  - Bi-temporal: rel properties t_event/t_ingestion/t_valid/t_invalid;
    `as_of(t)` becomes a Cypher WHERE clause on the rel.
  - Merge-as-alias: Kuzu doesn't have FK NULL semantics, so we'd encode
    canonical with a `CANONICALIZES` rel pointing winner ← loser.
  - Driver: `kuzu` PyPI package (Apache-2.0). Connect to a directory.

## Why we still ship the stub

`_resolve_store("kuzu:///path")` raises a clear NotImplementedError
with this same explanation, so users who try Kuzu in v0.1 get a
useful message rather than a confusing ImportError.
"""

from __future__ import annotations

from typing import NoReturn


def KuzuStore(path: str) -> NoReturn:  # noqa: N802  (matches the eventual class name)
    """Factory stub. Raises NotImplementedError immediately.

    Implemented as a function (not a class) so mypy doesn't expect it
    to satisfy the Store protocol — it can't satisfy it because it never
    returns. When L4.1 lands, this becomes a real class with the same
    name; existing call sites need no changes.
    """
    raise NotImplementedError(
        "KuzuStore is planned for L4.1, not v0.1. "
        "For v0.1, use sqlite (default) or postgres (multi-user). "
        "See module docstring for design notes and rationale."
    )
