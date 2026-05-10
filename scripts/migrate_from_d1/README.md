# Migrate from existing LifeGraph (D1) → lifegraph-kg

Three scripts that take the existing LifeGraph product's data (Cloudflare
D1 SQLite, flat `life_items.nodes` schema) and migrate it into a fresh
lifegraph-kg-shaped database (4-class entities + episodes + bi-temporal
edges + episode metadata).

The migration **re-extracts** every entry through lifegraph-kg's locked
v6 prompt rather than mechanically converting the legacy node format —
this gives us the v6 quality wins (no-translate, multi-predicate,
body_state metadata, Topic.kind discriminator) on the historical data.

## Pipeline

```
old D1 (life_items)
   │
   │  pull_all_from_d1.py    (wrangler d1 execute --json)
   ↓
data/raw_d1_full.json
   │
   │  migrate_to_lifegraph_kg.py    (calls lifegraph_kg.LifeGraph.log())
   │  → re-extracts via v6 prompt + persists
   ↓
data/migrated.db    (lifegraph-kg SQLite, ready for inspection)
   │
   │  verify.py
   ↓
report (counts, samples, type distribution, body_state cases, etc.)
```

## Privacy

Real user data. The `data/` subdir is gitignored. Don't paste the
contents into chats or commit.

## Cost / time

For the user with ~437 logs:

- Data pull: ~30s (single wrangler call)
- Re-extraction: ~30 min wall clock (rate-limited Anthropic calls,
  Sonnet 4.6, ~$3 total cost)
- Verification: instant

## Usage

```bash
# 0. Pull data from old D1 (uses wrangler in the lifegraph repo)
uv run python scripts/migrate_from_d1/pull_all_from_d1.py \\
    --user-id e4a81cd7-d276-41ce-a5b6-f45856ac431b

# 1. Migrate (re-extract through v6 + persist to local SQLite)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/migrate_from_d1/migrate_to_lifegraph_kg.py

# 2. Verify
uv run python scripts/migrate_from_d1/verify.py
```

## Rollback

The original D1 is never modified. Rollback = delete `data/migrated.db`.
