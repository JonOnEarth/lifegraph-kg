# lifegraph-kg examples

Four runnable scripts demonstrating the v0.1 API. Run from the repo root.

| # | File | Demonstrates |
|---|------|--------------|
| 1 | [`01_basic_log.py`](01_basic_log.py) | Log one entry; query Person/Place/Topic; episodes.mentioning |
| 2 | [`02_multi_episode.py`](02_multi_episode.py) | A week of entries; time-range queries; `query(Topic, kind="food").episodes()` pivot |
| 3 | [`03_bi_temporal.py`](03_bi_temporal.py) | Sara-moves-to-Tokyo: supersede + time-travel + audit-trail |
| 4 | [`04_hygiene.py`](04_hygiene.py) | Heuristic dedup proposals; auto-apply; canonical_id alias preserved |

## Running

Examples 1–3 need an Anthropic API key (the extractor calls Sonnet):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python examples/01_basic_log.py
uv run python examples/02_multi_episode.py
uv run python examples/03_bi_temporal.py
```

Example 4 uses the FakeClient pattern from tests so it runs without an API key:

```bash
uv run python examples/04_hygiene.py
```

## Cost

Each LLM call is roughly $0.005–$0.01 with Sonnet 4.6 at v6 prompt size.
Running examples 1–3 is well under $0.10.
