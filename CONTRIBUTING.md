# Contributing to lifegraph-kg

Thanks for considering it. This file covers the dev workflow and the
two things that make this project unusual: the **eval-driven design loop**
and the **canonical-ontology constraint**.

## Dev setup

```bash
git clone <this-repo>
cd lifegraph-kg
uv sync --extra dev               # core + dev tools
uv sync --extra dev --extra postgres   # + psycopg for Postgres tests
```

## The four CI gates (all must pass)

```bash
uv run ruff check .               # lint
uv run ruff format --check .      # format
uv run mypy lifegraph_kg tests    # strict type-check
uv run pytest                     # tests
```

A pre-merge run of all four is reasonable:

```bash
uv run ruff check . && uv run ruff format --check . \
  && uv run mypy lifegraph_kg tests && uv run pytest
```

## Eval-driven design

This project doesn't ship prompt or schema changes without measuring.
The methodology that built v0.1 was:

1. **Author a hypothesis** ("dropping Activity from the ontology should
   reduce node-count without losing query power").
2. **Pick a fixture set** that exercises the change. Synthetic for
   reproducibility, real-data (gitignored under `tests/eval/fixtures/private/`)
   for high-fidelity signal.
3. **Run the eval framework**:
   ```bash
   uv run python -m tests.eval.runners.run_all --validate-only
   uv run python -m tests.eval.runners.run_all
   ```
4. **For prompt changes specifically**, run head-to-head LLM-as-judge
   between the new and old version with randomized A/B order
   (see the `private/run_judge_*.py` scripts for the pattern).
5. **Diff metrics, write up the loss cases**, decide whether to ship.

The story matters: v3 of the extractor prompt regressed from v2 because
we over-engineered. v4 was a minimal patch on v2; v5 changed schema and
won on most cases but lost on multi-action; v6 fixed multi-predicate
support and locked. Each iteration was driven by **3-5 specific case
failures the judge surfaced** — not by guessing.

## The canonical-ontology constraint

The 4-class ontology (Person / Place / Project / Topic) and "verbs as
predicates / edges, not nodes" decision are **anchored to research**, not
invented for this project. Adding a new node-class needs a citation:

| If you want to add | Cite |
|---|---|
| A new entity class | Either show it's in PIMO / FOAF / CIDOC-CRM, or cite a personal-KG paper that adds it |
| A new Topic `kind` value | Make the case it's worth distinguishing in queries (a discriminator on Topic is cheap) |
| Promoting Topic{kind:health} to its own Health class | Show a Personal Health KG paper that justifies it |

Episode metadata (`body_state`, `sentiment`, `energy`) is the right
home for affect, body-state, and other "features of the episode itself"
in Conway's Self-Memory System sense. Don't add them as entity classes.

## Adding a new test fixture

```bash
# 1. Drop a JSON file in tests/eval/fixtures/<category>/<scenario>.json
# 2. Validate it loads against the Pydantic schema:
uv run python -m tests.eval.runners.run_extraction --validate-only

# 3. Run the runner to score against the library:
ANTHROPIC_API_KEY=sk-ant-... uv run python -m tests.eval.runners.run_extraction
```

CI picks up new fixtures automatically.

## Adding a new prompt iteration

Don't replace `lifegraph_kg/extract/prompt.py` directly. The pattern is:

1. Copy the current prompt to a private exploration script
   (template: `tests/eval/fixtures/private/run_preview_l1_v6.py`)
2. Iterate against the 30 real-data fixtures + synthetic
3. Run head-to-head judge against the previous version
4. Only after a win, port the new prompt to `extract/prompt.py`
5. Update tests/test_extraction.py if behavior changed
6. Update CHANGELOG.md

## Adding a new Store backend

Implement the `Store` Protocol in `lifegraph_kg/kg/store/<name>.py`. Add
a URI dispatch in `lifegraph_kg.kg._resolve_store`. Mirror the
SqliteStore tests (in `tests/test_kg_store.py`) — the protocol is the
contract; behaviors must match.

For backends that need running infrastructure (Postgres, Neo4j),
gate the integration tests on an env var (see the
`LIFEGRAPH_TEST_POSTGRES_URL` pattern in `tests/test_storage_drivers.py`).

## Pull request checklist

- [ ] All four CI gates green
- [ ] CHANGELOG.md entry in the `[Unreleased]` section
- [ ] If touching the ontology / schema: cite the canon source
- [ ] If touching the extractor prompt: include a head-to-head judge run
- [ ] If touching the public API: update README quickstart accordingly

## Contact

Issues and PRs on GitHub. For larger architectural conversations, file
an issue first to align before writing code.
