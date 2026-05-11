# lifegraph-kg

> A personal knowledge graph for autobiographical data — Person, Place, Project, Topic, Episode — with bi-temporal edges and a built-in hygiene engine.

`lifegraph-kg` is a memory framework for LLM agents that work with **personal lived experience**. Type natural-language entries; it extracts typed entities, persists them in a knowledge graph with bi-temporal edges, and ships with a built-in hygiene engine for canonicalization and dedup.

It targets the canonical PKG ontology established by [PIMO (NEPOMUK)](https://oscaf.sourceforge.net/pimo.html) and [Balog & Kenter (2019)](https://krisztianbalog.com/files/ictir2019-pkg.pdf), with affect/body-state on episodes (Conway's Self-Memory System) — distinct from generic agent-memory frameworks that store "what the agent should remember about a user."

Status: **0.1.0-dev.** Functionally complete, eval-validated, not yet on PyPI.

## Quickstart

```python
from datetime import datetime, UTC
from lifegraph_kg import LifeGraph, Person, Place, Topic

# Default: in-memory SQLite. Pass store="sqlite:///me.db" for persistence,
# or store="postgres://user:pass@host/db" for shared deployments.
lg = LifeGraph()

# Ingest: extracts + persists.
ep = lg.log(
    "Had ramen with Sara at Ippudo. Felt tired.",
    occurred_at=datetime(2026, 5, 9, 19, 0, tzinfo=UTC),
)
# ep.predicates  → ["ate"]
# ep.body_state  → "tired"
# ep.entities are persisted; query them through lg.query()

# Query entities
sara = lg.query(Person, key="sara").one()
foods = lg.query(Topic, kind="food").all()

# Episode-level recall
sara_episodes = lg.episodes.mentioning(sara)
last_week = lg.episodes.since(datetime(2026, 5, 1, tzinfo=UTC))
in_may = lg.episodes.between(may_1, may_31)
food_timeline = lg.query(Topic, kind="food").episodes()

# Bi-temporal: time-travel queries
facts_now  = lg.kg.edges_as_of(datetime.now(UTC))
facts_then = lg.kg.edges_as_of(datetime(2025, 12, 1, tzinfo=UTC))

# Bi-temporal: supersede instead of delete
lg.kg.invalidate_edge(edge_id, datetime.now(UTC))

# Hygiene: dedup with audit trail
proposals = lg.hygiene.propose(type_="Person")
applied   = lg.hygiene.auto_apply()  # only high-confidence merges
```

## Why this and not Graphiti / Mem0 / Letta?

Those are excellent. This one targets a different problem.

| Goal | Best fit |
|---|---|
| Agent memory for a chatbot — preferences, procedures, working set | **Letta**, **Mem0**, **Zep** |
| Knowledge graph for any agent — bring your own ontology | **Graphiti** |
| Personal autobiographical knowledge graph — your life, indexed by time/place/person | **lifegraph-kg** |

The differentiation is concrete:

- **Default life ontology pre-configured.** Person/Place/Project/Topic with kind discriminators. No Pydantic boilerplate to start.
- **Bi-temporal edges with audit-preserving supersede.** Same model as Graphiti, but on SQLite-default deploy.
- **Hygiene engine.** Heuristic dedup with proposal/apply pipeline; the loser entity stays in the DB as an alias (canonical_id), not a deletion. v0.2 adds embedding-based fuzzy match.
- **Anchored to PKG canon.** The ontology converges on the PIMO classes from 2006 + the [Balog & Kenter 2019](https://krisztianbalog.com/files/ictir2019-pkg.pdf) PKG paper + Conway's Self-Memory System — not invented for this project.

## What gets stored

```text
Episode  { text, occurred_at, source, predicates: list, body_state, sentiment, energy }
   │
   ├─[mentions]─→  Person, Place, Project, Topic{kind}   (4 entity classes)
   │
   └─[verb-edges]─→ same entities, bi-temporal:
                    t_event, t_ingestion, t_valid, t_invalid
                    (NULL t_invalid = currently valid)
```

| Layer | Why |
|-------|-----|
| **Episode** | The unit of record. DOLCE Perdurant; CIDOC E5 Event; NTCIR Moment; Conway ESK. The text + when + provenance. |
| **Entities** (Person/Place/Project/Topic) | The typed nodes — what you query against. PIMO + Balog canon. |
| **Edges** (verb-as-edge) | Bi-temporal; supersede instead of delete; preserves the audit trail. |
| **`predicates`, `body_state`, `sentiment`, `energy`** | Episode metadata, NOT separate node-classes. Affect is a feature of the episode (Conway), not a participant. |

## Storage backends

Same Store protocol across backends. Pick by URI:

| Backend | URI | Notes |
|---|---|---|
| **SQLite** (default) | `:memory:` or `sqlite:///path` | Zero-ops, single file. stdlib only. |
| **Postgres** | `postgres://user:pass@host/db` | `pip install 'lifegraph-kg[postgres]'`. Multi-user / shared deployments. PGlite (WASM Postgres) speaks the same protocol. |
| **Postgres + [Apache AGE](https://age.apache.org)** (v0.2) | same DSN | Native graph traversal — Cypher path queries on the same connection. |
| **Neo4j** (future) | — | For Graphiti users; depends on demand. |

## Roadmap

- **v0.1** — Extraction + episodic store + bi-temporal CRUD + hygiene engine. **Current state.**
- **v0.2** — Procedural memory as **human-habit extraction** ("you eat ramen on Tuesdays" mined from the episode stream). The differentiator vs. agent-memory frameworks, which use procedural memory for agent self-improvement. Plus embedding-based fuzzy hygiene.
- **v0.3** — Narrative views. Generated stories ("your year in food", "your week with Sara") synthesizing episodes.
- **TS mirror** — `@lifegraph/kg` on npm, with PGlite as the default storage backend (browser-compatible).

## Influences (cited inline in source)

| Source | What we took |
|---|---|
| **PIMO** (NEPOMUK 2006) | Person/Location/Project/Topic/Event as siblings — the original PKG canon |
| **Balog & Kenter (2019)** | The 3-class PKG agenda (agents, events, locations); confirmed our 4-class plan as canonical |
| **Conway's Self-Memory System (2000)** | Autobiographical memory layered as Lifetime Periods → General Events → ESK; affect as episode feature |
| **DOLCE / CIDOC-CRM** | Endurant/Perdurant split; E21 Person, E53 Place, E5 Event |
| **Graphiti** (Zep) | Episode model, bi-temporal edge invalidation, multi-backend driver pattern |
| **LangExtract** (Google) | The few-shot examples authoring surface, character-interval source grounding |
| **Memori** | The SQL-first thesis: for personal-scale data, p50 latency matters more than cluster scale |

Full lineage in [NOTICE](NOTICE).

## Methodology

The library was built TDD-first against a 6-iteration prompt-engineering cycle on **30 real-data cases**. The locked extractor prompt wins **22-8-0 vs the production legacy extractor** under LLM-as-judge head-to-head, with **0 substring violations** (no translation drift). The eval framework lives in [`tests/eval/`](tests/eval/) and includes scoring functions for extraction quality, bi-temporal CRUD correctness, hygiene precision/recall, recall@K for time/place/person queries, and performance.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the eval workflow.

## Install

```bash
# v0.1 dev: clone + install via uv
git clone https://github.com/<your>/lifegraph-kg
cd lifegraph-kg
uv sync --extra dev
```

After PyPI publish (coming soon):

```bash
pip install lifegraph-kg                          # SQLite-only
pip install 'lifegraph-kg[postgres]'              # + psycopg
```

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attributions.

## Name disambiguation

The name "LifeGraph" was previously used in Tominski & Aigner (2020), "LifeGraph: A Knowledge Graph for Lifelogs" (ACM). That work is a research artifact, not a product. This package is unrelated.
