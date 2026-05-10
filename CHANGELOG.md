# Changelog

All notable changes to `lifegraph-kg` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0-dev] — 2026-05-10 (in development)

The first functionally complete release. Extraction + episodic store +
bi-temporal CRUD + hygiene engine, anchored to the canonical PKG ontology
established by PIMO (NEPOMUK) and the [Balog & Kenter 2019](https://krisztianbalog.com/files/ictir2019-pkg.pdf)
PKG agenda.

### Added — Library

- **4-class default ontology** ([`lifegraph_kg.classes`](lifegraph_kg/classes.py)):
  Person, Place, Project, Topic with `kind` discriminator (food / media /
  health / object / org / idea / general). Frozen Pydantic models.
- **Extraction pipeline** ([`lifegraph_kg.extract`](lifegraph_kg/extract/)):
  v6 prompt with 6 inline few-shot examples (EN, CN, code-mixed) + substring
  assertion (no-translate enforcement) + Haiku critic pass. `extract(text)`
  returns `ExtractionResult` with predicates, body_state, sentiment, energy,
  and 4-class typed entities.
- **LLM provider abstraction** ([`lifegraph_kg.llm`](lifegraph_kg/llm/)):
  `LlmClient` Protocol with Anthropic default; tests inject mocks.
- **SQLite store** ([`lifegraph_kg.kg.store.sqlite`](lifegraph_kg/kg/store/sqlite.py)):
  default backend. Stdlib `sqlite3`, no third-party SQL deps. WAL +
  foreign keys. Two migrations (initial schema + hygiene additions).
- **Postgres store** ([`lifegraph_kg.kg.store.postgres`](lifegraph_kg/kg/store/postgres.py)):
  opt-in via `[postgres]` extra. Same Store protocol; same tests should
  pass. BIGINT for unix-ms timestamps (Y2038-safe).
- **Bi-temporal edges** ([`lifegraph_kg.kg.edge`](lifegraph_kg/kg/edge.py)):
  `t_event`, `t_ingestion`, `t_valid`, `t_invalid` columns. Supersede via
  `lg.kg.invalidate_edge()`; time-travel queries via `lg.kg.edges_as_of(t)`.
  Audit trail preserved (loser edges are invalidated, never deleted).
- **Episode model** ([`lifegraph_kg.kg.episode`](lifegraph_kg/kg/episode.py)):
  raw text + occurred_at + ingested_at + source + predicates list +
  scalar metadata (body_state / sentiment / energy). Conway's affect-as-feature.
- **LifeGraph facade** ([`lifegraph_kg.kg.__init__`](lifegraph_kg/kg/__init__.py)):
  `lg.log(text)` extracts + persists; `lg.query(Type, key=...)` returns
  `_EntityQuery` with `.one()` / `.all()` / `.first()` / `.episodes()`;
  `lg.episodes.{get, since, between, mentioning}`; `lg.kg.{invalidate_edge,
  edges_as_of, edges_for_episode}`; `lg.hygiene.{propose, apply, auto_apply}`.
- **Hygiene engine** ([`lifegraph_kg.hygiene`](lifegraph_kg/hygiene/)):
  heuristic dedup with three rules — `exact_normalized` (high confidence,
  auto-apply), `substring_qualifier` (Ippudo / Ippudo NYC), `edit_distance`
  (Damerau-Levenshtein ≤ 1, length ≥ 5, ASCII). Adversarial cases
  protected: never merges across types, never across Topic kinds, never
  on short strings. Apply path: redirect edges + mentions, set
  `canonical_id` on the loser entity (audit-preserving).

### Added — Eval framework

- **6-category eval suite** ([`tests/eval/`](tests/eval/)) with golden-set
  fixtures and pure-Python scoring: extraction (entity F1, type accuracy,
  grounding IoU, predicate F1, episode-metadata accuracy), temporal CRUD
  (as_of accuracy, supersede correctness, invalidate-not-delete), hygiene
  (dedup precision/recall, false-merge rate, grounding survival), recall
  (P@K, R@K, MRR for TPP queries), performance (latency stats, throughput,
  memory growth alpha), and cross-framework benchmark adapter shells.
- **Opt-in LLM-as-judge** ([`tests/eval/scoring/llm_judge.py`](tests/eval/scoring/llm_judge.py))
  for open-ended quality judgments. Pinned model + temperature for
  reproducibility.
- **Real-data validation cycle** (gitignored under `tests/eval/fixtures/private/`):
  6 prompt iterations on 30 real bilingual life-log entries, head-to-head
  judge runs across legacy / v1-v6. Locked v6 wins **22-8-0 vs production
  legacy** with **0 substring violations**.

### Added — Documentation

- README with canonical-PKG-ontology positioning, quickstart, three-layer
  storage model, storage-backend table.
- NOTICE with full attribution: Conway 2000, Pink 2025, EpMG benchmark,
  CoALA, Graphiti (Apache-2.0), LangExtract (Apache-2.0), Memori, the
  LifeGraph 2020 ACM disambiguation footnote.

### Decisions captured

- **Activity dropped as a node-class.** Verbs become edge labels (or
  `Episode.predicate`); Episode IS the activity instance. Matches every
  production memory framework + cognitive psych canon.
- **Health/Mood/BodyState as Episode metadata** instead of separate
  classes. Conway: affect is a feature of the episode, not a participant.
- **Topic with `kind` discriminator** absorbs Food / Media / named-Health /
  Object / Org / Idea. Cheap migration when a discriminator value
  outgrows Topic.
- **Apache 2.0 license** for the library (matches Graphiti). The
  separate (planned) platform repo will use AGPL.

### Removed (during development)

- Kuzu storage driver (project archived October 2025). Native graph
  traversal in v0.2 will go via Apache AGE on Postgres — same DSN.

### Not yet (v0.2+)

- Procedural-tier extraction (human habits from the episode stream)
- Embedding-based fuzzy hygiene
- TypeScript mirror (`@lifegraph/kg`)
- Apache AGE integration for native graph traversal
- LongMemEval / LoCoMo / EpMG benchmark adapters

[Unreleased]: https://github.com/lifegraph-os/lifegraph-kg/compare/v0.1.0...HEAD
[0.1.0-dev]: https://github.com/lifegraph-os/lifegraph-kg/releases/tag/v0.1.0
