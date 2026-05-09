# lifegraph-kg

> An autobiographical memory engine — typed schema with default life classes, indexed by time, place, and person.

`lifegraph-kg` is a memory framework for LLM agents that work with **personal lived experience**. It extracts typed entities from natural-language logs, stores them in a knowledge graph with bi-temporal edges, and ships with a built-in hygiene engine for canonicalization and dedup.

It's distinct from generic agent memory frameworks (Letta, Mem0) and generic agent knowledge graphs (Graphiti) in three ways:

1. **Default life schema.** Out of the box: `Person`, `Place`, `Activity`, `Food`, `Topic`, `Media`, `Health`, `Project`. No Pydantic boilerplate to use the package.
2. **Autobiographical indices.** Time, place, and person are first-class indices on every memory — matching how humans actually recall lived events (Conway 2000; Pink et al. 2025).
3. **Hygiene engine.** Canonical IDs, embedding-based dedup, sentiment-aware normalization. The thing that keeps your graph from drifting into noise.

Status: **pre-alpha.** Not yet on PyPI. APIs will change.

## Quickstart

```python
from lifegraph_kg import LifeGraph, classes as c

lg = LifeGraph(store="sqlite:///me.db")
lg.register(c.Person, c.Place, c.Activity, c.Food)
lg.register_user_class("Pet", parent=c.Entity, fields={"species": str, "name": str})

# Ingest: stores the episode AND extracts entities AND edges AND grounding
lg.log("Had ramen with Sara at Ippudo. Bumi (cat) seemed grumpy.",
       at="2026-05-09T19:00", source="telegram")

# Typed entity query
hits = lg.query(c.Person, name="Sara").related(c.Activity).since("2026-01-01")

# Episode-level recall
recent = lg.episodes.since("2026-05-01")
sara_episodes = lg.episodes.mentioning(c.Person, name="Sara")

# Time-travel
fact = lg.kg.where(c.Person, name="Sara").lives_in.as_of("2025-12-01")  # Berlin
fact = lg.kg.where(c.Person, name="Sara").lives_in.as_of("now")          # Tokyo
```

## What it stores (three layers + grounding)

| Layer | What | Why store it |
|-------|------|--------------|
| **Episodes** | Raw entries — original text + timestamp + source. | Audit, re-extraction when prompts improve, raw-recall. |
| **Entities** | Extracted typed nodes (Person:Sara, Place:Ippudo, …). | The typed knowledge graph; what you query against. |
| **Edges** | Relationships, bi-temporal: event-time `T`, ingestion-time `T'`, validity `(t_valid, t_invalid)`. | The "memory" — what was true when. |
| **Grounding** | `(entity_id, episode_id, char_start, char_end)` — every entity points back to the source span. | Audit, debug, UI highlights — without this, you have memories you can't trace. |

## Roadmap

- **v0.1** — Episodic + semantic memory. The package described above. (in development)
- **v0.2** — **Procedural memory as human-habit extraction.** "You eat ramen on Tuesdays" mined from the episode stream. Every other framework uses procedural memory for *agent* self-improvement; we use it for *human* habits. This is the long-term differentiator.
- **v0.3** — Narrative views. Generated stories ("your year in food", "your week with Sara") synthesized from episodes.
- **TS mirror** — `@lifegraph/kg` on npm, with PGlite as the default storage backend (browser-compatible, pgvector built-in).

## Storage backends

Same SQL across all backends (driver pattern, inspired by Graphiti):

| Backend | Status | Notes |
|---------|--------|-------|
| SQLite | default | zero-ops, single file, sub-50ms p50 |
| Postgres | optional | for shared deployments; covers PGlite via PG protocol |
| Kuzu | optional | embedded graph DB; richer graph queries (Cypher) |
| Neo4j | future | for users coming from Graphiti |

## Authoring extractions

Two paths into the same extractor — pick whichever matches your team:

**LangExtract-style few-shot (low ceremony, end-user friendly):**

```python
from lifegraph_kg.extract import examples

lg.log(text, examples=[
    examples.ExampleData(
        text="Met Alex for coffee at Blue Bottle.",
        extractions=[
            examples.Extraction(class_=c.Person, text="Alex"),
            examples.Extraction(class_=c.Place, text="Blue Bottle"),
            examples.Extraction(class_=c.Activity, text="coffee"),
        ],
    ),
])
```

**Pydantic schemas (Graphiti-style, type-safe):**

```python
from lifegraph_kg import classes as c
# All built-in classes are already Pydantic models. Subclass to extend:

class Pet(c.Entity):
    species: str
    name: str

lg.register(Pet)
```

## Influences

`lifegraph-kg` draws on prior art it tries to credit clearly:

- **Cognitive psych:** Conway (2000), the Self-Memory System model of autobiographical memory.
- **Recent LLM papers:** Pink et al. 2025 (arXiv:2502.06975) on episodic memory for long-term agents; the Episodic Memories Generation Benchmark (arXiv:2501.13121) defining episodes as `(time, space, entities, content)` 4-tuples; Sumers et al. 2023 (CoALA, arXiv:2309.02427) for the three-tier (episodic/semantic/procedural) cognitive model.
- **Engineering precedents:** Graphiti (Apache-2.0) for the episode model and bi-temporal edge invalidation; LangExtract (Apache-2.0) for the few-shot authoring surface and char-interval grounding; Memori for the SQL-first thesis.

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attributions.

## Name disambiguation

The name "LifeGraph" was previously used in Tominski & Aigner (2020), "LifeGraph: A Visual Analytics System for Personal Data" (ACM). That work is a research artifact, not a product. This package is unrelated.
