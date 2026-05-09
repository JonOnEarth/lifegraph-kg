# Episodic Memories Generation Benchmark (EpMG) adapter (planned)

Status: **stub.** Phase target: L5.

[Episodic Memories Generation Benchmark](https://arxiv.org/abs/2501.13121) (arXiv:2501.13121, 2025) defines an episode as a `(time, space, entities, content)` 4-tuple — exactly the framing this library leans into. Their benchmark generates synthetic episodes and tests recall against TPP-shaped queries.

## Why this benchmark in particular

EpMG is the benchmark that operationalizes the autobiographical-memory framing in the academic literature. Scoring well on it directly substantiates the package's positioning. It's also synthetic and freely redistributable (unlike LongMemEval/LoCoMo which are gated by license terms), which makes it the cheapest external benchmark to run.

## What this adapter will do (L5)

1. Generate or download the EpMG synthetic-episode set.
2. Ingest episodes via `lg.log(text, at=..., source="epmg-synthetic")`.
3. For each query (TPP-shaped), call the corresponding `lg.episodes.where(...)` API.
4. Score using EpMG's recall@K and exact-match rubrics, plus our `recall_summary()`.

Until L5, this directory is a placeholder.
