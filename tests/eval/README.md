# lifegraph-kg evaluation suite

Six categories of metrics, with golden-set fixtures and pure-Python scoring functions. Built TDD-style: every L1+ phase has a concrete eval target from day one.

## What's measured

| Category | Module | Phase target | Key metrics |
|----------|--------|--------------|-------------|
| **Extraction quality** | `scoring/extraction.py` | L1 | Entity F1 (micro + per-class); type accuracy; grounding overlap (IoU on char-intervals) |
| **Temporal CRUD** | `scoring/temporal.py` | L2 | Supersede correctness; `as_of` accuracy; invalidate-not-delete (history preserved) |
| **Hygiene** | `scoring/hygiene.py` | L3 | Dedup precision/recall; false-merge rate; grounding survival across merges |
| **Recall (TPP)** | `scoring/recall.py` | L2 | Precision@K, recall@K, MRR for time/place/person queries; latency p50/p95 |
| **Performance** | `scoring/perf.py` | L1+ | Log/query/episode latency p50/p95; throughput; memory at 10K / 100K / 1M |
| **Cross-framework** | `benchmarks/{longmemeval,locomo,epmg}/` | L5 | LongMemEval, LoCoMo, EpMG benchmark scores |

For categories with crisp golden answers (entity F1, latency), scoring is rule-based and deterministic. For open-ended judgments ("is this canonicalized name reasonable?"), an opt-in **LLM-as-judge** in `scoring/llm_judge.py` provides scores with model + temperature pinned for reproducibility.

## Running evals

```bash
# Run a single category
uv run python -m tests.eval.runners.run_extraction

# Run everything; emit a JSON report
uv run python -m tests.eval.runners.run_all --output tests/eval/reports/$(date +%Y-%m-%d).json

# Run with LLM judge enabled (requires ANTHROPIC_API_KEY)
LIFEGRAPH_EVAL_LLM_JUDGE=1 uv run python -m tests.eval.runners.run_all
```

Stubs error gracefully when their target phase isn't shipped yet — they emit `{"status": "not_yet_implemented", "phase": "L1"}` rather than crashing.

## Fixture format

All fixtures are JSON, validated against Pydantic schemas in `types.py` at load time. Each category has its own schema. Fixtures live in `fixtures/<category>/<scenario>.json` and follow this top-level structure:

```jsonc
{
  "id": "extraction-basic-001",
  "description": "Single sentence with one Person and one Place",
  "category": "extraction",
  "input": { ... },          // category-specific
  "expected": { ... },       // category-specific
  "metadata": {
    "source": "synthetic",
    "license": "Apache-2.0"
  }
}
```

## Synthetic-only data

All fixtures are programmatically generated or hand-authored toy data — no real user logs. This keeps the eval suite redistributable, deterministic, and privacy-clean. If you want to validate against your own data, see [docs/private_evals.md](../../docs/private_evals.md) (TBD).

## Reports

`runners/run_all.py` emits a typed `EvalReport` with per-category metrics, library version, timestamp, and (if LLM judge ran) judge model + temperature. Reports for each release are committed to `reports/` as a regression baseline.

```jsonc
{
  "lifegraph_kg_version": "0.0.1.dev0",
  "timestamp": "2026-05-09T11:00:00Z",
  "categories": {
    "extraction": { "f1": 0.92, "type_accuracy": 0.97, "grounding_iou": 0.88, "fixtures_run": 42 },
    "temporal":   { "supersede_correctness": 1.0, "as_of_accuracy": 0.95, "fixtures_run": 18 },
    ...
  },
  "ci_gate_pass": true
}
```

## Cross-framework benchmark adapters

Defer to L5. Each adapter is a thin shim that translates the external benchmark's input format into `lg.log(...)` calls and the benchmark's expected-output format into one of our scoring categories. Running these requires also running competitor libraries side-by-side; see `benchmarks/<name>/README.md` (per-adapter setup).

## Contributing

To add a new fixture:
1. Drop a JSON file in `fixtures/<category>/<scenario>.json` matching the category's Pydantic schema.
2. Run `uv run python -m tests.eval.runners.run_<category> --validate-only` to confirm it loads.
3. Open a PR — CI will pick up the new fixture automatically.
