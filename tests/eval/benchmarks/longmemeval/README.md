# LongMemEval adapter (planned)

Status: **stub.** Phase target: L5.

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) is a benchmark for long-term memory in LLM-based assistants — multi-session conversations where the assistant must recall facts from earlier sessions. It's the standard benchmark Mem0, Letta, and Zep cite in their papers.

## What this adapter will do (L5)

1. Download the LongMemEval dataset (subject to its license).
2. For each session in the dataset, replay the user-side messages as `lg.log(...)` calls.
3. For each evaluation question, query the `lifegraph-kg` store and assemble an answer (likely via the `lifegraph_kg.kg` query API + a thin LLM wrapper).
4. Score against the dataset's gold answers using the LongMemEval rubric (typically LLM-as-judge with a strong model).

## Why we care

Mem0 reports +26% LLM-as-Judge accuracy and 91% lower p95 latency vs OpenAI Memory on LongMemEval ([Mem0 blog](https://mem0.ai/blog/llm-judge-vs-memory-search/)). To position `lifegraph-kg` against that, we need the same benchmark on the same questions. Apples-to-apples requires running Mem0 and Letta side-by-side; that's the L5 setup work.

## Setup (when implemented)

```bash
# Download dataset (one-time)
uv run python -m tests.eval.benchmarks.longmemeval.download

# Run lifegraph-kg
uv run python -m tests.eval.benchmarks.longmemeval.run --library lifegraph-kg

# Run a competitor for comparison
uv run python -m tests.eval.benchmarks.longmemeval.run --library mem0
```

Until L5, this directory is a placeholder.
