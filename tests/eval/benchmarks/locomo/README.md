# LoCoMo adapter (planned)

Status: **stub.** Phase target: L5.

[LoCoMo](https://snap-research.github.io/locomo/) is the SIGIR-25 benchmark for long-conversation memory: multi-session dialogues with annotated questions probing temporal reasoning, single/multi-hop fact recall, and adversarial noise.

## What this adapter will do (L5)

1. Load the LoCoMo dataset.
2. For each conversation, ingest the dialogue turns as episodes (`source="dialogue"`).
3. For each question, query `lifegraph-kg` and assemble an answer.
4. Score using the LoCoMo rubric (LLM-as-judge against gold answers, plus type-stratified accuracy).

## Why we care

LoCoMo questions are stratified by reasoning type (single-hop, multi-hop, temporal, open-domain, adversarial). The temporal stratum is where our bi-temporal store should win against vector-only competitors — this benchmark validates that.

Until L5, this directory is a placeholder.
