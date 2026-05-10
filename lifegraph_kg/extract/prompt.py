# SPDX-License-Identifier: Apache-2.0
"""Extraction + critic prompts.

Locked from preview-L1 v6 — empirically the strongest version across 6
prompt iterations on 30 real-data cases (22-8-0 vs production legacy,
0 substring violations, multi-predicate coverage on 13/30 cases). See
`tests/eval/fixtures/private/run_preview_l1_v6.py` for the iteration
history.

Both prompts use Python `str.format()` placeholders. The double-brace
`{{...}}` escapes preserve the JSON examples in the prompt body.
"""

from __future__ import annotations

EXTRACTION_PROMPT = """\
You extract structured information from a personal life-log entry into a
*personal knowledge graph* — PIMO/PKG-canon style. 4 node-classes plus
episode metadata. No Activity class.

## Node classes (the ONLY entity types)

1. **Person**  — a named human: Sara, 璐萌, mom, 老板, 宝贝
2. **Place**   — a named location: Ippudo, Harvard, 迈阿密, the studio
3. **Project** — a named ongoing initiative with multi-episode lifespan or
                 sub-task structure: TimeWises, MED24课题, album launch.
                 One-shot deliverables → Topic, not Project.
4. **Topic**   — catch-all for any other referent. Always set `kind`:
                  food | media | health | object | org | idea | general

## Episode-level metadata (NOT entities)

- `predicates` — a LIST of main verbs/actions in the entry, normalized
                 lowercase: ["ate", "met", "fixed", "reviewed", ...].
                 Multi-action entries get multiple predicates.
                 An entry with one verb gets a list of one.
                 Don't conflate distinct actions ("查看 X 并完成 Y" → 2 predicates).
- `body_state` — bodily state if explicitly mentioned: "tired", "累了",
                 "energized", "sick", null otherwise.
- `sentiment`  — overall affect, ONLY if the text explicitly signals it.
                 Use "pos" / "neu" / "neg" / null. **DEFAULT TO NULL** —
                 don't infer neutral from absence of explicit affect.
                 "had ramen" → null (no affect cue).
                 "felt great" → "pos". "感到失落" → "neg". "效率一般" → "neu".
- `energy`     — energy level ONLY if explicitly signaled. "high"/"medium"/"low"
                 /null. Default null. "激情" → "high". "累了" → "low".

These DO NOT become entities. Body/affect words go in metadata fields,
never in entities.

## Output JSON shape (no preamble, no fences)

```
{{"predicates": ["<verb1>", "<verb2>", ...],
  "body_state": "<state>"|null,
  "sentiment": "pos"|"neu"|"neg"|null,
  "energy": "high"|"medium"|"low"|null,
  "entities": [
     {{"type": "Person|Place|Project|Topic", "kind": "<discriminator>"|null,
       "value": "<verbatim surface form>", "key": "<lowercase canonical>"}}
  ]}}
```

## CRITICAL RULES

1. NEVER TRANSLATE. The `value` field MUST be a verbatim substring of source.
2. NEVER include a value not in the source.
3. Verbs go in `predicates` as a LIST. Each distinct action = one entry.
4. Body/affect words go in `body_state`/`sentiment`/`energy`, never in entities.
5. Default sentiment/energy to NULL — don't infer from absence.
6. Only extract entities with referential weight. Generic mentions of an
   action's object can become Topic entities, but only if they're worth
   remembering.

## Few-shot examples

### Example 1 — single action, social
Input: "Met Alex at Blue Bottle for coffee. Felt energized."
Output:
{{"predicates": ["met"], "body_state": "energized", "sentiment": "pos", "energy": "high",
  "entities": [
    {{"type": "Person", "value": "Alex", "key": "alex"}},
    {{"type": "Place", "value": "Blue Bottle", "key": "blue-bottle"}},
    {{"type": "Topic", "kind": "food", "value": "coffee", "key": "coffee"}}
  ]}}

### Example 2 — multi-action, work, no affect cue
Input: "查看 gene list 并为 Matt 标出重要的 gene"
Output:
{{"predicates": ["reviewed", "annotated"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Person", "value": "Matt", "key": "matt"}},
    {{"type": "Topic", "kind": "general", "value": "gene list", "key": "gene-list"}},
    {{"type": "Topic", "kind": "general", "value": "gene", "key": "gene"}}
  ]}}

### Example 3 — Chinese, body state
Input: "累了，回家喂小猫并休息一会儿"
Output:
{{"predicates": ["went-home", "fed", "rested"], "body_state": "累了", "sentiment": null, "energy": "low",
  "entities": [
    {{"type": "Topic", "kind": "general", "value": "小猫", "key": "小猫"}}
  ]}}

### Example 4 — work, multi-action, code-mixed
Input: "刚刚更新了slide for rise star presentation, 并邮件联系了Tao更改摘要"
Output:
{{"predicates": ["updated", "emailed"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Project", "value": "rise star presentation", "key": "rise-star-presentation"}},
    {{"type": "Person", "value": "Tao", "key": "tao"}},
    {{"type": "Topic", "kind": "object", "value": "slide", "key": "slide"}},
    {{"type": "Topic", "kind": "idea", "value": "摘要", "key": "摘要"}}
  ]}}

### Example 5 — explicit affect signal
Input: "激情改文章并发送给老板"
Output:
{{"predicates": ["revised", "sent"], "body_state": null, "sentiment": "pos", "energy": "high",
  "entities": [
    {{"type": "Person", "value": "老板", "key": "老板"}},
    {{"type": "Topic", "kind": "general", "value": "文章", "key": "文章"}}
  ]}}

### Example 6 — neutral activity, no affect inference
Input: "Had ramen with Sara at Ippudo"
Output:
{{"predicates": ["ate"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Person", "value": "Sara", "key": "sara"}},
    {{"type": "Place", "value": "Ippudo", "key": "ippudo"}},
    {{"type": "Topic", "kind": "food", "value": "ramen", "key": "ramen"}}
  ]}}

## Now extract from this input

Input: {text}
"""


CRITIC_PROMPT = """\
You are validating a personal-knowledge-graph extraction. Check for issues.

Source text:
{text}

Extraction:
{extraction}

Issues to flag:
1. **translation** — any value that is a translation of source (not verbatim).
   Note: predicate verbs ARE normalized to lowercase English (e.g. "fixed"
   for "修复"). Don't flag this — only flag translations of entity values.
2. **not_in_source** — any entity value that is not a substring of the source
3. **missing_action** — a clearly-distinct action in the source not captured
   in `predicates` (e.g. "查看 X 并完成 Y" should produce 2 predicates)
4. **wrong_type** — verbs in entities (should be predicates); body/affect
   words in entities (should be metadata fields)
5. **affect_hallucination** — sentiment/energy set to non-null when the
   source has no explicit affect signal
6. **missing_entity** — an important named referent (Person, Place, named
   Project, named Media) not extracted

Respond ONLY with JSON:
{{"valid": true|false, "issues": [{{"kind": "<one of above>", "detail": "<brief>"}}]}}
"""
