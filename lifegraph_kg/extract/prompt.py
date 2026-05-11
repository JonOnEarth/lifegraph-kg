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
- `duration`   — minutes the entry took. Trust the text: convert any
                 stated duration in any language ("30分钟", "half an hour",
                 "1h", "26:14") to integer minutes. If the user did NOT
                 state a duration but the activity has a conventional one,
                 fall back to a calibration anchor and set
                 ``duration_inferred=true``. Anchors:
                   meal=30, workout=45, meeting=30, call=15-20, email=10,
                   errand=30, shower=15, sleep=480, commute=20.
                 Pure observations / emotions / open-ended thinking → null
                 (and duration_inferred must also be null — never set
                 duration_inferred=true with duration=null).
- `duration_inferred` — true when ``duration`` came from a calibration
                        anchor (UI renders "~30 min"). False when the
                        user explicitly stated it. Null when duration is
                        null.

Sentiment / energy / duration DO NOT become entities. They're metadata.

## Output JSON shape (no preamble, no fences)

```
{{"predicates": ["<verb1>", "<verb2>", ...],
  "body_state": "<state>"|null,
  "sentiment": "pos"|"neu"|"neg"|null,
  "energy": "high"|"medium"|"low"|null,
  "duration": <minutes int>|null,
  "duration_inferred": true|false|null,
  "time_mode": "absolute"|"floating"|null,
  "wall_clock_hour": 0-23|null,
  "wall_clock_minute": 0-59|null,
  "wall_clock_date": "MM-DD"|null,
  "entities": [
     {{"type": "Person|Place|Project|Topic", "kind": "<discriminator>"|null,
       "value": "<verbatim surface form>", "key": "<lowercase canonical>"}}
  ]}}
```

## time_mode (only for forward-looking intents — logs leave null)

- This entry is a **past event / observation** (logs): ``time_mode``
  MUST be null. wall_clock_* all null. Most inputs fall here.
- This entry is a **future intent / task** with a fixed physical moment
  (specific meeting, DDL, calendar event): ``time_mode = "absolute"``.
  wall_clock_* all null. The deadline phrase lives in the source text.
  Examples: "submit report by Friday", "lunch with Tom at 12pm tomorrow",
  "flight at 6:30am".
- This entry is a **future intent that follows the user's wall-clock**
  wherever they go (daily/weekly routine, habit, recurring chore):
  ``time_mode = "floating"`` AND ``wall_clock_hour`` + ``wall_clock_minute``
  MUST be filled from the explicit/implicit clock time. ``wall_clock_date``
  = "MM-DD" only for yearly anchors. Examples: "every day 8am meditate",
  "weekday morning workout", "call mom every Dec 25".

When in doubt between absolute and floating, prefer absolute (fixed
moments don't drift across timezones).

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

### Example 1 — single action, social, inferred duration
Input: "Met Alex at Blue Bottle for coffee. Felt energized."
Output:
{{"predicates": ["met"], "body_state": "energized", "sentiment": "pos", "energy": "high",
  "duration": 30, "duration_inferred": true,
  "time_mode": null, "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Person", "value": "Alex", "key": "alex"}},
    {{"type": "Place", "value": "Blue Bottle", "key": "blue-bottle"}},
    {{"type": "Topic", "kind": "food", "value": "coffee", "key": "coffee"}}
  ]}}

### Example 2 — multi-action, work, no duration cue
Input: "查看 gene list 并为 Matt 标出重要的 gene"
Output:
{{"predicates": ["reviewed", "annotated"], "body_state": null, "sentiment": null, "energy": null,
  "duration": null, "duration_inferred": null,
  "time_mode": null, "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Person", "value": "Matt", "key": "matt"}},
    {{"type": "Topic", "kind": "general", "value": "gene list", "key": "gene-list"}},
    {{"type": "Topic", "kind": "general", "value": "gene", "key": "gene"}}
  ]}}

### Example 3 — Chinese, body state, no duration
Input: "累了，回家喂小猫并休息一会儿"
Output:
{{"predicates": ["went-home", "fed", "rested"], "body_state": "累了", "sentiment": null, "energy": "low",
  "duration": null, "duration_inferred": null,
  "time_mode": null, "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Topic", "kind": "general", "value": "小猫", "key": "小猫"}}
  ]}}

### Example 4 — work, multi-action, code-mixed
Input: "刚刚更新了slide for rise star presentation, 并邮件联系了Tao更改摘要"
Output:
{{"predicates": ["updated", "emailed"], "body_state": null, "sentiment": null, "energy": null,
  "duration": null, "duration_inferred": null,
  "time_mode": null, "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Project", "value": "rise star presentation", "key": "rise-star-presentation"}},
    {{"type": "Person", "value": "Tao", "key": "tao"}},
    {{"type": "Topic", "kind": "object", "value": "slide", "key": "slide"}},
    {{"type": "Topic", "kind": "idea", "value": "摘要", "key": "摘要"}}
  ]}}

### Example 5 — explicit affect signal, stated duration
Input: "激情改文章一个小时并发送给老板"
Output:
{{"predicates": ["revised", "sent"], "body_state": null, "sentiment": "pos", "energy": "high",
  "duration": 60, "duration_inferred": false,
  "time_mode": null, "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Person", "value": "老板", "key": "老板"}},
    {{"type": "Topic", "kind": "general", "value": "文章", "key": "文章"}}
  ]}}

### Example 5b — stated duration in English
Input: "30-min coffee with Sarah this morning"
Output:
{{"predicates": ["had"], "body_state": null, "sentiment": null, "energy": null,
  "duration": 30, "duration_inferred": false,
  "time_mode": null, "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Person", "value": "Sarah", "key": "sarah"}},
    {{"type": "Topic", "kind": "food", "value": "coffee", "key": "coffee"}}
  ]}}

### Example 6 — neutral activity, inferred meal duration
Input: "Had ramen with Sara at Ippudo"
Output:
{{"predicates": ["ate"], "body_state": null, "sentiment": null, "energy": null,
  "duration": 30, "duration_inferred": true,
  "time_mode": null, "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Person", "value": "Sara", "key": "sara"}},
    {{"type": "Place", "value": "Ippudo", "key": "ippudo"}},
    {{"type": "Topic", "kind": "food", "value": "ramen", "key": "ramen"}}
  ]}}

### Example 7 — task, absolute (specific deadline)
Input: "Email Tao the Q3 report by Friday"
Output:
{{"predicates": ["email"], "body_state": null, "sentiment": null, "energy": null,
  "duration": null, "duration_inferred": null,
  "time_mode": "absolute", "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null,
  "entities": [
    {{"type": "Person", "value": "Tao", "key": "tao"}},
    {{"type": "Topic", "kind": "object", "value": "Q3 report", "key": "q3-report"}}
  ]}}

### Example 8 — task, floating (daily routine)
Input: "每天 8 点吃降压药"
Output:
{{"predicates": ["take"], "body_state": null, "sentiment": null, "energy": null,
  "duration": null, "duration_inferred": null,
  "time_mode": "floating", "wall_clock_hour": 8, "wall_clock_minute": 0, "wall_clock_date": null,
  "entities": [
    {{"type": "Topic", "kind": "health", "value": "降压药", "key": "降压药"}}
  ]}}

### Example 9 — task, floating with yearly anchor
Input: "Call mom on Dec 25 every year"
Output:
{{"predicates": ["call"], "body_state": null, "sentiment": null, "energy": null,
  "duration": null, "duration_inferred": null,
  "time_mode": "floating", "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": "12-25",
  "entities": [
    {{"type": "Person", "value": "mom", "key": "mom"}}
  ]}}

## Now extract from this input

Input: {text}
"""


TASK_EXTRACTION_PROMPT = """\
You extract structured information from a personal **task** entry — a
forward-looking intent the user wants to do. Tasks share most of the
schema with logs (Person/Place/Project/Topic entities + episode metadata)
but additionally surface lifecycle-relevant signals.

## Output JSON shape

```
{{"predicates": ["<verb1>", ...],
  "body_state": null,
  "sentiment": null,
  "energy": null,
  "entities": [{{"type": "...", "kind": "...", "value": "...", "key": "..."}}],
  "action_verb": "<the primary verb in imperative form>",
  "deadline_hint": "<verbatim deadline phrase from the text or null>",
  "priority_hint": "high"|"medium"|"low"|null,
  "gtd_context_hint": "@home"|"@work"|"@errands"|"@phone"|null,
  "time_mode": "absolute"|"floating",
  "wall_clock_hour": 0-23|null,
  "wall_clock_minute": 0-59|null,
  "wall_clock_date": "MM-DD"|null}}
```

## Task-specific rules

1. ``predicates`` is the imperative verb in normalized lowercase
   ("email", "buy", "review", "schedule"). Tasks have one primary verb
   most of the time; multi-action tasks ("email Tao AND update slide")
   produce multiple predicates.

2. ``action_verb`` is the same as the primary predicate but kept as a
   first-class field for the lifecycle UI (some tools render the verb
   prominently).

3. ``deadline_hint`` is a verbatim substring of the source describing
   timing ("by Friday", "next week", "before EOD", "tomorrow morning").
   The library parses this into an actual datetime separately — your
   job is just to surface the phrase.

4. ``priority_hint`` — only set if the source has explicit urgency
   markers: "URGENT", "ASAP", "high priority", "!!!". Default null.

5. ``gtd_context_hint`` — set if the source explicitly tags a context
   (@work, @home, @errands, @phone, @computer). Default null.

6. ``body_state``, ``sentiment``, ``energy`` are NOT used for tasks
   (tasks are intentions, not lived experience). Always null.

7. All other rules from log extraction apply: NEVER TRANSLATE,
   value MUST be a substring of source, entities use the 4 canonical
   classes with Topic.kind discriminator.

## time_mode inference (REQUIRED — pick one)

Pick ``absolute`` when the task is tied to a specific physical moment
that doesn't follow the user across timezones:

  ✓ specific people / meetings ("standup with Alice", "lunch with Tom")
  ✓ explicit DDL / delivery ("submit by Friday", "due before EOD")
  ✓ full date + time ("May 15 at 3pm")
  ✓ external constraints (flights, trains, interviews)

Pick ``floating`` when the task is a wall-clock habit the user wants to
follow wherever they are. wall-clock fields apply:

  ✓ daily/weekly/monthly recurrence with a clock time but no calendar
    date ("Every day at 8am meditate", "weekdays 7am workout")
  ✓ personal routine (eat-meds, wake-up, sleep, exercise, meditation)
  ✓ habit/streak ("drink water reminder", "stand up every hour")
  ✓ vague clock cues ("morning routine", "before bed")

Fallback: one-shot with explicit date → absolute. Recurring without
explicit date → floating. When genuinely ambiguous, prefer absolute
(the safer default — fixed moments don't drift).

When ``time_mode == "floating"``:
- ``wall_clock_hour`` (0-23) and ``wall_clock_minute`` (0-59) MUST be
  set from the explicit/implicit clock time.
- ``wall_clock_date`` = "MM-DD" ONLY for yearly anchors ("Dec 25
  birthday call"). For daily/weekly/monthly recurrence leave it null.

When ``time_mode == "absolute"``:
- wall_clock_* are all null.
- ``deadline_hint`` carries the verbatim phrase (parsing happens later).

## Examples

### Example 1 — absolute, basic task with DDL
Input: "Email Tao the Q3 report by Friday"
Output:
{{"predicates": ["email"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Person", "value": "Tao", "key": "tao"}},
    {{"type": "Topic", "kind": "object", "value": "Q3 report", "key": "q3-report"}}
  ],
  "action_verb": "email",
  "deadline_hint": "by Friday",
  "priority_hint": null,
  "gtd_context_hint": null,
  "time_mode": "absolute",
  "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null}}

### Example 2 — absolute, explicit priority + context
Input: "URGENT: review @work the security patch before EOD"
Output:
{{"predicates": ["review"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Topic", "kind": "object", "value": "security patch", "key": "security-patch"}}
  ],
  "action_verb": "review",
  "deadline_hint": "before EOD",
  "priority_hint": "high",
  "gtd_context_hint": "@work",
  "time_mode": "absolute",
  "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null}}

### Example 3 — absolute, Chinese task
Input: "明天下午之前给李发邮件确认会议"
Output:
{{"predicates": ["email", "confirm"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Person", "value": "李", "key": "李"}},
    {{"type": "Topic", "kind": "general", "value": "会议", "key": "会议"}}
  ],
  "action_verb": "email",
  "deadline_hint": "明天下午之前",
  "priority_hint": null,
  "gtd_context_hint": null,
  "time_mode": "absolute",
  "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": null}}

### Example 4 — floating, recurring chore (errands)
Input: "Buy groceries every Saturday morning"
Output:
{{"predicates": ["buy"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Topic", "kind": "general", "value": "groceries", "key": "groceries"}}
  ],
  "action_verb": "buy",
  "deadline_hint": "every Saturday morning",
  "priority_hint": null,
  "gtd_context_hint": "@errands",
  "time_mode": "floating",
  "wall_clock_hour": 9, "wall_clock_minute": 0, "wall_clock_date": null}}

### Example 5 — floating, daily routine
Input: "每天 8 点吃降压药"
Output:
{{"predicates": ["take"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Topic", "kind": "health", "value": "降压药", "key": "降压药"}}
  ],
  "action_verb": "take",
  "deadline_hint": "每天 8 点",
  "priority_hint": null,
  "gtd_context_hint": "@home",
  "time_mode": "floating",
  "wall_clock_hour": 8, "wall_clock_minute": 0, "wall_clock_date": null}}

### Example 6 — floating, yearly anchor
Input: "Call mom on her birthday every December 25"
Output:
{{"predicates": ["call"], "body_state": null, "sentiment": null, "energy": null,
  "entities": [
    {{"type": "Person", "value": "mom", "key": "mom"}}
  ],
  "action_verb": "call",
  "deadline_hint": "every December 25",
  "priority_hint": null,
  "gtd_context_hint": "@phone",
  "time_mode": "floating",
  "wall_clock_hour": null, "wall_clock_minute": null, "wall_clock_date": "12-25"}}

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
