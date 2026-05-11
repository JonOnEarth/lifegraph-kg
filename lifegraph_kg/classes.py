# SPDX-License-Identifier: Apache-2.0
"""The 4 default life-classes: Person, Place, Project, Topic.

Anchored to the canonical PKG ontology — PIMO (NEPOMUK 2006-2008),
Balog & Kenter "Personal Knowledge Graphs: A Research Agenda" (2019),
Conway's Self-Memory System (2000), DOLCE upper ontology (Endurant /
Perdurant), CIDOC-CRM (E21 Person, E53 Place, E5 Event).

Five independent traditions converge on the same minimal core:
Person, Place, Project, Topic (with `kind` discriminator), and Episode
(the Perdurant — defined separately as the unit-of-record).

There is intentionally NO Activity class. Verbs become predicates
(scalar metadata on Episode) rather than node-classes; the Episode
itself is the activity instance. NTCIR Lifelog and DOLCE both treat
verbs as edge labels on event-nodes; we follow that.

Health, Mood, BodyState are intentionally NOT classes either. Conway's
SMS is explicit: affect is a feature of the episode, not a participant.
Schema.org agrees — symptoms are properties (`signOrSymptom`), not class
instances. They live as scalar fields on Episode.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The Topic kind discriminator — absorbs Food, Media, Object, Org,
# Health-as-thing, ideas. Open-set in spirit but enum here for v0.1
# query reliability. Add values via a minor version bump if needed.
TopicKind = Literal[
    "general",
    "food",
    "media",
    "health",
    "object",
    "org",
    "idea",
]


class Entity(BaseModel):
    """Base entity — every node-class subclasses this.

    The `key` is a lowercase, hyphenated canonical form derived from
    `value`. The `value` is the verbatim surface form from the source
    text (substring-assertion enforced).

    ``user_id`` scopes the entity to a specific user — the dedup
    boundary is ``(user_id, type, key)``. Two users with a friend
    named "Sara" get distinct Person rows. Required (no default) so
    callers can't accidentally leak entities across tenants.
    """

    model_config = ConfigDict(frozen=True)

    # Tenancy: scopes the entity to a user. The dedup boundary is
    # ``(user_id, type, key)``. Default is empty so the extract pipeline
    # (which doesn't know the request's auth context) can construct
    # entities; ``LifeGraph.persist()`` always re-stamps with the
    # request user_id before write. Production reads see the real
    # user_id from the DB row.
    user_id: str = ""
    type: str
    key: str
    value: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class Person(Entity):
    """A named human — Person mention in autobiographical record.

    Anchored to: PIMO `Person`, FOAF `Person`, CIDOC-CRM `E21 Person`,
    schema.org `Person`. Universal across every PKG / cognitive-science
    tradition.

    Includes proper names ("Sara"), kinship terms ("mom", "妈妈"),
    role labels ("the boss", "老板"), and pet names ("宝贝") when used
    as a person reference.
    """

    type: Literal["Person"] = "Person"


class Place(Entity):
    """A named location — physical or virtual.

    Anchored to: schema.org `Place`, CIDOC `E53 Place`, PIMO `Location`,
    Tulving's "where" (episodic-memory primary index).

    We use **Place** (schema.org / CIDOC convention) over PIMO's
    "Location" — the latter reads as a property name. Includes
    restaurants, cities, buildings, descriptive locations like
    "the studio", "campus", virtual spaces like "Slack #general".
    """

    type: Literal["Place"] = "Place"


class Project(Entity):
    """A named ongoing initiative — multi-episode lifespan or sub-task structure.

    Anchored to: PIMO `Project`, Obsidian/Logseq PKM convention. Both a
    participant in episodes and a *container* for them — projects
    accumulate sub-events, deadlines, collaborators, artifacts.

    Promotion rule: a referent is a Project (vs. Topic) if it satisfies
    any 2 of: (1) named identity, (2) appears in ≥3 episodes spanning
    ≥7 days, (3) has sub-task structure. Default Topic, promote on
    evidence — cheap migration since `Project ⊆ Topic`.
    """

    type: Literal["Project"] = "Project"


class Topic(Entity):
    """The catch-all node-class — any referent that isn't Person, Place, or Project.

    Required `kind` discriminator: general | food | media | health |
    object | org | idea.

    `kind` rationale (with audit citations):
    - `food`   — ramen, coffee, 早饭 (no need for separate Food class;
                  schema.org has no top-level Food)
    - `media`  — books, films, TV shows ("五代十国的新电视剧"; we don't
                  use schema.org's `CreativeWork` since we're a journal,
                  not a catalog)
    - `health` — *named* medications, conditions, providers (Advil,
                  knee injury). Transient affect/body-state goes on
                  Episode metadata, not here.
    - `object` — physical artifacts (slide, laptop)
    - `org`    — companies, institutions (Harvard-the-institution, vs.
                  Harvard-the-campus which is a Place)
    - `idea`   — abstract concepts (UI bug, paper review, Q3 outlook)
    - `general` — anything else

    The `kind` discriminator pattern lets us start with one class and
    grow discriminator values (cheap migration) instead of growing
    class count (expensive migration). Schema.org and Wikidata use this
    pattern at much larger scale.
    """

    type: Literal["Topic"] = "Topic"
    kind: TopicKind = "general"
