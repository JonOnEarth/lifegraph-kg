# SPDX-License-Identifier: Apache-2.0
"""RestStore — PostgREST-backed Store implementation.

For Supabase deploys where the caller has only a project URL + service
role key (e.g. GitHub-OAuth Supabase logins where no DB password was
stored). Same Store protocol as SqliteStore / PostgresStore, but every
operation flows through Supabase's REST API instead of a Postgres wire
connection.

Trade-offs vs ``PostgresStore``:
  - **No transactions** — each table-mutation is its own HTTP call.
    ``save_episode`` is therefore not atomic; partial writes are
    possible on network failure. The library's two-phase save
    (episode + entities first, then edges) mitigates this — if the
    edges call fails, the episode + entities are still there, and a
    re-run with the same IDs is a no-op (``ON CONFLICT DO NOTHING``).
  - **Bandwidth** — every read is an HTTP roundtrip. Fine for
    low-volume single-user deploys; for high-QPS workloads use
    ``PostgresStore`` over the connection pooler.
  - **Schema is not initialized here** — the Supabase project must
    already have the lifegraph-kg schema applied (use the
    ``apply_migration`` MCP tool or paste
    ``scripts/migrate_from_d1/migrate_to_lifegraph_kg.py`` 's DDL into
    the SQL editor).

Usage:
    from lifegraph_kg import LifeGraph
    from lifegraph_kg.kg.store.rest import RestStore

    store = RestStore(
        project_url="https://<ref>.supabase.co",
        service_role_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    lg = LifeGraph(store=store)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from lifegraph_kg.classes import Person, Place, Project, Topic
from lifegraph_kg.extract.schema import EntityT
from lifegraph_kg.kg.edge import Edge
from lifegraph_kg.kg.episode import Episode


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


_LEGACY_USER = ""


def _entity_from_row(row: dict[str, Any]) -> EntityT:
    type_ = row["type"]
    common = {
        "user_id": row.get("user_id") or _LEGACY_USER,
        "key": row["key"],
        "value": row["value"],
        "attributes": (
            json.loads(row["attributes_json"])
            if isinstance(row.get("attributes_json"), str)
            else (row.get("attributes_json") or {})
        ),
    }
    if type_ == "Person":
        return Person(**common)
    if type_ == "Place":
        return Place(**common)
    if type_ == "Project":
        return Project(**common)
    if type_ == "Topic":
        return Topic(kind=row.get("kind") or "general", **common)
    raise ValueError(f"Unknown entity type in DB: {type_!r}")


def _episode_from_row(row: dict[str, Any]) -> Episode:
    occurred_at = _from_ms(row["occurred_at"])
    ingested_at = _from_ms(row["ingested_at"])
    assert occurred_at is not None
    assert ingested_at is not None
    preds_raw = row.get("predicates") or "[]"
    preds = json.loads(preds_raw) if isinstance(preds_raw, str) else preds_raw
    return Episode(
        id=row["id"],
        user_id=row.get("user_id") or _LEGACY_USER,
        text=row["text"],
        occurred_at=occurred_at,
        ingested_at=ingested_at,
        source=row.get("source") or "user",
        predicates=preds,
        body_state=row.get("body_state"),
        sentiment=row.get("sentiment"),
        energy=row.get("energy"),
        kind=row.get("kind") or "log",
        status=row.get("status") or "active",
        priority=row.get("priority"),
        deadline=_from_ms(row.get("deadline")),
        completed_at=_from_ms(row.get("completed_at")),
        recurrence=row.get("recurrence"),
        gtd_context=row.get("gtd_context"),
        action_verb=row.get("action_verb"),
    )


def _edge_from_row(row: dict[str, Any]) -> Edge:
    t_event = _from_ms(row["t_event"])
    t_ingestion = _from_ms(row["t_ingestion"])
    t_valid = _from_ms(row["t_valid"])
    assert t_event is not None
    assert t_ingestion is not None
    assert t_valid is not None
    return Edge(
        id=row["id"],
        user_id=row.get("user_id") or _LEGACY_USER,
        from_entity=row["from_entity"],
        to_entity=row["to_entity"],
        verb=row["verb"],
        episode_id=row["episode_id"],
        t_event=t_event,
        t_ingestion=t_ingestion,
        t_valid=t_valid,
        t_invalid=_from_ms(row.get("t_invalid")),
    )


class RestStore:
    """Supabase PostgREST backend implementing the Store protocol.

    Args:
        project_url: e.g. ``https://abc123.supabase.co``.
        service_role_key: the project's ``service_role`` API key (bypasses RLS).
        timeout: per-request timeout in seconds (default 30).
    """

    def __init__(
        self,
        *,
        project_url: str,
        service_role_key: str,
        timeout: float = 30.0,
    ) -> None:
        self._base = project_url.rstrip("/") + "/rest/v1"
        self._headers = {
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=timeout)

    # --- HTTP helpers ---

    def _get(self, path: str, **query: Any) -> list[dict[str, Any]]:
        r = self._client.get(self._base + path, headers=self._headers, params=query)
        r.raise_for_status()
        return r.json()

    def _post(
        self,
        path: str,
        body: list[dict[str, Any]] | dict[str, Any],
        *,
        on_conflict: str | None = None,
        resolution: str = "ignore-duplicates",
        return_repr: bool = True,
    ) -> list[dict[str, Any]]:
        url = self._base + path
        if on_conflict:
            url += f"?on_conflict={on_conflict}"
        h = dict(self._headers)
        prefer = [f"resolution={resolution}"]
        if return_repr:
            prefer.append("return=representation")
        else:
            prefer.append("return=minimal")
        h["Prefer"] = ",".join(prefer)
        r = self._client.post(url, headers=h, content=json.dumps(body, ensure_ascii=False))
        if r.status_code >= 300:
            raise RuntimeError(f"POST {path} failed {r.status_code}: {r.text[:300]}")
        return r.json() if return_repr and r.text else []

    def _patch(self, path: str, body: dict[str, Any], **query: Any) -> None:
        r = self._client.patch(
            self._base + path,
            headers={**self._headers, "Prefer": "return=minimal"},
            params=query,
            content=json.dumps(body, ensure_ascii=False),
        )
        if r.status_code >= 300:
            raise RuntimeError(f"PATCH {path} failed {r.status_code}: {r.text[:300]}")

    # --- schema ---

    def init_schema(self) -> None:
        """No-op for REST backend — Supabase schema is applied out-of-band
        (via ``mcp__supabase__apply_migration`` or the SQL editor). If
        you call this, it's a hint that you might be on the wrong code
        path; PostgresStore initializes its own schema, RestStore cannot."""

    # --- save (NON-atomic — caveat noted in module docstring) ---

    def save_episode(
        self,
        episode: Episode,
        entities: list[Any],
        edges: list[Edge],
    ) -> None:
        uid = episode.user_id
        # 1. Episode.
        ep_row = {
            "id": episode.id,
            "user_id": uid,
            "text": episode.text,
            "occurred_at": _to_ms(episode.occurred_at),
            "ingested_at": _to_ms(episode.ingested_at),
            "source": episode.source,
            "predicates": json.dumps(episode.predicates, ensure_ascii=False),
            "body_state": episode.body_state,
            "sentiment": episode.sentiment,
            "energy": episode.energy,
            "kind": episode.kind,
            "status": episode.status,
            "priority": episode.priority,
            "deadline": _to_ms(episode.deadline) if episode.deadline else None,
            "completed_at": _to_ms(episode.completed_at) if episode.completed_at else None,
            "recurrence": episode.recurrence,
            "gtd_context": episode.gtd_context,
            "action_verb": episode.action_verb,
        }
        self._post("/episodes", [ep_row], on_conflict="id", return_repr=False)

        # 2. Entities — POST with pre-generated IDs, then resolve (since
        # ON CONFLICT DO NOTHING doesn't return the existing row's ID).
        ent_rows = []
        for e in entities:
            ent_rows.append(
                {
                    "id": _new_id(),
                    "user_id": uid,
                    "type": e.type,
                    "kind": getattr(e, "kind", None),
                    "key": e.key,
                    "value": e.value,
                    "attributes_json": json.dumps(e.attributes, ensure_ascii=False),
                    "created_at": _to_ms(datetime.now(UTC)),
                }
            )
        if ent_rows:
            self._post("/entities", ent_rows, on_conflict="user_id,type,key", return_repr=False)

        # 3. Resolve entity IDs (the row may already have existed under
        # a different generated ID).
        for e in entities:
            resolved = self._get(
                "/entities",
                **{
                    "user_id": f"eq.{uid}",
                    "type": f"eq.{e.type}",
                    "key": f"eq.{e.key}",
                    "select": "id",
                    "limit": "1",
                },
            )
            if not resolved:
                continue
            entity_id = resolved[0]["id"]
            # 4. Mention link.
            self._post(
                "/entity_episode_mention",
                [{"entity_id": entity_id, "episode_id": episode.id, "user_id": uid}],
                on_conflict="entity_id,episode_id",
                return_repr=False,
            )

        # 5. Edges. Caller has already resolved entity IDs into edge.to_entity.
        if edges:
            edge_rows = [
                {
                    "id": e.id,
                    "user_id": e.user_id,
                    "from_entity": e.from_entity,
                    "to_entity": e.to_entity,
                    "verb": e.verb,
                    "episode_id": e.episode_id,
                    "t_event": _to_ms(e.t_event),
                    "t_ingestion": _to_ms(e.t_ingestion),
                    "t_valid": _to_ms(e.t_valid),
                    "t_invalid": _to_ms(e.t_invalid) if e.t_invalid else None,
                    "attributes_json": "{}",
                }
                for e in edges
            ]
            self._post("/edges", edge_rows, on_conflict="id", return_repr=False)

    def add_edges(self, edges: list[Edge]) -> None:
        if not edges:
            return
        edge_rows = [
            {
                "id": e.id,
                "user_id": e.user_id,
                "from_entity": e.from_entity,
                "to_entity": e.to_entity,
                "verb": e.verb,
                "episode_id": e.episode_id,
                "t_event": _to_ms(e.t_event),
                "t_ingestion": _to_ms(e.t_ingestion),
                "t_valid": _to_ms(e.t_valid),
                "t_invalid": _to_ms(e.t_invalid) if e.t_invalid else None,
                "attributes_json": "{}",
            }
            for e in edges
        ]
        self._post("/edges", edge_rows, on_conflict="id", return_repr=False)

    # --- episode reads ---

    def get_episode(self, episode_id: str) -> Episode | None:
        rows = self._get("/episodes", **{"id": f"eq.{episode_id}", "limit": "1"})
        return _episode_from_row(rows[0]) if rows else None

    def episodes_since(
        self, t: datetime, *, user_id: str, limit: int | None = None
    ) -> list[Episode]:
        q: dict[str, Any] = {
            "user_id": f"eq.{user_id}",
            "occurred_at": f"gte.{_to_ms(t)}",
            "order": "occurred_at.desc",
        }
        if limit is not None:
            q["limit"] = str(limit)
        return [_episode_from_row(r) for r in self._get("/episodes", **q)]

    def episodes_between(
        self,
        start: datetime,
        end: datetime,
        *,
        user_id: str,
        limit: int | None = None,
    ) -> list[Episode]:
        q: dict[str, Any] = {
            "user_id": f"eq.{user_id}",
            "and": f"(occurred_at.gte.{_to_ms(start)},occurred_at.lte.{_to_ms(end)})",
            "order": "occurred_at.desc",
        }
        if limit is not None:
            q["limit"] = str(limit)
        return [_episode_from_row(r) for r in self._get("/episodes", **q)]

    def episodes_mentioning(
        self, entity_id: str, *, limit: int | None = None
    ) -> list[Episode]:
        # Two-step: fetch mention rows, then batch-fetch episodes by id.
        q1: dict[str, Any] = {"entity_id": f"eq.{entity_id}", "select": "episode_id"}
        rows = self._get("/entity_episode_mention", **q1)
        ep_ids = [r["episode_id"] for r in rows]
        return self._episodes_by_ids(ep_ids, limit=limit) if ep_ids else []

    def episodes_mentioning_any(
        self, entity_ids: list[str], *, limit: int | None = None
    ) -> list[Episode]:
        if not entity_ids:
            return []
        # JOIN through entity_episode_mention: get distinct episode_ids
        # where entity_id IN (...), then fetch those episodes.
        q1 = {
            "entity_id": f"in.({','.join(entity_ids)})",
            "select": "episode_id",
        }
        rows = self._get("/entity_episode_mention", **q1)
        ep_ids = list({r["episode_id"] for r in rows})
        return self._episodes_by_ids(ep_ids, limit=limit) if ep_ids else []

    def _episodes_by_ids(
        self, episode_ids: list[str], limit: int | None = None
    ) -> list[Episode]:
        """Bulk-fetch episodes by their PKs, ordered most-recent-first."""
        if not episode_ids:
            return []
        q: dict[str, Any] = {
            "id": f"in.({','.join(episode_ids)})",
            "order": "occurred_at.desc",
        }
        if limit is not None:
            q["limit"] = str(limit)
        return [_episode_from_row(r) for r in self._get("/episodes", **q)]

    # --- entity reads ---

    def get_entity(self, entity_id: str) -> EntityT | None:
        rows = self._get("/entities", **{"id": f"eq.{entity_id}", "limit": "1"})
        return _entity_from_row(rows[0]) if rows else None

    def find_entity(self, type_: str, key: str, *, user_id: str) -> EntityT | None:
        rows = self._get(
            "/entities",
            **{
                "user_id": f"eq.{user_id}",
                "type": f"eq.{type_}",
                "key": f"eq.{key}",
                "limit": "1",
            },
        )
        return _entity_from_row(rows[0]) if rows else None

    def find_entity_id(self, type_: str, key: str, *, user_id: str) -> str | None:
        rows = self._get(
            "/entities",
            **{
                "user_id": f"eq.{user_id}",
                "type": f"eq.{type_}",
                "key": f"eq.{key}",
                "select": "id",
                "limit": "1",
            },
        )
        return rows[0]["id"] if rows else None

    def query_entities(
        self,
        *,
        user_id: str,
        type_: str | None = None,
        kind: str | None = None,
        key: str | None = None,
    ) -> list[EntityT]:
        q: dict[str, Any] = {"user_id": f"eq.{user_id}"}
        if type_ is not None:
            q["type"] = f"eq.{type_}"
        if kind is not None:
            q["kind"] = f"eq.{kind}"
        if key is not None:
            q["key"] = f"eq.{key}"
        return [_entity_from_row(r) for r in self._get("/entities", **q)]

    # --- bi-temporal CRUD ---

    def invalidate_edge(self, edge_id: str, t_invalid: datetime) -> None:
        # The DB-level guard "AND t_invalid IS NULL" can't be expressed
        # in a single PostgREST PATCH cleanly; we emulate by filtering on it.
        self._patch(
            "/edges",
            {"t_invalid": _to_ms(t_invalid)},
            **{"id": f"eq.{edge_id}", "t_invalid": "is.null"},
        )

    def edges_as_of(
        self, t: datetime, *, user_id: str, verb: str | None = None
    ) -> list[Edge]:
        ms = _to_ms(t)
        q: dict[str, Any] = {
            "user_id": f"eq.{user_id}",
            "t_valid": f"lte.{ms}",
            "or": f"(t_invalid.is.null,t_invalid.gt.{ms})",
        }
        if verb is not None:
            q["verb"] = f"eq.{verb}"
        return [_edge_from_row(r) for r in self._get("/edges", **q)]

    def edges_for_episode(self, episode_id: str) -> list[Edge]:
        q = {"episode_id": f"eq.{episode_id}", "order": "id"}
        return [_edge_from_row(r) for r in self._get("/edges", **q)]

    # --- hygiene (minimal — full merge flow lives in PostgresStore) ---

    def record_proposal(
        self,
        proposal_id: str,
        winner_id: str,
        loser_id: str,
        confidence: str,
        reason: str,
        detail: str = "",
    ) -> None:
        self._post(
            "/merge_proposals",
            [
                {
                    "id": proposal_id,
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "confidence": confidence,
                    "reason": reason,
                    "detail": detail,
                    "proposed_at": _to_ms(datetime.now(UTC)),
                }
            ],
            on_conflict="id",
            return_repr=False,
        )

    def apply_merge(self, winner_id: str, loser_id: str) -> None:
        # Re-point edges + mentions; mark loser canonical.
        if winner_id == loser_id:
            return
        self._patch(
            "/edges", {"to_entity": winner_id}, **{"to_entity": f"eq.{loser_id}"}
        )
        self._patch(
            "/edges", {"from_entity": winner_id}, **{"from_entity": f"eq.{loser_id}"}
        )
        # PostgREST has no "UPDATE OR IGNORE"; the conflict-handling path
        # in PostgresStore deletes overlapping mentions first. We approximate
        # by best-effort PATCH; duplicates surface as 409 which we ignore.
        try:
            self._patch(
                "/entity_episode_mention",
                {"entity_id": winner_id},
                **{"entity_id": f"eq.{loser_id}"},
            )
        except RuntimeError:
            pass
        self._patch(
            "/entities", {"canonical_id": winner_id}, **{"id": f"eq.{loser_id}"}
        )

    def mark_proposal_applied(self, proposal_id: str) -> None:
        self._patch(
            "/merge_proposals",
            {"applied_at": _to_ms(datetime.now(UTC))},
            **{"id": f"eq.{proposal_id}"},
        )

    def mark_proposal_rejected(self, proposal_id: str) -> None:
        self._patch(
            "/merge_proposals",
            {"rejected_at": _to_ms(datetime.now(UTC))},
            **{"id": f"eq.{proposal_id}"},
        )

    # --- task lifecycle ---

    def update_task_status(
        self,
        episode_id: str,
        status: str,
        completed_at: datetime | None = None,
    ) -> None:
        body: dict[str, Any] = {"status": status}
        if completed_at is not None:
            body["completed_at"] = _to_ms(completed_at)
        else:
            body["completed_at"] = None
        self._patch("/episodes", body, **{"id": f"eq.{episode_id}"})

    def query_tasks(
        self,
        *,
        user_id: str,
        status: str | None = None,
        priority: str | None = None,
        gtd_context: str | None = None,
        deadline_before: datetime | None = None,
        deadline_after: datetime | None = None,
    ) -> list[Episode]:
        q: dict[str, Any] = {"user_id": f"eq.{user_id}", "kind": "eq.task"}
        if status is not None:
            q["status"] = f"eq.{status}"
        if priority is not None:
            q["priority"] = f"eq.{priority}"
        if gtd_context is not None:
            q["gtd_context"] = f"eq.{gtd_context}"
        deadline_clauses: list[str] = []
        if deadline_before is not None:
            deadline_clauses.append(f"deadline.lt.{_to_ms(deadline_before)}")
        if deadline_after is not None:
            deadline_clauses.append(f"deadline.gt.{_to_ms(deadline_after)}")
        if deadline_clauses:
            q["and"] = f"({','.join(deadline_clauses)})"
        q["order"] = "deadline.asc.nullslast,occurred_at.desc"
        return [_episode_from_row(r) for r in self._get("/episodes", **q)]

    # --- mutation + bulk-list (Phase 7) ---

    def update_episode(
        self,
        episode_id: str,
        *,
        text: str | None = None,
        sentiment: str | None = None,
        energy: str | None = None,
        body_state: str | None = None,
        priority: str | None = None,
        deadline: datetime | None = None,
        recurrence: str | None = None,
        gtd_context: str | None = None,
        action_verb: str | None = None,
        source: str | None = None,
    ) -> None:
        body: dict[str, Any] = {}
        if text is not None:
            body["text"] = text
        if sentiment is not None:
            body["sentiment"] = sentiment
        if energy is not None:
            body["energy"] = energy
        if body_state is not None:
            body["body_state"] = body_state
        if priority is not None:
            body["priority"] = priority
        if deadline is not None:
            body["deadline"] = _to_ms(deadline)
        if recurrence is not None:
            body["recurrence"] = recurrence
        if gtd_context is not None:
            body["gtd_context"] = gtd_context
        if action_verb is not None:
            body["action_verb"] = action_verb
        if source is not None:
            body["source"] = source
        if not body:
            return
        self._patch("/episodes", body, **{"id": f"eq.{episode_id}"})

    def delete_episode(self, episode_id: str) -> None:
        """Cascade-delete via three separate DELETEs (PostgREST has no
        transaction across endpoints; we accept eventual-consistency in
        the rare interrupt-mid-delete case)."""
        # Mentions first (no FK fan-out), then edges (reference episode),
        # then the episode itself.
        h = {**self._headers, "Prefer": "return=minimal"}
        for path in ("/entity_episode_mention", "/edges"):
            r = self._client.delete(
                self._base + path,
                headers=h,
                params={"episode_id": f"eq.{episode_id}"},
            )
            if r.status_code >= 300:
                raise RuntimeError(
                    f"DELETE {path} failed {r.status_code}: {r.text[:200]}"
                )
        r = self._client.delete(
            self._base + "/episodes",
            headers=h,
            params={"id": f"eq.{episode_id}"},
        )
        if r.status_code >= 300:
            raise RuntimeError(
                f"DELETE /episodes failed {r.status_code}: {r.text[:200]}"
            )

    def list_episodes(
        self,
        *,
        user_id: str,
        kind: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Episode]:
        q: dict[str, Any] = {
            "user_id": f"eq.{user_id}",
            "order": "occurred_at.desc",
        }
        if kind is not None:
            q["kind"] = f"eq.{kind}"
        if status is not None:
            q["status"] = f"eq.{status}"
        if since is not None:
            q["occurred_at"] = f"gte.{_to_ms(since)}"
        if limit is not None:
            q["limit"] = str(limit)
        if offset:
            q["offset"] = str(offset)
        return [_episode_from_row(r) for r in self._get("/episodes", **q)]

    def close(self) -> None:
        self._client.close()
