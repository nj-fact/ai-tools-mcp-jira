from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx

from jira_agent.config import DEFAULT_CREDENTIAL_SOURCE, JiraConfig

# Jira Cloud exposes two REST families this client reaches: the platform API this class was
# originally built against, and the Agile API that carries boards, sprints, and backlog moves.
# Both honour the same per-project credential lookup; they differ only in this path prefix.
API_BASE_PATH = "/rest/api/3"
AGILE_BASE_PATH = "/rest/agile/1.0"

# The sprint custom field id probed on this Jira site (schema
# com.pyxis.greenhopper.jira:gh-sprint). Field ids are per-site, like story_points_field in
# config.py, but this one is a method parameter default rather than a config key — see
# get_issue_sprints.
DEFAULT_SPRINT_FIELD = "customfield_10020"

# The default field set both the CLI and the MCP server fetch when a caller names none of its
# own. Defined once here, rather than once per surface, so `cli.py`'s `search --json` and
# `mcp_server.py`'s `jira_search_issues` can never drift back to reporting different fields for
# the same query — which is exactly what happened when this list carried `parent` on one surface
# and not the other. `jira_get_issue` / the CLI's `issue` command add `parent` on top of this for
# their own single-issue default; see each surface's own default-fields helper.
DEFAULT_FIELDS = ["summary", "status", "assignee", "issuetype", "project", "updated", "parent"]


class JiraApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class JiraWriteScopeError(JiraApiError):
    """A single Jira write was given keys served by different sites or credentials."""


class SprintReassignmentError(RuntimeError):
    """A sprint-add would move at least one issue out of its current active/future sprint."""

    def __init__(self, sprint_id: int, report: list[dict[str, Any]]) -> None:
        at_risk = [entry for entry in report if entry["sprints_replaced"]]
        detail = "; ".join(
            f"{entry['key']} currently in "
            f"{', '.join(str(sprint.get('id')) for sprint in entry['sprints_replaced'])}"
            for entry in at_risk
        )
        super().__init__(
            f"Refusing to move issues to sprint {sprint_id}: Jira's sprint endpoint replaces an "
            f"issue's current active/future sprint ({detail}). Pass --allow-sprint-reassignment "
            "(CLI) or allow_sprint_reassignment=true (MCP) after reviewing the dry-run blast "
            "radius."
        )
        self.sprint_id = sprint_id
        self.report = report
        self.at_risk = at_risk


class JiraConcurrentEditError(JiraApiError):
    """A guarded whole-field rewrite found the issue had changed since it was read.

    Subclasses JiraApiError, not RuntimeError directly, so existing callers that already catch
    JiraApiError broadly (for example to report a failed write to an operator) keep catching
    this too, while a caller that cares specifically about a lost-update race can still isinstance
    against this narrower type and inspect expected_updated / found_updated.
    """

    def __init__(
        self,
        message: str,
        *,
        issue_key: str,
        expected_updated: str,
        found_updated: str,
    ) -> None:
        super().__init__(
            message,
            details={
                "issue_key": issue_key,
                "expected_updated": expected_updated,
                "found_updated": found_updated,
            },
        )
        self.issue_key = issue_key
        self.expected_updated = expected_updated
        self.found_updated = found_updated


class JiraMissingConcurrencyTokenError(JiraApiError):
    """A whole-field rewrite asked for the default concurrency guard, but the read that was
    supposed to carry the guard token (`fields.updated`) did not return a usable one.

    Distinct from JiraConcurrentEditError: that one fires when the guard successfully read a
    token and then found it stale immediately before the write. This one fires when there was no
    token to guard with in the first place. Treating a missing token as "no guard requested" —
    which is what passing `expect_updated=None` through to `_guarded_issue_put` does — would
    perform the exact unguarded overwrite the caller did not ask for, and report it as a guarded
    success. Raising here instead makes that refusal visible.
    """

    def __init__(self, *, issue_key: str) -> None:
        super().__init__(
            f"Issue {issue_key}: the concurrency guard is on but Jira's response did not include "
            "a usable 'updated' timestamp to guard the write with. Re-read the issue and retry, "
            "or pass the command's guard opt-out flag to proceed without the guard.",
            details={"issue_key": issue_key},
        )
        self.issue_key = issue_key


class SprintPreconditionError(RuntimeError):
    """`sprint-remove`'s precondition failed: at least one requested key is not currently in the
    named sprint. Raised before `move_issues_to_backlog` is ever called, so the whole batch is
    refused rather than moving the keys that did match while silently skipping the rest -- the
    same all-or-nothing contract the batched write itself already has.

    Carries the full membership report (see `sprint_membership_report`), not just the mismatched
    keys, so a caller can show exactly where every key currently is, not only that one was wrong.
    """

    def __init__(self, sprint_id: int | str, report: list[dict[str, Any]]) -> None:
        mismatched = [entry["key"] for entry in report if not entry["in_sprint"]]
        super().__init__(
            f"Refusing to remove from sprint {sprint_id}: not every key is currently in that "
            f"sprint (not in it: {', '.join(mismatched)})."
        )
        self.sprint_id = sprint_id
        self.report = report


def normalize_jira_numeric_id(value: int | str, *, name: str) -> int:
    """Return a positive Jira numeric id, refusing URL/path-shaped strings.

    IDs interpolated into REST paths must never be accepted as arbitrary strings: httpx resolves
    ``../`` path segments before sending the request, which can silently turn a sprint write into
    a different Agile endpoint.  Normalizing once at the public client boundary protects CLI,
    MCP, and direct Python callers alike.
    """
    if isinstance(value, bool):
        raise JiraApiError(f"{name} must be a positive integer Jira id, got {value!r}.")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value.strip()):
        normalized = int(value.strip())
    else:
        raise JiraApiError(f"{name} must be a positive integer Jira id, got {value!r}.")
    if normalized <= 0:
        raise JiraApiError(f"{name} must be a positive integer Jira id, got {value!r}.")
    return normalized


def normalize_jira_issue_key(value: str, *, name: str = "issue_key") -> str:
    """Return a canonical Jira issue key and reject URL/path-shaped input.

    Issue keys are interpolated into REST paths throughout this client. Accepting slashes,
    backslashes, or dot segments would let the HTTP client normalize a request onto a different
    resource before it is sent. Jira issue keys have a project-key prefix and a positive numeric
    sequence, so enforcing that public shape is both stricter and clearer than URL-escaping an
    invalid key.
    """
    if not isinstance(value, str):
        raise JiraApiError(f"{name} must be a Jira issue key like FC-123, got {value!r}.")
    normalized = value.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*-[1-9][0-9]*", normalized):
        raise JiraApiError(f"{name} must be a Jira issue key like FC-123, got {value!r}.")
    return normalized


def require_matching_jira_numeric_id(
    payload: dict[str, Any], expected_id: int | str, *, name: str
) -> int:
    """Fail closed unless a Jira response identifies the exact resource requested."""
    expected = normalize_jira_numeric_id(expected_id, name=name)
    if not isinstance(payload, dict):
        raise JiraApiError(
            f"Jira did not confirm {name} {expected} in the GET response.",
            details={f"requested_{name}": str(expected)},
        )
    try:
        returned = normalize_jira_numeric_id(payload.get("id"), name=f"returned {name}")
    except JiraApiError as exc:
        raise JiraApiError(
            f"Jira did not confirm {name} {expected} in the GET response.",
            details={f"requested_{name}": str(expected)},
        ) from exc
    if returned != expected:
        raise JiraApiError(
            f"Jira returned {name} {returned} while {expected} was requested.",
            details={
                f"requested_{name}": str(expected),
                f"returned_{name}": str(returned),
            },
        )
    return expected


def validate_attachment_upload_response(
    data: Any, *, status_code: int | None = None
) -> list[dict[str, Any]]:
    """Require at least one attachment object and a usable numeric id on every object."""
    if not isinstance(data, list) or not data:
        raise JiraApiError(
            "Jira returned an unexpected attachment payload",
            status_code=status_code,
            details={"attachment_count": len(data) if isinstance(data, list) else None},
        )
    for attachment in data:
        if not isinstance(attachment, dict):
            raise JiraApiError(
                "Jira returned an unexpected attachment payload",
                status_code=status_code,
                details={"attachment_count": len(data)},
            )
        try:
            normalize_jira_numeric_id(attachment.get("id"), name="attachment id")
        except JiraApiError as exc:
            raise JiraApiError(
                "Jira returned attachment metadata without a usable id",
                status_code=status_code,
                details={"attachment_count": len(data)},
            ) from exc
    return data


def validate_agile_issue_batch(issue_keys: list[str]) -> None:
    """Enforce Jira Agile's one-to-fifty issue batch contract locally."""
    if not issue_keys:
        raise JiraApiError("issue_keys must not be empty.")
    if len(issue_keys) > 50:
        raise JiraApiError(
            f"Jira Agile accepts at most 50 issues per sprint/backlog write; got {len(issue_keys)}."
        )
    for issue_key in issue_keys:
        normalize_jira_issue_key(issue_key)


def require_shared_write_scope(config: JiraConfig, issue_keys: Iterable[str]) -> None:
    """Refuse one write whose referenced keys resolve to different site/auth tuples."""
    keys = list(issue_keys)
    if not keys:
        return
    scopes: list[tuple[str, str, str]] = []
    details: list[dict[str, str]] = []
    for key in keys:
        credentials = config.credentials_for(key)
        scopes.append(
            (
                credentials.site_url,
                credentials.email,
                credentials.api_token.get_secret_value(),
            )
        )
        details.append(
            {
                "key": key,
                "credential_source": config.credential_source_for(key),
                "site_url": credentials.site_url,
            }
        )
    if any(scope != scopes[0] for scope in scopes[1:]):
        raise JiraWriteScopeError(
            "A single Jira write cannot span different sites or credential sets.",
            details={"scopes": details},
        )


def require_concurrency_token(
    issue_key: str, read_updated: Any, *, guard_enabled: bool
) -> str | None:
    """Return `read_updated` as the `expect_updated` guard token, honouring `guard_enabled`.

    Every whole-field-rewrite command reads `updated` in the same request as the content it is
    about to rewrite, then wants exactly this decision: pass `None` through unchanged when the
    guard is off, otherwise use the token that was read — unless that token is missing or not a
    string, in which case raise `JiraMissingConcurrencyTokenError` rather than silently falling
    back to an unguarded write. Centralized here so every guarded command applies the same rule.
    """
    if not guard_enabled:
        return None
    if not isinstance(read_updated, str) or not read_updated:
        raise JiraMissingConcurrencyTokenError(issue_key=issue_key)
    return read_updated


class JiraClient:
    def __init__(
        self,
        config: JiraConfig,
        *,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._clients: dict[tuple[str, str], httpx.Client] = {}
        self._owned_clients: list[httpx.Client] = []
        if http_client is not None:
            self._clients[(DEFAULT_CREDENTIAL_SOURCE, API_BASE_PATH)] = http_client

    def require_shared_write_scope(self, issue_keys: Iterable[str]) -> None:
        """Refuse an atomic write whose keys resolve to different Jira sites/authentication.

        Multi-key Agile writes and issue-link creation are one HTTP request.  Routing that
        request by only the first key is safe solely when every referenced key is served by the
        same site and credentials that performed each key's safety check.  The credential tuple
        is compared internally; error details deliberately expose only source names and sites,
        never email addresses or API tokens.
        """
        require_shared_write_scope(self.config, issue_keys)

    @staticmethod
    def validate_agile_issue_batch(issue_keys: list[str]) -> None:
        validate_agile_issue_batch(issue_keys)

    def _http_client_for(self, key: str | None, *, base_path: str = API_BASE_PATH) -> httpx.Client:
        """Return the HTTP client for one (credential set, API base) pair, cached per pair.

        The credential lookup is unchanged from before the Agile API was added: it is keyed by
        project or issue key exactly as ``config.credentials_for`` already resolves it. Only the
        cache key grew a second component so the same credential set can serve both a
        ``/rest/api/3`` client and a ``/rest/agile/1.0`` client without one evicting the other.
        """
        source = self.config.credential_source_for(key)
        cache_key = (source, base_path)
        client = self._clients.get(cache_key)
        if client is None:
            credentials = self.config.credentials_for(key)
            kwargs: dict[str, Any] = {
                "base_url": f"{credentials.site_url}{base_path}",
                "auth": (credentials.email, credentials.api_token.get_secret_value()),
                "timeout": self.config.timeout_seconds,
                "headers": {
                    "Accept": "application/json",
                    "User-Agent": "jira-agent/0.1",
                },
            }
            if self._transport is not None:
                kwargs["transport"] = self._transport
            client = httpx.Client(**kwargs)
            self._clients[cache_key] = client
            self._owned_clients.append(client)
        return client

    def close(self) -> None:
        """Close the HTTP clients this instance created; injected clients stay open."""
        for cache_key, client in list(self._clients.items()):
            if client in self._owned_clients:
                client.close()
                del self._clients[cache_key]
        self._owned_clients.clear()

    def __enter__(self) -> JiraClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def myself(self, *, key: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/myself", key=key)

    def get_issue(
        self,
        issue_key: str,
        *,
        fields: Iterable[str] | None = None,
        expand: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        issue_key = normalize_jira_issue_key(issue_key)
        params: dict[str, str] = {}
        if fields:
            params["fields"] = ",".join(fields)
        if expand:
            params["expand"] = ",".join(expand)
        return self._request("GET", f"/issue/{issue_key}", params=params, key=issue_key)

    def search_issues(
        self,
        jql: str,
        *,
        limit: int = 20,
        fields: Iterable[str] | None = None,
        next_page_token: str | None = None,
        project_key: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "jql": jql,
            "maxResults": limit,
        }
        if fields:
            params["fields"] = list(fields)
        if next_page_token:
            params["nextPageToken"] = next_page_token
        return self._request("GET", "/search/jql", params=params, key=project_key)

    _MAX_SEARCH_PAGES = 50

    def search_all_issues(
        self,
        jql: str,
        *,
        limit: int | None = None,
        fields: Iterable[str] | None = None,
        page_size: int = 100,
        project_key: str | None = None,
    ) -> dict[str, Any]:
        """Page through `/search/jql` by following `nextPageToken` until done or `limit` is met.

        This endpoint reports no total match count at all, only `issues`, `isLast`, and an opaque
        `nextPageToken` — see `approximate_issue_count` for the only way this API exposes a count.
        The returned dict therefore never carries a `total` key; inventing one here would be
        exactly the kind of fabricated number this method exists to avoid.

        Termination rule: the walk stops on the first page that comes back with no issues (there
        is nothing left to collect), on a page whose `isLast` is `True` (an explicit completeness
        signal), on a page carrying no `nextPageToken` (there is nothing left to resume with,
        whatever `isLast` said), or once `limit`/`_MAX_SEARCH_PAGES` is reached.

        The returned dict spells this fact `is_last`, snake_case like every other key this tool's
        own result documents use (`dry_run`, `error_type`, `expected_updated`,
        `comments_returned`), and is read from the *last page fetched* rather than defaulted:
        `True` when that page's `isLast` was `True`, `False` when it was explicitly `False` (the
        walk stopped early — by `limit` or page cap — while Jira still says more remain), and
        `None` when that page carried no `isLast` at all, meaning completeness is unknown rather
        than assumed. A missing signal must never be read as complete; that silent default is the
        exact defect this method exists to avoid. Only the *read* of the raw Jira page a few lines
        below keeps Jira's own `isLast` spelling, because that dict is Jira's response, not this
        method's result; `nextPageToken` is left as-is for the same reason, and because a caller
        passes it back verbatim as a request parameter.
        """
        issues: list[dict[str, Any]] = []
        is_last: bool | None = None
        next_token: str | None = None
        token: str | None = None
        for _ in range(self._MAX_SEARCH_PAGES):
            remaining = None if limit is None else limit - len(issues)
            if remaining is not None and remaining <= 0:
                break
            page_limit = page_size if remaining is None else min(page_size, remaining)
            page = self.search_issues(
                jql,
                limit=page_limit,
                fields=fields,
                next_page_token=token,
                project_key=project_key,
            )
            page_issues = page.get("issues") or []
            raw_is_last = page.get("isLast")
            is_last = None if raw_is_last is None else bool(raw_is_last)
            next_token = page.get("nextPageToken")
            if not page_issues:
                break
            issues.extend(page_issues)
            if is_last:
                # Explicit completeness signal: nothing more to fetch.
                break
            if not next_token:
                # No signal to resume with, regardless of what isLast said (or didn't say).
                # Looping again would just replay the same page and never terminate itself.
                break
            token = next_token

        result: dict[str, Any] = {
            "issues": issues,
            "returned": len(issues),
            "is_last": is_last,
        }
        if not is_last and next_token:
            result["nextPageToken"] = next_token
        return result

    def approximate_issue_count(self, jql: str, *, project_key: str | None = None) -> int | None:
        """Return Jira's approximate match count for a JQL query, or None if Jira left it out.

        This is a read in intent — it looks up a count, nothing more — but the endpoint Jira
        provides for it is a POST (`/search/approximate-count`), unlike every other lookup in this
        class. Naming and documenting that here is deliberate: a caller must not assume GET-like
        idempotency or cache-ability from the verb alone. This is also the only number this API
        gives for a `/search/jql` query, so the caller is responsible for labelling it approximate
        rather than presenting it as an exact total.

        Returns `None`, rather than fabricating a number, when Jira's response does not carry a
        `count` field. A caller that asked for this must report that absence explicitly (a
        `search --count` invocation that silently drops the field it was asked for is exactly the
        failure this return type exists to prevent) rather than treating a missing count the same
        as one that was never requested.
        """
        data = self._request(
            "POST",
            "/search/approximate-count",
            json={"jql": jql},
            key=project_key,
        )
        return data.get("count")

    def list_comments(
        self,
        issue_key: str,
        *,
        start_at: int = 0,
        limit: int = 20,
        order_by: str | None = None,
    ) -> dict[str, Any]:
        """Return one page of an issue's comments, oldest first unless `order_by` says otherwise.

        Jira's own default order for this endpoint is oldest-first, which is easy to mistake for
        "most recent" when only a handful are fetched. Pass ``order_by="-created"`` to get
        newest-first instead; the parameter is passed through as the ``orderBy`` query parameter
        only when set, so the unguarded default stays exactly what it was before.
        """
        issue_key = normalize_jira_issue_key(issue_key)
        params: dict[str, Any] = {"startAt": start_at, "maxResults": limit}
        if order_by:
            params["orderBy"] = order_by
        return self._request(
            "GET",
            f"/issue/{issue_key}/comment",
            params=params,
            key=issue_key,
        )

    _MAX_COMMENT_PAGES = 50

    def list_all_comments(
        self,
        issue_key: str,
        *,
        order_by: str | None = None,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """Page through every comment on an issue and return them as one page-shaped response.

        Pages with startAt/maxResults until the collected count reaches the API-reported
        ``total`` or a page comes back empty. Bounded to ``_MAX_COMMENT_PAGES`` requests: if a
        miscounted total or comments deleted mid-page ever kept this from converging, this
        returns the partial result instead of looping forever.

        The reported ``total`` is the FIRST page's ``total``, not the last's: a comment deleted
        between this method's first and last request would otherwise let a later, smaller
        ``total`` overwrite the first, and the two are then free to disagree with each other in
        either direction depending only on which page happened to run last. Keeping the first
        page's value is not enough by itself either — the exact same deletion can make even that
        first value smaller than the number of comments this method goes on to actually collect
        (reproducible synthetically with a page reporting ``total: 0`` alongside one comment).
        So the returned ``total`` is additionally floored at ``len(comments)``: it can disagree
        with Jira's own count, but it can never claim fewer comments exist than this method is
        handing back right now, which would read as ``comments_truncated: false`` alongside a
        ``comment_count`` smaller than ``comments_returned`` — a contradiction in the same report.
        """
        comments: list[dict[str, Any]] = []
        total: int | None = None
        start_at = 0
        for _ in range(self._MAX_COMMENT_PAGES):
            page = self.list_comments(
                issue_key, start_at=start_at, limit=page_size, order_by=order_by
            )
            page_comments = page.get("comments") or []
            if total is None and isinstance(page.get("total"), int):
                total = page["total"]
            if not page_comments:
                break
            comments.extend(page_comments)
            start_at += len(page_comments)
            if total is not None and len(comments) >= total:
                break
        reported_total = len(comments) if total is None else max(total, len(comments))
        return {"comments": comments, "total": reported_total}

    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        issue_key = normalize_jira_issue_key(issue_key)
        return self._request(
            "POST",
            f"/issue/{issue_key}/comment",
            json={"body": text_to_adf(body)},
            key=issue_key,
        )

    def get_comment(self, issue_key: str, comment_id: str) -> dict[str, Any]:
        issue_key = normalize_jira_issue_key(issue_key)
        normalized_comment_id = normalize_jira_numeric_id(comment_id, name="comment_id")
        result = self._request(
            "GET",
            f"/issue/{issue_key}/comment/{normalized_comment_id}",
            key=issue_key,
        )
        require_matching_jira_numeric_id(result, normalized_comment_id, name="comment_id")
        return result

    def update_comment(self, issue_key: str, comment_id: str, body: str) -> dict[str, Any]:
        """Replace the body of one existing comment.

        Jira keeps the original author and created instant and records the edit through
        updateAuthor and updated, so a correction stays attributable without a second comment.
        """
        issue_key = normalize_jira_issue_key(issue_key)
        comment_id = str(normalize_jira_numeric_id(comment_id, name="comment_id"))
        return self._request(
            "PUT",
            f"/issue/{issue_key}/comment/{comment_id}",
            json={"body": text_to_adf(body)},
            key=issue_key,
        )

    def delete_comment(self, issue_key: str, comment_id: str) -> dict[str, Any]:
        issue_key = normalize_jira_issue_key(issue_key)
        comment_id = str(normalize_jira_numeric_id(comment_id, name="comment_id"))
        return self._request(
            "DELETE",
            f"/issue/{issue_key}/comment/{comment_id}",
            key=issue_key,
        )

    def get_transitions(self, issue_key: str) -> dict[str, Any]:
        """List transitions Jira currently offers this account for `issue_key`.

        Always requests `expand=transitions.fields`, so every transition's `fields` metadata
        (which fields it requires, and each one's `allowedValues` when Jira has them) comes back
        in the same read that lists the transitions -- see `transition_required_fields_missing`
        and `transition_fields_payload`, which are the only things that read `fields`. Probed
        fact (FC-458): the metadata is per-transition, not a separate call.
        """
        issue_key = normalize_jira_issue_key(issue_key)
        return self._request(
            "GET",
            f"/issue/{issue_key}/transitions",
            params={"expand": "transitions.fields"},
            key=issue_key,
        )

    def transition_issue(
        self,
        issue_key: str,
        transition_id: str,
        *,
        fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform one transition, optionally supplying field values it requires.

        `fields` is only sent when non-empty, so a transition that needs none keeps posting the
        exact `{"transition": {"id": ...}}` body this always sent. See
        `transition_fields_payload` for how a caller's plain values are shaped against each
        field's own metadata before reaching here.
        """
        issue_key = normalize_jira_issue_key(issue_key)
        payload: dict[str, Any] = {"transition": {"id": transition_id}}
        if fields:
            payload["fields"] = fields
        return self._request(
            "POST",
            f"/issue/{issue_key}/transitions",
            json=payload,
            key=issue_key,
        )

    def create_metadata(
        self,
        *,
        project_key: str,
        issue_type_name: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "projectKeys": project_key,
            "expand": "projects.issuetypes.fields",
        }
        if issue_type_name:
            params["issuetypeNames"] = issue_type_name
        return self._request("GET", "/issue/createmeta", params=params, key=project_key)

    def get_edit_metadata(self, issue_key: str) -> dict[str, Any]:
        issue_key = normalize_jira_issue_key(issue_key)
        return self._request("GET", f"/issue/{issue_key}/editmeta", key=issue_key)

    def is_issue_type_allowed(self, issue_key: str, issue_type_name: str) -> bool:
        """Read editmeta and report whether `issue_type_name` is an allowed issuetype value.

        Lets a caller refuse an issue-type change up front with a readable message instead of
        sending a PUT that editmeta already shows Jira would reject.
        """
        metadata = self.get_edit_metadata(issue_key)
        needle = issue_type_name.strip().casefold()
        return any(name.casefold() == needle for name in issue_type_allowed_values(metadata))

    def create_issue(
        self,
        *,
        project_key: str,
        issue_type_name: str,
        summary: str,
        description: str | None = None,
        labels: list[str] | None = None,
        fix_versions: list[str] | None = None,
        parent_key: str | None = None,
        priority: str | None = None,
        story_points: float | None = None,
    ) -> dict[str, Any]:
        if parent_key is not None:
            parent_key = normalize_jira_issue_key(parent_key, name="parent_key")
            self.require_shared_write_scope([project_key, parent_key])
        return self._request(
            "POST",
            "/issue",
            json=create_issue_payload(
                project_key=project_key,
                issue_type_name=issue_type_name,
                summary=summary,
                description=description,
                labels=labels,
                fix_versions=fix_versions,
                parent_key=parent_key,
                priority=priority,
                story_points=story_points,
                story_points_field=self.config.story_points_field,
            ),
            key=project_key,
        )

    def assign_issue(self, issue_key: str, account_id: str | None) -> dict[str, Any]:
        issue_key = normalize_jira_issue_key(issue_key)
        return self._request(
            "PUT",
            f"/issue/{issue_key}/assignee",
            json={"accountId": account_id},
            key=issue_key,
        )

    def search_assignable_users(
        self,
        query: str,
        *,
        issue_key: str | None = None,
        project_key: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query": query,
            "maxResults": limit,
        }
        if issue_key:
            issue_key = normalize_jira_issue_key(issue_key)
            params["issueKey"] = issue_key
        if project_key:
            params["project"] = project_key
        return self._request(
            "GET",
            "/user/assignable/search",
            params=params,
            key=issue_key or project_key,
        )

    def issue_version(self, issue_key: str) -> str:
        """Return `fields.updated` for use as a whole-field-rewrite concurrency guard token.

        Jira Cloud exposes no ETag or other conditional-write token on the issue resource, so
        `updated` is the closest thing available: any field write moves it, including one made by
        another user, automation rule, or workflow post-function the caller never saw.

        Raises `JiraMissingConcurrencyTokenError`, not a bare `JiraApiError`, when the response
        carries no usable token. `require_concurrency_token` raises the same type for the same
        condition on every other guarded command's read; a caller that gets its token through
        this method instead (`human-brief`, `edit-issue`, `issue-type` on the CLI, and their MCP
        equivalents) deserves the identical `concurrent_edit_guard_unavailable` outcome, not a
        generic "Jira API error" exit that the other guarded commands do not produce for the
        same condition.
        """
        issue_key = normalize_jira_issue_key(issue_key)
        data = self._request(
            "GET", f"/issue/{issue_key}", params={"fields": "updated"}, key=issue_key
        )
        updated = (data.get("fields") or {}).get("updated")
        if not isinstance(updated, str) or not updated:
            raise JiraMissingConcurrencyTokenError(issue_key=issue_key)
        return updated

    def _guarded_issue_put(
        self,
        issue_key: str,
        fields: dict[str, Any],
        *,
        update: dict[str, Any] | None = None,
        expect_updated: str | None,
    ) -> dict[str, Any]:
        """PUT one issue's `fields` and/or `update` sections, checking `updated` immediately
        beforehand when `expect_updated` is given.

        This is a check-then-act narrowing of the race window, NOT an atomic compare-and-swap:
        Jira Cloud's issue PUT offers no conditional-write mechanism (no ETag / If-Match), so a
        change landing between the read below and the PUT that follows it is still silently lost.
        Passing `expect_updated=None` keeps the previously unguarded behaviour exactly as it was.

        `update` carries Jira's per-field `add`/`remove` array operations (see
        `edit_issue_payload`). Those are exactly the case `expect_updated` does not need to cover:
        Jira applies them against whatever the field holds server-side at write time, not against
        a value this process read earlier, so there is no read-modify-write for a stale token to
        protect. A caller that also supplies `fields` may still want the guard for THAT part of
        the same PUT; the token, when given, gates the whole request, `update` included, since
        Jira offers only one PUT per issue, not a way to guard one section and not the other.
        """
        issue_key = normalize_jira_issue_key(issue_key)
        if expect_updated is not None:
            found = self.issue_version(issue_key)
            if found != expect_updated:
                raise JiraConcurrentEditError(
                    f"Issue {issue_key} changed since it was read: expected updated="
                    f"{expect_updated!r}, found {found!r}. Re-read the issue and retry.",
                    issue_key=issue_key,
                    expected_updated=expect_updated,
                    found_updated=found,
                )
        body: dict[str, Any] = {}
        if fields:
            body["fields"] = fields
        if update:
            body["update"] = update
        return self._request("PUT", f"/issue/{issue_key}", json=body, key=issue_key)

    def edit_issue(
        self,
        issue_key: str,
        *,
        summary: str | None = None,
        description: str | None = None,
        labels: list[str] | None = None,
        fix_versions: list[str] | None = None,
        parent_key: str | None = None,
        clear_parent: bool = False,
        priority: str | None = None,
        story_points: float | None = None,
        labels_add: list[str] | None = None,
        labels_remove: list[str] | None = None,
        fix_versions_add: list[str] | None = None,
        fix_versions_remove: list[str] | None = None,
        expect_updated: str | None = None,
    ) -> dict[str, Any]:
        """Edit one issue. See `edit_issue_payload` for the `fields` vs. `update` split and the
        set-and-update-the-same-field refusal.

        Pass `expect_updated` only when this call includes a whole-field rewrite (`labels`,
        `fix_versions`, or any other keyword here) worth guarding. A call carrying only
        `labels_add`/`labels_remove`/`fix_versions_add`/`fix_versions_remove` needs no guard at
        all -- Jira applies those against its own current state -- so such a call should leave
        `expect_updated` at its default `None` rather than pay for a read that protects nothing.
        """
        if parent_key is not None:
            self.require_shared_write_scope([issue_key, parent_key])
        payload = edit_issue_payload(
            summary=summary,
            description=description,
            labels=labels,
            fix_versions=fix_versions,
            parent_key=parent_key,
            clear_parent=clear_parent,
            priority=priority,
            story_points=story_points,
            labels_add=labels_add,
            labels_remove=labels_remove,
            fix_versions_add=fix_versions_add,
            fix_versions_remove=fix_versions_remove,
            story_points_field=self.config.story_points_field,
        )
        return self._guarded_issue_put(
            issue_key,
            payload.get("fields") or {},
            update=payload.get("update"),
            expect_updated=expect_updated,
        )

    def edit_issue_adf_description(
        self,
        issue_key: str,
        description: dict[str, Any],
        *,
        expect_updated: str | None = None,
    ) -> dict[str, Any]:
        return self._guarded_issue_put(
            issue_key, {"description": description}, expect_updated=expect_updated
        )

    def set_issue_type(
        self,
        issue_key: str,
        issue_type_name: str,
        *,
        expect_updated: str | None = None,
    ) -> dict[str, Any]:
        """Change one issue's issue type, honouring the same concurrency guard as edit_issue.

        Call `is_issue_type_allowed` first to refuse early when `issue_type_name` is not among
        the target project's allowed issuetype values; this method sends the PUT unconditionally.
        """
        return self._guarded_issue_put(
            issue_key,
            {"issuetype": {"name": issue_type_name}},
            expect_updated=expect_updated,
        )

    def link_issues(
        self,
        *,
        outward_issue_key: str,
        inward_issue_key: str,
        link_type: str = "Relates",
        comment: str | None = None,
    ) -> dict[str, Any]:
        outward_issue_key = normalize_jira_issue_key(
            outward_issue_key, name="outward_issue_key"
        )
        inward_issue_key = normalize_jira_issue_key(inward_issue_key, name="inward_issue_key")
        self.require_shared_write_scope([outward_issue_key, inward_issue_key])
        return self._request(
            "POST",
            "/issueLink",
            json=issue_link_payload(
                outward_issue_key=outward_issue_key,
                inward_issue_key=inward_issue_key,
                link_type=link_type,
                comment=comment,
            ),
            key=outward_issue_key,
        )

    def delete_issue_link(self, link_id: str, *, issue_key: str) -> dict[str, Any]:
        """Delete one already-resolved issue link.

        Callers must validate the link against fresh issue reads before invoking this endpoint.
        ``issue_key`` only selects the correct per-project credential set.
        """
        issue_key = normalize_jira_issue_key(issue_key)
        link_id = str(normalize_jira_numeric_id(link_id, name="link_id"))
        return self._request("DELETE", f"/issueLink/{link_id}", key=issue_key)

    def add_attachment(self, issue_key: str, file_path: str | Path) -> list[dict[str, Any]]:
        issue_key = normalize_jira_issue_key(issue_key)
        path = Path(file_path)
        with path.open("rb") as handle:
            try:
                response = self._http_client_for(issue_key).request(
                    "POST",
                    f"/issue/{issue_key}/attachments",
                    headers={"X-Atlassian-Token": "no-check"},
                    files={"file": (path.name, handle)},
                )
            except httpx.HTTPError as exc:
                raise JiraApiError(
                    f"Jira request failed before receiving a response: {exc}"
                ) from exc

        if response.status_code >= 400:
            raise self._api_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise JiraApiError(
                "Jira returned a non-JSON response",
                status_code=response.status_code,
                details=response.text[:500],
            ) from exc
        return validate_attachment_upload_response(data, status_code=response.status_code)

    def delete_attachment(
        self,
        attachment_id: str,
        *,
        issue_key: str | None = None,
    ) -> dict[str, Any]:
        attachment_id = str(normalize_jira_numeric_id(attachment_id, name="attachment_id"))
        if issue_key is not None:
            issue_key = normalize_jira_issue_key(issue_key)
        return self._request("DELETE", f"/attachment/{attachment_id}", key=issue_key)

    def attachment_media_id(self, attachment_id: str, *, issue_key: str | None = None) -> str:
        attachment_id = str(normalize_jira_numeric_id(attachment_id, name="attachment_id"))
        if issue_key is not None:
            issue_key = normalize_jira_issue_key(issue_key)
        try:
            response = self._http_client_for(issue_key).get(
                f"/attachment/content/{attachment_id}",
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise JiraApiError(f"Jira request failed before receiving a response: {exc}") from exc

        if response.status_code >= 400:
            raise self._api_error(response)
        location = response.headers.get("location") or ""
        match = re.search(r"/file/([^/]+)/binary", location)
        if not match:
            raise JiraApiError(
                "Jira attachment redirect did not expose a media id",
                status_code=response.status_code,
                details={"location": location},
            )
        return match.group(1)

    def get_changelog(
        self,
        issue_key: str,
        *,
        start_at: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        issue_key = normalize_jira_issue_key(issue_key)
        return self._request(
            "GET",
            f"/issue/{issue_key}/changelog",
            params={"startAt": start_at, "maxResults": limit},
            key=issue_key,
        )

    # --- Agile API: boards, sprints, and backlog -----------------------------------------

    def list_boards(
        self,
        *,
        board_type: str | None = None,
        name: str | None = None,
        max_results: int = 50,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"maxResults": max_results}
        if board_type:
            params["type"] = board_type
        if name:
            params["name"] = name
        return self._request("GET", "/board", params=params, base_path=AGILE_BASE_PATH)

    def list_board_sprints(
        self,
        board_id: int | str,
        *,
        state: str | None = None,
        max_results: int = 50,
    ) -> dict[str, Any]:
        board_id = normalize_jira_numeric_id(board_id, name="board_id")
        params: dict[str, Any] = {"maxResults": max_results}
        if state:
            params["state"] = state
        return self._request(
            "GET",
            f"/board/{board_id}/sprint",
            params=params,
            base_path=AGILE_BASE_PATH,
        )

    def get_issue_sprints(
        self,
        issue_key: str,
        *,
        sprint_field: str = DEFAULT_SPRINT_FIELD,
    ) -> list[dict[str, Any]]:
        """Return the parsed sprint objects from an issue's sprint custom field.

        `sprint_field` defaults to the `customfield_10020` id probed on this Jira site (schema
        `com.pyxis.greenhopper.jira:gh-sprint`); the id is not guaranteed to match on another
        site, so a caller on a different site must pass its own. `config.sprint_field`
        (env `JIRA_SPRINT_FIELD`) is the config-resolved home for this default now, alongside
        `config.story_points_field`; callers that have a `JiraConfig` in hand should pass
        `config.sprint_field` explicitly rather than rely on this method's own default, which
        exists mainly for callers (and tests) with no config to resolve one from.

        Note this collapses three states an issue's sprint field can be in: absent from the
        response entirely, present with the value `null` (Jira's own shape for "in no sprint" —
        verified via `Sprint IS EMPTY` JQL), and present with a list. `null` and `[]` both mean
        "in no sprint" and both return `[]` here, same as an absent field. Read the field off
        `get_issue` directly instead when telling "field absent" apart from "genuinely no
        sprint" matters (see the CLI's `issue-sprint` command and `jira_get_issue_sprints`).
        """
        issue = self.get_issue(issue_key, fields=[sprint_field])
        value = (issue.get("fields") or {}).get(sprint_field)
        return list(value) if isinstance(value, list) else []

    def add_issues_to_sprint(self, sprint_id: int | str, issue_keys: list[str]) -> dict[str, Any]:
        sprint_id = normalize_jira_numeric_id(sprint_id, name="sprint_id")
        self.validate_agile_issue_batch(issue_keys)
        self.require_shared_write_scope(issue_keys)
        return self._request(
            "POST",
            f"/sprint/{sprint_id}/issue",
            json={"issues": list(issue_keys)},
            key=issue_keys[0] if issue_keys else None,
            base_path=AGILE_BASE_PATH,
        )

    def move_issues_to_backlog(self, issue_keys: list[str]) -> dict[str, Any]:
        self.validate_agile_issue_batch(issue_keys)
        self.require_shared_write_scope(issue_keys)
        return self._request(
            "POST",
            "/backlog/issue",
            json={"issues": list(issue_keys)},
            key=issue_keys[0] if issue_keys else None,
            base_path=AGILE_BASE_PATH,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        key: str | None = None,
        base_path: str = API_BASE_PATH,
    ) -> dict[str, Any]:
        try:
            response = self._http_client_for(key, base_path=base_path).request(
                method, path, params=params, json=json
            )
        except httpx.HTTPError as exc:
            raise JiraApiError(f"Jira request failed before receiving a response: {exc}") from exc

        if response.status_code >= 400:
            raise self._api_error(response)

        if not response.content:
            return {}

        try:
            data = response.json()
        except ValueError as exc:
            raise JiraApiError(
                "Jira returned a non-JSON response",
                status_code=response.status_code,
                details=response.text[:500],
            ) from exc

        if not isinstance(data, dict | list):
            raise JiraApiError(
                "Jira returned an unexpected JSON payload",
                status_code=response.status_code,
                details=data,
            )
        return data

    def _api_error(self, response: httpx.Response) -> JiraApiError:
        try:
            details: Any = response.json()
        except ValueError:
            details = response.text[:500]

        message = f"Jira API returned HTTP {response.status_code}"
        if isinstance(details, dict):
            errors = details.get("errorMessages") or details.get("errors")
            if errors:
                message = f"{message}: {errors}"

        return JiraApiError(message, status_code=response.status_code, details=details)


def is_in_sprint(sprints: list[dict[str, Any]], sprint_id: int | str) -> bool:
    """Whether one of the parsed sprint objects (as `get_issue_sprints` returns) has this
    sprint's id. Compared as strings since a caller may pass either an int (the CLI's own
    `sprint_id` argument type) or a str (the MCP surface accepts either).
    """
    target = str(sprint_id)
    return any(str(sprint.get("id")) == target for sprint in sprints if isinstance(sprint, dict))


def sprint_membership_report(
    client: JiraClient,
    issue_keys: Iterable[str],
    sprint_id: int | str,
    *,
    sprint_field: str,
) -> list[dict[str, Any]]:
    """Re-read each key's current sprint field and report whether `sprint_id` is among them.

    One shared read-and-decide step for two different callers that both need it: the
    `sprint-remove` precondition (refuse the whole batch up front if any key is not in the named
    sprint) and the post-write verification both `sprint-add` and `sprint-remove` run afterwards
    (the Agile write endpoints return 204 with no body, so this reread is what actually proves
    membership changed, not the response). Each entry carries the full parsed `sprints` list too,
    so a caller can report exactly where a mismatched key currently is, not only that it is not
    where expected.
    """
    report: list[dict[str, Any]] = []
    for issue_key in issue_keys:
        sprints = client.get_issue_sprints(issue_key, sprint_field=sprint_field)
        report.append(
            {
                "key": issue_key,
                "sprint_id": sprint_id,
                "sprints": sprints,
                "in_sprint": is_in_sprint(sprints, sprint_id),
            }
        )
    return report


def sprint_verification_results(
    report: list[dict[str, Any]], *, expected_in_sprint: bool
) -> list[dict[str, Any]]:
    """Attach `expected_in_sprint` / `verified` to a `sprint_membership_report` result.

    `expected_in_sprint=True` for `sprint-add` (the key must now be in the target sprint),
    `False` for `sprint-remove` (the key must no longer be in the sprint it was removed from).
    A 204 response proves only that Jira accepted the request, not that the issue ended up where
    asked; `verified` is the actual proof, from a fresh read taken after the write.
    """
    return [
        {
            **entry,
            "expected_in_sprint": expected_in_sprint,
            "verified": entry["in_sprint"] == expected_in_sprint,
        }
        for entry in report
    ]


def active_or_future_sprints(sprints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every non-closed (state `"future"` or `"active"`) entry in a parsed sprint list --
    exactly what `POST /rest/agile/1.0/backlog/issue` is documented to clear from an issue's
    sprint field. That endpoint takes no sprint id and is not scoped to one sprint at all: it is
    documented as equivalent to removing the FUTURE and ACTIVE sprint from each issue and leaves
    closed historical sprints untouched. Jira normally permits one current active/future sprint
    per issue. This helper nevertheless evaluates every non-closed entry returned by Jira so an
    anomalous response cannot hide additional destructive scope.
    """
    return [
        sprint for sprint in sprints if isinstance(sprint, dict) and sprint.get("state") != "closed"
    ]


def sprint_add_blast_radius(
    report: list[dict[str, Any]], sprint_id: int | str
) -> list[dict[str, Any]]:
    """Describe which current active/future sprint a sprint-add would replace.

    Jira names this operation "move issues to sprint", not an additive membership update. Closed
    historical sprint entries remain, while a different active/future sprint is replaced by the
    target. The report is read before both dry-run and execute so the caller sees that destructive
    effect and must opt in explicitly.
    """
    target = str(sprint_id)
    return [
        {
            **entry,
            "sprints_replaced": [
                sprint
                for sprint in active_or_future_sprints(entry["sprints"])
                if str(sprint.get("id")) != target
            ],
        }
        for entry in report
    ]


def sprint_add_verification_results(
    precondition: list[dict[str, Any]], post_write: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Verify target membership and departure from the replaced current sprint."""
    pre_by_key = {entry["key"]: entry for entry in precondition}
    results: list[dict[str, Any]] = []
    for entry in post_write:
        pre = pre_by_key[entry["key"]]
        replaced_ids = {str(sprint.get("id")) for sprint in pre["sprints_replaced"]}
        still_in_replaced = [
            sprint
            for sprint in entry["sprints"]
            if isinstance(sprint, dict) and str(sprint.get("id")) in replaced_ids
        ]
        results.append(
            {
                **entry,
                "pre_write_sprints": pre["sprints"],
                "sprints_replaced": pre["sprints_replaced"],
                "still_in_replaced_sprints": still_in_replaced,
                "expected_in_sprint": True,
                "verified": entry["in_sprint"] is True and not still_in_replaced,
            }
        )
    return results


def sprint_removal_blast_radius(
    report: list[dict[str, Any]], sprint_id: int | str
) -> list[dict[str, Any]]:
    """Attach the real `sprint-remove` blast radius to a `sprint_membership_report` result.

    Every entry gains `"sprints_removed"` (every non-closed sprint the key is actually in --
    what `move_issues_to_backlog` will really clear, per `active_or_future_sprints`) and
    `"other_sprints"` (`sprints_removed` minus `sprint_id` itself). `other_sprints` is non-empty
    exactly when the write would ALSO take the key out of a sprint nobody named -- the gap a
    named-sprint-only precondition and verification used to miss entirely. Jira normally permits
    one current active/future sprint, but the backlog endpoint still does not take a sprint id and
    the guard intentionally fails closed if Jira returns anomalous extra non-closed entries.
    """
    target = str(sprint_id)
    result: list[dict[str, Any]] = []
    for entry in report:
        sprints_removed = active_or_future_sprints(entry["sprints"])
        other_sprints = [s for s in sprints_removed if str(s.get("id")) != target]
        result.append({**entry, "sprints_removed": sprints_removed, "other_sprints": other_sprints})
    return result


def sprint_removal_verification_results(
    precondition: list[dict[str, Any]], post_write: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Verify a `sprint-remove` write against everything the command told the caller it would
    leave, not only the sprint id it was given.

    `precondition` and `post_write` are both `sprint_removal_blast_radius` results, read before
    and after the write. For each key, `still_in` names every sprint from the PRE-write
    `sprints_removed` (the full predicted blast radius: the named sprint plus every other
    non-closed one) that is still present in the fresh POST-write read; `verified` is True only
    when that list is empty AND the post-write entry also confirms the key left the named
    sprint specifically. That second condition is not redundant: it is what still catches a
    `sprint_id` that named a CLOSED sprint, which the endpoint never touches and which
    `active_or_future_sprints` therefore excludes from `sprints_removed` -- without it, naming a
    closed sprint that does nothing would read as trivially "verified" rather than as the no-op
    it actually was.
    """
    post_by_key = {entry["key"]: entry for entry in post_write}
    results: list[dict[str, Any]] = []
    for entry in precondition:
        post_entry = post_by_key[entry["key"]]
        removed_ids = {str(sprint.get("id")) for sprint in entry["sprints_removed"]}
        still_in = [
            sprint
            for sprint in post_entry["sprints"]
            if isinstance(sprint, dict) and str(sprint.get("id")) in removed_ids
        ]
        verified = not still_in and not post_entry["in_sprint"]
        results.append(
            {
                "key": entry["key"],
                "sprint_id": entry["sprint_id"],
                "pre_write_sprints": entry["sprints"],
                "sprints": post_entry["sprints"],
                "in_sprint": post_entry["in_sprint"],
                "sprints_removed": entry["sprints_removed"],
                "other_sprints": entry["other_sprints"],
                "still_in": still_in,
                "expected_in_sprint": False,
                "verified": verified,
            }
        )
    return results


class SprintMultiRemovalError(RuntimeError):
    """`sprint-remove`'s default refusal when the write would ALSO clear a non-closed sprint
    besides the one named. `POST /rest/agile/1.0/backlog/issue` removes every FUTURE and ACTIVE
    sprint Jira reports on an issue, not only `sprint_id`, so an anomalous response with another
    such sprint would silently lose that membership too unless this refuses first, before any
    write.

    Carries the full `sprint_removal_blast_radius` report, not just the at-risk keys, so a
    caller can show exactly what every key would additionally leave. `--allow-multi-sprint-removal`
    (CLI) / `allow_multi_sprint_removal=True` (MCP) opts in and proceeds anyway.
    """

    def __init__(self, sprint_id: int | str, report: list[dict[str, Any]]) -> None:
        at_risk = [entry for entry in report if entry["other_sprints"]]
        detail = "; ".join(
            f"{entry['key']} also in {', '.join(str(s.get('id')) for s in entry['other_sprints'])}"
            for entry in at_risk
        )
        super().__init__(
            f"Refusing to remove from sprint {sprint_id}: this would also clear at least one "
            f"key from a sprint nobody named ({detail}). Pass --allow-multi-sprint-removal "
            "(CLI) or allow_multi_sprint_removal=true (MCP) to proceed anyway."
        )
        self.sprint_id = sprint_id
        self.report = report
        self.at_risk = at_risk


def text_to_adf(text: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": _markdown_blocks_to_adf(text),
    }


def png_dimensions(path: str | Path) -> tuple[int, int]:
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    return struct.unpack(">II", header[16:24])


def media_single_adf(
    *,
    media_id: str,
    alt: str,
    width: int,
    height: int,
    collection: str = "",
) -> dict[str, Any]:
    return {
        "type": "mediaSingle",
        "attrs": {"layout": "center"},
        "content": [
            {
                "type": "media",
                "attrs": {
                    "id": media_id,
                    "type": "file",
                    "collection": collection,
                    "alt": alt,
                    "width": width,
                    "height": height,
                },
            }
        ],
    }


def replace_diagram_with_media(
    adf: dict[str, Any],
    *,
    media_id: str,
    alt: str,
    width: int,
    height: int,
    replace_media_id: str | None = None,
) -> dict[str, Any]:
    replacement = media_single_adf(media_id=media_id, alt=alt, width=width, height=height)
    content = list(adf.get("content") or [])
    if replace_media_id is not None:
        media_index = _media_single_index(content, replace_media_id)
        if media_index is None:
            raise ValueError(f"Description does not contain media id {replace_media_id!r}")
        content[media_index] = replacement
    else:
        mermaid_index = next(
            (
                index
                for index, block in enumerate(content)
                if block.get("type") == "codeBlock"
                and (block.get("attrs") or {}).get("language") == "mermaid"
            ),
            None,
        )

        if mermaid_index is not None:
            content[mermaid_index] = replacement
        else:
            media_index = next(
                (
                    index
                    for index, block in enumerate(content)
                    if block.get("type") == "mediaSingle"
                ),
                None,
            )
            if media_index is not None:
                content[media_index] = replacement
            else:
                content.append(replacement)

    return {**adf, "content": content}


def adf_contains_media_id(adf: Any, media_id: str) -> bool:
    """Return whether a Jira ADF description contains an inline media item with this id."""
    content = adf.get("content") if isinstance(adf, dict) else None
    return _media_single_index(content if isinstance(content, list) else [], media_id) is not None


def adf_media_ids(adf: Any) -> list[str]:
    """Return the inline media ids from top-level mediaSingle nodes in document order."""
    content = adf.get("content") if isinstance(adf, dict) else None
    if not isinstance(content, list):
        return []
    result: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "mediaSingle":
            continue
        for media in block.get("content") or []:
            media_id = (media.get("attrs") or {}).get("id") if isinstance(media, dict) else None
            if media_id is not None:
                result.append(str(media_id))
    return result


def _media_single_index(content: list[Any], media_id: str) -> int | None:
    for index, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") != "mediaSingle":
            continue
        if str(media_id) in adf_media_ids({"content": [block]}):
            return index
    return None


ADF_SECTION_PATCH_MODES = ("replace", "append", "prepend", "delete")
ADF_HEADING_LEVELS = range(1, 7)

# Inline nodes that carry their text in attrs instead of in a text child.
_ADF_INLINE_TEXT_ATTRS = {
    "emoji": ("text", "shortName"),
    "mention": ("text",),
    "status": ("text",),
    "date": ("timestamp",),
}


class AdfSectionPatchRefused(ValueError):
    """A section patch was refused by one of the two controls.

    Carries the lists as data so a caller never has to parse the sentence to recover.
    """

    def __init__(
        self,
        message: str,
        *,
        swallowed_headings: list[str],
        named_headings: list[str],
        whole_document: bool,
    ) -> None:
        super().__init__(message)
        self.swallowed_headings = list(swallowed_headings)
        self.named_headings = list(named_headings)
        self.whole_document = whole_document


class AdfPatchInvariantError(RuntimeError):
    """A rebuilt description failed the postcondition that only the matched slice may change."""


def patch_adf_description_section(
    description: Any,
    *,
    section: str,
    mode: str = "replace",
    content: str | None = None,
    include_heading: bool = True,
    expect_swallowed_headings: list[str] | None = None,
    allow_wide_rewrite: bool = False,
) -> dict[str, Any]:
    """Rewrite one section of an existing ADF description and leave every other node in place.

    A section is the matched heading node plus every following top-level node up to, but not
    including, the next heading whose level is the same or lower. Patching ``## A`` therefore
    covers its ``### A.1`` subsection and stops before ``## B``.

    ``replace`` and ``delete`` pass two independent controls. Neither inspects a node to judge
    whether it counts as content, and neither substitutes for the other.

    The first is arithmetic: when the slice is the entire content list, nothing outside it
    survives, and that is refused unless the caller passes ``allow_wide_rewrite``.

    The second is a naming duty. ``swallowed_headings`` is the ordered text of every heading
    inside the slice below the matched one, so it lists the sections the patch removes on top of
    the one named in ``section``. ``expect_swallowed_headings`` has to equal that list exactly,
    in order; absent, short, long, differing, or supplied when the slice swallows nothing is a
    mismatch and is refused, and the refusal prints both lists so the call can be corrected. A
    section is therefore removed only if the caller named it. A heading with no text is listed
    as ``<untitled>`` and is named with that same string.

    ``allow_wide_rewrite`` covers the first control only. It does not satisfy the naming duty.

    ``append`` and ``prepend`` remove nothing, so neither control applies to them.

    Naming the only section of a description carries no naming duty, because there is no
    unnamed section to lose. Control 1 still applies to it: an ordinary single-section
    description starts at index 0, so its slice is the whole content list and the flag is
    required. The patch goes through unflagged only when something precedes the section.

    Only the caller's Markdown is converted to ADF. The existing document is never round-tripped
    through Markdown, which would drop taskItem ids and states, tables, panels, and inline marks.
    Preserved nodes keep their localIds because they are the same objects; new checklist items
    are numbered past the highest task localId already in the document, so a spliced list cannot
    reuse an id that Jira addresses a checkbox by.

    ``include_heading`` applies to ``replace``: with ``True`` the caller's Markdown has to start
    with the new heading, because the matched heading node is part of the replaced slice; with
    ``False`` the original heading node stays and only the body below it changes. ``append`` and
    ``prepend`` always keep the original heading node.

    An ``attrs.level`` that is missing or is not an integer from 1 to 6 is never guessed. On the
    matched heading it is refused, and below the matched heading it ends the section instead of
    being skipped. No level outside the slice is consulted at all.

    Returns the rebuilt document together with the counts that show which slice moved.
    """
    if mode not in ADF_SECTION_PATCH_MODES:
        raise ValueError(
            f"Unknown section patch mode: {mode}. Use one of {', '.join(ADF_SECTION_PATCH_MODES)}."
        )
    body = (content or "").strip()
    if mode == "delete":
        if body:
            raise ValueError("Mode delete removes the section and takes no content.")
    elif not body:
        raise ValueError(f"Mode {mode} requires the Markdown content of the new section.")

    source = _adf_description_content(description)
    before = list(source)
    start, end = _adf_section_bounds(source, section)
    heading = source[start]
    heading_text = adf_heading_text(heading).strip()
    level = _strict_heading_level(heading)

    whole_document = start == 0 and end == len(source)
    swallowed = [_heading_name(node) for node in source[start + 1 : end] if _is_heading(node)]
    if mode in {"replace", "delete"}:
        refusals = []
        if whole_document and not allow_wide_rewrite:
            refusals.append(
                f"Nothing outside the slice would survive: section {heading_text!r} covers all "
                f"{len(source)} top-level nodes, so mode {mode} rewrites the entire field. Pass "
                "allow_wide_rewrite to do that deliberately."
            )
        named = [name.strip() for name in expect_swallowed_headings or []]
        if named != swallowed:
            refusals.append(
                f"Swallowed-section mismatch: the slice removes {_names(swallowed)} on top of "
                f"{heading_text!r}, and the call named {_names(named)}. Pass "
                "expect_swallowed_headings with exactly those names, in that order."
            )
        if refusals:
            raise AdfSectionPatchRefused(
                " ".join(refusals),
                swallowed_headings=swallowed,
                named_headings=named,
                whole_document=whole_document,
            )

    if not include_heading and mode != "replace":
        raise ValueError(
            f"include_heading applies to mode replace only; mode {mode} always keeps the "
            "existing heading node."
        )

    blocks = _markdown_blocks_to_adf(content, task_id_start=_next_task_id(source)) if body else []
    if mode == "replace" and include_heading and not _is_heading(blocks[0] if blocks else None):
        raise ValueError(
            f"Mode replace with include_heading replaces the {heading_text!r} heading node, so "
            "the content has to start with its own heading. Add the heading to the content, or "
            "pass include_heading=false to keep the existing heading and change only the body."
        )

    if mode == "replace":
        new_nodes = list(blocks) if include_heading else [heading, *blocks]
    elif mode == "append":
        new_nodes = [*source[start:end], *blocks]
    elif mode == "prepend":
        new_nodes = [heading, *blocks, *source[start + 1 : end]]
    else:
        new_nodes = []

    patched = _rebuild_with_section(source, start=start, end=end, new_nodes=new_nodes)
    if not patched:
        raise ValueError(
            f"Mode {mode} on section {heading_text!r} would leave the description with no "
            "content, and this tool refuses to read such a description back. Replace the "
            "section with placeholder content instead of removing it."
        )
    assert_adf_slice_replacement(
        before,
        patched,
        start=start,
        end=end,
        new_nodes=new_nodes,
        document=description,
    )

    return {
        "document": {**description, "content": patched},
        "heading": heading_text,
        "level": level,
        "mode": mode,
        "start_index": start,
        "end_index": end,
        "replaced_node_count": end - start,
        "new_node_count": len(new_nodes),
        "preserved_before": start,
        "preserved_after": len(source) - end,
        "original_node_count": len(source),
        "patched_node_count": len(patched),
        "whole_document": whole_document,
        "swallowed_headings": swallowed,
        "new_nodes": new_nodes,
    }


def _rebuild_with_section(
    source: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    new_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the patched content. Its result is checked, never trusted."""
    return [*source[:start], *new_nodes, *source[end:]]


def assert_adf_slice_replacement(
    before: list[dict[str, Any]],
    patched: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    new_nodes: list[dict[str, Any]],
    document: Any = None,
) -> None:
    """Check a rebuilt content list against the node identities captured before the rebuild.

    A rebuild that re-renders an untouched node, drops or reorders one, works from bounds that
    are not a slice of the input, or edits the caller's document in place fails here instead of
    reaching Jira. Pass document to also assert that the input was left alone.

    This checks how the slice was rebuilt. It does not judge how wide the slice is, and it
    never refuses a patch for being too broad; patch_adf_description_section decides that
    separately from the extent of the slice.
    """
    if not 0 <= start <= end <= len(before):
        raise AdfPatchInvariantError(
            f"Slice bounds {start}..{end} are not a slice of {len(before)} top-level nodes."
        )

    if document is not None:
        current = document.get("content") if isinstance(document, dict) else None
        if (
            not isinstance(current, list)
            or len(current) != len(before)
            or any(node is not original for node, original in zip(current, before, strict=True))
        ):
            raise AdfPatchInvariantError(
                "Slice replacement changed the document it was given instead of building a new one."
            )

    expected_count = len(before) - (end - start) + len(new_nodes)
    if len(patched) != expected_count:
        raise AdfPatchInvariantError(
            f"Slice replacement produced {len(patched)} top-level nodes instead of "
            f"{expected_count}."
        )

    kept_after = len(before) - end
    if any(
        node is not original for node, original in zip(patched[:start], before[:start], strict=True)
    ):
        raise AdfPatchInvariantError(
            f"Slice replacement rebuilt one of the {start} top-level node(s) before the slice."
        )
    if any(
        node is not original
        for node, original in zip(patched[len(patched) - kept_after :], before[end:], strict=True)
    ):
        raise AdfPatchInvariantError(
            f"Slice replacement rebuilt one of the {kept_after} top-level node(s) after the slice."
        )
    if any(
        node is not original
        for node, original in zip(
            patched[start : len(patched) - kept_after], new_nodes, strict=True
        )
    ):
        raise AdfPatchInvariantError(
            "Slice replacement did not place the new nodes in the matched slice."
        )


def _adf_description_content(description: Any) -> list[dict[str, Any]]:
    if description is None:
        raise ValueError("The issue has no description, so no section can be patched.")
    if not isinstance(description, dict) or description.get("type") != "doc":
        raise ValueError(
            "The issue description is not an ADF doc node, so no section can be patched."
        )
    content = description.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("The issue description is empty, so no section can be patched.")
    return list(content)


def find_adf_section_bounds(
    content: list[dict[str, Any]],
    section: str,
) -> tuple[int, int] | None:
    """Return the [start, end) top-level slice of one section, or None when it is absent.

    The heading is matched the way markdown_section matches one: leading hashes optional, case
    insensitive, whitespace-trimmed. An ambiguous heading and a matched heading without a usable
    level are refused, and an unreadable level further down ends the section.
    """
    wanted = _normalized_section_name(section)
    matches = [
        index
        for index, node in enumerate(content)
        if _is_heading(node) and adf_heading_text(node).strip().casefold() == wanted
    ]
    if len(matches) > 1:
        raise ValueError(f"Description section is ambiguous: {section}")
    if not matches:
        return None

    start = matches[0]
    level = _strict_heading_level(content[start])
    end = start + 1
    while end < len(content):
        node = content[end]
        if _is_heading(node):
            boundary = _heading_level(node)
            if boundary is None or boundary <= level:
                break
        end += 1
    return start, end


def _adf_section_bounds(content: list[dict[str, Any]], section: str) -> tuple[int, int]:
    bounds = find_adf_section_bounds(content, section)
    if bounds is None:
        headings = [adf_heading_text(node).strip() for node in content if _is_heading(node)]
        raise ValueError(
            f"Description section not found: {section}. "
            f"Existing headings: {', '.join(headings) if headings else 'none'}"
        )
    return bounds


def _normalized_section_name(section: str) -> str:
    wanted = re.sub(r"^#{1,6}\s+", "", section.strip()).strip().casefold()
    if not wanted:
        raise ValueError("A section heading is required.")
    return wanted


_TASK_LOCAL_ID = re.compile(r"^task(?:-list)?-(\d+)$")


def _next_task_id(nodes: Iterable[Any]) -> int:
    """Return a task localId number that cannot collide with one already in the document.

    _markdown_blocks_to_adf numbers taskList and taskItem localIds from a shared counter that
    starts at 1. Content spliced into a description that already holds task-1 would repeat it,
    and Jira addresses a checkbox by its localId, so new content starts past the highest number
    present anywhere in the document.
    """
    highest = 0
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        attrs = node.get("attrs")
        if isinstance(attrs, dict):
            match = _TASK_LOCAL_ID.match(str(attrs.get("localId") or ""))
            if match:
                highest = max(highest, int(match.group(1)))
        children = node.get("content")
        if isinstance(children, list):
            stack.extend(children)
    return highest + 1


UNTITLED_HEADING = "<untitled>"


def _heading_name(node: Any) -> str:
    """Name one heading for the swallowed list. A heading with no text is still nameable."""
    return adf_heading_text(node).strip() or UNTITLED_HEADING


def _names(headings: list[str]) -> str:
    """Render a heading list as a JSON array, in full.

    Never truncate and never join bare: the caller has to copy this back, so a name containing a
    comma has to stay one name, and an empty list has to be distinguishable from a list holding
    the word that would otherwise describe it.
    """
    return json.dumps(headings, ensure_ascii=False)


def _is_heading(node: Any) -> bool:
    return isinstance(node, dict) and node.get("type") == "heading"


def _strict_heading_level(node: dict[str, Any]) -> int:
    level = _heading_level(node)
    if level is None:
        raise ValueError(
            f"Heading {adf_heading_text(node).strip()!r} has no usable attrs.level between 1 "
            "and 6, so the extent of its section cannot be determined."
        )
    return level


def _heading_level(node: Any) -> int | None:
    """Return an ADF heading level, or None when it is missing or unusable.

    No caller may guess a level. The matched heading is refused and an unreadable heading below
    it ends the section, so a malformed level can never widen the slice a patch rewrites.
    """
    attrs = node.get("attrs") if isinstance(node, dict) else None
    level = attrs.get("level") if isinstance(attrs, dict) else None
    if isinstance(level, bool) or not isinstance(level, int):
        return None
    return level if level in ADF_HEADING_LEVELS else None


def adf_heading_text(node: Any) -> str:
    """Return the readable text of one ADF node, including inline nodes that hide it in attrs."""
    if not isinstance(node, dict):
        return ""
    node_type = str(node.get("type") or "")
    if node_type == "text":
        return str(node.get("text") or "")
    attrs = node.get("attrs")
    if isinstance(attrs, dict):
        for key in _ADF_INLINE_TEXT_ATTRS.get(node_type, ()):
            value = attrs.get(key)
            if value not in (None, ""):
                return str(value)
    return "".join(adf_heading_text(child) for child in node.get("content") or [])


def _markdown_blocks_to_adf(text: str, *, task_id_start: int = 1) -> list[dict[str, Any]]:
    lines = text.splitlines() or [""]
    blocks: list[dict[str, Any]] = []
    index = 0
    task_index = task_id_start

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            index += 1
            continue

        if re.fullmatch(r"-{3,}", stripped):
            blocks.append({"type": "rule"})
            index += 1
            continue

        code_match = re.match(r"^```([A-Za-z0-9_-]+)?\s*$", stripped)
        if code_match:
            language = code_match.group(1)
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "```":
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            block: dict[str, Any] = {
                "type": "codeBlock",
                "content": [{"type": "text", "text": "\n".join(code_lines)}],
            }
            if language:
                block["attrs"] = {"language": language}
            blocks.append(block)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            blocks.append(
                {
                    "type": "heading",
                    "attrs": {"level": min(len(heading_match.group(1)), 6)},
                    "content": _inline_text_to_adf(heading_match.group(2)),
                }
            )
            index += 1
            continue

        if _is_table_start(lines, index):
            table, index = _table_to_adf(lines, index)
            blocks.append(table)
            continue

        task_match = re.match(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$", line)
        if task_match:
            items: list[dict[str, Any]] = []
            while index < len(lines):
                item_match = re.match(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$", lines[index])
                if not item_match:
                    break
                state = "DONE" if item_match.group(1).lower() == "x" else "TODO"
                item_text, index = _consume_list_item(lines, index, item_match.group(2))
                items.append(
                    {
                        "type": "taskItem",
                        "attrs": {"localId": f"task-{task_index}", "state": state},
                        "content": _inline_text_to_adf(item_text),
                    }
                )
                task_index += 1
                index = _skip_blank_lines_before_item(
                    lines,
                    index,
                    r"^\s*[-*]\s+\[([ xX])\]\s+",
                )
            blocks.append(
                {
                    "type": "taskList",
                    "attrs": {"localId": f"task-list-{task_index - len(items)}"},
                    "content": items,
                }
            )
            continue

        bullet_match = re.match(r"^\s*[-*]\s+(?!\[[ xX]\]\s)(.+?)\s*$", line)
        if bullet_match:
            items: list[dict[str, Any]] = []
            while index < len(lines):
                item_match = re.match(
                    r"^\s*[-*]\s+(?!\[[ xX]\]\s)(.+?)\s*$",
                    lines[index],
                )
                if not item_match:
                    break
                item_text, index = _consume_list_item(lines, index, item_match.group(1))
                items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": _inline_text_to_adf(item_text),
                            }
                        ],
                    }
                )
                index = _skip_blank_lines_before_item(
                    lines,
                    index,
                    r"^\s*[-*]\s+(?!\[[ xX]\]\s)",
                )
            blocks.append({"type": "bulletList", "content": items})
            continue

        ordered_match = re.match(r"^\s*(\d+)[.)]\s+(.+?)\s*$", line)
        if ordered_match:
            order = int(ordered_match.group(1))
            items = []
            while index < len(lines):
                item_match = re.match(r"^\s*(\d+)[.)]\s+(.+?)\s*$", lines[index])
                if not item_match:
                    break
                item_text, index = _consume_list_item(lines, index, item_match.group(2))
                items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": _inline_text_to_adf(item_text),
                            }
                        ],
                    }
                )
                index = _skip_blank_lines_before_item(lines, index, r"^\s*\d+[.)]\s+")
            blocks.append(
                {
                    "type": "orderedList",
                    "attrs": {"order": order},
                    "content": items,
                }
            )
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines) and not _is_block_start(lines, index):
            candidate = lines[index].strip()
            if re.match(r"^\*\*[^*]+:\*\*", candidate):
                break
            paragraph_lines.append(candidate)
            index += 1
        blocks.append(
            {
                "type": "paragraph",
                "content": _inline_text_to_adf(" ".join(paragraph_lines)),
            }
        )

    return blocks or [{"type": "paragraph"}]


def _inline_text_to_adf(text: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    cursor = 0
    pattern = re.compile(
        r"`(?P<code>[^`]+)`"
        r"|\*\*(?P<strong>.+?)\*\*"
        r"|\[(?P<link_text>[^\]]+)\]\((?P<href>(?:https?://|mailto:)[^)\s]+)\)"
    )
    for match in pattern.finditer(text):
        if match.start() > cursor:
            parts.append({"type": "text", "text": text[cursor : match.start()]})
        if match.group("code") is not None:
            parts.append(
                {
                    "type": "text",
                    "text": match.group("code"),
                    "marks": [{"type": "code"}],
                }
            )
        elif match.group("strong") is not None:
            parts.append(
                {
                    "type": "text",
                    "text": match.group("strong"),
                    "marks": [{"type": "strong"}],
                }
            )
        else:
            parts.append(
                {
                    "type": "text",
                    "text": match.group("link_text"),
                    "marks": [
                        {
                            "type": "link",
                            "attrs": {"href": match.group("href")},
                        }
                    ],
                }
            )
        cursor = match.end()
    if cursor < len(text):
        parts.append({"type": "text", "text": text[cursor:]})
    return parts or [{"type": "text", "text": ""}]


def _consume_list_item(lines: list[str], index: int, first_line: str) -> tuple[str, int]:
    parts = [first_line.strip()]
    index += 1
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or _is_block_start(lines, index):
            break
        if not re.match(r"^\s{2,}\S", line):
            break
        parts.append(stripped)
        index += 1
    return " ".join(parts), index


def _skip_blank_lines_before_item(
    lines: list[str],
    index: int,
    item_pattern: str,
) -> int:
    candidate = index
    while candidate < len(lines) and not lines[candidate].strip():
        candidate += 1
    if candidate > index and candidate < len(lines):
        if re.match(item_pattern, lines[candidate]):
            return candidate
    return index


def _is_block_start(lines: list[str], index: int) -> bool:
    line = lines[index]
    stripped = line.strip()
    return bool(
        not stripped
        or re.fullmatch(r"-{3,}", stripped)
        or re.match(r"^```", stripped)
        or re.match(r"^#{1,6}\s+", line)
        or _is_table_start(lines, index)
        or re.match(r"^\s*[-*]\s+", line)
        or re.match(r"^\s*\d+[.)]\s+", line)
    )


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return (
        _is_table_row(lines[index])
        and _is_table_row(lines[index + 1])
        and _is_table_separator(lines[index + 1])
    )


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_to_adf(lines: list[str], index: int) -> tuple[dict[str, Any], int]:
    header = _table_cells(lines[index])
    rows: list[dict[str, Any]] = [_table_row_to_adf(header, header=True)]
    index += 2

    while index < len(lines) and _is_table_row(lines[index]):
        rows.append(_table_row_to_adf(_table_cells(lines[index]), header=False))
        index += 1

    table = {
        "type": "table",
        "attrs": {"isNumberColumnEnabled": False, "layout": "default"},
        "content": rows,
    }
    return table, index


def _table_row_to_adf(cells: list[str], *, header: bool) -> dict[str, Any]:
    cell_type = "tableHeader" if header else "tableCell"
    return {
        "type": "tableRow",
        "content": [
            {
                "type": cell_type,
                "attrs": {},
                "content": [
                    {
                        "type": "paragraph",
                        "content": _inline_text_to_adf(cell),
                    }
                ],
            }
            for cell in cells
        ],
    }


DEFAULT_STORY_POINTS_FIELD = "customfield_10028"


def _story_points_value(story_points: float) -> float | int:
    """Jira stores story points as a number. Keep whole values integral."""
    if story_points < 0:
        raise ValueError("Story points must not be negative")
    return int(story_points) if float(story_points).is_integer() else story_points


def create_issue_payload(
    *,
    project_key: str,
    issue_type_name: str,
    summary: str,
    description: str | None = None,
    labels: list[str] | None = None,
    fix_versions: list[str] | None = None,
    parent_key: str | None = None,
    priority: str | None = None,
    story_points: float | None = None,
    story_points_field: str = DEFAULT_STORY_POINTS_FIELD,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "issuetype": {"name": issue_type_name},
        "summary": summary,
    }
    if description:
        fields["description"] = text_to_adf(description)
    if labels is not None:
        fields["labels"] = labels
    if fix_versions is not None:
        fields["fixVersions"] = [{"name": version} for version in fix_versions]
    if parent_key:
        fields["parent"] = {"key": parent_key}
    if priority:
        fields["priority"] = {"name": priority}
    if story_points is not None:
        fields[story_points_field] = _story_points_value(story_points)
    return {"fields": fields}


def edit_issue_payload(
    *,
    summary: str | None = None,
    description: str | None = None,
    labels: list[str] | None = None,
    fix_versions: list[str] | None = None,
    parent_key: str | None = None,
    clear_parent: bool = False,
    priority: str | None = None,
    story_points: float | None = None,
    labels_add: list[str] | None = None,
    labels_remove: list[str] | None = None,
    fix_versions_add: list[str] | None = None,
    fix_versions_remove: list[str] | None = None,
    story_points_field: str = DEFAULT_STORY_POINTS_FIELD,
) -> dict[str, Any]:
    """Build the PUT body `edit_issue` sends: a `fields` section (whole-field replacement) and,
    when atomic array operations are requested, an `update` section alongside it. Either section
    is omitted outright when empty, so the returned dict is exactly the body `edit_issue` sends,
    never a preview that carries a section the real request does not.

    `fields` entries replace a whole field's value outright. Jira applies them unconditionally,
    so a concurrent editor's own change to that same field is silently lost the moment this
    fields dict also names it -- exactly the race `expect_updated` (see `_guarded_issue_put`)
    narrows for a caller that opts in.

    `labels_add`/`labels_remove` and `fix_versions_add`/`fix_versions_remove` instead build
    Jira's `update` array operations. Probed read-only against this Jira site's `editmeta`:
    `labels` advertises `operations: ["add", "set", "remove"]` with string items, and
    `fixVersions` advertises `operations: ["set", "add", "remove"]` with version-object items
    (`{"name": ...}`). Jira applies `add`/`remove` against whatever the field holds server-side
    at write time, not against a value read earlier, so there is no read-modify-write here for a
    stale `expect_updated` token to protect -- and none should be passed for a call built only
    from these four parameters (see `edit_issue`'s docstring).

    A value named in BOTH the add and the remove side of the same field (for example
    `labels_add=["x"], labels_remove=["x"]`) resolves to added, never removed: the matching
    remove operation is dropped before either list reaches the `update` section. This is the
    same add-wins precedence `workflows.drift_label_operations` applies for the identical
    conflict (`remove_labels -= set(add_labels)`), so the two call paths agree on one outcome
    for the same input rather than leaving it to whichever order Jira happens to apply
    operations within one array -- an order nobody here has established.

    Every value in `labels_add`/`labels_remove`/`fix_versions_add`/`fix_versions_remove` must be
    non-blank: an empty or whitespace-only entry is refused locally with `ValueError` naming the
    parameter, rather than sent to Jira for it to reject.

    Jira rejects a PUT that both sets and updates the SAME field, so passing `labels` together
    with `labels_add`/`labels_remove` (or `fix_versions` together with
    `fix_versions_add`/`fix_versions_remove`) raises ValueError here rather than sending that
    doomed request. A whole-array write on one field together with an atomic operation on a
    DIFFERENT field is fine and produces `fields` and `update` side by side -- only the SAME
    field may not appear in both. The CLI and the MCP server each pre-check the same conflict
    first, using their own option/parameter spellings, so a caller sees a message naming what
    they actually typed; this is the shared defence underneath both, and for any other caller of
    this function.
    """
    if clear_parent and parent_key is not None:
        raise ValueError("Use either parent_key or clear_parent, not both")
    for _param_name, _values in (
        ("labels_add", labels_add),
        ("labels_remove", labels_remove),
        ("fix_versions_add", fix_versions_add),
        ("fix_versions_remove", fix_versions_remove),
    ):
        for _value in _values or []:
            if not _value.strip():
                raise ValueError(
                    f"{_param_name} must not contain an empty or whitespace-only value."
                )
    fields: dict[str, Any] = {}
    if summary is not None:
        fields["summary"] = summary
    if description is not None:
        fields["description"] = text_to_adf(description)
    if labels is not None:
        fields["labels"] = labels
    if fix_versions is not None:
        fields["fixVersions"] = [{"name": version} for version in fix_versions]
    if clear_parent:
        fields["parent"] = None
    elif parent_key is not None:
        fields["parent"] = {"key": parent_key}
    if priority is not None:
        fields["priority"] = {"name": priority}
    if story_points is not None:
        fields[story_points_field] = _story_points_value(story_points)

    update: dict[str, Any] = {}
    if labels_add or labels_remove:
        if labels is not None:
            raise ValueError(
                "Cannot set and update `labels` in the same request: Jira rejects a PUT that "
                "both sets a field via `fields` and updates it via `update`. Pass either "
                "labels (the whole array) or labels_add/labels_remove (atomic operations), not "
                "both."
            )
        labels_add_values = list(labels_add or [])
        labels_remove_values = [
            value for value in labels_remove or [] if value not in labels_add_values
        ]
        update["labels"] = [{"add": value} for value in labels_add_values] + [
            {"remove": value} for value in labels_remove_values
        ]
    if fix_versions_add or fix_versions_remove:
        if fix_versions is not None:
            raise ValueError(
                "Cannot set and update `fixVersions` in the same request: Jira rejects a PUT "
                "that both sets a field via `fields` and updates it via `update`. Pass either "
                "fix_versions (the whole array) or fix_versions_add/fix_versions_remove (atomic "
                "operations), not both."
            )
        fix_versions_add_values = list(fix_versions_add or [])
        fix_versions_remove_values = [
            value for value in fix_versions_remove or [] if value not in fix_versions_add_values
        ]
        update["fixVersions"] = [
            {"add": {"name": value}} for value in fix_versions_add_values
        ] + [{"remove": {"name": value}} for value in fix_versions_remove_values]

    if not fields and not update:
        raise ValueError("At least one editable field must be provided")
    payload: dict[str, Any] = {}
    if fields:
        payload["fields"] = fields
    if update:
        payload["update"] = update
    return payload


def issue_link_payload(
    *,
    outward_issue_key: str,
    inward_issue_key: str,
    link_type: str = "Relates",
    comment: str | None = None,
) -> dict[str, Any]:
    """Build Jira's inverse REST representation of the requested business direction.

    ``outward_issue_key`` is the work item that displays the link type's outward label (for
    example, "blocks"). Jira's create endpoint names the *other* endpoint ``outwardIssue``.
    Jira attaches an inline ``comment`` to that REST ``outwardIssue``, so with this business-name
    interface the comment lands on ``inward_issue_key``. CLI/MCP surface and audit that target
    explicitly rather than implying the first key receives it.
    """
    payload: dict[str, Any] = {
        "type": {"name": link_type},
        "inwardIssue": {"key": outward_issue_key},
        "outwardIssue": {"key": inward_issue_key},
    }
    if comment:
        payload["comment"] = {"body": text_to_adf(comment)}
    return payload


def issue_type_allowed_values(metadata: dict[str, Any]) -> list[str]:
    """Extract the allowed issue-type names from one `get_edit_metadata` response.

    A single place both the CLI's `issue-type-options` command and the MCP
    `jira_issue_type_options` tool call, so they report the same list under the same key
    (`allowed_values`) rather than two names for the one fact `editmeta` gives them.
    """
    issuetype_field = (metadata.get("fields") or {}).get("issuetype") or {}
    return [
        str(value.get("name"))
        for value in issuetype_field.get("allowedValues") or []
        if isinstance(value, dict) and value.get("name")
    ]


def select_transition(
    data: dict[str, Any],
    transition_name_or_id: str,
) -> dict[str, Any] | None:
    transitions_data = data.get("transitions") or []
    if not isinstance(transitions_data, list):
        return None

    needle = transition_name_or_id.casefold()
    for transition in transitions_data:
        if not isinstance(transition, dict):
            continue
        transition_id = str(transition.get("id") or "")
        name = str(transition.get("name") or "")
        if transition_id == transition_name_or_id or name.casefold() == needle:
            return transition
    return None


# Jira's three status categories, in workflow order. Probed fact: `statusCategory.id` is 2/4/3
# for To Do/In Progress/Done respectively -- NOT ordinal -- so this module orders by NAME, never
# by that numeric id. Keys are casefolded; lookups through `status_category_rank` casefold too.
_STATUS_CATEGORY_ORDER = {"to do": 0, "in progress": 1, "done": 2}


def status_category_rank(category_name: str | None) -> int | None:
    """Rank one Jira status-category NAME on the fixed To Do < In Progress < Done order.

    Returns None for a missing or unrecognized name rather than guessing a position: an absent
    `statusCategory`, a typo, or a future fourth category must read as "cannot be ordered", never
    as the smallest or largest rank.
    """
    if not category_name:
        return None
    return _STATUS_CATEGORY_ORDER.get(category_name.strip().casefold())


def issue_status_category(issue: dict[str, Any]) -> str | None:
    """Read `fields.status.statusCategory.name` off one issue GET response, or None."""
    fields = issue.get("fields") or {}
    status = fields.get("status") or {}
    category = status.get("statusCategory") or {}
    name = category.get("name")
    return str(name) if name else None


def transition_target_category(transition: dict[str, Any]) -> str | None:
    """Read `to.statusCategory.name` off one transition entry, or None."""
    target = transition.get("to") or {}
    category = target.get("statusCategory") or {}
    name = category.get("name")
    return str(name) if name else None


def describe_transition_options(transitions_data: dict[str, Any]) -> list[dict[str, Any]]:
    """List every transition Jira currently offers: id, name, target status, target category.

    Shared by both surfaces' "the name I was given did not match" and "Jira offered none at all"
    refusals, so a caller sees exactly what to try next instead of a dead end. Probed fact: two of
    three real FC issues checked returned an EMPTY transitions list here -- that reflects the
    authenticated account's current permissions against the workflow, not a dead workflow, and
    callers are told so explicitly when the list is empty (see each surface's own refusal text).
    """
    transitions_list = transitions_data.get("transitions")
    if not isinstance(transitions_list, list):
        return []
    options: list[dict[str, Any]] = []
    for transition in transitions_list:
        if not isinstance(transition, dict):
            continue
        target = transition.get("to") or {}
        options.append(
            {
                "id": transition.get("id"),
                "name": transition.get("name"),
                "to": target.get("name"),
                "to_category": (target.get("statusCategory") or {}).get("name"),
            }
        )
    return options


class TransitionDirectionRefusedError(RuntimeError):
    """Refuses a transition whose target status category is EARLIER than the issue's current one
    on the fixed To Do < In Progress < Done order, or whose direction could not be determined at
    all because a status-category name was missing from Jira's response (`direction="unknown"`).

    This is a check on the category information Jira's own response happens to carry, NOT a
    safety guarantee about the site's actual workflow graph: a workflow can be configured so a
    forward-looking category still represents a real regression in practice, and this check
    cannot see that. Only the caller's explicit opt-in
    (`--allow-backwards-transition` / `allow_backwards_transition=true`) overrides it, for either
    `direction`.
    """

    def __init__(
        self,
        *,
        issue_key: str,
        transition_id: str,
        transition_name: str,
        current_category: str | None,
        target_category: str | None,
        direction: str,
    ) -> None:
        if direction == "backward":
            detail = (
                f"transition {transition_name!r} ({transition_id}) would move {issue_key} from "
                f"category {current_category!r} back to {target_category!r}"
            )
        elif direction == "unknown":
            detail = (
                f"the direction of transition {transition_name!r} ({transition_id}) on "
                f"{issue_key} could not be determined (current category={current_category!r}, "
                f"target category={target_category!r})"
            )
        else:
            raise ValueError(f"Unrecognized direction: {direction!r}")
        super().__init__(f"Refusing {detail}.")
        self.issue_key = issue_key
        self.transition_id = transition_id
        self.transition_name = transition_name
        self.current_category = current_category
        self.target_category = target_category
        self.direction = direction


def check_transition_direction(
    *,
    issue_key: str,
    transition: dict[str, Any],
    current_category: str | None,
    allow_backwards: bool,
) -> None:
    """Raise TransitionDirectionRefusedError unless `transition` moves `issue_key` forward or
    keeps it in the same status category, or the caller explicitly passed `allow_backwards=True`.

    Called identically from both surfaces' dry run and their `--execute` / `execute=true` path,
    so a clean dry run predicts the real one. "Won't Do" moving To Do straight to Done is FORWARD
    and is allowed without the flag; a same-category move is allowed too. Only a target category
    ranked EARLIER than the current one, or a direction this function cannot determine at all
    (either category name missing or unrecognized), is refused.
    """
    target_category = transition_target_category(transition)
    current_rank = status_category_rank(current_category)
    target_rank = status_category_rank(target_category)

    direction: str | None
    if current_rank is None or target_rank is None:
        direction = "unknown"
    elif target_rank < current_rank:
        direction = "backward"
    else:
        direction = None

    if direction is not None and not allow_backwards:
        raise TransitionDirectionRefusedError(
            issue_key=issue_key,
            transition_id=str(transition.get("id") or ""),
            transition_name=str(transition.get("name") or ""),
            current_category=current_category,
            target_category=target_category,
            direction=direction,
        )


class TransitionFieldsMissingError(RuntimeError):
    """Refuses a transition Jira marks as requiring one or more fields the caller did not supply.

    `missing` is a list of `{"field": <field id>, "allowed_values": [str, ...] | None}` -- the
    field's `allowedValues` names when Jira's `expand=transitions.fields` provided them (see
    FC-458's probed `resolution` field), or None when it did not. Raised before any write, from
    both surfaces' dry run and their execute path, so a caller sees exactly which `--field
    NAME=VALUE` (CLI) / `fields` entry (MCP) to add, instead of a transition that fails outright
    at Jira -- or, worse, one that silently could never be performed at all.
    """

    def __init__(self, *, transition_name: str, missing: list[dict[str, Any]]) -> None:
        parts = []
        for entry in missing:
            allowed = entry.get("allowed_values")
            if allowed:
                parts.append(f"{entry['field']} (allowed: {', '.join(allowed)})")
            else:
                parts.append(str(entry["field"]))
        super().__init__(
            f"Transition {transition_name!r} requires field(s) not supplied: {', '.join(parts)}."
        )
        self.transition_name = transition_name
        self.missing = missing


def transition_required_fields_missing(
    transition: dict[str, Any],
    supplied_fields: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return every field `transition`'s own `fields` metadata marks required and
    `supplied_fields` does not name with a non-empty value, each with its `allowedValues` names
    when Jira provided any.

    Reads `transition["fields"]`, populated only when the transitions read used
    `expand=transitions.fields` (see `JiraClient.get_transitions`); a transition with no `fields`
    key has nothing to check against and this returns an empty list, exactly as if nothing were
    required.
    """
    fields_meta = transition.get("fields")
    if not isinstance(fields_meta, dict):
        return []
    supplied = supplied_fields or {}
    missing_value = object()
    missing: list[dict[str, Any]] = []
    for name, meta in fields_meta.items():
        if not isinstance(meta, dict) or not meta.get("required"):
            continue
        value = supplied.get(name, missing_value)
        # `0` and `False` can be legitimate values for required numeric/boolean fields. `None`
        # and a blank string cannot satisfy a required transition field, though. Treating either
        # as present only postpones the refusal until after the real write is attempted.
        if value is not missing_value and value is not None and (
            not isinstance(value, str) or bool(value.strip())
        ):
            continue
        allowed_values = meta.get("allowedValues")
        allowed_names = None
        if isinstance(allowed_values, list):
            allowed_names = [
                str(value.get("name"))
                for value in allowed_values
                if isinstance(value, dict) and value.get("name")
            ] or None
        missing.append({"field": name, "allowed_values": allowed_names})
    return sorted(missing, key=lambda entry: str(entry["field"]))


def transition_fields_payload(
    transition: dict[str, Any],
    supplied_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """Shape caller-supplied transition field values against `transition`'s own `fields`
    metadata, keyed exactly as Jira named each field.

    A field whose own schema has type `resolution` is sent as `{"name": value}`, the shape Jira's
    transition POST expects for that reference field; every other field is sent as the plain
    supplied value. The transition metadata can expose `allowedValues` for many other kinds of
    fields (including an array option), for which `{"name": value}` is not a valid universal
    representation. Therefore this decision keys on the schema type, never merely on the presence
    of `allowedValues`, and never on the field id.
    """
    fields_meta = transition.get("fields")
    if not isinstance(fields_meta, dict):
        fields_meta = {}
    payload: dict[str, Any] = {}
    for name, value in (supplied_fields or {}).items():
        meta = fields_meta.get(name)
        schema = meta.get("schema") if isinstance(meta, dict) else None
        if isinstance(schema, dict) and schema.get("type") == "resolution":
            payload[name] = {"name": value}
        else:
            payload[name] = value
    return payload
