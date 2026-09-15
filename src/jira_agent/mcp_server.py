from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from jira_agent.audit import audit_committed_write, audit_write
from jira_agent.client import (
    DEFAULT_FIELDS,
    DEFAULT_SPRINT_FIELD,
    DEFAULT_STORY_POINTS_FIELD,
    AdfPatchInvariantError,
    AdfSectionPatchRefused,
    JiraApiError,
    JiraClient,
    JiraConcurrentEditError,
    JiraMissingConcurrencyTokenError,
    SprintMultiRemovalError,
    SprintPreconditionError,
    SprintReassignmentError,
    TransitionDirectionRefusedError,
    TransitionFieldsMissingError,
    adf_contains_media_id,
    check_transition_direction,
    create_issue_payload,
    describe_transition_options,
    edit_issue_payload,
    issue_link_payload,
    issue_status_category,
    issue_type_allowed_values,
    normalize_jira_issue_key,
    normalize_jira_numeric_id,
    patch_adf_description_section,
    png_dimensions,
    replace_diagram_with_media,
    require_concurrency_token,
    require_matching_jira_numeric_id,
    require_shared_write_scope,
    select_transition,
    sprint_add_blast_radius,
    sprint_add_verification_results,
    sprint_membership_report,
    sprint_removal_blast_radius,
    sprint_removal_verification_results,
    transition_fields_payload,
    transition_required_fields_missing,
    validate_agile_issue_batch,
    validate_attachment_upload_response,
)
from jira_agent.config import load_config
from jira_agent.image_gallery import (
    image_gallery_blocks,
    load_image_gallery_manifest,
    newest_attachments_by_filename,
    upsert_top_level_section,
)
from jira_agent.mermaid import (
    discover_mermaid_runtime,
    embed_mermaid_diagrams,
    mermaid_sources_from_adf,
    render_mermaid_png,
)
from jira_agent.reporting import adf_to_text, agent_plan_comment_body, fetch_agent_report
from jira_agent.safety import (
    JiraWriteSafetyError,
    ensure_issue_write_allowed,
    ensure_project_write_allowed,
    write_scope_summary,
)
from jira_agent.workflows import (
    drift_label_operations,
    drift_reconcile_comment_body,
    factory_contract_comment_body,
    issue_contains_link_id,
    read_text_source,
    resolve_directed_issue_link,
    resolve_png_attachment,
)

mcp = FastMCP("jira-agent")


def _client() -> JiraClient:
    return JiraClient(load_config())


def _fields_or_default(fields: list[str] | None) -> list[str]:
    """The shared default field set, imported from `client.DEFAULT_FIELDS` so this surface and
    the CLI's own `DEFAULT_FIELDS` cannot drift apart again — see
    `test_default_fields_are_the_single_shared_list` in `test_surface_parity.py`, plus the
    config-resolved sprint field appended on top.

    This already includes `parent`; there is no longer a separate single-issue default, because
    the divergence that used to require one (this surface's list-shaped tools omitting `parent`
    while the CLI's did not) is exactly the inconsistency this shared list closes. The sprint
    field is deliberately NOT folded into `DEFAULT_FIELDS` itself: that constant is shared with
    the CLI and every other caller of it, and the sprint field id is instance-specific, resolved
    from config at call time, not a literal safe to hardcode into a shared list. An explicit
    `fields` argument is used exactly as given, with no field appended to it — matching
    `test_jira_get_issue_explicit_fields_are_not_augmented_with_parent`'s existing rule for
    `parent`.
    """
    if fields:
        return fields
    return [*DEFAULT_FIELDS, _sprint_field()]


def _story_points_field() -> str:
    """Dry runs must show the same field id the write would use, but must not fail
    when the environment is incomplete."""
    try:
        return load_config().story_points_field
    except Exception:
        return DEFAULT_STORY_POINTS_FIELD


def _sprint_field() -> str:
    """Same contract as `_story_points_field`: show the field id a write/read would actually
    use, without failing a caller (for example a dry run, or the default-fields augmentation in
    `_fields_or_default`) just because the environment is incomplete."""
    try:
        return load_config().sprint_field
    except Exception:
        return DEFAULT_SPRINT_FIELD


def _error(exc: JiraApiError) -> dict[str, Any]:
    return {
        "ok": False,
        "status_code": exc.status_code,
        "message": str(exc),
        "details": exc.details,
    }


def _safety_error(exc: JiraWriteSafetyError) -> dict[str, Any]:
    return {
        "ok": False,
        "message": str(exc),
        "error_type": "write_safety",
    }


def _transition_not_found_error(
    *, issue_key: str, requested: str, transitions_data: dict[str, Any]
) -> dict[str, Any]:
    """Shape an unmatched transition name/id by listing what IS available, per issue -- the same
    refusal `cli._exit_transition_not_found` builds, so both surfaces read identically. An EMPTY
    list gets its own message rather than silence: two of three FC issues probed against the
    live instance returned an empty transitions list here, meaning no permission or no
    applicable transition for the authenticated account right now, not the end of the workflow.
    """
    options = describe_transition_options(transitions_data)
    if options:
        message = f"No transition found for: {requested}"
    else:
        message = (
            f"Jira returned no transitions for {issue_key} for the authenticated account. This "
            "means no permission or no applicable transition right now, not the end of the "
            "workflow."
        )
    return {
        "ok": False,
        "error_type": "transition_not_found",
        "issue": issue_key,
        "requested": requested,
        "available_transitions": options,
        "message": message,
    }


def _transition_direction_error(exc: TransitionDirectionRefusedError) -> dict[str, Any]:
    """Shape a TransitionDirectionRefusedError the same way `cli._exit_transition_direction_refused`
    does, so a caller sees the identical fact regardless of which surface answered it.
    """
    return {
        "ok": False,
        "error_type": "transition_direction_refused",
        "issue": exc.issue_key,
        "transition": {"id": exc.transition_id, "name": exc.transition_name},
        "current_category": exc.current_category,
        "target_category": exc.target_category,
        "direction": exc.direction,
        "message": str(exc),
    }


def _transition_fields_missing_error(exc: TransitionFieldsMissingError) -> dict[str, Any]:
    """Shape a TransitionFieldsMissingError the same way `cli._exit_transition_fields_missing`
    does, so a caller sees the identical fact regardless of which surface answered it.
    """
    return {
        "ok": False,
        "error_type": "transition_fields_missing",
        "transition": exc.transition_name,
        "missing_fields": exc.missing,
        "message": str(exc),
    }


def _concurrent_edit_error(exc: JiraConcurrentEditError) -> dict[str, Any]:
    """Shape a JiraConcurrentEditError as a structured, actionable refusal.

    Names both timestamps at the top level, rather than leaving the caller to dig them out of
    a generic error's `details`, and keeps the underlying message, which already tells the
    caller to re-read the issue and retry.
    """
    return {
        "ok": False,
        "error_type": "concurrent_edit",
        "message": str(exc),
        "issue": exc.issue_key,
        "expected_updated": exc.expected_updated,
        "found_updated": exc.found_updated,
    }


def _cleanup_uploaded_attachments(config: Any, issue_key: str, attachment_ids: list[str]) -> None:
    """Best-effort delete of attachments a call uploaded when the write that would have
    referenced them never happened. Shared by jira_render_mermaid_diagrams's guard-refusal
    paths and its pre-existing generic-failure path, so a refused concurrency guard leaves the
    issue exactly as it stood before the call, the same way a transport failure already did.
    """
    if not attachment_ids:
        return
    try:
        with JiraClient(config) as cleanup_client:
            for attachment_id in attachment_ids:
                try:
                    cleanup_client.delete_attachment(attachment_id, issue_key=issue_key)
                except JiraApiError:
                    pass
    except JiraApiError:
        pass


def _delete_attachments_best_effort(
    client: JiraClient, issue_key: str, attachment_ids: list[str]
) -> tuple[list[str], list[dict[str, str]]]:
    """Delete superseded attachments without turning a committed description write into failure."""
    deleted: list[str] = []
    failures: list[dict[str, str]] = []
    for attachment_id in attachment_ids:
        try:
            client.delete_attachment(attachment_id, issue_key=issue_key)
            deleted.append(attachment_id)
        except JiraApiError as exc:
            failures.append({"attachment_id": attachment_id, "message": str(exc)})
    return deleted, failures


def _guard_unavailable_error(exc: JiraMissingConcurrencyTokenError) -> dict[str, Any]:
    """Shape a JiraMissingConcurrencyTokenError as the one `concurrent_edit_guard_unavailable`
    outcome every guarded tool returns for "no usable token to guard with", whichever call site
    raised it: `require_concurrency_token` on a same-request read, or `issue_version` on a
    dedicated one. `jira_patch_description_section` built this document inline before this
    shared helper existed; every guarded tool added since routes through it instead, so the
    condition cannot drift back to being tested three different ways across the file.
    """
    return {
        "ok": False,
        "error_type": "concurrent_edit_guard_unavailable",
        "issue": exc.issue_key,
        "message": str(exc),
    }


# /search/jql itself reports no match count at all (only `issues`, `isLast`, and an opaque
# `nextPageToken`); see JiraClient.search_all_issues. This bound is the tool surface's own
# per-page size, used only to decide when a requested `limit` needs more than one request.
_SEARCH_PAGE_SIZE = 100


def _normalize_is_last(payload: dict[str, Any]) -> dict[str, Any]:
    """Rename an aggregated search payload's completeness key to this file's `is_last`.

    `client.search_all_issues` already returns `is_last`, not Jira's own `isLast`; that
    migration is complete. This stays tolerant of either spelling on input anyway, as a cheap
    defence against a future regression on that side, and always emits `is_last` on output.
    Never invents a value: a payload carrying neither key stays `None`, rather than the `True`
    this used to default to, so an aggregation that never confirmed completeness is never
    reported as complete.
    """
    result = dict(payload)
    if "is_last" in result:
        value = result.pop("is_last")
        result.pop("isLast", None)
    elif "isLast" in result:
        value = result.pop("isLast")
    else:
        value = None
    result["is_last"] = value
    return result


def _search_result(
    client: JiraClient,
    jql: str,
    *,
    limit: int,
    fields: list[str] | None,
    project: str | None = None,
    next_page_token: str | None = None,
) -> dict[str, Any]:
    """Run one JQL search and shape the result without ever inventing a match count or claim.

    Always returns `returned` (the issue count actually in this response) and `is_last`, and
    never a `total` — `/search/jql` does not provide one. `is_last` is read from Jira's own
    `isLast` field on a single page and renamed to this file's snake_case convention; a
    delegated multi-page result from `search_all_issues` is normalized the same way regardless
    of which spelling it returns (see `_normalize_is_last`). Either path reports `is_last` as
    `None`, never `True`, when the response carried no completeness signal at all — reporting
    completeness the API never confirmed is exactly the fabricated certainty this avoids. A
    `limit` above one page pages automatically via `search_all_issues` unless the caller is
    already resuming a manual walk with `next_page_token`, in which case exactly one page is
    fetched.
    """
    if next_page_token is None and limit > _SEARCH_PAGE_SIZE:
        aggregated = client.search_all_issues(
            jql,
            limit=limit,
            fields=fields,
            project_key=project,
            page_size=_SEARCH_PAGE_SIZE,
        )
        return _normalize_is_last(aggregated)
    page = client.search_issues(
        jql,
        limit=limit,
        fields=fields,
        next_page_token=next_page_token,
        project_key=project,
    )
    issues = page.get("issues") or []
    raw_is_last = page.get("isLast")
    result: dict[str, Any] = {
        "issues": issues,
        "returned": len(issues),
        "is_last": bool(raw_is_last) if raw_is_last is not None else None,
    }
    next_token = page.get("nextPageToken")
    if next_token:
        result["nextPageToken"] = next_token
    return result


def _run_jql(jql: str, limit: int = 20, project: str | None = None) -> dict[str, Any]:
    try:
        with _client() as client:
            return {
                "ok": True,
                "jql": jql,
                "result": _search_result(
                    client, jql, limit=limit, fields=_fields_or_default(None), project=project
                ),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_myself(project: str | None = None) -> dict[str, Any]:
    """Return the authenticated Jira user. Pass a project key to check its credential set."""
    try:
        with _client() as client:
            return {"ok": True, "user": client.myself(key=project)}
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_get_issue(key: str, fields: list[str] | None = None) -> dict[str, Any]:
    """Fetch one Jira issue by key.

    Defaults to DEFAULT_FIELDS, matching the CLI, which already includes `parent`. `parent` is
    not part of Jira's own default field list, so passing an explicit `fields` list must name it
    to see it.
    """
    try:
        with _client() as client:
            issue = client.get_issue(key, fields=_fields_or_default(fields))
            return {"ok": True, "issue": issue}
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_search_issues(
    jql: str,
    limit: int = 20,
    fields: list[str] | None = None,
    next_page_token: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Search Jira issues with JQL. Pass a project key to search with its credential set.

    `/search/jql` reports no match count at all, only `issues`, `isLast`, and an opaque
    `nextPageToken`; this never invents one. The result carries `returned` (the issue count in
    this response) and `is_last` (renamed from Jira's `isLast`; `None` when the response gave
    no completeness signal, never defaulted to `True`), plus `nextPageToken` when more remain
    and you are not already paging past it automatically. A `limit` above 100 pages through
    every matching issue up to that limit on its own; pass `next_page_token` yourself only to
    resume a manual page walk.
    """
    try:
        with _client() as client:
            return {
                "ok": True,
                "result": _search_result(
                    client,
                    jql,
                    limit=limit,
                    fields=_fields_or_default(fields),
                    next_page_token=next_page_token,
                    project=project,
                ),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_recent_issues(days: int = 30, limit: int = 20) -> dict[str, Any]:
    """Fetch recently updated Jira issues with a bounded JQL query."""
    jql = f"updated >= -{days}d ORDER BY updated DESC"
    return jira_search_issues(jql, limit=limit)


@mcp.tool
def jira_mine(project: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Fetch unresolved issues assigned to the authenticated user."""
    clauses = ["assignee = currentUser()", "resolution = Unresolved"]
    if project:
        clauses.insert(0, f"project = {project}")
    return _run_jql(" AND ".join(clauses) + " ORDER BY updated DESC", limit=limit, project=project)


@mcp.tool
def jira_fc_open(limit: int = 20) -> dict[str, Any]:
    """Fetch open FC issues."""
    return _run_jql("project = FC AND statusCategory != Done ORDER BY updated DESC", limit=limit)


@mcp.tool
def jira_by_status(status: str, project: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Fetch issues by status."""
    escaped_status = status.replace('"', '\\"')
    clauses = [f'status = "{escaped_status}"']
    if project:
        clauses.insert(0, f"project = {project}")
    return _run_jql(" AND ".join(clauses) + " ORDER BY updated DESC", limit=limit, project=project)


@mcp.tool
def jira_test_scope_issues(limit: int = 5) -> dict[str, Any]:
    """Fetch issues from the configured test scope. Defaults to the FC queue/project."""
    try:
        config = load_config()
        with JiraClient(config) as client:
            return {
                "ok": True,
                "test_jql": config.test_jql,
                "result": _search_result(
                    client,
                    config.test_jql,
                    limit=limit,
                    fields=[*DEFAULT_FIELDS, config.sprint_field],
                ),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_create_issue(
    project_key: str,
    issue_type_name: str,
    summary: str,
    description: str | None = None,
    description_file: str | None = None,
    description_section: str | None = None,
    labels: list[str] | None = None,
    fix_versions: list[str] | None = None,
    parent_key: str | None = None,
    priority: str | None = None,
    story_points: float | None = None,
    execute: bool = False,
    render_mermaid: bool = True,
) -> dict[str, Any]:
    """Create an issue and automatically embed Mermaid diagrams. Defaults to dry-run.

    description_section is an input filter, not an in-place edit. It selects one heading body out
    of the Markdown supplied in description or description_file, and that extract becomes the
    entire description field. To change one section of an issue that already exists, use
    jira_patch_description_section. A parent_key is validated as an existing issue in the same
    Jira site and credential scope before either dry-run success or a write is reported.
    """
    config = load_config()
    try:
        ensure_project_write_allowed(config, project_key)
        body = (
            read_text_source(description, description_file, section=description_section)
            if description or description_file
            else None
        )
        payload = create_issue_payload(
            project_key=project_key,
            issue_type_name=issue_type_name,
            summary=summary,
            description=body,
            labels=labels,
            fix_versions=fix_versions,
            parent_key=parent_key,
            priority=priority,
            story_points=story_points,
            story_points_field=_story_points_field(),
        )
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}

    description_adf = payload["fields"].get("description")
    diagram_count = (
        len(mermaid_sources_from_adf(description_adf))
        if render_mermaid and isinstance(description_adf, dict)
        else 0
    )

    if parent_key is not None:
        try:
            require_shared_write_scope(config, [project_key, parent_key])
            with JiraClient(config) as parent_client:
                parent_client.get_issue(parent_key, fields=["project"])
        except JiraApiError as exc:
            return _error(exc)

    if not execute:
        return {
            "ok": True,
            "dry_run": True,
            "payload": payload,
            "mermaid_diagrams": diagram_count,
            "message": "Pass execute=true to create this issue in Jira.",
        }

    try:
        with JiraClient(config) as client:
            issue_result = client.create_issue(
                project_key=project_key,
                issue_type_name=issue_type_name,
                summary=summary,
                description=body,
                labels=labels,
                fix_versions=fix_versions,
                parent_key=parent_key,
                priority=priority,
                story_points=story_points,
            )
            create_audit_warning = audit_committed_write(
                config,
                action="create_issue",
                issue_key=issue_result.get("key"),
                payload=payload,
                result=issue_result,
                writer=audit_write,
            )
            result: dict[str, Any] = {"ok": True, "issue": issue_result}
            if create_audit_warning:
                result.update(
                    audit_warning=create_audit_warning,
                    partial_success=True,
                )
            if diagram_count and issue_result.get("key") and isinstance(description_adf, dict):
                try:
                    created_key = str(issue_result["key"])

                    def _audit_mermaid_commit(commit: dict[str, Any]) -> dict[str, Any] | None:
                        return audit_committed_write(
                            config,
                            action="render_mermaid_diagrams",
                            issue_key=created_key,
                            payload={
                                "files": commit["files"],
                                "attachment_ids": commit["attachment_ids"],
                                "cleanup_pending_attachment_ids": commit[
                                    "cleanup_pending_attachment_ids"
                                ],
                                "deleted_attachment_ids": [],
                            },
                            result=commit["result"],
                            writer=audit_write,
                        )

                    mermaid_result = embed_mermaid_diagrams(
                        client,
                        issue_key=created_key,
                        description=description_adf,
                        output_dir=Path(config.audit_log_path).parent
                        / "diagrams"
                        / created_key,
                        on_description_committed=_audit_mermaid_commit,
                    )
                    result["mermaid"] = mermaid_result
                except (JiraApiError, OSError, RuntimeError, ValueError) as exc:
                    result["mermaid"] = {"ok": False, "message": str(exc)}
            return result
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_create_test_issue(
    summary: str,
    description: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Create an issue in the configured test project. Defaults to dry-run."""
    config = load_config()
    payload = create_issue_payload(
        project_key=config.test_project_key,
        issue_type_name=config.test_issue_type,
        summary=summary,
        description=description,
    )
    try:
        ensure_project_write_allowed(config, config.test_project_key)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    if not execute:
        return {
            "ok": True,
            "dry_run": True,
            "payload": payload,
            "message": "Pass execute=true to create this issue in Jira.",
        }

    try:
        with JiraClient(config) as client:
            issue_result = client.create_issue(
                project_key=config.test_project_key,
                issue_type_name=config.test_issue_type,
                summary=summary,
                description=description,
            )
            create_audit_warning = audit_committed_write(
                config,
                action="create_issue",
                issue_key=issue_result.get("key"),
                payload=payload,
                result=issue_result,
                writer=audit_write,
            )
            result: dict[str, Any] = {
                "ok": True,
                "issue": issue_result,
            }
            if create_audit_warning:
                result.update(
                    audit_warning=create_audit_warning,
                    partial_success=True,
                )
            return result
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_get_test_issue() -> dict[str, Any]:
    """Fetch the configured integration-test issue."""
    config = load_config()
    if not config.test_issue_key:
        return {"ok": False, "message": "JIRA_TEST_ISSUE_KEY is not configured."}
    try:
        with JiraClient(config) as client:
            return {
                "ok": True,
                "issue": client.get_issue(config.test_issue_key, fields=DEFAULT_FIELDS),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_assign_issue(
    key: str | None = None,
    account_id: str | None = None,
    unassigned: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Assign a Jira issue. Defaults to the configured test issue and dry-run."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }

    try:
        with JiraClient(config) as client:
            me = client.myself(key=issue_key)
            target_account_id = None if unassigned else account_id or str(me["accountId"])
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "target_account_id": target_account_id,
                    "target_user": None if unassigned else me.get("displayName"),
                    "message": "Pass execute=true to assign this issue in Jira.",
                }

            ensure_issue_write_allowed(config, client, issue_key)
            result = client.assign_issue(issue_key, target_account_id)
            audit_write(
                config,
                action="assign_issue",
                issue_key=issue_key,
                payload={"accountId": target_account_id},
                result=result,
            )
            return {
                "ok": True,
                "issue": issue_key,
                "target_account_id": target_account_id,
            }
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_agent_report(
    key: str | None = None,
    comments_limit: int | Literal["all"] = 5,
    changelog_limit: int = 0,
) -> dict[str, Any]:
    """Return compact issue context for agents.

    comments_limit controls how many comments the report carries, never how many the issue
    actually has: the report always carries `comment_count` (the API's true total),
    `comments_returned` (how many are actually attached), and `comments_truncated` (true
    whenever comments_returned is less than comment_count), so a reader never has to infer one
    from the other. Both an integer and the literal "all" return comments newest first: an
    integer fetches that many of the newest comments, and "all" pages through every comment and
    returns them newest first too, rather than the API's native oldest-first order. The report
    also carries `comments_order`, always "newest-first", so a caller never has to infer the
    order from the count alone.

    comments_limit must be a non-negative integer, or the literal "all"; a negative integer is
    refused with a structured error rather than reaching Jira as a negative maxResults.
    """
    if isinstance(comments_limit, int) and comments_limit < 0:
        return {
            "ok": False,
            "message": (
                f"comments_limit must be a non-negative integer or 'all', got {comments_limit!r}."
            ),
        }

    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }

    try:
        with JiraClient(config) as client:
            return {
                "ok": True,
                "report": fetch_agent_report(
                    client,
                    issue_key=issue_key,
                    site_url=config.site_url,
                    comment_limit=comments_limit,
                    changelog_limit=changelog_limit,
                    write_scope=config.test_issue_key,
                ),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_edit_issue(
    key: str | None = None,
    summary: str | None = None,
    description: str | None = None,
    description_file: str | None = None,
    description_section: str | None = None,
    labels: list[str] | None = None,
    fix_versions: list[str] | None = None,
    parent_key: str | None = None,
    priority: str | None = None,
    story_points: float | None = None,
    labels_add: list[str] | None = None,
    labels_remove: list[str] | None = None,
    fix_versions_add: list[str] | None = None,
    fix_versions_remove: list[str] | None = None,
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Edit issue fields with Markdown-to-ADF rendering. Defaults to dry-run.

    description and description_file replace the whole description field. description_section does
    not narrow that write: it is an input filter that selects one heading body out of the Markdown
    you supply, and that extract then becomes the entire description, dropping every other section
    the issue had. To change one section and keep the rest, use jira_patch_description_section.

    labels and fix_versions each replace the whole array outright: omitting one leaves that field
    untouched, an empty list clears it, and a concurrent editor's own change to the same field is
    silently lost when this call also names it -- guard_concurrent_edit exists for exactly this.

    labels_add/labels_remove and fix_versions_add/fix_versions_remove are atomic instead: Jira
    applies each one against whatever the field currently holds server-side (`editmeta`
    advertises `add`/`remove` operations for both `labels` and `fixVersions` on this site), so
    they cannot lose a concurrent editor's own change and never need a guard token. labels cannot
    be combined with labels_add/labels_remove, and fix_versions cannot be combined with
    fix_versions_add/fix_versions_remove: Jira rejects a PUT that both sets and updates the same
    field, and this refuses the same way before sending it.

    guard_concurrent_edit defaults to true: when this call includes at least one whole-field
    write (summary, description, labels, fix_versions, parent_key, priority, or story_points),
    the issue's `updated` timestamp is read right before the write and passed back as a
    concurrency guard, so a change that lands between that read and this write is refused with
    error_type "concurrent_edit" naming both timestamps, instead of being silently overwritten.
    When that dedicated read itself carries no usable `updated`, this refuses with error_type
    "concurrent_edit_guard_unavailable" instead of a generic error, the same outcome every other
    guarded tool produces for the same condition. Pass false to skip the guard. A call with no
    whole-field write at all -- only labels_add/labels_remove/fix_versions_add/
    fix_versions_remove -- skips this read entirely regardless of guard_concurrent_edit: there is
    nothing there for a token to protect. A parent_key is validated as an existing issue in the
    same Jira site and credential scope as the child before either dry-run success or a write is
    reported.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }

    if labels is not None and (labels_add or labels_remove):
        return {
            "ok": False,
            "message": (
                "Cannot combine labels with labels_add/labels_remove: Jira rejects a PUT that "
                "both sets and updates the same field. labels replaces the whole array and "
                "carries the concurrency guard; labels_add/labels_remove are atomic and carry "
                "none. Use one form or the other."
            ),
        }
    if fix_versions is not None and (fix_versions_add or fix_versions_remove):
        return {
            "ok": False,
            "message": (
                "Cannot combine fix_versions with fix_versions_add/fix_versions_remove: Jira "
                "rejects a PUT that both sets and updates the same field. fix_versions replaces "
                "the whole array and carries the concurrency guard; "
                "fix_versions_add/fix_versions_remove are atomic and carry none. Use one form "
                "or the other."
            ),
        }

    try:
        body = (
            read_text_source(description, description_file, section=description_section)
            if description or description_file
            else None
        )
        payload = edit_issue_payload(
            summary=summary,
            description=body,
            labels=labels,
            fix_versions=fix_versions,
            parent_key=parent_key,
            priority=priority,
            story_points=story_points,
            labels_add=labels_add,
            labels_remove=labels_remove,
            fix_versions_add=fix_versions_add,
            fix_versions_remove=fix_versions_remove,
            story_points_field=config.story_points_field,
        )
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}

    try:
        with JiraClient(config) as client:
            if parent_key is not None:
                require_shared_write_scope(config, [issue_key, parent_key])
                client.get_issue(parent_key, fields=["project"])
            safety = ensure_issue_write_allowed(config, client, issue_key)
            expect_updated = (
                client.issue_version(issue_key)
                if guard_concurrent_edit and payload.get("fields")
                else None
            )
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "payload": payload,
                    "safety": safety,
                    "write_scope": write_scope_summary(config),
                    "expect_updated": expect_updated,
                    "message": "Pass execute=true to edit this issue in Jira.",
                }

            client.edit_issue(
                issue_key,
                summary=summary,
                description=body,
                labels=labels,
                fix_versions=fix_versions,
                parent_key=parent_key,
                priority=priority,
                story_points=story_points,
                labels_add=labels_add,
                labels_remove=labels_remove,
                fix_versions_add=fix_versions_add,
                fix_versions_remove=fix_versions_remove,
                expect_updated=expect_updated,
            )
            audit_write(
                config,
                action="edit_issue",
                issue_key=issue_key,
                payload=payload,
                result={},
            )
            return {"ok": True, "issue": issue_key}
    except JiraConcurrentEditError as exc:
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_clear_issue_parent(
    key: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Clear an issue parent explicitly. Defaults to dry-run and verifies the reread.

    G2 decision: this deliberately carries no concurrency guard. The whole-field rewrites the
    guard protects (description, labels) read an existing value, transform it, and write the
    transformed whole field back, so an unrelated concurrent change to that same field is
    silently discarded when the stale read overwrites it. Clearing a parent is not that shape:
    it sets one relationship field to a single, specific, verified target state (null) rather
    than folding a read into a computed replacement, and the reread below independently confirms
    that target state was reached regardless of what any concurrent editor did to `parent` in
    between. A guard would only add a refusal for a condition this command already tolerates by
    construction and already verifies.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }

    payload = edit_issue_payload(clear_parent=True, story_points_field=config.story_points_field)
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            current = client.get_issue(issue_key, fields=["parent"])
            before = (current.get("fields") or {}).get("parent")
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "before": before,
                    "payload": payload,
                    "safety": safety,
                    "write_scope": write_scope_summary(config),
                    "message": "Pass execute=true to clear this issue parent in Jira.",
                }

            result = client.edit_issue(issue_key, clear_parent=True)
            verified = client.get_issue(issue_key, fields=["parent"])
            after = (verified.get("fields") or {}).get("parent")
            audit_write(
                config,
                action="clear_parent",
                issue_key=issue_key,
                payload={"before": before, "payload": payload},
                result={"edit": result, "verified_parent": after},
            )
            if after is not None:
                return {
                    "ok": False,
                    "message": f"Parent clear was not verified for {issue_key}: {after!r}",
                    "issue": issue_key,
                    "verified_parent": after,
                }
            return {"ok": True, "issue": issue_key, "verified_parent": None}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}


_PATCH_SUMMARY_KEYS = (
    "heading",
    "level",
    "mode",
    "start_index",
    "end_index",
    "replaced_node_count",
    "new_node_count",
    "preserved_before",
    "preserved_after",
    "original_node_count",
    "patched_node_count",
    "whole_document",
    "swallowed_headings",
)


def _patch_summary(patch: dict[str, Any]) -> dict[str, Any]:
    """Return the counts that show at a glance whether a payload patches or replaces."""
    return {key: patch[key] for key in _PATCH_SUMMARY_KEYS}


@mcp.tool
def jira_patch_description_section(
    key: str,
    section: str,
    content: str | None = None,
    content_file: str | None = None,
    mode: str = "replace",
    include_heading: bool = True,
    expect_swallowed_headings: list[str] | None = None,
    allow_wide_rewrite: bool = False,
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Rewrite one section of an issue's existing description. Defaults to dry-run.

    Use this whenever only part of a description changes. Unlike jira_edit_issue, this tool reads
    the current ADF description and rewrites only the matched section. Every top-level node
    outside that section is carried over unchanged, keeping its taskItem ids and states, tables,
    panels, and inline marks. New checklist items are numbered past the highest task localId
    already in the description, so they cannot collide with a preserved one.

    The section is the matched heading plus every following node up to the next heading of the
    same or a lower level, so "## A" also covers its "### A.1" and stops before "## B". A heading
    with no such peer below it takes everything after it, whatever sits in front of it, so
    naming the document title can mean the whole field.

    replace and delete pass two independent controls, and neither substitutes for the other.

    First, arithmetic: if the slice is the whole content list, nothing outside it survives, and
    that is refused unless you pass allow_wide_rewrite.

    Second, a naming duty. swallowed_headings is the ordered text of every heading inside the
    slice below the one you named, so it is the list of extra sections the patch removes. Pass
    expect_swallowed_headings equal to that list, in that order. Absent, short, long, different,
    or supplied when the slice swallows nothing is a mismatch and is refused, and the refusal
    prints both lists so you can correct the call. A section is removed only if you named it. A
    heading with no text is listed and named as <untitled>.

    allow_wide_rewrite covers the first control only; it does not satisfy the naming duty.
    append and prepend remove nothing, so neither control applies to them. Naming the only
    section of a description carries no naming duty, because there is no unnamed section to
    lose, but control 1 still applies to it: an ordinary single-section description starts at
    index 0, so its slice is the whole content list and the flag is required. It goes through
    unflagged only when something precedes the section.

    A refusal returns error_type "section_patch_refused" together with swallowed_headings and
    expect_swallowed_headings as lists, so the correction is data and never a parsed sentence.

    guard_concurrent_edit defaults to true. The issue's `updated` timestamp is read in the same
    request that already reads the description, so the token belongs to that exact read, and is
    passed back as a concurrency guard on the write: an edit that lands in between is refused
    with error_type "concurrent_edit" naming both timestamps, instead of being silently
    reverted. When that same request's response omits `updated` entirely, there is no token to
    guard with; this refuses with error_type "concurrent_edit_guard_unavailable" rather than
    silently falling back to an unconditional write, and never issues a second read to get one,
    since a second read is exactly the straddling read the same-request design avoids. Pass
    false to skip the guard and go back to the previous unconditional read-then-overwrite
    behaviour.

    section names a top-level heading and is matched with leading # characters optional, case
    insensitively, and whitespace-trimmed. An unknown heading and an ambiguous heading are
    refused. An ADF level that is missing or out of range is never guessed: on the matched
    heading it is refused, and below it the section ends there. No level outside the slice is
    consulted.

    mode is replace (default), append, prepend, or delete. content or content_file carries the new
    Markdown and is required unless mode is delete. include_heading applies to replace: leave it
    true and start content with its own heading, or set it false to keep the existing heading node
    and change only the body below it; passing false with another mode is refused, because append
    and prepend always keep the heading. A patch that would leave the description with no content
    at all is refused.

    The dry run reports the matched heading, the replaced index range, and the top-level node
    counts before and after, so a destructive payload is visible before anything is written.
    """
    try:
        if mode == "delete":
            if content is not None or content_file is not None:
                return {
                    "ok": False,
                    "message": "Mode delete removes the section and takes no content.",
                }
            body = None
        else:
            body = read_text_source(content, content_file)
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}

    config = load_config()
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, key)
            issue = client.get_issue(key, fields=["description", "updated"])
            issue_fields = issue.get("fields") or {}
            read_updated = issue_fields.get("updated")
            # require_concurrency_token treats anything that is not a non-empty string as "no
            # usable token" -- not just None. A bare `read_updated is None` check let
            # `updated: ""` (a falsy but non-None value) through as `expect_updated=""`, which
            # then compared unequal to whatever the pre-PUT reread found and raised a misleading
            # "concurrent_edit" refusal instead of this guard-unavailable one.
            expect_updated = require_concurrency_token(
                key, read_updated, guard_enabled=guard_concurrent_edit
            )
            patch = patch_adf_description_section(
                issue_fields.get("description"),
                section=section,
                mode=mode,
                content=body,
                include_heading=include_heading,
                expect_swallowed_headings=expect_swallowed_headings,
                allow_wide_rewrite=allow_wide_rewrite,
            )
            summary = _patch_summary(patch)
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": key,
                    **summary,
                    "new_nodes": patch["new_nodes"],
                    "safety": safety,
                    "write_scope": write_scope_summary(config),
                    "expect_updated": expect_updated,
                    "message": "Pass execute=true to patch this section in Jira.",
                }

            result = client.edit_issue_adf_description(
                key, patch["document"], expect_updated=expect_updated
            )
            audit_write(
                config,
                action="patch_description_section",
                issue_key=key,
                payload={
                    "section": section,
                    "include_heading": include_heading,
                    "expect_swallowed_headings": expect_swallowed_headings,
                    "allow_wide_rewrite": allow_wide_rewrite,
                    "guard_concurrent_edit": guard_concurrent_edit,
                    **summary,
                },
                result=result,
            )
            return {"ok": True, "issue": key, **summary}
    except AdfSectionPatchRefused as exc:
        return {
            "ok": False,
            "error_type": "section_patch_refused",
            "message": str(exc),
            "swallowed_headings": exc.swallowed_headings,
            "expect_swallowed_headings": exc.named_headings,
            "whole_document": exc.whole_document,
        }
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}
    except AdfPatchInvariantError as exc:
        return {"ok": False, "message": str(exc), "error_type": "patch_invariant"}
    except JiraConcurrentEditError as exc:
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_agent_plan_comment(
    key: str | None = None,
    plan: str = "Review the issue context and proceed with the next verified action.",
    next_step: str | None = None,
    risk: str | None = None,
    result: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Prepare or add a standardized agent plan comment. Defaults to dry-run."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }

    try:
        with JiraClient(config) as client:
            report = fetch_agent_report(
                client,
                issue_key=issue_key,
                site_url=config.site_url,
                comment_limit=1,
                write_scope=config.test_issue_key,
            )
            body = agent_plan_comment_body(
                report,
                plan=plan,
                next_step=next_step,
                risk=risk,
                result=result,
            )
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "body": body,
                    "message": "Pass execute=true to add this comment in Jira.",
                }

            ensure_issue_write_allowed(config, client, issue_key)
            comment_result = client.add_comment(issue_key, body)
            audit_write(
                config,
                action="agent_plan_comment",
                issue_key=issue_key,
                payload={"body": body},
                result={"id": comment_result.get("id")},
            )
            return {"ok": True, "comment": comment_result}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_write_human_brief(
    body: str,
    key: str | None = None,
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Write a Human Brief into the Jira description. Defaults to dry-run.

    This replaces the whole description field, so a concurrent edit to that field would
    otherwise be silently overwritten. guard_concurrent_edit defaults to true: the issue's
    `updated` timestamp is read right before the write and passed back as a concurrency guard,
    matching jira_edit_issue. A change landing between that read and the write is refused with
    error_type "concurrent_edit"; a read with no usable `updated` refuses with error_type
    "concurrent_edit_guard_unavailable". Pass false to skip the guard.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    payload = edit_issue_payload(description=body)
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "payload": payload,
                    "safety": safety,
                    "message": "Pass execute=true to write the Human Brief.",
                }
            expect_updated = None
            if guard_concurrent_edit:
                read_updated = client.get_issue(issue_key, fields=["updated"])
                expect_updated = require_concurrency_token(
                    issue_key,
                    (read_updated.get("fields") or {}).get("updated"),
                    guard_enabled=True,
                )
            result = client.edit_issue(issue_key, description=body, expect_updated=expect_updated)
            audit_write(
                config,
                action="human_brief",
                issue_key=issue_key,
                payload={**payload, "guard_concurrent_edit": guard_concurrent_edit},
                result=result,
            )
            return {"ok": True, "issue": issue_key}
    except JiraConcurrentEditError as exc:
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_store_factory_contract(
    body: str | None = None,
    key: str | None = None,
    file_path: str | None = None,
    mode: str = "comment",
    execute: bool = False,
) -> dict[str, Any]:
    """Store a Factory Contract as a comment or attachment. Defaults to dry-run."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    if mode not in {"comment", "attachment"}:
        return {"ok": False, "message": "mode must be comment or attachment"}
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if mode == "attachment":
                if not file_path:
                    return {"ok": False, "message": "file_path is required for attachment mode"}
                if not execute:
                    return {
                        "ok": True,
                        "dry_run": True,
                        "issue": issue_key,
                        "file_path": file_path,
                        "safety": safety,
                    }
                result = validate_attachment_upload_response(
                    client.add_attachment(issue_key, file_path)
                )
                audit_write(
                    config,
                    action="factory_contract_attachment",
                    issue_key=issue_key,
                    payload={"file": file_path},
                    result={"attachments": result},
                )
                return {"ok": True, "attachments": result}

            if not body:
                return {"ok": False, "message": "body is required for comment mode"}
            comment = factory_contract_comment_body(body, source_name=file_path)
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "body": comment,
                    "safety": safety,
                }
            result = client.add_comment(issue_key, comment)
            audit_write(
                config,
                action="factory_contract_comment",
                issue_key=issue_key,
                payload={"body": comment},
                result={"id": result.get("id")},
            )
            return {"ok": True, "comment": result}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_post_drift_reconcile_comment(
    summary: str,
    key: str | None = None,
    decision: str | None = None,
    details: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Post a Drift Reconcile comment. Defaults to dry-run."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    body = drift_reconcile_comment_body(summary=summary, decision=decision, details=details)
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "body": body,
                    "safety": safety,
                }
            result = client.add_comment(issue_key, body)
            audit_write(
                config,
                action="drift_reconcile_comment",
                issue_key=issue_key,
                payload={"body": body},
                result={"id": result.get("id")},
            )
            return {"ok": True, "comment": result}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_update_drift_labels(
    key: str | None = None,
    add: list[str] | None = None,
    remove: list[str] | None = None,
    clear: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Add or remove drift-* labels atomically. Defaults to dry-run.

    add and remove need no read of the issue at all: each named label becomes its own Jira
    `add`/`remove` operation on `labels` (`editmeta` advertises both for this field), applied
    server-side against whatever the array currently holds, so two agents adding different drift
    labels at the same time cannot lose each other's work. This tool carries NO concurrency
    guard on any path -- there is no whole-array read-modify-write left for a stale `updated`
    token to protect.

    clear=True is the one path that still reads the issue first, once, to find out which labels
    currently start with `drift-`; only those become explicit `remove` operations. A drift label
    added by someone else after that read is never named and survives, the same way add/remove
    alone already leave everything they were not told about untouched. That read still needs no
    guard: it only decides which `remove` operations to send, and each one is itself safe
    against a concurrent edit once sent.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            existing: list[str] | None = None
            if clear:
                issue = client.get_issue(issue_key, fields=["labels"])
                existing = (issue.get("fields") or {}).get("labels") or []
            add_labels, remove_labels = drift_label_operations(
                add=add, remove=remove, existing_labels=existing, clear=clear
            )
            if not add_labels and not remove_labels:
                # Nothing is written either way, but every other early return on this surface
                # carries `dry_run`, and a caller branching on that key for a byte-identical
                # execute=True vs. execute=False response would get a KeyError without it.
                return {
                    "ok": True,
                    "dry_run": not execute,
                    "issue": issue_key,
                    "labels_added": [],
                    "labels_removed": [],
                    "message": "Nothing to do: no labels to add or remove.",
                }
            payload = edit_issue_payload(
                labels_add=add_labels or None, labels_remove=remove_labels or None
            )
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "payload": payload,
                    "safety": safety,
                    "write_scope": write_scope_summary(config),
                }
            # Atomic add/remove: never pass expect_updated. There is no whole-array rewrite here
            # for a stale token to protect (see this tool's docstring and drift_label_operations).
            result = client.edit_issue(
                issue_key,
                labels_add=add_labels or None,
                labels_remove=remove_labels or None,
                expect_updated=None,
            )
            audit_write(
                config,
                action="drift_labels",
                issue_key=issue_key,
                payload=payload,
                result=result,
            )
            return {
                "ok": True,
                "issue": issue_key,
                "labels_added": add_labels,
                "labels_removed": remove_labels,
            }
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}


@mcp.tool
def jira_list_comments(
    key: str,
    limit: int | Literal["all"] = 20,
    start_at: int = 0,
    order_by: str | None = "-created",
) -> dict[str, Any]:
    """List comments for one Jira issue, NEWEST first by default.

    Jira's own default order for this endpoint is oldest-first, which is easy to mistake for
    "most recent" when only a handful come back: on a long issue the first page of the default
    order holds only the oldest comments, so a freshly written one stays invisible. The dominant
    use of this tool is checking whether your own comment landed, so order_by defaults here to
    "-created" (newest first) instead of Jira's native order; pass order_by=None to get that
    native oldest-first order back.

    limit="all" pages through every comment on the issue and ignores start_at. An integer limit
    returns one page of that size starting at start_at.

    The response carries `returned` (the number of comments actually in this response) and
    `total` (the API's real total comment count) as separate fields; never infer one from the
    other.

    limit (when an integer) must be at least 1, and start_at must be non-negative, matching the
    CLI's `comments` command; either violation is refused with a structured error rather than
    reaching Jira as a negative maxResults/startAt.
    """
    if isinstance(limit, int) and limit < 1:
        return {"ok": False, "message": f"limit must be at least 1 or 'all', got {limit!r}."}
    if start_at < 0:
        return {"ok": False, "message": f"start_at must be non-negative, got {start_at!r}."}
    try:
        with _client() as client:
            if limit == "all":
                page = client.list_all_comments(key, order_by=order_by)
            else:
                page = client.list_comments(key, start_at=start_at, limit=limit, order_by=order_by)
            comments = page.get("comments") or []
            return {
                "ok": True,
                "comments": comments,
                "returned": len(comments),
                "total": page.get("total"),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_user_search(
    query: str,
    key: str | None = None,
    project: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search assignable Jira users."""
    config = load_config()
    issue_key = key or config.test_issue_key
    project_key = project or (None if issue_key else config.test_project_key)
    try:
        with JiraClient(config) as client:
            return {
                "ok": True,
                "users": client.search_assignable_users(
                    query,
                    issue_key=issue_key,
                    project_key=project_key,
                    limit=limit,
                ),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_assign_user(
    query: str,
    key: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Assign an issue by searching exactly one active assignable user."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    try:
        with JiraClient(config) as client:
            users = client.search_assignable_users(query, issue_key=issue_key, limit=10)
            active_users = [user for user in users if user.get("active")]
            if len(active_users) != 1:
                return {
                    "ok": False,
                    "message": (
                        f"Expected exactly one active assignable user, got {len(active_users)}."
                    ),
                    "users": active_users,
                }
            user = active_users[0]
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "user": user,
                    "message": "Pass execute=true to assign this issue in Jira.",
                }

            ensure_issue_write_allowed(config, client, issue_key)
            result = client.assign_issue(issue_key, str(user["accountId"]))
            audit_write(
                config,
                action="assign_user",
                issue_key=issue_key,
                payload={"query": query, "accountId": user.get("accountId")},
                result=result,
            )
            return {"ok": True, "issue": issue_key, "user": user}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_add_comment(key: str, body: str, execute: bool = False) -> dict[str, Any]:
    """Add a comment to one Jira issue. Defaults to dry-run unless execute is true."""
    if not execute:
        return {
            "ok": True,
            "dry_run": True,
            "issue": key,
            "body": body,
            "message": "Pass execute=true to write this comment to Jira.",
        }

    try:
        with _client() as client:
            ensure_issue_write_allowed(client.config, client, key)
            comment_result = client.add_comment(key, body)
            audit_write(
                client.config,
                action="add_comment",
                issue_key=key,
                payload={"body": body},
                result={"id": comment_result.get("id")},
            )
            return {"ok": True, "comment": comment_result}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_update_comment(
    key: str,
    comment_id: str,
    body: str,
    execute: bool = False,
) -> dict[str, Any]:
    """Replace the body of one existing comment on a Jira issue. Defaults to dry-run.

    Use this to correct a comment in place instead of stacking a second comment that contradicts
    the first. Jira keeps the original author and created instant and records the edit separately,
    so the correction stays attributable. The dry run returns the current body so the replacement
    can be compared against what it overwrites.

    `current_body` is the current body flattened to plain text with the same flattener the CLI
    uses, matching the CLI's `edit-comment` dry run rather than exposing raw ADF under a name
    that reads as plain text. The raw ADF document is still available, under `current_body_adf`.
    The replacement text is `proposed_body`, matching the CLI's own key for the identical fact
    (this used to be spelled `new_body` here only).
    """
    try:
        key = normalize_jira_issue_key(key)
        comment_id = str(normalize_jira_numeric_id(comment_id, name="comment_id"))
        with _client() as client:
            safety = ensure_issue_write_allowed(client.config, client, key)
            try:
                current = client.get_comment(key, comment_id)
                require_matching_jira_numeric_id(current, comment_id, name="comment_id")
            except JiraApiError as exc:
                if exc.status_code == 404:
                    return {
                        "ok": False,
                        "message": f"Comment {comment_id} is not present on {key}.",
                    }
                raise
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": key,
                    "comment_id": comment_id,
                    "current_author": (current.get("author") or {}).get("displayName"),
                    "current_body": adf_to_text(current.get("body")).strip(),
                    "current_body_adf": current.get("body"),
                    "proposed_body": body,
                    "safety": safety,
                    "message": "Pass execute=true to replace this comment body in Jira.",
                }

            result = client.update_comment(key, comment_id, body)
            audit_write(
                client.config,
                action="update_comment",
                issue_key=key,
                payload={"comment_id": comment_id, "body": body},
                result={"id": result.get("id"), "updated": result.get("updated")},
            )
            return {"ok": True, "comment": result}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_list_transitions(key: str) -> dict[str, Any]:
    """List available transitions for one Jira issue.

    An EMPTY `transitions.transitions` list is not necessarily the end of the workflow: it means
    Jira offered none of its own applicable transitions to the authenticated account right now,
    which can mean no permission just as easily as no applicable transition (probed against the
    real instance: two of three FC issues checked hit exactly this). The response also carries
    each transition's `fields` metadata (required flags and `allowedValues`), via
    `expand=transitions.fields`, so a caller can see what `jira_transition_issue`'s `fields`
    argument would need before ever calling it.
    """
    try:
        with _client() as client:
            return {"ok": True, "transitions": client.get_transitions(key)}
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_link_issue(
    outward_key: str,
    inward_key: str,
    link_type: str = "Relates",
    comment: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Link two Jira issues by business direction. Defaults to dry-run.

    outward_key displays the link type's outward label (for example, "blocks"); inward_key
    displays its inward label (for example, "is blocked by"). Jira's create-link REST mapping is
    inverse to those business names, and its optional inline comment is written to the REST
    outwardIssue -- therefore to this tool's ``inward_key`` (the second argument). The dry run,
    result, and audit payload expose that exact ``comment_issue_key``.
    """
    config = load_config()
    payload = issue_link_payload(
        outward_issue_key=outward_key,
        inward_issue_key=inward_key,
        link_type=link_type,
        comment=comment,
    )
    try:
        with JiraClient(config) as client:
            require_shared_write_scope(config, [outward_key, inward_key])
            outward_safety = ensure_issue_write_allowed(config, client, outward_key)
            inward_safety = ensure_issue_write_allowed(config, client, inward_key)
            comment_issue_key = inward_key if comment else None
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "payload": payload,
                    "outward_safety": outward_safety,
                    "inward_safety": inward_safety,
                    **(
                        {"comment_issue_key": comment_issue_key}
                        if comment_issue_key is not None
                        else {}
                    ),
                    "message": "Pass execute=true to create this issue link in Jira.",
                }

            result = client.link_issues(
                outward_issue_key=outward_key,
                inward_issue_key=inward_key,
                link_type=link_type,
                comment=comment,
            )
            audit_write(
                config,
                action="link_issue",
                issue_key=comment_issue_key or outward_key,
                payload={**payload, "comment_issue_key": comment_issue_key},
                result=result,
            )
            response = {
                "ok": True,
                "outward_key": outward_key,
                "inward_key": inward_key,
            }
            if comment_issue_key is not None:
                response["comment_issue_key"] = comment_issue_key
            return response
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_delete_issue_link(
    outward_key: str,
    inward_key: str,
    link_type: str = "Relates",
    link_id: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Delete one validated directed issue link. Defaults to dry-run.

    Fresh reads of both endpoints must identify one link with the requested type and direction.
    Pass link_id to additionally pin the expected Jira id; otherwise ambiguity is refused.
    """
    try:
        outward_key = normalize_jira_issue_key(outward_key, name="outward_key")
        inward_key = normalize_jira_issue_key(inward_key, name="inward_key")
        if link_id is not None:
            link_id = str(normalize_jira_numeric_id(link_id, name="link_id"))
    except JiraApiError as exc:
        return _error(exc)
    config = load_config()
    try:
        with JiraClient(config) as client:
            require_shared_write_scope(config, [outward_key, inward_key])
            outward_safety = ensure_issue_write_allowed(config, client, outward_key)
            inward_safety = ensure_issue_write_allowed(config, client, inward_key)
            outward_issue = client.get_issue(outward_key, fields=["issuelinks"])
            inward_issue = client.get_issue(inward_key, fields=["issuelinks"])
            resolved = resolve_directed_issue_link(
                outward_issue=outward_issue,
                inward_issue=inward_issue,
                outward_key=outward_key,
                inward_key=inward_key,
                link_type=link_type,
                link_id=link_id,
            )
            resolved["id"] = str(normalize_jira_numeric_id(resolved["id"], name="link_id"))
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "resolved_link": resolved,
                    "outward_safety": outward_safety,
                    "inward_safety": inward_safety,
                    "write_scope": write_scope_summary(config),
                    "message": "Pass execute=true to delete this exact issue link in Jira.",
                }

            result = client.delete_issue_link(resolved["id"], issue_key=outward_key)
            verified_outward = client.get_issue(outward_key, fields=["issuelinks"])
            verified_inward = client.get_issue(inward_key, fields=["issuelinks"])
            still_present = {
                outward_key: issue_contains_link_id(verified_outward, resolved["id"]),
                inward_key: issue_contains_link_id(verified_inward, resolved["id"]),
            }
            audit_write(
                config,
                action="delete_issue_link",
                issue_key=outward_key,
                payload={"resolved_link": resolved},
                result={"delete": result, "still_present": still_present},
            )
            if any(still_present.values()):
                return {
                    "ok": False,
                    "message": f"Issue-link deletion was not verified for {resolved['id']}.",
                    "resolved_link": resolved,
                    "still_present": still_present,
                }
            return {"ok": True, "resolved_link": resolved, "still_present": still_present}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}


@mcp.tool
def jira_attach_file(
    file_path: str,
    key: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Attach a local file to a Jira issue. Defaults to dry-run."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "file_path": file_path,
                    "safety": safety,
                    "message": "Pass execute=true to upload this attachment to Jira.",
                }

            result = validate_attachment_upload_response(
                client.add_attachment(issue_key, file_path)
            )
            audit_write(
                config,
                action="attach",
                issue_key=issue_key,
                payload={"file": file_path},
                result={"attachments": result},
            )
            return {"ok": True, "attachments": result}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_delete_attachment(
    attachment_id: str,
    key: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Delete one attachment from an allowed Jira issue. Defaults to dry-run."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    try:
        issue_key = normalize_jira_issue_key(issue_key)
        attachment_id = str(normalize_jira_numeric_id(attachment_id, name="attachment_id"))
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["attachment"])
            attachments = issue.get("fields", {}).get("attachment") or []
            attachment = next(
                (item for item in attachments if str(item.get("id")) == attachment_id),
                None,
            )
            if attachment is None:
                return {
                    "ok": False,
                    "message": f"Attachment {attachment_id} is not present on {issue_key}.",
                }
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "attachment": attachment,
                    "safety": safety,
                    "message": "Pass execute=true to delete this attachment from Jira.",
                }
            result = client.delete_attachment(attachment_id, issue_key=issue_key)
            audit_write(
                config,
                action="delete_attachment",
                issue_key=issue_key,
                payload={
                    "attachment_id": attachment_id,
                    "filename": attachment.get("filename"),
                },
                result=result,
            )
            return {"ok": True, "issue": issue_key, "deleted_attachment_id": attachment_id}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_render_mermaid_diagrams(
    key: str | None = None,
    filenames: list[str] | None = None,
    alt_texts: list[str] | None = None,
    output_dir: str | None = None,
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Render Mermaid code blocks locally, upload PNGs, and embed them. Defaults to dry-run.

    This rewrites the whole description field. guard_concurrent_edit defaults to true, but like
    jira_embed_diagram, the token cannot come from the initial read (the one that finds the
    Mermaid blocks to render): uploading each rendered PNG is itself a write that moves this
    issue's own `updated` on the Jira side (see G1), so a token read before the uploads, or a
    description built from that same initial read, would already be out of date by the write.

    Instead, after every upload this re-reads the description and `updated` together in one
    request and builds the write from THAT read. The guard therefore covers only the span from
    this post-upload read to the write: a concurrent edit landing before that read is
    incorporated into the write rather than lost, because the write is built from the fresh
    document, not the initial one. This is still check-then-act, not an atomic compare-and-swap
    — Jira Cloud has no conditional write on this resource, so a change landing between the
    post-upload read and the write is still lost. A refused write, or any other failure once an
    upload has happened, cleans up the attachments already uploaded in this call.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }

    uploaded_ids: list[str] = []
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["description", "attachment"])
            description = issue.get("fields", {}).get("description") or {
                "type": "doc",
                "version": 1,
                "content": [],
            }
            sources = mermaid_sources_from_adf(description)
            if not sources:
                return {"ok": False, "message": f"No Mermaid code blocks found on {issue_key}."}

            names = filenames or [
                f"{issue_key.lower()}-mermaid-{index}.png" for index in range(1, len(sources) + 1)
            ]
            alts = alt_texts or [f"Mermaid diagram {index}" for index in range(1, len(sources) + 1)]
            if len(names) != len(sources) or len(alts) != len(sources):
                return {
                    "ok": False,
                    "message": "filenames and alt_texts must match the Mermaid block count.",
                }
            if len(set(names)) != len(names):
                return {"ok": False, "message": "Diagram filenames must be unique."}
            for name in names:
                path = Path(name)
                if path.name != name or path.suffix.lower() != ".png":
                    return {
                        "ok": False,
                        "message": f"Diagram filename must be a plain PNG filename: {name}",
                    }

            runtime = discover_mermaid_runtime()
            destination = (
                Path(output_dir)
                if output_dir
                else (Path(config.audit_log_path).parent / "diagrams" / issue_key)
            )
            paths = [destination / name for name in names]
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "diagram_count": len(sources),
                    "files": [str(path) for path in paths],
                    "runtime": {
                        "node": str(runtime.node),
                        "cli": str(runtime.cli),
                        "browser": str(runtime.browser),
                    },
                    "safety": safety,
                    "message": "Pass execute=true to render, upload, and embed the diagrams.",
                }

            for source, path in zip(sources, paths, strict=True):
                render_mermaid_png(source, path, runtime=runtime)

            attachments = issue.get("fields", {}).get("attachment") or []
            previous_ids = {
                str(attachment.get("id"))
                for attachment in attachments
                if attachment.get("filename") in names
            }
            new_attachments: list[dict[str, Any]] = []
            media: list[tuple[str, str, int, int]] = []
            for path, alt in zip(paths, alts, strict=True):
                uploaded = client.add_attachment(issue_key, path)
                if not uploaded:
                    raise JiraApiError(f"Jira returned no attachment metadata for {path.name}.")
                attachment = uploaded[0]
                attachment_id = str(attachment.get("id"))
                uploaded_ids.append(attachment_id)
                new_attachments.append(attachment)
                media_id = client.attachment_media_id(attachment_id, issue_key=issue_key)
                width, height = png_dimensions(path)
                media.append((media_id, alt, width, height))

            # Every upload above already changed this issue's `updated` on the Jira side, so
            # content and guard token are read together HERE, after the last mutation this call
            # makes before its write, rather than built from the (now possibly stale) first read
            # above.
            fresh_issue = client.get_issue(issue_key, fields=["description", "updated"])
            fresh_fields = fresh_issue.get("fields") or {}
            updated_description = fresh_fields.get("description") or {
                "type": "doc",
                "version": 1,
                "content": [],
            }
            for media_id, alt, width, height in media:
                updated_description = replace_diagram_with_media(
                    updated_description,
                    media_id=media_id,
                    alt=alt,
                    width=width,
                    height=height,
                )
            expect_updated = require_concurrency_token(
                issue_key, fresh_fields.get("updated"), guard_enabled=guard_concurrent_edit
            )
            result = client.edit_issue_adf_description(
                issue_key, updated_description, expect_updated=expect_updated
            )
            committed_attachment_ids = list(uploaded_ids)
            to_delete = sorted(previous_ids - set(committed_attachment_ids))
            # The write above succeeded, so every id in uploaded_ids is now referenced by the
            # live description. Clear it (after computing to_delete, which must still exclude
            # them) so a failure in the OLD-attachment cleanup below cannot trigger this call's
            # own except clauses into deleting the attachment(s) the write just committed.
            uploaded_ids.clear()
            audit_warning = audit_committed_write(
                config,
                action="render_mermaid_diagrams",
                issue_key=issue_key,
                payload={
                    "files": [str(path) for path in paths],
                    "attachment_ids": committed_attachment_ids,
                    "cleanup_pending_attachment_ids": to_delete,
                    "deleted_attachment_ids": [],
                    "guard_concurrent_edit": guard_concurrent_edit,
                },
                result=result,
                writer=audit_write,
            )
            deleted, cleanup_failures = _delete_attachments_best_effort(
                client, issue_key, to_delete
            )
            return {
                "ok": True,
                "issue": issue_key,
                "attachments": new_attachments,
                "deleted_attachment_ids": deleted,
                "cleanup_failures": cleanup_failures,
                "audit_warning": audit_warning,
                "partial_success": bool(audit_warning or cleanup_failures),
            }
    # JiraApiError (and its subclasses JiraConcurrentEditError and JiraMissingConcurrencyTokenError)
    # is itself a RuntimeError, and JiraWriteSafetyError is *also* a RuntimeError — directly
    # (safety.py), not through JiraApiError — so all of these Jira-specific clauses must come
    # BEFORE the generic (OSError, RuntimeError, ValueError) catch-all below. Reversed, that
    # catch-all would swallow every one of them first, exactly as it did before this fix: the
    # attachment cleanup on failure was unreachable dead code for any real Jira API failure.
    except JiraConcurrentEditError as exc:
        _cleanup_uploaded_attachments(config, issue_key, uploaded_ids)
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        _cleanup_uploaded_attachments(config, issue_key, uploaded_ids)
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        _cleanup_uploaded_attachments(config, issue_key, uploaded_ids)
        return _error(exc)
    except (OSError, RuntimeError, ValueError) as exc:
        _cleanup_uploaded_attachments(config, issue_key, uploaded_ids)
        return {"ok": False, "message": str(exc)}


@mcp.tool
def jira_embed_diagram(
    file_path: str,
    key: str | None = None,
    alt: str = "Architecture diagram",
    delete_filenames: list[str] | None = None,
    delete_attachment_ids: list[str] | None = None,
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Upload a PNG diagram and embed it in the Jira description. Defaults to dry-run.

    This rewrites the whole description field. guard_concurrent_edit defaults to true, but
    unlike jira_embed_existing_png_attachment and jira_upsert_image_gallery, the token cannot
    come from the first read (the one that fetches attachments to resolve delete_filenames and
    populate the dry-run response): uploading the file is itself a write that moves this issue's
    own `updated` on the Jira side (see G1), so a description and token both taken from that
    first read would already be out of date by the write.

    Instead, after the upload this re-reads the description and `updated` together in one
    request and builds the write from THAT read. The guard therefore covers only the span from
    this post-upload read to the write: a concurrent edit landing before that read is
    incorporated into the write rather than lost, because the write is built from the fresh
    document, not the first-read one. This is still check-then-act, not an atomic
    compare-and-swap — Jira Cloud has no conditional write on this resource, so a change landing
    between the post-upload read and the write is still lost. A refused write, or any other
    failure once the upload has happened, cleans up the attachment already uploaded in this call.
    Pass false to skip the guard.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    try:
        width, height = png_dimensions(file_path)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}

    delete_names = set(delete_filenames or [])
    try:
        delete_ids = {
            str(normalize_jira_numeric_id(value, name="delete_attachment_id"))
            for value in (delete_attachment_ids or [])
        }
    except JiraApiError as exc:
        return _error(exc)
    uploaded_ids: list[str] = []
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["description", "attachment"])
            attachments = issue.get("fields", {}).get("attachment") or []
            current_attachment_ids = {
                str(attachment.get("id"))
                for attachment in attachments
                if isinstance(attachment, dict) and attachment.get("id") is not None
            }
            unknown_delete_ids = delete_ids - current_attachment_ids
            if unknown_delete_ids:
                return {
                    "ok": False,
                    "message": (
                        "Every delete_attachment_id must belong to the target issue; not present "
                        f"on {issue_key}: {', '.join(sorted(unknown_delete_ids))}."
                    ),
                }
            for attachment in attachments:
                if attachment.get("filename") in delete_names:
                    delete_ids.add(str(attachment.get("id")))
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "file_path": file_path,
                    "dimensions": {"width": width, "height": height},
                    "delete_attachment_ids": sorted(delete_ids),
                    "safety": safety,
                    "message": "Pass execute=true to upload and embed this diagram.",
                }
            uploaded = client.add_attachment(issue_key, file_path)
            if not uploaded:
                raise JiraApiError(
                    f"Jira returned no attachment metadata for {Path(file_path).name}."
                )
            attachment = uploaded[0]
            attachment_id = str(attachment.get("id"))
            uploaded_ids.append(attachment_id)
            media_id = client.attachment_media_id(attachment_id, issue_key=issue_key)
            # The upload above already changed this issue's `updated` on the Jira side, so
            # content and guard token are read together HERE, after the last mutation this call
            # makes before its write, rather than built from the first (now possibly stale) read
            # above.
            fresh_issue = client.get_issue(issue_key, fields=["description", "updated"])
            fresh_fields = fresh_issue.get("fields") or {}
            description = replace_diagram_with_media(
                fresh_fields.get("description") or {"type": "doc", "version": 1, "content": []},
                media_id=media_id,
                alt=alt,
                width=width,
                height=height,
            )
            expect_updated = require_concurrency_token(
                issue_key, fresh_fields.get("updated"), guard_enabled=guard_concurrent_edit
            )
            result = client.edit_issue_adf_description(
                issue_key, description, expect_updated=expect_updated
            )
            # The write above succeeded, so the new attachment is now referenced by the live
            # description. Clear it so a failure while deleting OTHER, already-superseded
            # attachments below cannot trigger this call's own except clauses into deleting the
            # attachment the write just committed.
            uploaded_ids.clear()
            cleanup_pending_ids = sorted(delete_ids - {attachment_id})
            audit_warning = audit_committed_write(
                config,
                action="embed_diagram",
                issue_key=issue_key,
                payload={
                    "file": file_path,
                    "attachment_id": attachment_id,
                    "media_id": media_id,
                    "cleanup_pending_attachment_ids": cleanup_pending_ids,
                    "deleted_attachment_ids": [],
                    "dimensions": {"width": width, "height": height},
                    "guard_concurrent_edit": guard_concurrent_edit,
                },
                result=result,
                writer=audit_write,
            )
            deleted, cleanup_failures = _delete_attachments_best_effort(
                client, issue_key, cleanup_pending_ids
            )
            return {
                "ok": True,
                "issue": issue_key,
                "attachment": attachment,
                "media_id": media_id,
                "deleted_attachment_ids": deleted,
                "cleanup_failures": cleanup_failures,
                "audit_warning": audit_warning,
                "partial_success": bool(audit_warning or cleanup_failures),
            }
    except JiraConcurrentEditError as exc:
        _cleanup_uploaded_attachments(config, issue_key, uploaded_ids)
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        _cleanup_uploaded_attachments(config, issue_key, uploaded_ids)
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        _cleanup_uploaded_attachments(config, issue_key, uploaded_ids)
        return _error(exc)


@mcp.tool
def jira_embed_existing_png_attachment(
    filename: str,
    key: str | None = None,
    attachment_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
    alt: str = "Architecture diagram",
    replace_existing: bool = False,
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Embed one existing PNG attachment without uploading a duplicate. Defaults to dry-run.

    filename must identify exactly one current PNG attachment unless attachment_id pins the match.
    width and height are required because display dimensions are never guessed. When
    replace_existing is true, exactly one inline media item must belong to an attachment with the
    same filename; no unrelated media is replaced.

    This rewrites the whole description field. guard_concurrent_edit defaults to true: unlike
    jira_embed_diagram, embedding an already-uploaded attachment performs no Jira-changing call
    between the read and the write, so the same read that fetches the description and
    attachments also fetches `updated`, and that token guards the write. Pass false to skip it.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    if width is None or height is None or width < 1 or height < 1:
        return {"ok": False, "message": "width and height must both be positive integers."}

    try:
        issue_key = normalize_jira_issue_key(issue_key)
        if attachment_id is not None:
            attachment_id = str(normalize_jira_numeric_id(attachment_id, name="attachment_id"))
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["description", "attachment", "updated"])
            expect_updated = require_concurrency_token(
                issue_key,
                (issue.get("fields") or {}).get("updated"),
                guard_enabled=guard_concurrent_edit,
            )
            attachment = resolve_png_attachment(
                issue, filename=filename, attachment_id=attachment_id
            )
            resolved_id = str(normalize_jira_numeric_id(attachment["id"], name="attachment_id"))
            media_id = client.attachment_media_id(resolved_id, issue_key=issue_key)
            existing_description = issue.get("fields", {}).get("description") or {
                "type": "doc",
                "version": 1,
                "content": [],
            }
            replace_media_id: str | None = None
            if replace_existing:
                same_filename = [
                    item
                    for item in issue.get("fields", {}).get("attachment") or []
                    if isinstance(item, dict) and item.get("filename") == filename
                ]
                embedded_ids = [
                    client.attachment_media_id(str(item["id"]), issue_key=issue_key)
                    for item in same_filename
                    if item.get("id") is not None
                ]
                matches = [
                    item_id
                    for item_id in embedded_ids
                    if adf_contains_media_id(existing_description, item_id)
                ]
                if len(matches) != 1:
                    return {
                        "ok": False,
                        "message": (
                            "Expected exactly one embedded media item from attachments named "
                            f"{filename!r}; found {len(matches)}."
                        ),
                    }
                replace_media_id = matches[0]

            payload = {
                "attachment_id": resolved_id,
                "filename": filename,
                "media_id": media_id,
                "dimensions": {"width": width, "height": height},
                "replace_media_id": replace_media_id,
            }
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "attachment": attachment,
                    "payload": payload,
                    "safety": safety,
                    "write_scope": write_scope_summary(config),
                    "message": "Pass execute=true to embed this existing attachment in Jira.",
                }

            description = replace_diagram_with_media(
                existing_description,
                media_id=media_id,
                alt=alt,
                width=width,
                height=height,
                replace_media_id=replace_media_id,
            )
            result = client.edit_issue_adf_description(
                issue_key, description, expect_updated=expect_updated
            )
            verified = client.get_issue(issue_key, fields=["description", "attachment"])
            verified_attachment = resolve_png_attachment(
                verified, filename=filename, attachment_id=resolved_id
            )
            verified_description = (verified.get("fields") or {}).get("description") or {}
            embedded = adf_contains_media_id(verified_description, media_id)
            audit_write(
                config,
                action="embed_existing_png",
                issue_key=issue_key,
                payload={**payload, "guard_concurrent_edit": guard_concurrent_edit},
                result={
                    "edit": result,
                    "verified_attachment_id": verified_attachment["id"],
                    "embedded": embedded,
                },
            )
            if not embedded:
                return {
                    "ok": False,
                    "message": (
                        f"Existing attachment {resolved_id} was not found inline after reread."
                    ),
                    "attachment_id": resolved_id,
                }
            return {
                "ok": True,
                "issue": issue_key,
                "attachment": verified_attachment,
                "media_id": media_id,
                "embedded": True,
            }
    except JiraConcurrentEditError as exc:
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}


@mcp.tool
def jira_upsert_image_gallery(
    manifest_path: str,
    key: str | None = None,
    documentation_filename: str | None = None,
    before_heading: str = "Child Issues",
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Insert or replace a linked screenshot gallery using existing Jira attachments.

    This rewrites the whole description field. guard_concurrent_edit defaults to true: this
    tool performs no Jira-changing call between the read and the write, so the same read that
    fetches the description and attachments also fetches `updated`, and that token guards the
    write. Pass false to skip it.
    """
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {"ok": False, "message": "No issue key provided."}
    try:
        manifest = load_image_gallery_manifest(manifest_path)
        filenames = {
            str(row["file"]) for section in manifest["sections"] for row in section["rows"]
        }
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["description", "attachment", "updated"])
            expect_updated = require_concurrency_token(
                issue_key,
                (issue.get("fields") or {}).get("updated"),
                guard_enabled=guard_concurrent_edit,
            )
            attachments = issue.get("fields", {}).get("attachment") or []
            by_name = newest_attachments_by_filename(attachments)
            missing = sorted(filenames - set(by_name))
            if documentation_filename and documentation_filename not in by_name:
                missing.append(documentation_filename)
            if missing:
                return {"ok": False, "message": "Missing Jira attachments.", "missing": missing}
            blocks = image_gallery_blocks(
                manifest,
                attachment_urls={name: str(by_name[name]["content"]) for name in filenames},
                documentation_url=(
                    str(by_name[documentation_filename]["content"])
                    if documentation_filename
                    else None
                ),
            )
            description = upsert_top_level_section(
                issue.get("fields", {}).get("description")
                or {"type": "doc", "version": 1, "content": []},
                title=str(manifest["title"]),
                blocks=blocks,
                before_heading=before_heading,
            )
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "title": manifest["title"],
                    "screenshot_count": len(filenames),
                    "before_heading": before_heading,
                    "safety": safety,
                }
            result = client.edit_issue_adf_description(
                issue_key, description, expect_updated=expect_updated
            )
            audit_write(
                config,
                action="upsert_image_gallery",
                issue_key=issue_key,
                payload={
                    "manifest": manifest_path,
                    "filenames": sorted(filenames),
                    "documentation_filename": documentation_filename,
                    "before_heading": before_heading,
                    "guard_concurrent_edit": guard_concurrent_edit,
                },
                result=result,
            )
            return {"ok": True, "issue": issue_key, "screenshot_count": len(filenames)}
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}
    except JiraConcurrentEditError as exc:
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_changelog(key: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Return recent changelog entries for an issue."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
        }
    try:
        with JiraClient(config) as client:
            return {"ok": True, "changelog": client.get_changelog(issue_key, limit=limit)}
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_transition_issue(
    key: str,
    transition_name_or_id: str,
    execute: bool = False,
    allow_backwards_transition: bool = False,
    fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Transition one Jira issue by transition name or ID. Defaults to dry-run.

    Refuses (`error_type: "transition_direction_refused"`) by default when the chosen
    transition's target status category is EARLIER than the issue's current one on the fixed
    To Do < In Progress < Done order, or when either category could not be read from Jira's
    response at all -- pass allow_backwards_transition=true to proceed deliberately. This checks
    the category information Jira's response happens to carry, not a safety guarantee about the
    site's actual workflow: a differently configured workflow can make a forward-looking category
    a real regression anyway.

    Refuses (`error_type: "transition_fields_missing"`) before writing when the transition
    requires a field `fields` did not supply, naming it and, when Jira provided them, its allowed
    values. Pass `fields` as `{field_id: value}`; a reference field (like FC-458's `resolution`)
    is shaped as Jira's `{"name": value}` from that field's own metadata, not by special-casing
    the field id `resolution`.

    An unmatched `transition_name_or_id` returns `error_type: "transition_not_found"` together
    with every transition Jira DID offer (id, name, target status, target category), so a caller
    can pick one instead of guessing why the name did not match -- including the case where Jira
    offered none at all, which means no permission or no applicable transition right now, not the
    end of the workflow.

    Both refusals above run identically in the dry run and under execute=true, so a clean dry run
    predicts the real one.
    """
    try:
        with _client() as client:
            transitions = client.get_transitions(key)
            selected = select_transition(transitions, transition_name_or_id)
            if selected is None:
                return _transition_not_found_error(
                    issue_key=key,
                    requested=transition_name_or_id,
                    transitions_data=transitions,
                )

            issue = client.get_issue(key, fields=["status"])
            current_category = issue_status_category(issue)

            try:
                check_transition_direction(
                    issue_key=key,
                    transition=selected,
                    current_category=current_category,
                    allow_backwards=allow_backwards_transition,
                )
            except TransitionDirectionRefusedError as exc:
                return _transition_direction_error(exc)

            missing = transition_required_fields_missing(selected, fields)
            if missing:
                fields_exc = TransitionFieldsMissingError(
                    transition_name=str(selected.get("name") or ""), missing=missing
                )
                return _transition_fields_missing_error(fields_exc)

            fields_payload = transition_fields_payload(selected, fields)

            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": key,
                    "transition": selected,
                    "current_category": current_category,
                    "fields": fields_payload,
                    "message": "Pass execute=true to transition this issue in Jira.",
                }

            ensure_issue_write_allowed(client.config, client, key)
            client.transition_issue(key, str(selected["id"]), fields=fields_payload or None)
            audit_write(
                client.config,
                action="transition_issue",
                issue_key=key,
                payload={"transition": selected, "fields": fields_payload},
                result={},
            )
            return {"ok": True, "issue": key, "transition": selected, "fields": fields_payload}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_delete_comment(key: str, comment_id: str, execute: bool = False) -> dict[str, Any]:
    """Delete one comment from a Jira issue. Defaults to dry-run.

    Reads the comment first, matching the jira_update_comment convention, so both the dry run
    and a refusal on a missing comment show what would be deleted. `current_body` is flattened
    to plain text with the same flattener the CLI uses, matching its `delete-comment` dry run;
    the raw ADF document is still available, under `current_body_adf`.
    """
    try:
        key = normalize_jira_issue_key(key)
        comment_id = str(normalize_jira_numeric_id(comment_id, name="comment_id"))
        with _client() as client:
            safety = ensure_issue_write_allowed(client.config, client, key)
            try:
                current = client.get_comment(key, comment_id)
                require_matching_jira_numeric_id(current, comment_id, name="comment_id")
            except JiraApiError as exc:
                if exc.status_code == 404:
                    return {
                        "ok": False,
                        "message": f"Comment {comment_id} is not present on {key}.",
                    }
                raise
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": key,
                    "comment_id": comment_id,
                    "current_author": (current.get("author") or {}).get("displayName"),
                    "current_body": adf_to_text(current.get("body")).strip(),
                    "current_body_adf": current.get("body"),
                    "safety": safety,
                    "message": "Pass execute=true to delete this comment in Jira.",
                }

            client.delete_comment(key, comment_id)
            audit_write(
                client.config,
                action="delete_comment",
                issue_key=key,
                payload={"comment_id": comment_id},
                result={},
            )
            return {"ok": True, "issue": key, "deleted_comment_id": comment_id}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_issue_type_options(key: str) -> dict[str, Any]:
    """List one issue's allowed issue-type values for a type change.

    Reads editmeta, the same source jira_set_issue_type checks before writing, so an agent can
    see what an issue may change to without guessing and hitting the refusal. The key is
    `allowed_values`, matching the CLI's `issue-type-options` output and the Jira editmeta field
    (`issuetype.allowedValues`) it comes from — this used to be spelled `allowed_issue_types`
    here only, so the same fact had two names depending on which surface answered it.
    """
    try:
        with _client() as client:
            metadata = client.get_edit_metadata(key)
            allowed_values = issue_type_allowed_values(metadata)
            return {"ok": True, "issue": key, "allowed_values": allowed_values}
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_set_issue_type(
    key: str,
    issue_type_name: str,
    guard_concurrent_edit: bool = True,
    execute: bool = False,
) -> dict[str, Any]:
    """Change one issue's issue type. Defaults to dry-run.

    Checks the target issue's editmeta first and refuses immediately when issue_type_name is not
    among its allowed issuetype values, instead of sending a PUT Jira would reject. Guarded by
    the same concurrency check as jira_edit_issue and jira_patch_description_section: pass
    guard_concurrent_edit=false to skip it. A read with no usable `updated` refuses with
    error_type "concurrent_edit_guard_unavailable", the same outcome every other guarded tool
    produces for the same condition.
    """
    try:
        with _client() as client:
            safety = ensure_issue_write_allowed(client.config, client, key)
            if not client.is_issue_type_allowed(key, issue_type_name):
                return {
                    "ok": False,
                    "message": f"{issue_type_name!r} is not an allowed issue type for {key}.",
                }
            expect_updated = client.issue_version(key) if guard_concurrent_edit else None
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": key,
                    "issue_type_name": issue_type_name,
                    "expect_updated": expect_updated,
                    "safety": safety,
                    "message": "Pass execute=true to change this issue's type in Jira.",
                }

            client.set_issue_type(key, issue_type_name, expect_updated=expect_updated)
            audit_write(
                client.config,
                action="set_issue_type",
                issue_key=key,
                payload={
                    "issue_type_name": issue_type_name,
                    "guard_concurrent_edit": guard_concurrent_edit,
                },
                result={},
            )
            return {"ok": True, "issue": key, "issue_type_name": issue_type_name}
    except JiraConcurrentEditError as exc:
        return _concurrent_edit_error(exc)
    except JiraMissingConcurrencyTokenError as exc:
        return _guard_unavailable_error(exc)
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_list_boards(
    board_type: str | None = None,
    name: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """List Jira Agile boards, optionally filtered by type (for example "scrum") or name.

    `max_results` defaults to 50 and this never fetches a second page, so a site with more
    boards than that has more than this response shows. Matching the CLI's `boards` command,
    the `boards` document carries `returned` (the board count actually in this response) and
    `is_last` (Jira's own `isLast`, `None` when the response carried no such signal) alongside
    the raw fields; neither surface invents a `total` Jira did not report.
    """
    try:
        with _client() as client:
            boards = client.list_boards(board_type=board_type, name=name, max_results=max_results)
            values = boards.get("values") or []
            return {
                "ok": True,
                "boards": {**boards, "returned": len(values), "is_last": boards.get("isLast")},
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_list_board_sprints(
    board_id: int | str,
    state: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """List sprints on one Jira Agile board. state optionally filters to active/future/closed.

    `max_results` defaults to 50 and this never fetches a second page, so a board with more
    sprints than that has more than this response shows. Matching the CLI's `sprints` command,
    the `sprints` document carries `returned` and `is_last` (Jira's own `isLast`, `None` when
    absent) alongside the raw fields; neither surface invents a `total` Jira did not report.
    """
    try:
        with _client() as client:
            sprints = client.list_board_sprints(board_id, state=state, max_results=max_results)
            values = sprints.get("values") or []
            return {
                "ok": True,
                "sprints": {
                    **sprints,
                    "returned": len(values),
                    "is_last": sprints.get("isLast"),
                },
            }
    except JiraApiError as exc:
        return _error(exc)


# Sentinel so a sprint-field key genuinely absent from `fields` is distinguishable from one
# present with a `None` or empty-list value; mirrors the CLI's own `_SPRINT_FIELD_MISSING`.
_SPRINT_FIELD_MISSING = object()


def _sprint_field_or_missing(fields: dict[str, Any], sprint_field: str) -> Any:
    """Read a sprint custom field, distinguishing absent from present-but-empty.

    `dict.get(key, _SPRINT_FIELD_MISSING)` returns the sentinel only when the key was never in
    Jira's response at all, so a caller can tell "this site's sprint field id is probably wrong,
    or not visible to this credential" apart from "the field is there and this issue really has
    no sprint" — two states `JiraClient.get_issue_sprints` documents that it collapses into the
    same empty list.
    """
    return fields.get(sprint_field, _SPRINT_FIELD_MISSING)


@mcp.tool
def jira_get_issue_sprints(key: str, sprint_field: str | None = None) -> dict[str, Any]:
    """Return the sprint objects on one issue's sprint field.

    sprint_field defaults to config.sprint_field (JIRA_SPRINT_FIELD), resolved from config at
    call time -- customfield_10020, the id probed on this Jira site, unless overridden; pass a
    different id if this site's Agile sprint field uses another custom field id.

    `sprint_field_present` tells apart two states an empty `sprints` list alone cannot: the
    field absent from Jira's response entirely (this site's sprint field id is probably
    different, or the field is not visible with the caller's permissions) and the field present
    -- with `null` (this Jira site's own shape for "in no sprint", verified via `Sprint IS EMPTY`
    JQL) or an empty list -- meaning the issue genuinely is in no sprint. Reads the issue
    directly rather than through `JiraClient.get_issue_sprints`, which documents that it
    collapses all three into the same `[]`, matching the CLI's `issue-sprint` command, which
    makes the same choice for the same reason.
    """
    try:
        with _client() as client:
            field = sprint_field or client.config.sprint_field
            issue = client.get_issue(key, fields=[field])
            raw = _sprint_field_or_missing(issue.get("fields") or {}, field)
            field_present = raw is not _SPRINT_FIELD_MISSING
            return {
                "ok": True,
                "issue": key,
                "sprint_field": field,
                "sprint_field_present": field_present,
                "sprints": raw if isinstance(raw, list) else [],
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_add_issues_to_sprint(
    sprint_id: int | str,
    issue_keys: list[str],
    execute: bool = False,
    allow_sprint_reassignment: bool = False,
) -> dict[str, Any]:
    """Move one or more issues to a sprint. Defaults to dry-run.

    Jira's endpoint is destructive move semantics, not additive membership: when a key is in a
    different active/future sprint, it leaves that sprint. Every invocation therefore reads the
    current sprint field before the dry run or write. ``precondition[].sprints_replaced`` exposes
    the blast radius, and the whole batch is refused unless ``allow_sprint_reassignment=true``.
    Closed historical sprint entries are retained. Jira accepts at most 50 keys per call.

    The Agile add endpoint returns 204 with no body, so Jira accepting the request is not proof
    the issue ended up in the sprint. The write is audited as soon as Jira accepts it -- before
    the post-write reread runs -- so a write that landed is never left with no audit record just
    because that reread afterwards failed. After execute=true, every key's sprint field is
    re-read and `results` reports, per key, whether it is now confirmed in `sprint_id`
    (`verified`). `ok` is false and a key can show `verified: false` even though the write itself
    succeeded, if the reread does not confirm it -- that reread, not the 204, is what this tool
    treats as proof. If the reread itself fails, this returns `ok: false`,
    `error_type: "sprint_verification_unavailable"`, and `verified: null` (there is no
    `results` without a successful reread), saying the write was accepted but could not be
    confirmed, rather than reporting the same shape as a genuine write failure that never wrote
    anything.
    """
    if not issue_keys:
        return {"ok": False, "message": "issue_keys must not be empty."}
    try:
        sprint_id = normalize_jira_numeric_id(sprint_id, name="sprint_id")
        with _client() as client:
            validate_agile_issue_batch(issue_keys)
            require_shared_write_scope(client.config, issue_keys)
            safety = {
                issue_key: ensure_issue_write_allowed(client.config, client, issue_key)
                for issue_key in issue_keys
            }
            sprint_field = client.config.sprint_field
            precondition = sprint_add_blast_radius(
                sprint_membership_report(client, issue_keys, sprint_id, sprint_field=sprint_field),
                sprint_id,
            )
            if any(entry["sprints_replaced"] for entry in precondition) and not (
                allow_sprint_reassignment
            ):
                exc = SprintReassignmentError(sprint_id, precondition)
                return {
                    "ok": False,
                    "error_type": "sprint_reassignment_refused",
                    "message": str(exc),
                    "sprint_id": sprint_id,
                    "report": precondition,
                }
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "sprint_id": sprint_id,
                    "issue_keys": issue_keys,
                    "safety": safety,
                    "precondition": precondition,
                    "allow_sprint_reassignment": allow_sprint_reassignment,
                    "message": "Pass execute=true to move these issues to the sprint in Jira.",
                }

            client.add_issues_to_sprint(sprint_id, issue_keys)

            verification_error: JiraApiError | None = None
            results: list[dict[str, Any]] | None = None
            try:
                results = sprint_add_verification_results(
                    precondition,
                    sprint_membership_report(
                        client, issue_keys, sprint_id, sprint_field=sprint_field
                    ),
                )
            except JiraApiError as exc:
                verification_error = exc

            audit_write(
                client.config,
                action="add_issues_to_sprint",
                issue_key=issue_keys[0],
                payload={
                    "sprint_id": sprint_id,
                    "issue_keys": issue_keys,
                    "precondition": precondition,
                    "allow_sprint_reassignment": allow_sprint_reassignment,
                },
                result=(
                    {"verification": results}
                    if verification_error is None
                    else {"verified": None, "verification_error": str(verification_error)}
                ),
            )

            if verification_error is not None:
                return {
                    "ok": False,
                    "error_type": "sprint_verification_unavailable",
                    "sprint_id": sprint_id,
                    "issue_keys": issue_keys,
                    "verified": None,
                    "message": (
                        "The sprint-add write was accepted by Jira, but the post-write "
                        f"verification reread failed, so it could not be confirmed: "
                        f"{verification_error}"
                    ),
                }

            ok = all(entry["verified"] for entry in results)
            return {
                "ok": ok,
                "sprint_id": sprint_id,
                "issue_keys": issue_keys,
                "results": results,
            }
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_move_issues_to_backlog(
    sprint_id: int | str,
    issue_keys: list[str],
    execute: bool = False,
    allow_multi_sprint_removal: bool = False,
) -> dict[str, Any]:
    """Remove one or more issues from sprint_id, moving them to the backlog. Defaults to dry-run.

    BREAKING CHANGE: this tool used to take only issue_keys and post them straight to the
    backlog endpoint. `sprint_id` is now required, and it is a PRECONDITION THIS TOOL ENFORCES,
    NOT A FILTER JIRA APPLIES: the underlying endpoint (`POST /rest/agile/1.0/backlog/issue`)
    takes no sprint id at all. It is documented as equivalent to removing the FUTURE and ACTIVE
    sprints from the given issues -- ALL of them, on every key in the batch, not only the one
    this tool was told to check. Closed sprints are unaffected. Jira normally permits only one
    current active/future sprint per issue; this tool nevertheless evaluates every non-closed
    entry Jira returns so anomalous data cannot hide destructive scope.

    Before any write (including the dry run, so it predicts what execute=true does), every key's
    current sprint field is re-read, and the whole batch is refused -- via
    `error_type: "sprint_precondition_failed"`, nothing written for any key -- if even one key is
    not currently in `sprint_id`; `report` lists exactly which sprint each mismatched key is
    actually in. Separately, and also by default, the whole batch is refused (via
    `error_type: "sprint_multi_removal_refused"`) if any key sits in another non-closed sprint
    besides `sprint_id`: without `allow_multi_sprint_removal=true`, that key would silently lose
    a sprint membership nobody named. `dry_run=true` already lists, per key, every sprint it will
    actually leave (`sprints_removed`), not only `sprint_id`, so this predicts the real blast
    radius rather than under-reporting it.

    The endpoint returns 204 with no body. The write is audited as soon as Jira accepts it --
    before the post-write reread runs -- so a write that landed is never left with no audit
    record just because that reread afterwards failed. After execute=true, every key's sprint
    field is re-read again and `results` reports, per key, whether it left every sprint the dry
    run said it would (not merely `sprint_id`), naming any of those it is still in (`still_in`).
    `ok` is false and a key can show `verified: false` even though the write itself succeeded, if
    it still verifies as present in one of them afterwards. If the reread itself fails, this
    returns `ok: false`, `error_type: "sprint_verification_unavailable"`, and `verified: null`,
    saying the write was accepted but could not be confirmed.
    """
    if not issue_keys:
        return {"ok": False, "message": "issue_keys must not be empty."}
    try:
        sprint_id = normalize_jira_numeric_id(sprint_id, name="sprint_id")
        with _client() as client:
            validate_agile_issue_batch(issue_keys)
            require_shared_write_scope(client.config, issue_keys)
            safety = {
                issue_key: ensure_issue_write_allowed(client.config, client, issue_key)
                for issue_key in issue_keys
            }

            sprint_field = client.config.sprint_field
            precondition = sprint_removal_blast_radius(
                sprint_membership_report(client, issue_keys, sprint_id, sprint_field=sprint_field),
                sprint_id,
            )
            if any(not entry["in_sprint"] for entry in precondition):
                exc = SprintPreconditionError(sprint_id, precondition)
                return {
                    "ok": False,
                    "error_type": "sprint_precondition_failed",
                    "message": str(exc),
                    "sprint_id": sprint_id,
                    "report": precondition,
                }

            at_risk = [entry for entry in precondition if entry["other_sprints"]]
            if at_risk and not allow_multi_sprint_removal:
                exc = SprintMultiRemovalError(sprint_id, precondition)
                return {
                    "ok": False,
                    "error_type": "sprint_multi_removal_refused",
                    "message": str(exc),
                    "sprint_id": sprint_id,
                    "report": precondition,
                }

            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "sprint_id": sprint_id,
                    "issue_keys": issue_keys,
                    "safety": safety,
                    "precondition": precondition,
                    "message": "Pass execute=true to move these issues to the backlog in Jira.",
                }

            client.move_issues_to_backlog(issue_keys)

            verification_error: JiraApiError | None = None
            results: list[dict[str, Any]] | None = None
            try:
                post_write = sprint_removal_blast_radius(
                    sprint_membership_report(
                        client, issue_keys, sprint_id, sprint_field=sprint_field
                    ),
                    sprint_id,
                )
                results = sprint_removal_verification_results(precondition, post_write)
            except JiraApiError as exc:
                verification_error = exc

            audit_write(
                client.config,
                action="move_issues_to_backlog",
                issue_key=issue_keys[0],
                payload={
                    "sprint_id": sprint_id,
                    "issue_keys": issue_keys,
                    "pre_write_sprints": {entry["key"]: entry["sprints"] for entry in precondition},
                },
                result=(
                    {"verification": results}
                    if verification_error is None
                    else {"verified": None, "verification_error": str(verification_error)}
                ),
            )

            if verification_error is not None:
                return {
                    "ok": False,
                    "error_type": "sprint_verification_unavailable",
                    "sprint_id": sprint_id,
                    "issue_keys": issue_keys,
                    "verified": None,
                    "message": (
                        "The sprint-remove write was accepted by Jira, but the post-write "
                        f"verification reread failed, so it could not be confirmed: "
                        f"{verification_error}"
                    ),
                }

            ok = all(entry["verified"] for entry in results)
            return {
                "ok": ok,
                "sprint_id": sprint_id,
                "issue_keys": issue_keys,
                "results": results,
            }
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_config_fields() -> dict[str, Any]:
    """Show the custom field ids this tool has resolved from config. Read-only, no Jira call.

    Mirrors the CLI's `config-fields` command. `story_points_field` and `sprint_field` are both
    instance-specific Jira custom field ids -- resolved the same way in `config.py`, from a
    `JIRA_STORY_POINTS_FIELD` / `JIRA_SPRINT_FIELD` env var, or this site's probed default
    (`customfield_10028` / `customfield_10020`) -- and this tool exists so a caller can check
    which ids are actually in effect instead of guessing, before running anything that writes.

    On this site: customfield_10028 is "Story Points", populated and editable for project FC,
    and is the correct story_points_field default. customfield_10016 is "Story point estimate"
    -- it exists, but is unused and not editable for FC, and must never be configured instead.
    """
    config = load_config()
    return {
        "ok": True,
        "story_points_field": config.story_points_field,
        "sprint_field": config.sprint_field,
    }


def main() -> None:
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
