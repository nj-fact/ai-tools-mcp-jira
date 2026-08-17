from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from jira_agent.audit import audit_write
from jira_agent.client import (
    DEFAULT_STORY_POINTS_FIELD,
    JiraApiError,
    JiraClient,
    create_issue_payload,
    edit_issue_payload,
    issue_link_payload,
    png_dimensions,
    replace_diagram_with_media,
    select_transition,
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
    mermaid_sources_from_adf,
    render_mermaid_png,
)
from jira_agent.reporting import agent_plan_comment_body, fetch_agent_report
from jira_agent.safety import (
    JiraWriteSafetyError,
    ensure_issue_write_allowed,
    ensure_project_write_allowed,
    write_scope_summary,
)
from jira_agent.workflows import (
    drift_reconcile_comment_body,
    factory_contract_comment_body,
    read_text_source,
    update_drift_labels,
)

mcp = FastMCP("jira-agent")

DEFAULT_FIELDS = ["summary", "status", "assignee", "issuetype", "project", "updated"]


def _client() -> JiraClient:
    return JiraClient(load_config())


def _fields_or_default(fields: list[str] | None) -> list[str]:
    return fields or DEFAULT_FIELDS


def _story_points_field() -> str:
    """Dry runs must show the same field id the write would use, but must not fail
    when the environment is incomplete."""
    try:
        return load_config().story_points_field
    except Exception:
        return DEFAULT_STORY_POINTS_FIELD


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


def _run_jql(jql: str, limit: int = 20) -> dict[str, Any]:
    try:
        with _client() as client:
            return {
                "ok": True,
                "jql": jql,
                "result": client.search_issues(jql, limit=limit, fields=DEFAULT_FIELDS),
            }
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_myself() -> dict[str, Any]:
    """Return the authenticated Jira user."""
    try:
        with _client() as client:
            return {"ok": True, "user": client.myself()}
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_get_issue(key: str, fields: list[str] | None = None) -> dict[str, Any]:
    """Fetch one Jira issue by key."""
    try:
        with _client() as client:
            return {"ok": True, "issue": client.get_issue(key, fields=_fields_or_default(fields))}
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_search_issues(
    jql: str,
    limit: int = 20,
    fields: list[str] | None = None,
    next_page_token: str | None = None,
) -> dict[str, Any]:
    """Search Jira issues with JQL."""
    try:
        with _client() as client:
            return {
                "ok": True,
                "result": client.search_issues(
                    jql,
                    limit=limit,
                    fields=_fields_or_default(fields),
                    next_page_token=next_page_token,
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
    return _run_jql(" AND ".join(clauses) + " ORDER BY updated DESC", limit=limit)


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
    return _run_jql(" AND ".join(clauses) + " ORDER BY updated DESC", limit=limit)


@mcp.tool
def jira_test_scope_issues(limit: int = 5) -> dict[str, Any]:
    """Fetch issues from the configured test scope. Defaults to the FC queue/project."""
    try:
        config = load_config()
        with JiraClient(config) as client:
            return {
                "ok": True,
                "test_jql": config.test_jql,
                "result": client.search_issues(
                    config.test_jql,
                    limit=limit,
                    fields=DEFAULT_FIELDS,
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
    parent_key: str | None = None,
    priority: str | None = None,
    story_points: float | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Create a Jira issue with Markdown-to-ADF rendering. Defaults to dry-run."""
    try:
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
            parent_key=parent_key,
            priority=priority,
            story_points=story_points,
            story_points_field=_story_points_field(),
        )
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}

    if not execute:
        return {
            "ok": True,
            "dry_run": True,
            "payload": payload,
            "message": "Pass execute=true to create this issue in Jira.",
        }

    try:
        config = load_config()
        with JiraClient(config) as client:
            ensure_project_write_allowed(config, project_key)
            issue_result = client.create_issue(
                project_key=project_key,
                issue_type_name=issue_type_name,
                summary=summary,
                description=body,
                labels=labels,
                parent_key=parent_key,
                priority=priority,
                story_points=story_points,
            )
            audit_write(
                config,
                action="create_issue",
                issue_key=issue_result.get("key"),
                payload=payload,
                result=issue_result,
            )
            return {"ok": True, "issue": issue_result}
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
    if not execute:
        return {
            "ok": True,
            "dry_run": True,
            "payload": payload,
            "message": "Pass execute=true to create this issue in Jira.",
        }

    try:
        with JiraClient(config) as client:
            ensure_project_write_allowed(config, config.test_project_key)
            issue_result = client.create_issue(
                project_key=config.test_project_key,
                issue_type_name=config.test_issue_type,
                summary=summary,
                description=description,
            )
            audit_write(
                config,
                action="create_issue",
                issue_key=issue_result.get("key"),
                payload=payload,
                result=issue_result,
            )
            return {
                "ok": True,
                "issue": issue_result,
            }
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
            me = client.myself()
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
    comments_limit: int = 5,
    changelog_limit: int = 0,
) -> dict[str, Any]:
    """Return compact issue context for agents."""
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
    parent_key: str | None = None,
    priority: str | None = None,
    story_points: float | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Edit issue fields with Markdown-to-ADF rendering. Defaults to dry-run."""
    config = load_config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        return {
            "ok": False,
            "message": "No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.",
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
            parent_key=parent_key,
            priority=priority,
            story_points=story_points,
            story_points_field=config.story_points_field,
        )
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}

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
                    "write_scope": write_scope_summary(config),
                    "message": "Pass execute=true to edit this issue in Jira.",
                }

            client.edit_issue(
                issue_key,
                summary=summary,
                description=body,
                labels=labels,
                parent_key=parent_key,
                priority=priority,
                story_points=story_points,
            )
            audit_write(
                config,
                action="edit_issue",
                issue_key=issue_key,
                payload=payload,
                result={},
            )
            return {"ok": True, "issue": issue_key}
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
    execute: bool = False,
) -> dict[str, Any]:
    """Write a Human Brief into the Jira description. Defaults to dry-run."""
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
            result = client.edit_issue(issue_key, description=body)
            audit_write(
                config,
                action="human_brief",
                issue_key=issue_key,
                payload=payload,
                result=result,
            )
            return {"ok": True, "issue": issue_key}
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
                result = client.add_attachment(issue_key, file_path)
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
    """Set or remove drift-* labels. Defaults to dry-run."""
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
            issue = client.get_issue(issue_key, fields=["labels"])
            existing = (issue.get("fields") or {}).get("labels") or []
            labels = update_drift_labels(existing, add=add, remove=remove, clear=clear)
            payload = edit_issue_payload(labels=labels)
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": issue_key,
                    "before": existing,
                    "after": labels,
                    "payload": payload,
                    "safety": safety,
                }
            result = client.edit_issue(issue_key, labels=labels)
            audit_write(
                config,
                action="drift_labels",
                issue_key=issue_key,
                payload={"before": existing, "after": labels},
                result=result,
            )
            return {"ok": True, "issue": issue_key, "labels": labels}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)
    except ValueError as exc:
        return {"ok": False, "message": str(exc)}


@mcp.tool
def jira_list_comments(key: str, limit: int = 20, start_at: int = 0) -> dict[str, Any]:
    """List comments for one Jira issue."""
    try:
        with _client() as client:
            return {
                "ok": True,
                "comments": client.list_comments(key, start_at=start_at, limit=limit),
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
def jira_list_transitions(key: str) -> dict[str, Any]:
    """List available transitions for one Jira issue."""
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
    """Link two Jira issues. Defaults to dry-run."""
    config = load_config()
    payload = issue_link_payload(
        outward_issue_key=outward_key,
        inward_issue_key=inward_key,
        link_type=link_type,
        comment=comment,
    )
    try:
        with JiraClient(config) as client:
            outward_safety = ensure_issue_write_allowed(config, client, outward_key)
            inward_safety = ensure_issue_write_allowed(config, client, inward_key)
            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "payload": payload,
                    "outward_safety": outward_safety,
                    "inward_safety": inward_safety,
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
                issue_key=outward_key,
                payload=payload,
                result=result,
            )
            return {"ok": True, "outward_key": outward_key, "inward_key": inward_key}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


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

            result = client.add_attachment(issue_key, file_path)
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
            result = client.delete_attachment(attachment_id)
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
    execute: bool = False,
) -> dict[str, Any]:
    """Render Mermaid code blocks locally, upload PNGs, and embed them. Defaults to dry-run."""
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
            updated_description = description
            for path, alt in zip(paths, alts, strict=True):
                uploaded = client.add_attachment(issue_key, path)
                attachment = uploaded[0]
                attachment_id = str(attachment.get("id"))
                uploaded_ids.append(attachment_id)
                new_attachments.append(attachment)
                media_id = client.attachment_media_id(attachment_id)
                width, height = png_dimensions(path)
                updated_description = replace_diagram_with_media(
                    updated_description,
                    media_id=media_id,
                    alt=alt,
                    width=width,
                    height=height,
                )

            result = client.edit_issue_adf_description(issue_key, updated_description)
            deleted: list[str] = []
            for attachment_id in sorted(previous_ids - set(uploaded_ids)):
                client.delete_attachment(attachment_id)
                deleted.append(attachment_id)
            audit_write(
                config,
                action="render_mermaid_diagrams",
                issue_key=issue_key,
                payload={
                    "files": [str(path) for path in paths],
                    "attachment_ids": uploaded_ids,
                    "deleted_attachment_ids": deleted,
                },
                result=result,
            )
            return {
                "ok": True,
                "issue": issue_key,
                "attachments": new_attachments,
                "deleted_attachment_ids": deleted,
            }
    except (OSError, RuntimeError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        if uploaded_ids:
            try:
                with JiraClient(config) as cleanup_client:
                    for attachment_id in uploaded_ids:
                        cleanup_client.delete_attachment(attachment_id)
            except JiraApiError:
                pass
        return _error(exc)


@mcp.tool
def jira_embed_diagram(
    file_path: str,
    key: str | None = None,
    alt: str = "Architecture diagram",
    delete_filenames: list[str] | None = None,
    delete_attachment_ids: list[str] | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Upload a PNG diagram and embed it in the Jira description. Defaults to dry-run."""
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
    delete_ids = set(delete_attachment_ids or [])
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["description", "attachment"])
            attachments = issue.get("fields", {}).get("attachment") or []
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
            attachment = uploaded[0]
            attachment_id = str(attachment.get("id"))
            media_id = client.attachment_media_id(attachment_id)
            description = replace_diagram_with_media(
                issue.get("fields", {}).get("description")
                or {"type": "doc", "version": 1, "content": []},
                media_id=media_id,
                alt=alt,
                width=width,
                height=height,
            )
            result = client.edit_issue_adf_description(issue_key, description)
            deleted: list[str] = []
            for item_id in sorted(delete_ids - {attachment_id}):
                client.delete_attachment(item_id)
                deleted.append(item_id)
            audit_write(
                config,
                action="embed_diagram",
                issue_key=issue_key,
                payload={
                    "file": file_path,
                    "attachment_id": attachment_id,
                    "media_id": media_id,
                    "deleted_attachment_ids": deleted,
                    "dimensions": {"width": width, "height": height},
                },
                result=result,
            )
            return {
                "ok": True,
                "issue": issue_key,
                "attachment": attachment,
                "media_id": media_id,
                "deleted_attachment_ids": deleted,
            }
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


@mcp.tool
def jira_upsert_image_gallery(
    manifest_path: str,
    key: str | None = None,
    documentation_filename: str | None = None,
    before_heading: str = "Child Issues",
    execute: bool = False,
) -> dict[str, Any]:
    """Insert or replace a linked screenshot gallery using existing Jira attachments."""
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
            issue = client.get_issue(issue_key, fields=["description", "attachment"])
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
            result = client.edit_issue_adf_description(issue_key, description)
            audit_write(
                config,
                action="upsert_image_gallery",
                issue_key=issue_key,
                payload={
                    "manifest": manifest_path,
                    "filenames": sorted(filenames),
                    "documentation_filename": documentation_filename,
                    "before_heading": before_heading,
                },
                result=result,
            )
            return {"ok": True, "issue": issue_key, "screenshot_count": len(filenames)}
    except (OSError, ValueError) as exc:
        return {"ok": False, "message": str(exc)}
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
) -> dict[str, Any]:
    """Transition one Jira issue by transition name or ID. Defaults to dry-run."""
    try:
        with _client() as client:
            transitions = client.get_transitions(key)
            selected = select_transition(transitions, transition_name_or_id)
            if selected is None:
                return {
                    "ok": False,
                    "message": f"No transition found for: {transition_name_or_id}",
                    "transitions": transitions,
                }

            if not execute:
                return {
                    "ok": True,
                    "dry_run": True,
                    "issue": key,
                    "transition": selected,
                    "message": "Pass execute=true to transition this issue in Jira.",
                }

            ensure_issue_write_allowed(client.config, client, key)
            client.transition_issue(key, str(selected["id"]))
            audit_write(
                client.config,
                action="transition_issue",
                issue_key=key,
                payload={"transition": selected},
                result={},
            )
            return {"ok": True, "issue": key, "transition": selected}
    except JiraWriteSafetyError as exc:
        return _safety_error(exc)
    except JiraApiError as exc:
        return _error(exc)


def main() -> None:
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
