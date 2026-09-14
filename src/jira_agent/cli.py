from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from jira_agent.audit import audit_write
from jira_agent.client import (
    DEFAULT_FIELDS,
    AdfPatchInvariantError,
    AdfSectionPatchRefused,
    JiraApiError,
    JiraClient,
    JiraConcurrentEditError,
    JiraMissingConcurrencyTokenError,
    SprintMultiRemovalError,
    SprintPreconditionError,
    adf_contains_media_id,
    create_issue_payload,
    edit_issue_payload,
    issue_link_payload,
    issue_type_allowed_values,
    patch_adf_description_section,
    png_dimensions,
    replace_diagram_with_media,
    require_concurrency_token,
    select_transition,
    sprint_membership_report,
    sprint_removal_blast_radius,
    sprint_removal_verification_results,
    sprint_verification_results,
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
)
from jira_agent.reporting import adf_to_text, agent_plan_comment_body, fetch_agent_report
from jira_agent.safety import (
    JiraWriteSafetyError,
    ensure_issue_write_allowed,
    ensure_project_write_allowed,
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

app = typer.Typer(
    help="Small local Jira API tool for agent workflows.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)
# Structured output is data, not presentation. A markup interpreter rewrites a name containing
# [...], word-wrap injects newlines into long values, and highlighting adds ANSI on a terminal;
# all three silently change bytes a caller is expected to copy back. This console opts out of
# every one of them in one named place instead of at each call site.
data_console = Console(markup=False, highlight=False, soft_wrap=True)
# Table cells hold Jira text too, and rich reads markup in them: a summary "Costs [in EUR/m2]"
# renders as "Costs". Tables still need width handling, so this console keeps wrapping and only
# drops the interpretation.
table_console = Console(markup=False, highlight=False)

# The literal every command below uses as its own `--fields` default, computed once so every
# comparison against "the caller did not narrow --fields" reads the same value `DEFAULT_FIELDS`
# itself would produce. Commands that request this exact default (rather than an explicit
# narrower --fields) also get the config-resolved sprint field appended to what they ask Jira
# for -- see `issue` and `_run_search`. It is never folded into `DEFAULT_FIELDS` itself: that
# constant is shared with the MCP surface and other non-issue/search callers, and the sprint
# field id is instance-specific, resolved from config at call time, not a shared literal.
_DEFAULT_FIELDS_CSV = ",".join(DEFAULT_FIELDS)


def _client() -> JiraClient:
    try:
        config = load_config()
    except Exception as exc:
        _print_error("Config error:", exc)
        raise typer.Exit(2) from exc
    return JiraClient(config)


def _config():
    try:
        return load_config()
    except Exception as exc:
        _print_error("Config error:", exc)
        raise typer.Exit(2) from exc


def _print_json(data: Any) -> None:
    """Print structured output byte-exactly. Never through a formatter."""
    data_console.print(json.dumps(data, indent=2, ensure_ascii=False))


def _emit_json(data: Any) -> None:
    """Write exactly one JSON document to stdout and nothing else.

    This is the call every ``--json`` code path uses for its result. By the time it runs, every
    prose line for that invocation has already gone to stderr through ``_say``, so stdout ends
    up carrying only this document, with no `find('{')` slicing required on the caller's side.
    """
    _print_json(data)


def _say(message: str, *, json_mode: bool) -> None:
    """Print one line of human prose: stdout normally, stderr when the caller passed --json.

    A --json caller needs stdout to hold nothing but the JSON document, so every prose line a
    command would otherwise print goes through this one named place instead of a conditional at
    each call site. With ``json_mode=False`` this is exactly ``console.print``, so it changes
    nothing about normal-mode output.
    """
    (error_console if json_mode else console).print(message)


def _print_error(label: str, detail: object) -> None:
    """Print a labelled error without letting the detail be read as markup.

    Every detail here is untrusted: exception text carries Jira heading and issue text, and a
    name like "[/red] Detail" raises MarkupError inside the handler that was meant to report it.
    """
    error_console.print(f"[red]{label}[/red] {escape(str(detail))}")


def _field_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    fields = [item.strip() for item in value.split(",") if item.strip()]
    return fields or None


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items


# Sentinel so a sprint-field key genuinely absent from a `fields` dict is distinguishable from
# one present with a `None` or empty-list value. `dict.get(key, _SPRINT_FIELD_MISSING)` returns
# this object only when the key was never in the response at all. Shared by `_sprint_cell`,
# `issue-sprint`, and `_warn_missing_fields`'s callers, so all three treat "absent" the same way.
_SPRINT_FIELD_MISSING = object()


def _warn_missing_fields(
    requested_fields: list[str] | None, returned_fields: set[str], *, json_mode: bool
) -> list[str]:
    """Compare what a command asked Jira for against what actually came back, and print the one
    shared "Jira did not return requested field(s)" banner when something is missing.

    Originally written for `issue` alone; `_run_search` (and therefore `search`, `mine`,
    `fc-open`, `by-status`, `recent`, and `test-scope`) reuses this exact function rather than
    growing a second copy of the same check, so a caller of any of those commands gets the same
    warning `issue` already gave when, for example, `JIRA_SPRINT_FIELD` names a field id this
    Jira site does not recognize -- Jira returns HTTP 200 and just leaves the field out, with no
    error of its own to warn the caller instead.
    """
    if not requested_fields:
        return []
    missing_fields = [name for name in requested_fields if name not in returned_fields]
    if missing_fields:
        _say(
            "[yellow]Jira did not return requested field(s):[/yellow] "
            f"{', '.join(missing_fields)}",
            json_mode=json_mode,
        )
    return missing_fields


def _sprint_cell(fields: dict[str, Any], sprint_field: str) -> str:
    """Render an issue's sprint field as its name(s) and state, never the raw Jira object.

    Keeps the same three states `issue-sprint` keeps apart, rather than collapsing "we don't
    know" into "no sprint": the field genuinely absent from this row's response (this Jira
    site's sprint field id may differ from `sprint_field`, or `JIRA_SPRINT_FIELD` may be
    misconfigured -- Jira returns HTTP 200 and silently drops an unknown field id, with no error
    of its own) renders distinctly from `null` or an empty list, Jira's own shape for "in no
    sprint" on this site, verified via `Sprint IS EMPTY` JQL. Collapsing those two used to mean
    a wrong field id rendered every row as "in no sprint" with nothing to say otherwise; see the
    per-call-site "Jira did not return requested field(s)" warning (`_warn_missing_fields`) for
    the accompanying banner this cell alone cannot carry.
    """
    raw = fields.get(sprint_field, _SPRINT_FIELD_MISSING)
    if raw is _SPRINT_FIELD_MISSING:
        return "? (field not returned)"
    sprints = raw if isinstance(raw, list) else []
    if not sprints:
        return "-"
    return ", ".join(
        f"{sprint.get('name')} ({sprint.get('state')})"
        for sprint in sprints
        if isinstance(sprint, dict)
    )


def _issue_row(issue: dict[str, Any], *, sprint_field: str | None = None) -> dict[str, str]:
    fields = issue.get("fields") or {}
    status = fields.get("status") or {}
    assignee = fields.get("assignee") or {}
    issue_type = fields.get("issuetype") or {}
    project = fields.get("project") or {}
    row = {
        "key": str(issue.get("key") or ""),
        "summary": str(fields.get("summary") or ""),
        "status": str(status.get("name") or ""),
        "assignee": str(assignee.get("displayName") or "Unassigned"),
        "type": str(issue_type.get("name") or ""),
        "project": str(project.get("key") or ""),
        "updated": str(fields.get("updated") or ""),
    }
    if sprint_field:
        row["sprint"] = _sprint_cell(fields, sprint_field)
    return row


def _print_issues_table(issues: list[dict[str, Any]], *, sprint_field: str | None = None) -> None:
    """Render the shared issue table. `sprint_field` is opt-in per call site (only `issue` and
    the search-backed commands pass it) rather than baked into every caller of this shared
    renderer, since a caller such as `test-issue` that never requested the sprint field would
    otherwise render an always-blank column for a field it never asked Jira for.
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Assignee", no_wrap=True)
    table.add_column("Summary")
    if sprint_field:
        table.add_column("Sprint")
    for issue in issues:
        row = _issue_row(issue, sprint_field=sprint_field)
        values = [row["key"], row["status"], row["assignee"], row["summary"]]
        if sprint_field:
            values.append(row["sprint"])
        table.add_row(*values)
    table_console.print(table)


def _print_create_metadata(data: dict[str, Any]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Project", no_wrap=True)
    table.add_column("Issue Type", no_wrap=True)
    table.add_column("ID", no_wrap=True)
    table.add_column("Required Fields")
    for project in data.get("projects") or []:
        for issue_type in project.get("issuetypes") or []:
            fields = issue_type.get("fields") or {}
            required = [
                str(field.get("name") or field_id)
                for field_id, field in fields.items()
                if field.get("required")
            ]
            table.add_row(
                str(project.get("key") or ""),
                str(issue_type.get("name") or ""),
                str(issue_type.get("id") or ""),
                ", ".join(required),
            )
    table_console.print(table)


def _adf_to_text(node: Any) -> str:
    return adf_to_text(node)


def _print_agent_report(report: dict[str, Any]) -> None:
    console.print(
        f"[bold]{escape(str(report['key']))}[/bold] {escape(str(report.get('summary') or ''))}"
    )
    console.print(report["url"])
    console.print(f"status: {(report.get('status') or {}).get('name')}")
    assignee = report.get("assignee") or {}
    console.print(f"assignee: {assignee.get('displayName') or 'Unassigned'}")
    console.print(
        f"project: {(report.get('project') or {}).get('key')} / "
        f"{(report.get('project') or {}).get('name')}"
    )
    console.print(f"type: {report.get('issue_type')}")
    console.print(f"updated: {report.get('updated')}")
    if report.get("write_scope"):
        console.print(f"write scope: {report['write_scope']}")

    description = report.get("description")
    if description:
        console.print("\n[bold]Description[/bold]")
        console.print(description)

    console.print("\n[bold]Comments[/bold]")
    comment_count = report.get("comment_count")
    comments_returned = report.get("comments_returned")
    comments_truncated = report.get("comments_truncated")
    comments_order = report.get("comments_order")
    if comments_returned is not None and comment_count is not None:
        status_word = "truncated" if comments_truncated else "complete"
        # G4: `--comments all` used to return oldest-first in human mode and now returns
        # newest-first, with `comments_order` reaching only `--json`. Stating the order next to
        # the count here means a reader who "learned the last line is the newest" is corrected
        # in the same place they read the count, not left to discover the flip by misreading a
        # comment's date.
        order_suffix = f", {comments_order}" if comments_order else ""
        console.print(
            f"comments: {comments_returned} of {comment_count} returned "
            f"({status_word}{order_suffix})"
        )
    comments = report.get("comments") or []
    if comments:
        for comment in comments:
            author = comment.get("author") or {}
            console.print(
                f"- {comment.get('id')} by {author.get('displayName')} "
                f"at {comment.get('created')}: {comment.get('body')}"
            )
    else:
        console.print("No comments returned.")

    console.print("\n[bold]Transitions[/bold]")
    transitions = report.get("transitions") or []
    if transitions:
        table = Table(show_header=True, header_style="bold")
        table.add_column("ID", no_wrap=True)
        table.add_column("Name", no_wrap=True)
        table.add_column("To")
        for transition in transitions:
            target = transition.get("to") or {}
            table.add_row(
                str(transition.get("id") or ""),
                str(transition.get("name") or ""),
                str(target.get("name") or ""),
            )
        table_console.print(table)
    else:
        console.print("No transitions returned.")

    changelog = report.get("changelog") or []
    if changelog:
        console.print("\n[bold]Changelog[/bold]")
        for entry in changelog:
            author = entry.get("author") or {}
            item_text = "; ".join(
                f"{item.get('field')}: {item.get('from')} -> {item.get('to')}"
                for item in entry.get("items") or []
            )
            console.print(f"- {entry.get('created')} by {author.get('displayName')}: {item_text}")


def _mcp_config_payload() -> dict[str, Any]:
    return {
        "mcpServers": {
            "jira-agent": {
                "command": sys.executable,
                "args": ["-m", "jira_agent.mcp_server"],
                "cwd": str(Path.cwd()),
            }
        }
    }


def _print_users_table(users: list[dict[str, Any]]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Display Name")
    table.add_column("Email")
    table.add_column("Account ID")
    table.add_column("Active", no_wrap=True)
    for user in users:
        table.add_row(
            str(user.get("displayName") or ""),
            str(user.get("emailAddress") or ""),
            str(user.get("accountId") or ""),
            str(user.get("active")),
        )
    table_console.print(table)


def _read_audit_records(path: str, limit: int) -> list[dict[str, Any]]:
    audit_path = Path(path)
    if not audit_path.exists():
        return []
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines[-limit:] if line.strip()]
    return records


@app.command()
def myself(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Show the authenticated Jira user."""
    with _client() as client:
        try:
            data = client.myself()
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json(data)
        return

    console.print(f"{data.get('displayName')} <{data.get('emailAddress', 'email hidden')}>")
    console.print(f"accountId: {data.get('accountId')}")
    console.print(f"timeZone: {data.get('timeZone')}")


@app.command("user-search")
def user_search(
    query: Annotated[str, typer.Argument(help="Name or email query.")],
    key: Annotated[
        str | None,
        typer.Option("--issue", help="Optional issue key for assignable-user search."),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", help="Optional project key for assignable-user search."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum users to return."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Search assignable Jira users."""
    config = _config()
    issue_key = key or config.test_issue_key
    project_key = project or (None if issue_key else config.test_project_key)
    with JiraClient(config) as client:
        try:
            users = client.search_assignable_users(
                query,
                issue_key=issue_key,
                project_key=project_key,
                limit=limit,
            )
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json(users)
        return
    _print_users_table(users)


@app.command()
def issue(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    fields: Annotated[
        str | None,
        typer.Option(
            "--fields",
            help=(
                "Comma-separated Jira fields to fetch. Left at its default, this also asks for "
                "the configured sprint field (config.sprint_field / JIRA_SPRINT_FIELD); an "
                "explicit --fields is used exactly as given, with no field added to it."
            ),
        ),
    ] = _DEFAULT_FIELDS_CSV,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Fetch one Jira issue."""
    with _client() as client:
        sprint_field = client.config.sprint_field
        if fields == _DEFAULT_FIELDS_CSV:
            requested_fields: list[str] | None = [*DEFAULT_FIELDS, sprint_field]
        else:
            requested_fields = _field_list(fields)
        try:
            data = client.get_issue(key, fields=requested_fields)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    returned_fields = set((data.get("fields") or {}).keys())
    _warn_missing_fields(requested_fields, returned_fields, json_mode=json_output)

    if json_output:
        _emit_json(data)
        return

    # Only render a Sprint column when the sprint field was actually part of what was requested:
    # an explicit --fields that left it out must not render an always-blank column that looks
    # like "checked, no sprint" rather than "not asked for".
    render_sprint_field = (
        sprint_field if requested_fields and sprint_field in requested_fields else None
    )
    _print_issues_table([data], sprint_field=render_sprint_field)


@app.command()
def comments(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum comments to return. Ignored with --all."),
    ] = 20,
    start_at: Annotated[
        int,
        typer.Option("--start-at", min=0, help="Pagination start offset. Ignored with --all."),
    ] = 0,
    all_comments: Annotated[
        bool,
        typer.Option(
            "--all", help="Page through every comment on the issue, ignoring --limit/--start-at."
        ),
    ] = False,
    order: Annotated[
        str,
        typer.Option(
            "--order",
            help=(
                "'newest' or 'oldest'. Default is 'newest', a deliberate change from Jira's "
                "natural oldest-first order: the common case is verifying a comment just written."
            ),
        ),
    ] = "newest",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """List comments for one Jira issue, newest first by default."""
    if order not in {"newest", "oldest"}:
        error_console.print(
            f"[red]Invalid --order value:[/red] {order!r} (use 'newest' or 'oldest')"
        )
        raise typer.Exit(2)
    order_by = "-created" if order == "newest" else None

    with _client() as client:
        try:
            if all_comments:
                data = client.list_all_comments(key, order_by=order_by)
            else:
                data = client.list_comments(key, start_at=start_at, limit=limit, order_by=order_by)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    comments_data = data.get("comments") or []
    if not isinstance(comments_data, list):
        error_console.print("[red]Unexpected Jira response: comments is not a list[/red]")
        raise typer.Exit(1)
    returned = len(comments_data)

    if json_output:
        _emit_json({**data, "returned": returned})
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Author", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Body")
    for comment in comments_data:
        author = comment.get("author") or {}
        table.add_row(
            str(comment.get("id") or ""),
            str(author.get("displayName") or ""),
            str(comment.get("created") or ""),
            _adf_to_text(comment.get("body")).strip(),
        )
    table_console.print(table)
    total = data.get("total")
    if isinstance(total, int):
        console.print(f"returned {returned} of {total} comments")


@app.command()
def add_comment(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    body: Annotated[str, typer.Argument(help="Comment text.")],
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually write the comment to Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Add a comment to one Jira issue. Defaults to dry-run."""
    if not execute:
        _say("[yellow]Dry run:[/yellow] would add this comment", json_mode=json_output)
        _say(f"issue: {key}", json_mode=json_output)
        _say(body, json_mode=json_output)
        if json_output:
            _emit_json({"issue": key, "body": body, "dry_run": True})
        _say("Pass --execute to write it to Jira.", json_mode=json_output)
        return

    with _client() as client:
        try:
            ensure_issue_write_allowed(client.config, client, key)
            data = client.add_comment(key, body)
            audit_write(
                client.config,
                action="add_comment",
                issue_key=key,
                payload={"body": body},
                result={"id": data.get("id")},
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Added comment {data.get('id')} to {key}.")


@app.command("edit-comment")
def edit_comment(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    comment_id: Annotated[str, typer.Argument(help="Jira comment id.")],
    body: Annotated[
        str | None,
        typer.Option("--body", help="New comment text."),
    ] = None,
    body_file: Annotated[
        Path | None,
        typer.Option("--body-file", help="UTF-8 Markdown file with the new comment text."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually rewrite the comment in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Replace one comment's body on one Jira issue. Defaults to dry-run.

    Shows the comment's current body next to the proposed replacement, both in the dry-run
    prose and, under --json, in the JSON document, because a caller correcting a comment needs
    to see exactly what they are about to overwrite before committing to it.
    """
    try:
        new_body = read_text_source(body, body_file)
    except (OSError, ValueError) as exc:
        _print_error("Edit error:", exc)
        raise typer.Exit(2) from exc

    with _client() as client:
        try:
            current = client.get_comment(key, comment_id)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

        current_body = _adf_to_text(current.get("body")).strip()

        if not execute:
            _say("[yellow]Dry run:[/yellow] would replace this comment", json_mode=json_output)
            _say(f"issue: {key}", json_mode=json_output)
            _say(f"comment: {comment_id}", json_mode=json_output)
            _say("current body:", json_mode=json_output)
            _say(current_body, json_mode=json_output)
            _say("proposed body:", json_mode=json_output)
            _say(new_body, json_mode=json_output)
            if json_output:
                _emit_json(
                    {
                        "issue": key,
                        "comment": comment_id,
                        "current_body": current_body,
                        "proposed_body": new_body,
                        "dry_run": True,
                    }
                )
            _say("Pass --execute to write it to Jira.", json_mode=json_output)
            return

        try:
            ensure_issue_write_allowed(client.config, client, key)
            data = client.update_comment(key, comment_id, new_body)
            audit_write(
                client.config,
                action="edit_comment",
                issue_key=key,
                payload={
                    "comment_id": comment_id,
                    "previous_body": current_body,
                    "body": new_body,
                },
                result={"id": data.get("id")},
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Edited comment {comment_id} on {key}.")


@app.command("delete-comment")
def delete_comment(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    comment_id: Annotated[str, typer.Argument(help="Jira comment id.")],
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually delete the comment in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Delete one comment from one Jira issue. Defaults to dry-run.

    Deleting a comment is irreversible, so the dry run prints the full current body: it is the
    only chance the caller gets to notice a wrong comment id before the comment is gone for good.
    """
    with _client() as client:
        try:
            current = client.get_comment(key, comment_id)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

        current_body = _adf_to_text(current.get("body")).strip()

        if not execute:
            _say("[yellow]Dry run:[/yellow] would delete this comment", json_mode=json_output)
            _say(f"issue: {key}", json_mode=json_output)
            _say(f"comment: {comment_id}", json_mode=json_output)
            _say("current body:", json_mode=json_output)
            _say(current_body, json_mode=json_output)
            if json_output:
                _emit_json(
                    {
                        "issue": key,
                        "comment": comment_id,
                        "current_body": current_body,
                        "dry_run": True,
                    }
                )
            _say("Pass --execute to delete it from Jira.", json_mode=json_output)
            return

        try:
            ensure_issue_write_allowed(client.config, client, key)
            data = client.delete_comment(key, comment_id)
            audit_write(
                client.config,
                action="delete_comment",
                issue_key=key,
                payload={"comment_id": comment_id, "body": current_body},
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({"issue": key, "comment": comment_id, **data, "dry_run": False})
        return

    console.print(f"Deleted comment {comment_id} from {key}.")


@app.command()
def recent(
    days: Annotated[
        int,
        typer.Option("--days", min=1, max=365, help="Look back this many days."),
    ] = 30,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum issues to return; pages past one page."),
    ] = 20,
    fields: Annotated[
        str | None,
        typer.Option("--fields", help="Comma-separated Jira fields to fetch."),
    ] = _DEFAULT_FIELDS_CSV,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the curated result document, not Jira's raw response.",
        ),
    ] = False,
) -> None:
    """Show recently updated Jira issues."""
    jql = f"updated >= -{days}d ORDER BY updated DESC"
    _run_search(
        jql,
        limit=limit,
        fields=fields,
        next_page_token=None,
        json_output=json_output,
    )


@app.command()
def mine(
    project: Annotated[
        str | None,
        typer.Option("--project", help="Optional project key filter."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum issues to return; pages past one page."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the curated result document, not Jira's raw response.",
        ),
    ] = False,
) -> None:
    """Show unresolved issues assigned to the authenticated user."""
    clauses = ["assignee = currentUser()", "resolution = Unresolved"]
    if project:
        clauses.insert(0, f"project = {project}")
    jql = " AND ".join(clauses) + " ORDER BY updated DESC"
    _run_search(
        jql,
        limit=limit,
        fields=_DEFAULT_FIELDS_CSV,
        next_page_token=None,
        json_output=json_output,
    )


@app.command("assigned-to-me")
def assigned_to_me(
    project: Annotated[
        str | None,
        typer.Option("--project", help="Optional project key filter."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum issues to return; pages past one page."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the curated result document, not Jira's raw response.",
        ),
    ] = False,
) -> None:
    """Alias for mine."""
    mine(project=project, limit=limit, json_output=json_output)


@app.command("fc-open")
def fc_open(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum issues to return; pages past one page."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the curated result document, not Jira's raw response.",
        ),
    ] = False,
) -> None:
    """Show open FC issues."""
    jql = "project = FC AND statusCategory != Done ORDER BY updated DESC"
    _run_search(
        jql,
        limit=limit,
        fields=_DEFAULT_FIELDS_CSV,
        next_page_token=None,
        json_output=json_output,
    )


@app.command("by-status")
def by_status(
    status: Annotated[str, typer.Argument(help="Jira status name.")],
    project: Annotated[
        str | None,
        typer.Option("--project", help="Optional project key filter."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum issues to return; pages past one page."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the curated result document, not Jira's raw response.",
        ),
    ] = False,
) -> None:
    """Search issues by status."""
    escaped_status = status.replace('"', '\\"')
    clauses = [f'status = "{escaped_status}"']
    if project:
        clauses.insert(0, f"project = {project}")
    jql = " AND ".join(clauses) + " ORDER BY updated DESC"
    _run_search(
        jql,
        limit=limit,
        fields=_DEFAULT_FIELDS_CSV,
        next_page_token=None,
        json_output=json_output,
    )


@app.command()
def search(
    jql: Annotated[str, typer.Argument(help="JQL query.")],
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help=(
                "Maximum issues to return; pages automatically past one page. "
                "Ignored when --all is set (--all wins over --limit)."
            ),
        ),
    ] = 20,
    all_issues: Annotated[
        bool,
        typer.Option(
            "--all", help="Page through every matching issue. Overrides --limit when both are set."
        ),
    ] = False,
    count: Annotated[
        bool,
        typer.Option(
            "--count",
            help=(
                "Also report Jira's approximate match count via one extra POST to "
                "/search/approximate-count. Opt-in because of that extra request."
            ),
        ),
    ] = False,
    fields: Annotated[
        str | None,
        typer.Option("--fields", help="Comma-separated Jira fields to fetch."),
    ] = _DEFAULT_FIELDS_CSV,
    next_page_token: Annotated[
        str | None,
        typer.Option(
            "--next-page-token",
            help="Jira enhanced search nextPageToken, for a caller paging one page at a time.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the curated result document, not Jira's raw response.",
        ),
    ] = False,
) -> None:
    """Search Jira issues with JQL.

    Jira's enhanced search endpoint never reports a total match count, only the issues it
    returned and whether that was the last page. This command reports exactly that (`returned`,
    `is_last`) instead of inventing a number the API never gave it. Pass --count for Jira's
    separate, explicitly approximate count.
    """
    _run_search(
        jql,
        limit=limit,
        fields=fields,
        next_page_token=next_page_token,
        json_output=json_output,
        all_pages=all_issues,
        count=count,
    )


@app.command("test-scope")
def test_scope(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=20, help="Maximum test-scope issues to return."),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the curated result document, not Jira's raw response.",
        ),
    ] = False,
) -> None:
    """Run a read-only smoke query against the configured Jira test scope.

    Reports the same curated shape as every other search-backed command (`returned`, `is_last`,
    never a fabricated `total`); see `_run_search`.
    """
    config = _config()
    _say(f"test JQL: {config.test_jql}", json_mode=json_output)
    _run_search(
        config.test_jql,
        limit=limit,
        fields=_DEFAULT_FIELDS_CSV,
        next_page_token=None,
        json_output=json_output,
    )


@app.command("create-meta")
def create_meta(
    project_key: Annotated[
        str | None,
        typer.Option("--project", help="Jira project key. Defaults to JIRA_TEST_PROJECT_KEY."),
    ] = None,
    issue_type: Annotated[
        str | None,
        typer.Option("--issue-type", help="Optional issue type filter."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Show create metadata for a Jira project."""
    config = _config()
    project = project_key or config.test_project_key
    with JiraClient(config) as client:
        try:
            data = client.create_metadata(project_key=project, issue_type_name=issue_type)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json(data)
        return

    _print_create_metadata(data)


@app.command("create-test-issue")
def create_test_issue(
    summary: Annotated[str, typer.Argument(help="Summary for the FC test issue.")],
    description: Annotated[
        str | None,
        typer.Option("--description", help="Optional issue description."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually create the issue in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Create a test issue in the configured Jira test project. Defaults to dry-run."""
    config = _config()
    payload = create_issue_payload(
        project_key=config.test_project_key,
        issue_type_name=config.test_issue_type,
        summary=summary,
        description=description,
    )

    if not execute:
        _say("[yellow]Dry run:[/yellow] would create this Jira test issue", json_mode=json_output)
        _say(f"project: {config.test_project_key}", json_mode=json_output)
        _say(f"issue type: {config.test_issue_type}", json_mode=json_output)
        if json_output:
            _emit_json({**payload, "dry_run": True})
        else:
            _print_json(payload)
        _say("Pass --execute to create it in Jira.", json_mode=json_output)
        return

    with JiraClient(config) as client:
        try:
            ensure_project_write_allowed(config, config.test_project_key)
            data = client.create_issue(
                project_key=config.test_project_key,
                issue_type_name=config.test_issue_type,
                summary=summary,
                description=description,
            )
            audit_write(
                config,
                action="create_issue",
                issue_key=data.get("key"),
                payload=payload,
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Created Jira test issue {data.get('key')} ({data.get('id')}).")


@app.command("create-issue")
def create_issue(
    summary: Annotated[str, typer.Argument(help="Issue summary.")],
    issue_type: Annotated[str, typer.Option("--issue-type", help="Jira issue type name.")],
    project_key: Annotated[
        str | None,
        typer.Option("--project", help="Jira project key. Defaults to JIRA_TEST_PROJECT_KEY."),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option("--description", help="Markdown description text."),
    ] = None,
    description_file: Annotated[
        Path | None,
        typer.Option("--description-file", help="UTF-8 Markdown description file."),
    ] = None,
    description_section: Annotated[
        str | None,
        typer.Option(
            "--description-section",
            help=(
                "Select one heading body from the supplied Markdown. The extract becomes the "
                "whole description field; use patch-description-section to edit one section "
                "of an existing description in place."
            ),
        ),
    ] = None,
    labels: Annotated[
        str | None,
        typer.Option("--labels", help="Comma-separated labels."),
    ] = None,
    parent: Annotated[
        str | None,
        typer.Option("--parent", help="Optional parent issue key."),
    ] = None,
    priority: Annotated[
        str | None,
        typer.Option("--priority", help="Optional Jira priority name."),
    ] = None,
    story_points: Annotated[
        float | None,
        typer.Option("--story-points", help="Optional story-point estimate."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually create the issue in Jira."),
    ] = False,
    render_mermaid: Annotated[
        bool,
        typer.Option(
            "--render-mermaid/--no-render-mermaid",
            help="Render Mermaid code blocks to PNG, attach, and embed them.",
        ),
    ] = True,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Create a Jira issue with Markdown-to-ADF rendering. Defaults to dry-run."""
    config = _config()
    project = project_key or config.test_project_key
    try:
        body = (
            read_text_source(description, description_file, section=description_section)
            if description or description_file
            else None
        )
    except (OSError, ValueError) as exc:
        _print_error("Description error:", exc)
        raise typer.Exit(2) from exc

    try:
        payload = create_issue_payload(
            project_key=project,
            issue_type_name=issue_type,
            summary=summary,
            description=body,
            labels=_csv_list(labels),
            parent_key=parent,
            priority=priority,
            story_points=story_points,
            story_points_field=config.story_points_field,
        )
    except ValueError as exc:
        _print_error("Create error:", exc)
        raise typer.Exit(2) from exc
    description_adf = payload["fields"].get("description")
    diagram_count = (
        len(mermaid_sources_from_adf(description_adf))
        if render_mermaid and isinstance(description_adf, dict)
        else 0
    )
    if not execute:
        _say("[yellow]Dry run:[/yellow] would create this Jira issue", json_mode=json_output)
        if json_output:
            _emit_json({**payload, "dry_run": True})
        else:
            _print_json(payload)
        if diagram_count:
            _say(
                f"Mermaid diagrams to render after creation: {diagram_count}",
                json_mode=json_output,
            )
        _say("Pass --execute to create it in Jira.", json_mode=json_output)
        return

    with JiraClient(config) as client:
        try:
            ensure_project_write_allowed(config, project)
            data = client.create_issue(
                project_key=project,
                issue_type_name=issue_type,
                summary=summary,
                description=body,
                labels=_csv_list(labels),
                parent_key=parent,
                priority=priority,
                story_points=story_points,
            )
            audit_write(
                config,
                action="create_issue",
                issue_key=data.get("key"),
                payload=payload,
                result=data,
            )
            mermaid_result: dict[str, Any] | None = None
            if diagram_count and data.get("key") and isinstance(description_adf, dict):
                try:
                    mermaid_result = embed_mermaid_diagrams(
                        client,
                        issue_key=str(data["key"]),
                        description=description_adf,
                        output_dir=Path(config.audit_log_path).parent
                        / "diagrams"
                        / str(data["key"]),
                    )
                    audit_write(
                        config,
                        action="render_mermaid_diagrams",
                        issue_key=str(data["key"]),
                        payload={
                            "files": mermaid_result["files"],
                            "attachment_ids": mermaid_result["attachment_ids"],
                            "deleted_attachment_ids": mermaid_result["deleted_attachment_ids"],
                        },
                        result=mermaid_result["result"],
                    )
                except (JiraApiError, OSError, RuntimeError, ValueError) as exc:
                    mermaid_result = {"ok": False, "message": str(exc)}
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        if mermaid_result is None:
            _emit_json({**data, "dry_run": False})
        else:
            _emit_json({"issue": data, "mermaid": mermaid_result, "dry_run": False})
        return
    console.print(f"Created Jira issue {data.get('key')} ({data.get('id')}).")
    if mermaid_result is not None:
        if mermaid_result.get("ok"):
            console.print(f"Embedded {mermaid_result['diagram_count']} Mermaid diagram(s).")
        else:
            error_console.print(
                f"[yellow]Warning:[/yellow] issue was created, but Mermaid embedding failed: "
                f"{mermaid_result['message']}"
            )


@app.command("render-mermaid")
def render_mermaid(
    key: Annotated[str, typer.Argument(help="Jira issue key.")],
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Directory for rendered PNG files."),
    ] = None,
    concurrency_guard: Annotated[
        bool,
        typer.Option(
            "--concurrency-guard/--no-concurrency-guard",
            help=(
                "Uploading each rendered PNG is itself a write that moves this issue's own "
                "`updated` on the Jira side, so the token cannot come from the read that finds "
                "the Mermaid blocks to render. Instead, after every upload this re-reads the "
                "description and `updated` together in one request, builds the write from THAT "
                "read, and refuses the write (exit code 4) if `updated` no longer matches "
                "immediately before it. This is a check-then-act narrowing of the race window, "
                "not an atomic compare-and-swap: Jira Cloud has no conditional write here, so a "
                "change landing in that narrower window is still lost. Default: guarded."
            ),
        ),
    ] = True,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Render, upload, and embed the diagrams in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the machine-readable result."),
    ] = False,
) -> None:
    """Render and embed all Mermaid blocks in an existing issue. Defaults to dry-run.

    This rewrites the whole description field, so a concurrent edit to that field would
    otherwise be silently overwritten. By default the write is guarded, but like `embed-diagram`,
    the token cannot come from the same read that finds the Mermaid blocks to render: uploading
    each rendered PNG is itself a write that moves this issue's own `updated` on the Jira side.
    The guard token is instead read together with a fresh description immediately after the last
    upload, so the write is built from that same fresh document. This means a concurrent edit
    landing before that post-upload read is incorporated into the write rather than lost; it is
    not an atomic compare-and-swap, and a change landing between that read and the write is still
    lost, since Jira Cloud offers no conditional write on this resource. Pass
    --no-concurrency-guard for the old unguarded behaviour.
    """
    config = _config()
    try:
        with JiraClient(config) as client:
            safety = ensure_issue_write_allowed(config, client, key)
            issue = client.get_issue(key, fields=["description", "attachment"])
            description_adf = issue.get("fields", {}).get("description") or {
                "type": "doc",
                "version": 1,
                "content": [],
            }
            sources = mermaid_sources_from_adf(description_adf)
            if not sources:
                error_console.print(
                    f"[yellow]No Mermaid code blocks found on {escape(str(key))}.[/yellow]"
                )
                raise typer.Exit(1)

            runtime = discover_mermaid_runtime()
            destination = output_dir or (Path(config.audit_log_path).parent / "diagrams" / key)
            if not execute:
                result = {
                    "ok": True,
                    "dry_run": True,
                    "issue": key,
                    "diagram_count": len(sources),
                    "output_dir": str(destination),
                    "runtime": {
                        "node": str(runtime.node),
                        "cli": str(runtime.cli),
                        "browser": str(runtime.browser),
                    },
                    "safety": safety,
                }
                if json_output:
                    _emit_json(result)
                else:
                    console.print("[yellow]Dry run:[/yellow] would render Mermaid diagrams")
                    console.print(f"issue: {key}")
                    console.print(f"Mermaid diagrams: {len(sources)}")
                    console.print(f"output directory: {destination}")
                    console.print("Pass --execute to render, upload, and embed them.")
                return

            result = embed_mermaid_diagrams(
                client,
                issue_key=key,
                description=description_adf,
                attachments=issue.get("fields", {}).get("attachment") or [],
                output_dir=destination,
                runtime=runtime,
                guard_concurrent_edit=concurrency_guard,
            )
            audit_write(
                config,
                action="render_mermaid_diagrams",
                issue_key=key,
                payload={
                    "files": result["files"],
                    "attachment_ids": result["attachment_ids"],
                    "deleted_attachment_ids": result["deleted_attachment_ids"],
                    "concurrency_guard": concurrency_guard,
                },
                result=result["result"],
            )
    except JiraConcurrentEditError as exc:
        _exit_concurrent_edit_error(exc, json_mode=json_output)
    except JiraMissingConcurrencyTokenError as exc:
        _exit_missing_concurrency_token_error(exc, json_mode=json_output)
    except JiraWriteSafetyError as exc:
        _exit_safety_error(exc, json_mode=json_output)
    except JiraApiError as exc:
        _exit_api_error(exc, json_mode=json_output)
    except (OSError, RuntimeError, ValueError) as exc:
        _print_error("Mermaid error:", exc)
        raise typer.Exit(1) from exc

    if json_output:
        _emit_json({**result, "dry_run": False})
        return
    console.print(f"Embedded {result['diagram_count']} Mermaid diagram(s) in {key}.")


@app.command("test-issue")
def test_issue(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Fetch the configured integration-test issue."""
    config = _config()
    if not config.test_issue_key:
        error_console.print("[red]JIRA_TEST_ISSUE_KEY is not configured.[/red]")
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            data = client.get_issue(config.test_issue_key, fields=DEFAULT_FIELDS)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json(data)
        return

    _print_issues_table([data])


@app.command("agent-report")
def agent_report(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    comments_limit: Annotated[
        str,
        typer.Option(
            "--comments",
            help=(
                "Newest comments to include: a non-negative integer, or 'all' to page through "
                "every comment. Both return comments newest first (report['comments_order'] "
                "always reads 'newest-first')."
            ),
        ),
    ] = "5",
    changelog_limit: Annotated[
        int,
        typer.Option("--changelog", min=0, max=100, help="Maximum changelog entries to include."),
    ] = 0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Build a compact agent-friendly report for one Jira issue."""
    comment_limit: int | str
    if comments_limit == "all":
        comment_limit = "all"
    else:
        try:
            parsed_comments_limit = int(comments_limit)
        except ValueError:
            parsed_comments_limit = None
        if parsed_comments_limit is None or parsed_comments_limit < 0:
            error_console.print(
                "[red]Invalid --comments value:[/red] "
                f"{comments_limit!r} (use a non-negative integer or 'all')"
            )
            raise typer.Exit(2)
        comment_limit = parsed_comments_limit

    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            report = fetch_agent_report(
                client,
                issue_key=issue_key,
                site_url=config.site_url,
                comment_limit=comment_limit,
                changelog_limit=changelog_limit,
                write_scope=config.test_issue_key,
            )
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json(report)
        return

    _print_agent_report(report)


@app.command("agent-plan-comment")
def agent_plan_comment(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    plan: Annotated[
        str,
        typer.Option("--plan", help="Plan text for the comment."),
    ] = "Review the issue context and proceed with the next verified action.",
    next_step: Annotated[
        str | None,
        typer.Option("--next-step", help="Optional next step."),
    ] = None,
    risk: Annotated[
        str | None,
        typer.Option("--risk", help="Optional risk or blocker note."),
    ] = None,
    result: Annotated[
        str | None,
        typer.Option("--result", help="Optional result note."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually add the comment in Jira."),
    ] = False,
) -> None:
    """Prepare or add a standardized agent plan comment."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
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
                console.print("[yellow]Dry run:[/yellow] would add this agent plan comment")
                console.print(f"issue: {issue_key}")
                console.print(body)
                console.print("Pass --execute to write it to Jira.")
                return

            ensure_issue_write_allowed(config, client, issue_key)
            data = client.add_comment(issue_key, body)
            audit_write(
                config,
                action="agent_plan_comment",
                issue_key=issue_key,
                payload={"body": body},
                result={"id": data.get("id")},
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Added agent plan comment {data.get('id')} to {issue_key}.")


@app.command("human-brief")
def human_brief(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option("--text", help="Brief text to write into the Jira description."),
    ] = None,
    file_path: Annotated[
        Path | None,
        typer.Option("--file", exists=True, dir_okay=False, readable=True, help="Brief file."),
    ] = None,
    concurrency_guard: Annotated[
        bool,
        typer.Option(
            "--concurrency-guard/--no-concurrency-guard",
            help=(
                "Capture the issue's `updated` timestamp immediately before the write and "
                "refuse the write (exit code 4) if it no longer matches. This is a check-then-"
                "act narrowing of the race window, not an atomic compare-and-swap: Jira Cloud "
                "has no conditional write here, so a change landing in that window is still "
                "lost. Default: guarded."
            ),
        ),
    ] = True,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually update the Jira description."),
    ] = False,
) -> None:
    """Write a Human Brief into the Jira description. Defaults to dry-run.

    This replaces the whole description field, so a concurrent edit to that field would
    otherwise be silently overwritten. By default the write is guarded the same way `edit-issue`
    is: immediately before the write, it captures the issue's `updated` timestamp with
    `issue_version`, which itself refuses (before any write is attempted) if Jira's response
    does not carry a usable timestamp to guard with. The client then re-checks that same
    timestamp again right before the PUT and refuses the write (exit code 4) if it no longer
    matches. Pass --no-concurrency-guard for the old unguarded behaviour.
    """
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    try:
        body = read_text_source(text, file_path)
        payload = edit_issue_payload(description=body)
    except ValueError as exc:
        _print_error("Human brief error:", exc)
        raise typer.Exit(2) from exc

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would write Human Brief")
                console.print(f"issue: {issue_key}")
                console.print(f"safety: {safety}")
                _print_json(payload)
                console.print("Pass --execute to write it to Jira.")
                return

            # Read as late as possible, immediately before the write, to keep the race window
            # this narrows as small as this process can make it. issue_version raises
            # JiraMissingConcurrencyTokenError when it cannot read a usable token — itself a
            # JiraApiError only via subclassing, not a bare one — so a missing token refuses
            # here rather than silently falling back to an unguarded write.
            expect_updated = client.issue_version(issue_key) if concurrency_guard else None

            data = client.edit_issue(issue_key, description=body, expect_updated=expect_updated)
            audit_write(
                config,
                action="human_brief",
                issue_key=issue_key,
                payload={**payload, "concurrency_guard": concurrency_guard},
                result=data,
            )
        except JiraConcurrentEditError as exc:
            _exit_concurrent_edit_error(exc, json_mode=False)
        except JiraMissingConcurrencyTokenError as exc:
            _exit_missing_concurrency_token_error(exc, json_mode=False)
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Wrote Human Brief into {issue_key}.")


@app.command("factory-contract")
def factory_contract(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option("--text", help="Factory Contract text."),
    ] = None,
    file_path: Annotated[
        Path | None,
        typer.Option("--file", exists=True, dir_okay=False, readable=True, help="Contract file."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Storage mode: comment or attachment."),
    ] = "comment",
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually write to Jira."),
    ] = False,
) -> None:
    """Store a Factory Contract as a comment or attachment. Defaults to dry-run."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    if mode not in {"comment", "attachment"}:
        error_console.print("[red]--mode must be comment or attachment.[/red]")
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if mode == "attachment":
                if file_path is None:
                    error_console.print("[red]--file is required for attachment mode.[/red]")
                    raise typer.Exit(2)
                if not execute:
                    console.print("[yellow]Dry run:[/yellow] would attach Factory Contract")
                    console.print(f"issue: {issue_key}")
                    console.print(f"file: {file_path}")
                    console.print(f"safety: {safety}")
                    console.print("Pass --execute to upload it to Jira.")
                    return
                result = client.add_attachment(issue_key, file_path)
                audit_write(
                    config,
                    action="factory_contract_attachment",
                    issue_key=issue_key,
                    payload={"file": str(file_path)},
                    result={"attachments": result},
                )
                console.print(f"Attached Factory Contract to {issue_key}.")
                return

            contract = read_text_source(text, file_path)
            body = factory_contract_comment_body(
                contract,
                source_name=file_path.name if file_path else None,
            )
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would add Factory Contract comment")
                console.print(f"issue: {issue_key}")
                console.print(f"safety: {safety}")
                console.print(body)
                console.print("Pass --execute to write it to Jira.")
                return
            result = client.add_comment(issue_key, body)
            audit_write(
                config,
                action="factory_contract_comment",
                issue_key=issue_key,
                payload={"body": body},
                result={"id": result.get("id")},
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)
        except ValueError as exc:
            _print_error("Factory Contract error:", exc)
            raise typer.Exit(2) from exc

    console.print(f"Added Factory Contract comment to {issue_key}.")


@app.command("drift-reconcile-comment")
def drift_reconcile_comment(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    summary: Annotated[str, typer.Option("--summary", help="Drift summary.")] = "",
    decision: Annotated[
        str | None,
        typer.Option("--decision", help="Optional reconcile decision."),
    ] = None,
    details: Annotated[
        str | None,
        typer.Option("--details", help="Optional detail text."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually add the comment in Jira."),
    ] = False,
) -> None:
    """Post a Drift Reconcile comment. Defaults to dry-run."""
    if not summary.strip():
        error_console.print("[red]--summary is required.[/red]")
        raise typer.Exit(2)
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    body = drift_reconcile_comment_body(summary=summary, decision=decision, details=details)
    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would add Drift Reconcile comment")
                console.print(f"issue: {issue_key}")
                console.print(f"safety: {safety}")
                console.print(body)
                console.print("Pass --execute to write it to Jira.")
                return
            result = client.add_comment(issue_key, body)
            audit_write(
                config,
                action="drift_reconcile_comment",
                issue_key=issue_key,
                payload={"body": body},
                result={"id": result.get("id")},
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Added Drift Reconcile comment to {issue_key}.")


@app.command("drift-labels")
def drift_labels(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    add: Annotated[
        str | None,
        typer.Option("--add", help="Comma-separated drift labels to add."),
    ] = None,
    remove: Annotated[
        str | None,
        typer.Option("--remove", help="Comma-separated drift labels to remove."),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="Remove all existing drift-* labels."),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually update labels in Jira."),
    ] = False,
) -> None:
    """Add or remove drift-* labels atomically. Defaults to dry-run.

    `--add` and `--remove` need no read of the issue at all: each named label becomes its own
    Jira `add`/`remove` operation on `labels` (see `editmeta`'s advertised operations for that
    field), applied server-side against whatever the array currently holds, so two agents adding
    different drift labels at the same time cannot lose each other's work. This command carries
    NO concurrency guard on any path -- there is no whole-array read-modify-write left for a
    stale `updated` token to protect.

    `--clear` is the one path that still reads the issue first, once, to find out which labels
    currently start with `drift-`; only those become explicit `remove` operations. A drift label
    added by someone else after that read is never named and survives, the same way `--add`/
    `--remove` alone already leave everything they were not told about untouched. That read
    still needs no guard: it only decides which `remove` operations to send, and each one is
    itself safe against a concurrent edit once sent.
    """
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            existing: list[str] | None = None
            if clear:
                issue = client.get_issue(issue_key, fields=["labels"])
                existing = (issue.get("fields") or {}).get("labels") or []
            add_labels, remove_labels = drift_label_operations(
                add=_csv_list(add),
                remove=_csv_list(remove),
                existing_labels=existing,
                clear=clear,
            )
            if not add_labels and not remove_labels:
                console.print("[yellow]Nothing to do:[/yellow] no labels to add or remove.")
                return
            payload = edit_issue_payload(
                labels_add=add_labels or None, labels_remove=remove_labels or None
            )
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would update drift labels")
                console.print(f"issue: {issue_key}")
                console.print(f"safety: {safety}")
                _print_json({"payload": payload})
                console.print("Pass --execute to write it to Jira.")
                return
            # Atomic add/remove: never pass expect_updated. There is no whole-array rewrite here
            # for a stale token to protect (see the docstring above and drift_label_operations).
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
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)
        except ValueError as exc:
            _print_error("Drift label error:", exc)
            raise typer.Exit(2) from exc

    console.print(f"Updated drift labels on {issue_key}.")


@app.command("edit-issue")
def edit_issue(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    summary: Annotated[
        str | None,
        typer.Option("--summary", help="New issue summary."),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option("--description", help="New Markdown issue description."),
    ] = None,
    description_file: Annotated[
        Path | None,
        typer.Option("--description-file", help="UTF-8 Markdown description file."),
    ] = None,
    description_section: Annotated[
        str | None,
        typer.Option(
            "--description-section",
            help=(
                "Select one heading body from the supplied Markdown. The extract becomes the "
                "whole description field; use patch-description-section to edit one section "
                "of an existing description in place."
            ),
        ),
    ] = None,
    labels: Annotated[
        str | None,
        typer.Option(
            "--labels",
            help=(
                "Comma-separated labels to set, replacing the whole array. Whole-array form: "
                "this one CAN silently drop a label a concurrent editor added, so it carries the "
                "concurrency guard (see --concurrency-guard). Cannot be combined with "
                "--add-label/--remove-label."
            ),
        ),
    ] = None,
    add_label: Annotated[
        list[str] | None,
        typer.Option(
            "--add-label",
            help=(
                "Add one label without touching the rest (repeatable). Atomic form: applied "
                "against whatever labels the issue currently has, so it cannot lose a "
                "concurrent editor's own change and carries NO concurrency guard. Cannot be "
                "combined with --labels."
            ),
        ),
    ] = None,
    remove_label: Annotated[
        list[str] | None,
        typer.Option(
            "--remove-label",
            help=(
                "Remove one label without touching the rest (repeatable). Same atomic form as "
                "--add-label, so it carries NO concurrency guard. Cannot be combined with "
                "--labels."
            ),
        ),
    ] = None,
    fix_version: Annotated[
        list[str] | None,
        typer.Option(
            "--fix-version",
            help=(
                "Fix version name to set. Repeat to set several; pass none to leave unchanged. "
                "Whole-array form: this one CAN silently drop a fix version a concurrent editor "
                "added, so it carries the concurrency guard (see --concurrency-guard). Cannot be "
                "combined with --add-fix-version/--remove-fix-version."
            ),
        ),
    ] = None,
    add_fix_version: Annotated[
        list[str] | None,
        typer.Option(
            "--add-fix-version",
            help=(
                "Add one fix version without touching the rest (repeatable). Atomic form: "
                "applied against whatever fix versions the issue currently has, so it carries "
                "NO concurrency guard. Cannot be combined with --fix-version."
            ),
        ),
    ] = None,
    remove_fix_version: Annotated[
        list[str] | None,
        typer.Option(
            "--remove-fix-version",
            help=(
                "Remove one fix version without touching the rest (repeatable). Same atomic "
                "form as --add-fix-version, so it carries NO concurrency guard. Cannot be "
                "combined with --fix-version."
            ),
        ),
    ] = None,
    parent: Annotated[
        str | None,
        typer.Option("--parent", help="New parent issue key."),
    ] = None,
    priority: Annotated[
        str | None,
        typer.Option("--priority", help="New Jira priority name."),
    ] = None,
    story_points: Annotated[
        float | None,
        typer.Option("--story-points", help="New story-point estimate."),
    ] = None,
    concurrency_guard: Annotated[
        bool,
        typer.Option(
            "--concurrency-guard/--no-concurrency-guard",
            help=(
                "Read the issue's `updated` timestamp immediately before the write and refuse "
                "the write (exit code 4) if it no longer matches. This is a check-then-act "
                "narrowing of the race window, not an atomic compare-and-swap: Jira Cloud has "
                "no conditional write here, so a change landing in that window is still lost. "
                "Default: guarded."
            ),
        ),
    ] = True,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually edit the issue in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Edit issue fields with Markdown-to-ADF rendering. Defaults to dry-run.

    `--description` and friends, plus the whole-array `--labels`/`--fix-version`, each replace
    the whole field, so a concurrent edit to that same field would otherwise be silently
    overwritten. By default a call that sets one of those is guarded: this command has to add a
    dedicated read, since edit-issue previously read nothing before writing, so immediately
    before attempting the write it captures the issue's `updated` timestamp with `issue_version`,
    as late as possible to keep the window small. The client then re-checks that same timestamp
    again right before the PUT and refuses the write (exit code 4) if it no longer matches. This
    narrows the race window to a check-then-act gap; it is not an atomic compare-and-swap, since
    Jira Cloud offers no conditional write here, so a change landing inside that narrowed window
    is still lost. Pass --no-concurrency-guard for the old unguarded behaviour.

    `--add-label`/`--remove-label` and `--add-fix-version`/`--remove-fix-version` are atomic
    instead: Jira applies them against whatever the array currently holds server-side (see
    `editmeta`'s advertised `add`/`remove` operations for `labels` and `fixVersions`), so they
    cannot lose a concurrent editor's own change and never need `expect_updated`. A call that
    mixes one of these with an unrelated whole-field option (for example `--summary` together
    with `--add-label`) still reads and checks `updated` for the whole-field part; only a call
    with no whole-field option at all skips that read entirely. `--labels` cannot be combined
    with `--add-label`/`--remove-label`, and `--fix-version` cannot be combined with
    `--add-fix-version`/`--remove-fix-version`: Jira rejects a PUT that both sets and updates the
    same field.
    """
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)

    if labels is not None and (add_label or remove_label):
        _print_error(
            "Edit error:",
            "Cannot combine --labels with --add-label/--remove-label: Jira rejects a PUT that "
            "both sets and updates the same field. --labels replaces the whole array and "
            "carries the concurrency guard; --add-label/--remove-label are atomic and carry "
            "none. Use one form or the other.",
        )
        raise typer.Exit(2)
    if fix_version and (add_fix_version or remove_fix_version):
        _print_error(
            "Edit error:",
            "Cannot combine --fix-version with --add-fix-version/--remove-fix-version: Jira "
            "rejects a PUT that both sets and updates the same field. --fix-version replaces "
            "the whole array and carries the concurrency guard; --add-fix-version/"
            "--remove-fix-version are atomic and carry none. Use one form or the other.",
        )
        raise typer.Exit(2)

    for _option_name, _values in (
        ("--add-label", add_label),
        ("--remove-label", remove_label),
        ("--add-fix-version", add_fix_version),
        ("--remove-fix-version", remove_fix_version),
    ):
        if _values and any(not value.strip() for value in _values):
            _print_error(
                "Edit error:",
                f"{_option_name} must not be empty or whitespace-only. Pass a real label or "
                "fix version name, or omit the option entirely.",
            )
            raise typer.Exit(2)

    try:
        body = (
            read_text_source(description, description_file, section=description_section)
            if description or description_file
            else None
        )
        payload = edit_issue_payload(
            summary=summary,
            description=body,
            labels=_csv_list(labels),
            fix_versions=fix_version,
            parent_key=parent,
            priority=priority,
            story_points=story_points,
            labels_add=add_label,
            labels_remove=remove_label,
            fix_versions_add=add_fix_version,
            fix_versions_remove=remove_fix_version,
            story_points_field=config.story_points_field,
        )
    except (OSError, ValueError) as exc:
        _print_error("Edit error:", exc)
        raise typer.Exit(2) from exc

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                _say("[yellow]Dry run:[/yellow] would edit issue", json_mode=json_output)
                _say(f"issue: {issue_key}", json_mode=json_output)
                _say(f"safety: {safety}", json_mode=json_output)
                if json_output:
                    _emit_json({**payload, "dry_run": True})
                else:
                    _print_json(payload)
                _say("Pass --execute to write it to Jira.", json_mode=json_output)
                return

            # edit-issue previously read nothing before writing. issue_version is the dedicated,
            # minimal read added for the guard: it fetches only `updated`, taken as late as
            # possible (right before the write call below) so the window this narrows — between
            # this read and the client's own immediate re-check before its PUT — stays as small
            # as this process can make it. Skipped entirely when the request carries no
            # whole-field write at all (`payload.get("fields")` empty or absent -- absent is how
            # edit_issue_payload now represents "nothing to guard", the same way it already
            # omitted an empty `update`): atomic-only add/remove operations need no token to
            # guard with, so there is nothing here worth the read.
            expect_updated = (
                client.issue_version(issue_key)
                if concurrency_guard and payload.get("fields")
                else None
            )

            data = client.edit_issue(
                issue_key,
                summary=summary,
                description=body,
                labels=_csv_list(labels),
                fix_versions=fix_version,
                parent_key=parent,
                priority=priority,
                story_points=story_points,
                labels_add=add_label,
                labels_remove=remove_label,
                fix_versions_add=add_fix_version,
                fix_versions_remove=remove_fix_version,
                expect_updated=expect_updated,
            )
            audit_write(
                config,
                action="edit_issue",
                issue_key=issue_key,
                payload={**payload, "concurrency_guard": concurrency_guard},
                result=data,
            )
        except JiraConcurrentEditError as exc:
            _exit_concurrent_edit_error(exc, json_mode=json_output)
        except JiraMissingConcurrencyTokenError as exc:
            _exit_missing_concurrency_token_error(exc, json_mode=json_output)
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Edited {issue_key}.")


@app.command("clear-parent")
def clear_parent(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually clear the issue parent in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the verified Jira JSON response."),
    ] = False,
) -> None:
    """Clear an issue parent explicitly. Defaults to dry-run and verifies the reread.

    G2/G8 decision: this deliberately carries no concurrency guard, for the same reason as its
    MCP equivalent, `jira_clear_issue_parent`: it sets one relationship field to a single,
    specific, verified target state (null) rather than reading and rewriting a whole computed
    field, and the reread below independently confirms that target state regardless of what a
    concurrent editor did to `parent` in between.
    """
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)

    payload = edit_issue_payload(clear_parent=True, story_points_field=config.story_points_field)
    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            current = client.get_issue(issue_key, fields=["parent"])
            before = (current.get("fields") or {}).get("parent")
            if not execute:
                _say("[yellow]Dry run:[/yellow] would clear issue parent", json_mode=json_output)
                _say(f"issue: {issue_key}", json_mode=json_output)
                _say(f"safety: {safety}", json_mode=json_output)
                if json_output:
                    _emit_json({"before": before, "payload": payload, "dry_run": True})
                else:
                    _print_json({"before": before, "payload": payload})
                _say("Pass --execute to clear the parent in Jira.", json_mode=json_output)
                return

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
                raise ValueError(f"Parent clear was not verified for {issue_key}: {after!r}")
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)
        except ValueError as exc:
            error_console.print(f"[red]Parent clear error:[/red] {exc}")
            raise typer.Exit(1) from exc

    if json_output:
        _emit_json({**verified, "dry_run": False})
        return
    console.print(f"Cleared parent of {issue_key}.")


@app.command("patch-description-section")
def patch_description_section(
    key: Annotated[str, typer.Argument(help="Jira issue key.")],
    section: Annotated[
        str,
        typer.Option("--section", help="Heading of the section to patch in the live description."),
    ],
    content: Annotated[
        str | None,
        typer.Option("--content", help="Markdown for the new section."),
    ] = None,
    content_file: Annotated[
        Path | None,
        typer.Option("--content-file", help="UTF-8 Markdown file for the new section."),
    ] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="replace, append, prepend, or delete."),
    ] = "replace",
    include_heading: Annotated[
        bool,
        typer.Option(
            "--include-heading/--no-include-heading",
            help=(
                "With --mode replace only: the content carries its own heading, or keep the "
                "existing heading node and change only the body."
            ),
        ),
    ] = True,
    expect_swallowed_heading: Annotated[
        list[str] | None,
        typer.Option(
            "--expect-swallowed-heading",
            help="Name one section the slice removes besides the one named. Repeat, in order.",
        ),
    ] = None,
    allow_wide_rewrite: Annotated[
        bool,
        typer.Option(
            "--allow-wide-rewrite",
            help="Permit a slice that is the whole description. It grants nothing else.",
        ),
    ] = False,
    concurrency_guard: Annotated[
        bool,
        typer.Option(
            "--concurrency-guard/--no-concurrency-guard",
            help=(
                "Capture the issue's `updated` timestamp in the same read as the description, "
                "then refuse the write (exit code 4) if it no longer matches immediately before "
                "the write. This is a check-then-act narrowing of the race window, not an "
                "atomic compare-and-swap: Jira Cloud has no conditional write here, so a change "
                "landing in that window is still lost. Default: guarded."
            ),
        ),
    ] = True,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually patch the section in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Rewrite one section of an issue's existing description. Defaults to dry-run.

    Only the matched section changes; every top-level node outside it is carried over unchanged.
    A heading with no peer of the same or a lower level below it takes everything after it,
    whatever sits in front of it, so naming the document title can mean the whole field.
    `--mode replace` and `--mode delete` pass two independent controls. A slice that is the
    whole content list needs `--allow-wide-rewrite`. Every heading inside the slice below the
    one you named has to be named too, with a repeated `--expect-swallowed-heading` in document
    order; the refusal prints the exact list. `--allow-wide-rewrite` does not satisfy the naming
    duty. The description is read and written back whole, so a naive read-then-write still risks
    silently reverting a concurrent edit. By default this command guards against exactly that:
    the read that fetches the description also fetches `updated` in the same request, so the
    guard token belongs to the same read as the content rather than to a second read that could
    itself straddle a change; the write then fails with exit code 4 if `updated` no longer
    matches immediately beforehand. This narrows the race window but does not close it — Jira
    Cloud offers no conditional write on this resource, so an edit landing inside that narrowed
    window is still lost. Pass --no-concurrency-guard for the old unguarded behaviour. Use this
    command instead of `edit-issue --description-section`, which always replaces the whole field.
    """
    config = _config()
    try:
        if mode == "delete":
            if content is not None or content_file is not None:
                error_console.print(
                    "[red]Patch error:[/red] Mode delete removes the section and takes no content."
                )
                raise typer.Exit(2)
            body = None
        else:
            body = read_text_source(content, content_file)
    except (OSError, ValueError) as exc:
        _print_error("Patch error:", exc)
        raise typer.Exit(2) from exc

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, key)
            # Fetch "updated" in the SAME request as "description" so the guard token belongs
            # to the read that produced the content being patched, rather than to a second,
            # independent read that could itself straddle a concurrent change.
            issue = client.get_issue(key, fields=["description", "updated"])
            issue_fields = issue.get("fields") or {}
            read_updated = issue_fields.get("updated")
            expect_updated = require_concurrency_token(
                key, read_updated, guard_enabled=concurrency_guard
            )
            patch = patch_adf_description_section(
                issue_fields.get("description"),
                section=section,
                mode=mode,
                content=body,
                include_heading=include_heading,
                expect_swallowed_headings=expect_swallowed_heading,
                allow_wide_rewrite=allow_wide_rewrite,
            )
        except AdfSectionPatchRefused as exc:
            _print_error("Patch refused:", exc)
            if json_output:
                _emit_json(
                    {
                        "error_type": "section_patch_refused",
                        "swallowed_headings": exc.swallowed_headings,
                        "expect_swallowed_headings": exc.named_headings,
                        "whole_document": exc.whole_document,
                        "dry_run": True,
                    }
                )
            raise typer.Exit(2) from exc
        except (OSError, ValueError) as exc:
            _print_error("Patch error:", exc)
            raise typer.Exit(2) from exc
        except AdfPatchInvariantError as exc:
            _print_error("Patch invariant error:", exc)
            raise typer.Exit(2) from exc
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraMissingConcurrencyTokenError as exc:
            _exit_missing_concurrency_token_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

        summary = {
            "issue": key,
            "heading": patch["heading"],
            "level": patch["level"],
            "mode": patch["mode"],
            "start_index": patch["start_index"],
            "end_index": patch["end_index"],
            "replaced_node_count": patch["replaced_node_count"],
            "new_node_count": patch["new_node_count"],
            "preserved_before": patch["preserved_before"],
            "preserved_after": patch["preserved_after"],
            "original_node_count": patch["original_node_count"],
            "patched_node_count": patch["patched_node_count"],
            "whole_document": patch["whole_document"],
            "swallowed_headings": patch["swallowed_headings"],
        }

        if not execute:
            _say(
                "[yellow]Dry run:[/yellow] would patch one description section",
                json_mode=json_output,
            )
            _say(f"safety: {safety}", json_mode=json_output)
            if json_output:
                _emit_json({**summary, "new_nodes": patch["new_nodes"], "dry_run": True})
            else:
                _print_json(summary)
                _print_json(patch["new_nodes"])
            _say("Pass --execute to write it to Jira.", json_mode=json_output)
            return

        try:
            data = client.edit_issue_adf_description(
                key, patch["document"], expect_updated=expect_updated
            )
            audit_write(
                config,
                action="patch_description_section",
                issue_key=key,
                payload={
                    "section": section,
                    "include_heading": include_heading,
                    "expect_swallowed_headings": expect_swallowed_heading,
                    "allow_wide_rewrite": allow_wide_rewrite,
                    "concurrency_guard": concurrency_guard,
                    **summary,
                },
                result=data,
            )
        except JiraConcurrentEditError as exc:
            _exit_concurrent_edit_error(exc, json_mode=json_output)
        except JiraMissingConcurrencyTokenError as exc:
            _exit_missing_concurrency_token_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Patched section {patch['heading']!r} of {key}.")


@app.command("mcp-config")
def mcp_config() -> None:
    """Print the local MCP client configuration block."""
    _emit_json(_mcp_config_payload())


@app.command("config-fields")
def config_fields(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the resolved field ids as JSON."),
    ] = False,
) -> None:
    """Show the custom field ids this tool has resolved from config. Read-only, no Jira call.

    `story_points_field` and `sprint_field` are both instance-specific Jira custom field ids --
    a story-points id that is right on one Jira site can be someone else's sprint field on
    another -- resolved the same way in one place, `config.py`: a `JIRA_STORY_POINTS_FIELD` /
    `JIRA_SPRINT_FIELD` env var, or this site's probed default (`customfield_10028` /
    `customfield_10020`). Two incorrect readings came from guessing a custom field id instead of
    checking it; this command exists so a caller can check instead of guess, before running
    anything that writes.

    On this site: `customfield_10028` is "Story Points", populated and editable for project FC,
    and is the correct `story_points_field` default. `customfield_10016` is "Story point
    estimate" -- it exists, but is unused and not editable for FC, and must never be the
    configured `story_points_field` on this site.
    """
    config = _config()
    if json_output:
        _emit_json(
            {
                "story_points_field": config.story_points_field,
                "sprint_field": config.sprint_field,
            }
        )
        return
    console.print(f"story_points_field: {config.story_points_field}")
    console.print(f"sprint_field: {config.sprint_field}")


@app.command("smoke-write")
def smoke_write(
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Run write smoke actions. Without this, only dry-run/read checks run.",
        ),
    ] = False,
    transition_name: Annotated[
        str | None,
        typer.Option("--transition", help="Optional transition name or ID to test."),
    ] = None,
) -> None:
    """Run a write-capability smoke check against the configured test issue."""
    config = _config()
    if not config.test_issue_key:
        error_console.print("[red]JIRA_TEST_ISSUE_KEY is not configured.[/red]")
        raise typer.Exit(2)

    issue_key = config.test_issue_key
    comment = "jira-agent smoke-write validation."
    with JiraClient(config) as client:
        try:
            me = client.myself()
            issue_data = client.get_issue(issue_key, fields=DEFAULT_FIELDS)
            transitions_data = client.get_transitions(issue_key)
            selected = (
                select_transition(transitions_data, transition_name) if transition_name else None
            )

            console.print(f"issue: {issue_key}")
            _print_issues_table([issue_data])
            console.print(f"comment: {comment}")
            console.print(f"assignee target: {me.get('displayName')} ({me.get('accountId')})")
            if transition_name:
                if selected:
                    target = selected.get("to") or {}
                    console.print(
                        f"transition target: {selected.get('name')} "
                        f"({selected.get('id')}) -> {target.get('name')}"
                    )
                else:
                    _print_error("No transition found for:", transition_name)
                    raise typer.Exit(1)

            if not execute:
                console.print("[yellow]Dry run:[/yellow] no Jira writes were performed.")
                console.print("Pass --execute to add the comment and assign the issue.")
                return

            ensure_issue_write_allowed(config, client, issue_key)
            added_comment = client.add_comment(issue_key, comment)
            client.assign_issue(issue_key, str(me["accountId"]))
            audit_write(
                config,
                action="smoke_write",
                issue_key=issue_key,
                payload={
                    "comment": comment,
                    "accountId": str(me["accountId"]),
                    "transition": selected,
                },
                result={"comment_id": added_comment.get("id")},
            )
            console.print(f"added comment: {added_comment.get('id')}")
            console.print("assigned issue to authenticated user")
            if selected:
                client.transition_issue(issue_key, str(selected["id"]))
                console.print(f"transitioned issue with {selected.get('name')}")
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)


@app.command()
def assign(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    account_id: Annotated[
        str | None,
        typer.Option(
            "--account-id",
            help="Target Jira accountId. Defaults to the authenticated user.",
        ),
    ] = None,
    unassigned: Annotated[
        bool,
        typer.Option("--unassigned", help="Assign the issue to Unassigned."),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually assign the issue in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Assign a Jira issue. Defaults to dry-run."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            me = client.myself()
            target_account_id = None if unassigned else account_id or str(me["accountId"])
            if not execute:
                _say("[yellow]Dry run:[/yellow] would assign issue", json_mode=json_output)
                _say(f"issue: {issue_key}", json_mode=json_output)
                _say(
                    f"target accountId: {target_account_id or 'Unassigned'}",
                    json_mode=json_output,
                )
                if not unassigned and not account_id:
                    _say(f"target user: {me.get('displayName')}", json_mode=json_output)
                if json_output:
                    _emit_json(
                        {
                            "issue": issue_key,
                            "accountId": target_account_id,
                            "dry_run": True,
                        }
                    )
                _say("Pass --execute to write it to Jira.", json_mode=json_output)
                return

            ensure_issue_write_allowed(config, client, issue_key)
            data = client.assign_issue(issue_key, target_account_id)
            audit_write(
                config,
                action="assign_issue",
                issue_key=issue_key,
                payload={"accountId": target_account_id},
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Assigned {issue_key} to {target_account_id or 'Unassigned'}.")


@app.command("assign-user")
def assign_user(
    query: Annotated[str, typer.Argument(help="Name or email query for assignable user.")],
    key: Annotated[
        str | None,
        typer.Option("--issue", help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually assign the issue in Jira."),
    ] = False,
) -> None:
    """Assign an issue by searching an assignable user."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            users = client.search_assignable_users(query, issue_key=issue_key, limit=10)
            active_users = [user for user in users if user.get("active")]
            if len(active_users) != 1:
                _print_users_table(active_users)
                error_console.print(
                    "[red]Expected exactly one active assignable user, "
                    f"got {len(active_users)}.[/red]"
                )
                raise typer.Exit(1)
            user = active_users[0]
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would assign issue")
                console.print(f"issue: {issue_key}")
                console.print(f"user: {user.get('displayName')} ({user.get('accountId')})")
                console.print("Pass --execute to write it to Jira.")
                return

            ensure_issue_write_allowed(config, client, issue_key)
            data = client.assign_issue(issue_key, str(user["accountId"]))
            audit_write(
                config,
                action="assign_user",
                issue_key=issue_key,
                payload={"query": query, "accountId": user.get("accountId")},
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Assigned {issue_key} to {user.get('displayName')}.")


@app.command("link-issue")
def link_issue(
    outward_key: Annotated[
        str,
        typer.Argument(
            help="Issue displaying the link type's outward label, for example 'blocks'."
        ),
    ],
    inward_key: Annotated[
        str,
        typer.Argument(
            help="Issue displaying the link type's inward label, for example 'is blocked by'."
        ),
    ],
    link_type: Annotated[
        str,
        typer.Option("--type", help="Jira issue link type name."),
    ] = "Relates",
    comment: Annotated[
        str | None,
        typer.Option("--comment", help="Optional comment sent with the Jira link."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually create the issue link in Jira."),
    ] = False,
) -> None:
    """Link two Jira issues by their business direction. Defaults to dry-run."""
    config = _config()
    payload = issue_link_payload(
        outward_issue_key=outward_key,
        inward_issue_key=inward_key,
        link_type=link_type,
        comment=comment,
    )
    with JiraClient(config) as client:
        try:
            outward_safety = ensure_issue_write_allowed(config, client, outward_key)
            inward_safety = ensure_issue_write_allowed(config, client, inward_key)
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would create issue link")
                console.print(f"outward safety: {outward_safety}")
                console.print(f"inward safety: {inward_safety}")
                _print_json(payload)
                console.print("Pass --execute to write it to Jira.")
                return

            data = client.link_issues(
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
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Linked {outward_key} to {inward_key} with {link_type}.")


@app.command("delete-issue-link")
def delete_issue_link(
    outward_key: Annotated[str, typer.Argument(help="Expected outward issue key.")],
    inward_key: Annotated[str, typer.Argument(help="Expected inward issue key.")],
    link_type: Annotated[
        str,
        typer.Option("--type", help="Exact Jira issue link type name."),
    ] = "Relates",
    link_id: Annotated[
        str | None,
        typer.Option(
            "--link-id",
            help="Optional exact Jira link id. Omit only when the fresh reads identify one link.",
        ),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually delete the validated issue link in Jira."),
    ] = False,
) -> None:
    """Delete one validated directed issue link. Defaults to dry-run."""
    config = _config()
    with JiraClient(config) as client:
        try:
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
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would delete issue link")
                console.print(f"outward safety: {outward_safety}")
                console.print(f"inward safety: {inward_safety}")
                _print_json({"resolved_link": resolved})
                console.print("Pass --execute to delete this exact link in Jira.")
                return

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
                raise ValueError(
                    f"Issue-link deletion was not verified for {resolved['id']}: {still_present}"
                )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)
        except ValueError as exc:
            error_console.print(f"[red]Issue-link deletion error:[/red] {exc}")
            raise typer.Exit(1) from exc

    console.print(f"Deleted {link_type} link {resolved['id']} from {outward_key} to {inward_key}.")


@app.command("attach")
def attach(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    file_path: Annotated[
        Path | None,
        typer.Option("--file", exists=True, dir_okay=False, readable=True, help="File to attach."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually upload the attachment to Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Attach a file to a Jira issue. Defaults to dry-run."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    if file_path is None:
        error_console.print("[red]--file is required.[/red]")
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                _say("[yellow]Dry run:[/yellow] would attach file", json_mode=json_output)
                _say(f"issue: {issue_key}", json_mode=json_output)
                _say(f"file: {file_path}", json_mode=json_output)
                _say(f"safety: {safety}", json_mode=json_output)
                if json_output:
                    _emit_json({"issue": issue_key, "file": str(file_path), "dry_run": True})
                _say("Pass --execute to upload it to Jira.", json_mode=json_output)
                return

            data = client.add_attachment(issue_key, file_path)
            audit_write(
                config,
                action="attach",
                issue_key=issue_key,
                payload={"file": str(file_path), "size": file_path.stat().st_size},
                result={"attachments": data},
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({"attachments": data, "dry_run": False})
        return
    for attachment in data:
        console.print(f"Attached {attachment.get('filename')} ({attachment.get('id')}).")


@app.command("delete-attachment")
def delete_attachment(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    attachment_id: Annotated[
        str | None,
        typer.Option("--attachment-id", help="Attachment id to delete."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually delete the attachment from Jira."),
    ] = False,
) -> None:
    """Delete one attachment from an allowed Jira issue. Defaults to dry-run."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    if not attachment_id:
        error_console.print("[red]--attachment-id is required.[/red]")
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["attachment"])
            attachments = issue.get("fields", {}).get("attachment") or []
            attachment = next(
                (item for item in attachments if str(item.get("id")) == attachment_id),
                None,
            )
            if attachment is None:
                error_console.print(
                    f"[red]Attachment {attachment_id} is not present on {issue_key}.[/red]"
                )
                raise typer.Exit(2)
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would delete attachment")
                console.print(f"issue: {issue_key}")
                console.print(f"attachment: {attachment.get('filename')} ({attachment_id})")
                console.print(f"safety: {safety}")
                console.print("Pass --execute to delete it from Jira.")
                return

            data = client.delete_attachment(attachment_id, issue_key=issue_key)
            audit_write(
                config,
                action="delete_attachment",
                issue_key=issue_key,
                payload={"attachment_id": attachment_id, "filename": attachment.get("filename")},
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Deleted attachment {attachment_id} from {issue_key}.")


@app.command("embed-diagram")
def embed_diagram(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    file_path: Annotated[
        Path | None,
        typer.Option("--file", exists=True, dir_okay=False, readable=True, help="PNG file."),
    ] = None,
    alt: Annotated[
        str,
        typer.Option("--alt", help="Alt text for the embedded diagram."),
    ] = "Architecture diagram",
    delete_filename: Annotated[
        str | None,
        typer.Option(
            "--delete-filename",
            help="Comma-separated existing attachment filenames to remove.",
        ),
    ] = None,
    delete_attachment_id: Annotated[
        str | None,
        typer.Option(
            "--delete-attachment-id",
            help="Comma-separated existing attachment ids to remove.",
        ),
    ] = None,
    concurrency_guard: Annotated[
        bool,
        typer.Option(
            "--concurrency-guard/--no-concurrency-guard",
            help=(
                "Uploading the file is itself a write that moves this issue's own `updated`, so "
                "the token cannot come from the FIRST read (the one that only resolves "
                "--delete-filename and feeds the dry-run display). Instead, after the upload, "
                "this re-reads the description and `updated` together in one request and builds "
                "the write from THAT read, then refuses the write (exit code 4) if `updated` no "
                "longer matches immediately before it. This is a check-then-act narrowing of the "
                "race window, not an atomic compare-and-swap: Jira Cloud has no conditional write "
                "here, so a change landing in that narrower window is still lost. "
                "Default: guarded."
            ),
        ),
    ] = True,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Upload, embed, and clean up in Jira."),
    ] = False,
) -> None:
    """Upload a PNG diagram and embed it in the Jira description. Defaults to dry-run.

    This rewrites the whole description field, so a concurrent edit to that field would
    otherwise be silently overwritten. By default the write is guarded, but unlike
    `embed-existing-png` and `upsert-image-gallery`, the token cannot come from the FIRST read
    (the one that fetches attachments to resolve `--delete-filename` and populate the dry-run
    display): uploading the file is itself a write that moves this issue's own `updated` on the
    Jira side, so a token read before the upload is stale by construction, and a description
    built from that same first read would already be out of date by the time of the write.

    Instead, after the upload this re-reads the description and `updated` together in one
    request, and builds the write from THAT read. The guard therefore covers only the span from
    this post-upload read to the write: a concurrent edit landing before that read is
    incorporated into the write rather than lost, because the description being rewritten is the
    fresh one, not the first-read one. This is still check-then-act, not an atomic
    compare-and-swap — Jira Cloud has no conditional write on this resource, so a change landing
    between the post-upload read and the write is still lost. Pass --no-concurrency-guard for the
    old unguarded behaviour.
    """
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    if file_path is None:
        error_console.print("[red]--file is required.[/red]")
        raise typer.Exit(2)
    try:
        width, height = png_dimensions(file_path)
    except ValueError as exc:
        _print_error("Diagram error:", exc)
        raise typer.Exit(2) from exc

    delete_names = set(_csv_list(delete_filename) or [])
    delete_ids = set(_csv_list(delete_attachment_id) or [])

    with JiraClient(config) as client:
        uploaded_ids: list[str] = []
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            # First read: legitimately only for resolving --delete-filename to attachment ids
            # and for the dry-run display. Its `description` is never written; see the write
            # below, which reads a fresh copy after the upload.
            issue = client.get_issue(issue_key, fields=["description", "attachment"])
            attachments = issue.get("fields", {}).get("attachment") or []
            for attachment in attachments:
                if attachment.get("filename") in delete_names:
                    delete_ids.add(str(attachment.get("id")))
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would embed diagram")
                console.print(f"issue: {issue_key}")
                console.print(f"file: {file_path}")
                console.print(f"dimensions: {width}x{height}")
                console.print(f"delete attachment ids: {sorted(delete_ids) or 'none'}")
                console.print(f"safety: {safety}")
                console.print("Pass --execute to upload and embed it in Jira.")
                return

            uploaded = client.add_attachment(issue_key, file_path)
            attachment = uploaded[0]
            attachment_id = str(attachment.get("id"))
            uploaded_ids.append(attachment_id)
            media_id = client.attachment_media_id(attachment_id, issue_key=issue_key)
            # The upload above already changed this issue's `updated` on the Jira side, so
            # content and guard token are read together HERE, after the last mutation this
            # command makes before its write, rather than built from the first (now possibly
            # stale) read above. This is what makes the guard meaningful: the token and the
            # document it is meant to protect come from the same request.
            fresh_issue = client.get_issue(issue_key, fields=["description", "updated"])
            fresh_fields = fresh_issue.get("fields") or {}
            existing_description = fresh_fields.get("description") or {
                "type": "doc",
                "version": 1,
                "content": [],
            }
            description = replace_diagram_with_media(
                existing_description,
                media_id=media_id,
                alt=alt,
                width=width,
                height=height,
            )
            expect_updated = require_concurrency_token(
                issue_key, fresh_fields.get("updated"), guard_enabled=concurrency_guard
            )
            result = client.edit_issue_adf_description(
                issue_key, description, expect_updated=expect_updated
            )
        except JiraConcurrentEditError as exc:
            _cleanup_uploaded_attachments(client, issue_key, uploaded_ids)
            _exit_concurrent_edit_error(exc, json_mode=False)
        except JiraMissingConcurrencyTokenError as exc:
            _cleanup_uploaded_attachments(client, issue_key, uploaded_ids)
            _exit_missing_concurrency_token_error(exc, json_mode=False)
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _cleanup_uploaded_attachments(client, issue_key, uploaded_ids)
            _exit_api_error(exc)

        # The write above already succeeded and the new attachment is now referenced by the
        # live description, so a failure from here on (deleting OTHER, already-superseded
        # attachments) must not roll back the upload this call just committed.
        deleted: list[str] = []
        try:
            for item_id in sorted(delete_ids - {attachment_id}):
                client.delete_attachment(item_id, issue_key=issue_key)
                deleted.append(item_id)
            audit_write(
                config,
                action="embed_diagram",
                issue_key=issue_key,
                payload={
                    "file": str(file_path),
                    "attachment_id": attachment_id,
                    "media_id": media_id,
                    "deleted_attachment_ids": deleted,
                    "dimensions": {"width": width, "height": height},
                    "concurrency_guard": concurrency_guard,
                },
                result=result,
            )
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(
        f"Embedded {file_path.name} ({attachment_id}) into {issue_key}; "
        f"deleted {len(deleted)} old attachment(s)."
    )


@app.command("embed-existing-png")
def embed_existing_png(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    filename: Annotated[
        str | None,
        typer.Option("--filename", help="Exact existing PNG attachment filename."),
    ] = None,
    attachment_id: Annotated[
        str | None,
        typer.Option("--attachment-id", help="Pin an exact attachment id when filenames repeat."),
    ] = None,
    width: Annotated[
        int | None,
        typer.Option("--width", min=1, help="Verified PNG display width in pixels."),
    ] = None,
    height: Annotated[
        int | None,
        typer.Option("--height", min=1, help="Verified PNG display height in pixels."),
    ] = None,
    alt: Annotated[
        str,
        typer.Option("--alt", help="Alt text for the embedded diagram."),
    ] = "Architecture diagram",
    replace_existing: Annotated[
        bool,
        typer.Option(
            "--replace-existing",
            help=(
                "Replace exactly one currently embedded media item from an attachment of this "
                "filename."
            ),
        ),
    ] = False,
    concurrency_guard: Annotated[
        bool,
        typer.Option(
            "--concurrency-guard/--no-concurrency-guard",
            help=(
                "Capture the issue's `updated` timestamp in the same read as the description, "
                "then refuse the write (exit code 4) if it no longer matches immediately before "
                "the write. This is a check-then-act narrowing of the race window, not an "
                "atomic compare-and-swap: Jira Cloud has no conditional write here, so a change "
                "landing in that window is still lost. Default: guarded."
            ),
        ),
    ] = True,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually embed the existing attachment in Jira."),
    ] = False,
) -> None:
    """Embed one existing PNG attachment without uploading a duplicate. Defaults to dry-run.

    This rewrites the whole description field, so a concurrent edit to that field would
    otherwise be silently overwritten. By default the write is guarded: the same read that
    fetches the description and attachments also fetches `updated`, so the guard token belongs
    to that read rather than to a second one that could itself straddle a change; the write then
    refuses (exit code 4) if `updated` no longer matches immediately beforehand. Pass
    --no-concurrency-guard for the old unguarded behaviour.
    """
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    if not filename:
        error_console.print("[red]--filename is required.[/red]")
        raise typer.Exit(2)
    if width is None or height is None:
        error_console.print(
            "[red]--width and --height are required; dimensions are never guessed.[/red]"
        )
        raise typer.Exit(2)

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            issue = client.get_issue(issue_key, fields=["description", "attachment", "updated"])
            expect_updated = require_concurrency_token(
                issue_key,
                (issue.get("fields") or {}).get("updated"),
                guard_enabled=concurrency_guard,
            )
            attachment = resolve_png_attachment(
                issue, filename=filename, attachment_id=attachment_id
            )
            resolved_id = str(attachment["id"])
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
                    raise ValueError(
                        "Expected exactly one embedded media item from attachments named "
                        f"{filename!r}; "
                        f"found {len(matches)}."
                    )
                replace_media_id = matches[0]

            payload = {
                "attachment_id": resolved_id,
                "filename": filename,
                "media_id": media_id,
                "dimensions": {"width": width, "height": height},
                "replace_media_id": replace_media_id,
            }
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would embed existing PNG attachment")
                console.print(f"issue: {issue_key}")
                console.print(f"safety: {safety}")
                _print_json(payload)
                console.print("Pass --execute to embed this existing attachment in Jira.")
                return

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
                payload={**payload, "concurrency_guard": concurrency_guard},
                result={
                    "edit": result,
                    "verified_attachment_id": verified_attachment["id"],
                    "embedded": embedded,
                },
            )
            if not embedded:
                raise ValueError(
                    f"Existing attachment {resolved_id} was not found inline after rereading "
                    f"{issue_key}."
                )
        except JiraConcurrentEditError as exc:
            _exit_concurrent_edit_error(exc, json_mode=False)
        except JiraMissingConcurrencyTokenError as exc:
            _exit_missing_concurrency_token_error(exc, json_mode=False)
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)
        except ValueError as exc:
            error_console.print(f"[red]Existing PNG embed error:[/red] {exc}")
            raise typer.Exit(1) from exc

    console.print(f"Embedded existing {filename} ({resolved_id}) into {issue_key}.")


@app.command("upsert-image-gallery")
def upsert_image_gallery(
    key: Annotated[str, typer.Argument(help="Jira issue key.")],
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, readable=True),
    ],
    documentation_filename: Annotated[
        str | None,
        typer.Option("--documentation-filename"),
    ] = None,
    before_heading: Annotated[
        str,
        typer.Option("--before-heading"),
    ] = "Child Issues",
    concurrency_guard: Annotated[
        bool,
        typer.Option(
            "--concurrency-guard/--no-concurrency-guard",
            help=(
                "Capture the issue's `updated` timestamp in the same read as the description, "
                "then refuse the write (exit code 4) if it no longer matches immediately before "
                "the write. This is a check-then-act narrowing of the race window, not an "
                "atomic compare-and-swap: Jira Cloud has no conditional write here, so a change "
                "landing in that window is still lost. Default: guarded."
            ),
        ),
    ] = True,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually update the Jira description."),
    ] = False,
) -> None:
    """Insert or replace a linked screenshot gallery using existing Jira attachments.

    This rewrites the whole description field, so a concurrent edit to that field would
    otherwise be silently overwritten. By default the write is guarded: the same read that
    fetches the description and attachments also fetches `updated`, so the guard token belongs
    to that read rather than to a second one that could itself straddle a change; the write then
    refuses (exit code 4) if `updated` no longer matches immediately beforehand. Pass
    --no-concurrency-guard for the old unguarded behaviour.
    """
    config = _config()
    try:
        manifest = load_image_gallery_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        _print_error("Gallery manifest error:", exc)
        raise typer.Exit(2) from exc

    filenames = {str(row["file"]) for section in manifest["sections"] for row in section["rows"]}
    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, key)
            issue = client.get_issue(key, fields=["description", "attachment", "updated"])
            expect_updated = require_concurrency_token(
                key,
                (issue.get("fields") or {}).get("updated"),
                guard_enabled=concurrency_guard,
            )
            attachments = issue.get("fields", {}).get("attachment") or []
            by_name = newest_attachments_by_filename(attachments)
            missing = sorted(filenames - set(by_name))
            if documentation_filename and documentation_filename not in by_name:
                missing.append(documentation_filename)
            if missing:
                error_console.print(
                    f"[red]Missing Jira attachments on {key}:[/red] {', '.join(missing)}"
                )
                raise typer.Exit(2)

            urls = {name: str(by_name[name]["content"]) for name in filenames}
            documentation_url = (
                str(by_name[documentation_filename]["content"]) if documentation_filename else None
            )
            blocks = image_gallery_blocks(
                manifest,
                attachment_urls=urls,
                documentation_url=documentation_url,
            )
            description = upsert_top_level_section(
                issue.get("fields", {}).get("description")
                or {"type": "doc", "version": 1, "content": []},
                title=str(manifest["title"]),
                blocks=blocks,
                before_heading=before_heading,
            )
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would upsert image gallery")
                console.print(f"issue: {key}")
                console.print(f"title: {manifest['title']}")
                console.print(f"screenshots: {len(filenames)}")
                console.print(f"before heading: {before_heading}")
                console.print(f"safety: {safety}")
                console.print("Pass --execute to update the Jira description.")
                return

            result = client.edit_issue_adf_description(
                key, description, expect_updated=expect_updated
            )
            audit_write(
                config,
                action="upsert_image_gallery",
                issue_key=key,
                payload={
                    "manifest": str(manifest_path),
                    "filenames": sorted(filenames),
                    "documentation_filename": documentation_filename,
                    "before_heading": before_heading,
                    "concurrency_guard": concurrency_guard,
                },
                result=result,
            )
        except ValueError as exc:
            _print_error("Gallery error:", exc)
            raise typer.Exit(2) from exc
        except JiraConcurrentEditError as exc:
            _exit_concurrent_edit_error(exc, json_mode=False)
        except JiraMissingConcurrencyTokenError as exc:
            _exit_missing_concurrency_token_error(exc, json_mode=False)
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Updated image gallery '{manifest['title']}' on {key}.")


@app.command()
def changelog(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum changelog entries to return."),
    ] = 10,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Show recent changelog entries for an issue."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)
    with JiraClient(config) as client:
        try:
            data = client.get_changelog(issue_key, limit=limit)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json(data)
        return
    for entry in data.get("values") or []:
        author = entry.get("author") or {}
        console.print(f"{entry.get('created')} by {author.get('displayName')}")
        for item in entry.get("items") or []:
            console.print(
                f"  - {item.get('field')}: {item.get('fromString')} -> {item.get('toString')}"
            )


@app.command("audit-log")
def audit_log(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum audit records to show."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Show local write audit records."""
    config = _config()
    records = _read_audit_records(config.audit_log_path, limit)
    if json_output:
        _emit_json(records)
        return
    if not records:
        console.print("No audit records.")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Timestamp")
    table.add_column("Action")
    table.add_column("Issue")
    for record in records:
        table.add_row(
            str(record.get("timestamp")),
            str(record.get("action")),
            str(record.get("issue_key")),
        )
    table_console.print(table)


@app.command()
def transitions(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """List available transitions for one Jira issue."""
    with _client() as client:
        try:
            data = client.get_transitions(key)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json(data)
        return

    transitions_data = data.get("transitions") or []
    if not isinstance(transitions_data, list):
        error_console.print("[red]Unexpected Jira response: transitions is not a list[/red]")
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("To")
    for transition in transitions_data:
        target = transition.get("to") or {}
        table.add_row(
            str(transition.get("id") or ""),
            str(transition.get("name") or ""),
            str(target.get("name") or ""),
        )
    table_console.print(table)


@app.command()
def transition(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    transition_name_or_id: Annotated[
        str,
        typer.Argument(help="Transition name or transition ID."),
    ],
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually transition the issue in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Transition one Jira issue. Defaults to dry-run."""
    with _client() as client:
        try:
            transition_data = client.get_transitions(key)
            selected = select_transition(transition_data, transition_name_or_id)
            if selected is None:
                _print_error("No transition found for:", transition_name_or_id)
                raise typer.Exit(1)

            if not execute:
                target = selected.get("to") or {}
                _say("[yellow]Dry run:[/yellow] would transition issue", json_mode=json_output)
                _say(f"issue: {key}", json_mode=json_output)
                _say(
                    f"transition: {selected.get('name')} ({selected.get('id')})",
                    json_mode=json_output,
                )
                _say(f"target status: {target.get('name')}", json_mode=json_output)
                if json_output:
                    _emit_json({"issue": key, "transition": selected, "dry_run": True})
                _say("Pass --execute to write it to Jira.", json_mode=json_output)
                return

            ensure_issue_write_allowed(client.config, client, key)
            data = client.transition_issue(key, str(selected["id"]))
            audit_write(
                client.config,
                action="transition_issue",
                issue_key=key,
                payload={"transition": selected},
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Transitioned {key} with {selected.get('name')} ({selected.get('id')}).")


# --- Sprint and board commands (Agile API) ------------------------------------------------


@app.command()
def boards(
    board_type: Annotated[
        str | None,
        typer.Option("--type", help="Filter by board type, for example scrum or kanban."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Filter by board name."),
    ] = None,
    max_results: Annotated[
        int,
        typer.Option("--max-results", min=1, max=100, help="Maximum boards to return."),
    ] = 50,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """List Jira Agile boards, so a caller can find a board id without the UI.

    `--max-results` defaults to 50 and this command never fetches a second page, so a site with
    more boards than that has more than this command shows. Both output modes say so instead of
    staying silent about it: the human table is followed by a `returned N; more remain` or
    `returned N; complete` line taken from the response's own `isLast`, and the `--json` document
    carries the same fact as `returned`/`is_last` alongside the raw response. No total is
    fabricated; Jira's own `total`, when present, is reported and labelled as coming from Jira.
    """
    with _client() as client:
        try:
            data = client.list_boards(board_type=board_type, name=name, max_results=max_results)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    values = data.get("values") or []
    returned = len(values)
    is_last = data.get("isLast")
    raw_total = data.get("total")
    total = raw_total if isinstance(raw_total, int) else None

    if json_output:
        _emit_json({**data, "returned": returned, "is_last": is_last})
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Name")
    table.add_column("Type", no_wrap=True)
    for board in values:
        table.add_row(
            str(board.get("id") or ""),
            str(board.get("name") or ""),
            str(board.get("type") or ""),
        )
    table_console.print(table)
    if is_last is True:
        console.print(f"returned {returned} board(s); complete")
    elif is_last is False:
        console.print(f"returned {returned} board(s); more remain")
    else:
        console.print(f"returned {returned} board(s)")
    if total is not None:
        console.print(f"total: {total} (from Jira's response)")


@app.command()
def sprints(
    board_id: Annotated[
        int,
        typer.Option("--board", help="Board id. Use `boards` to find it."),
    ],
    state: Annotated[
        str | None,
        typer.Option("--state", help="Filter by sprint state: active, future, or closed."),
    ] = None,
    max_results: Annotated[
        int,
        typer.Option("--max-results", min=1, max=100, help="Maximum sprints to return."),
    ] = 50,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """List one board's sprints.

    `--max-results` defaults to 50 and this command never fetches a second page, so a board with
    more sprints than that has more than this command shows. Both output modes say so instead of
    staying silent about it: the human table is followed by a `returned N; more remain` or
    `returned N; complete` line taken from the response's own `isLast`, and the `--json` document
    carries the same fact as `returned`/`is_last` alongside the raw response. No total is
    fabricated; Jira's own `total`, when present, is reported and labelled as coming from Jira.
    """
    with _client() as client:
        try:
            data = client.list_board_sprints(board_id, state=state, max_results=max_results)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    values = data.get("values") or []
    returned = len(values)
    is_last = data.get("isLast")
    raw_total = data.get("total")
    total = raw_total if isinstance(raw_total, int) else None

    if json_output:
        _emit_json({**data, "returned": returned, "is_last": is_last})
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Name")
    table.add_column("State", no_wrap=True)
    table.add_column("Start", no_wrap=True)
    table.add_column("End", no_wrap=True)
    for sprint in values:
        # The table is a glance view, so dates are trimmed to their day; --json still carries
        # the full Jira timestamp. A full-width Name column matters more here: an operator moving
        # 13 issues at once reads sprint names, not sprint timestamps, to pick the right one.
        table.add_row(
            str(sprint.get("id") or ""),
            str(sprint.get("name") or ""),
            str(sprint.get("state") or ""),
            str(sprint.get("startDate") or "")[:10],
            str(sprint.get("endDate") or "")[:10],
        )
    table_console.print(table)
    if is_last is True:
        console.print(f"returned {returned} sprint(s); complete")
    elif is_last is False:
        console.print(f"returned {returned} sprint(s); more remain")
    else:
        console.print(f"returned {returned} sprint(s)")
    if total is not None:
        console.print(f"total: {total} (from Jira's response)")


@app.command("issue-sprint")
def issue_sprint(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    sprint_field: Annotated[
        str | None,
        typer.Option(
            "--sprint-field",
            help=(
                "Sprint custom field id. Defaults to config.sprint_field "
                "(JIRA_SPRINT_FIELD), resolved from config at call time rather than a fixed "
                "literal, so a site override takes effect without passing this every time."
            ),
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Print the curated result document (issue, sprint_field, "
                "sprint_field_present, sprints), not Jira's raw response."
            ),
        ),
    ] = False,
) -> None:
    """Show which sprint(s) one issue is in. Read-only, no --execute.

    Renders three distinct states instead of collapsing any two of them: the sprint field
    absent from the Jira response entirely, the field present with the value `null` or an empty
    list (the issue is in no sprint -- this Jira site returns `null`, not `[]`, verified via
    `Sprint IS EMPTY` JQL, and both render identically here), and the field present with one or
    more sprints.
    """
    with _client() as client:
        field = sprint_field or client.config.sprint_field
        try:
            issue = client.get_issue(key, fields=[field])
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    fields = issue.get("fields") or {}
    raw = fields.get(field, _SPRINT_FIELD_MISSING)
    field_present = raw is not _SPRINT_FIELD_MISSING
    sprint_list = raw if isinstance(raw, list) else []

    if not field_present:
        _say(
            f"{key}: the Jira response did not include the sprint field ({field}).",
            json_mode=json_output,
        )
    elif not sprint_list:
        _say(f"{key} is in no sprint.", json_mode=json_output)
    else:
        _say(f"{key} is in {len(sprint_list)} sprint(s):", json_mode=json_output)
        for sprint in sprint_list:
            _say(
                f"- {sprint.get('id')} {sprint.get('name')} ({sprint.get('state')})",
                json_mode=json_output,
            )

    if json_output:
        _emit_json(
            {
                "issue": key,
                "sprint_field": field,
                "sprint_field_present": field_present,
                "sprints": sprint_list,
            }
        )


@app.command("sprint-add")
def sprint_add(
    sprint_id: Annotated[int, typer.Argument(help="Target Jira sprint id.")],
    keys: Annotated[
        list[str],
        typer.Argument(help="One or more Jira issue keys to add to the sprint."),
    ],
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually add the issues to the sprint."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Add one or more issues to a sprint. Defaults to dry-run.

    This is a single batched POST (`add_issues_to_sprint` sends every key in one request), not a
    per-issue loop, so there is no "key 7 of 13 failed" partial state to report on the Jira side.
    The safety gate is checked for every key before the dry run reports success, not only before
    `--execute` writes: a clean dry run must predict what `--execute` does, and a dry run that
    never checked the gate could report success for a batch `--execute` goes on to refuse.

    The Agile add endpoint returns 204 with no body, so a 200 from Jira proves only that the
    request was accepted, not that each issue actually ended up in the sprint. The write is
    audited as soon as Jira accepts it -- before the post-write reread runs -- so a write that
    landed is never left with no audit record just because that reread afterwards failed. After
    `--execute`, every key's sprint field is re-read and `results` (per key, under `--json`; one
    line each otherwise) reports whether it is now confirmed in `sprint_id` -- a key that is not
    exits non-zero even though the write itself returned success. If the reread itself fails
    (a transient Jira error, for instance), the write is reported as accepted but unverified
    (`verified: null`) rather than as a failed write, and the command exits with the
    verification-failure code, not the generic API-error one.
    """
    if not keys:
        _print_error("No issue keys given:", "pass at least one issue key.")
        raise typer.Exit(2)

    with _client() as client:
        try:
            # Check every key's write permission before writing any of them, on both the dry run
            # and the real write. The write itself is one batched call, so this loop is what keeps
            # a rejection on key N from following a write that already happened for keys 1..N-1:
            # nothing is written until every key in the batch has cleared the gate.
            safety = [
                ensure_issue_write_allowed(client.config, client, issue_key) for issue_key in keys
            ]
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)

        if not execute:
            _say(
                f"[yellow]Dry run:[/yellow] would add {len(keys)} issue(s) to sprint {sprint_id}",
                json_mode=json_output,
            )
            for issue_key in keys:
                _say(f"- {issue_key} -> sprint {sprint_id}", json_mode=json_output)
            _say(f"safety: {safety}", json_mode=json_output)
            if json_output:
                _emit_json(
                    {
                        "sprint_id": sprint_id,
                        "issue_keys": list(keys),
                        "safety": safety,
                        "dry_run": True,
                    }
                )
            _say("Pass --execute to write it to Jira.", json_mode=json_output)
            return

        sprint_field = client.config.sprint_field
        try:
            data = client.add_issues_to_sprint(sprint_id, list(keys))
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

        verification_error: JiraApiError | None = None
        results: list[dict[str, Any]] | None = None
        try:
            results = sprint_verification_results(
                sprint_membership_report(client, keys, sprint_id, sprint_field=sprint_field),
                expected_in_sprint=True,
            )
        except JiraApiError as exc:
            verification_error = exc

        audit_write(
            client.config,
            action="sprint_add",
            issue_key=None,
            payload={"sprint_id": sprint_id, "issue_keys": list(keys)},
            result=(
                {**data, "verification": results}
                if verification_error is None
                else {**data, "verified": None, "verification_error": str(verification_error)}
            ),
        )

    if verification_error is not None:
        _print_error(
            "Verification unavailable:",
            "the write was accepted by Jira but the post-write reread failed, so it could not "
            f"be confirmed: {verification_error}",
        )
        if json_output:
            _emit_json(
                {
                    "ok": False,
                    "error_type": "sprint_verification_unavailable",
                    "sprint_id": sprint_id,
                    "issue_keys": list(keys),
                    "verified": None,
                    "message": (
                        "The sprint-add write was accepted by Jira, but the post-write "
                        "verification reread failed, so it could not be confirmed."
                    ),
                    "dry_run": False,
                }
            )
        raise typer.Exit(SPRINT_VERIFICATION_EXIT_CODE)

    assert results is not None
    failures = [entry for entry in results if not entry["verified"]]
    for entry in results:
        if entry["verified"]:
            _say(f"- {entry['key']}: verified in sprint {sprint_id}", json_mode=json_output)
        else:
            _say(
                f"- {entry['key']}: [red]FAILURE[/red] -- not in sprint {sprint_id} after write "
                f"(currently in: {_sprint_locations(entry['sprints'])})",
                json_mode=json_output,
            )

    if json_output:
        _emit_json(
            {
                "sprint_id": sprint_id,
                "issue_keys": list(keys),
                "result": data,
                "results": results,
                "ok": not failures,
                "dry_run": False,
            }
        )
        if failures:
            raise typer.Exit(SPRINT_VERIFICATION_EXIT_CODE)
        return

    if failures:
        console.print(
            f"[red]{len(failures)} of {len(keys)} issue(s) did not verify in sprint "
            f"{sprint_id} after the write.[/red]"
        )
        raise typer.Exit(SPRINT_VERIFICATION_EXIT_CODE)

    console.print(f"Added {len(keys)} issue(s) to sprint {sprint_id}, verified by reread.")


@app.command("sprint-remove")
def sprint_remove(
    sprint_id: Annotated[
        int,
        typer.Argument(help="Sprint id every key must currently be in, checked before any write."),
    ],
    keys: Annotated[
        list[str],
        typer.Argument(help="One or more Jira issue keys to move to the backlog."),
    ],
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually move the issues to the backlog."),
    ] = False,
    allow_multi_sprint_removal: Annotated[
        bool,
        typer.Option(
            "--allow-multi-sprint-removal",
            help=(
                "Proceed even when a key also sits in a non-closed sprint besides SPRINT_ID. "
                "The backlog endpoint clears that sprint too; without this flag, such a key "
                "refuses the whole batch before any write."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Remove one or more issues from SPRINT_ID, moving them to the backlog. Defaults to dry-run.

    BREAKING CHANGE: this command used to take only issue keys (`sprint-remove KEY...`) and post
    them straight to the backlog endpoint. `SPRINT_ID` is now a required first argument, and it
    is a PRECONDITION THIS COMMAND ENFORCES, NOT A FILTER JIRA APPLIES: the underlying endpoint
    (`POST /rest/agile/1.0/backlog/issue`) takes no sprint id at all. It is documented as
    equivalent to removing the FUTURE and ACTIVE sprints from the given issues -- ALL of them,
    on every key in the batch, not only the one this command was told to check. Closed sprints
    are unaffected. An issue can be in more than one non-closed sprint at once (a long-lived
    issue on a board that has moved through several sprints), so "the named sprint" and "every
    sprint this write actually clears" are not the same set, and this command never conflates
    them.

    Before any write, every key's current sprint field is re-read, and the WHOLE batch is
    refused -- nothing written for any key -- if even one key is not currently in `SPRINT_ID`,
    listing exactly which sprint each mismatched key is actually in. Separately, and also by
    default, the whole batch is refused if any key sits in another non-closed sprint besides
    `SPRINT_ID`: without `--allow-multi-sprint-removal`, that key would silently lose a sprint
    membership nobody named. This is a check immediately before the write, the same
    check-then-act race the write-scope gate already has, not an atomic guarantee that nothing
    changes in between.

    Like sprint-add, the write itself is one batched POST, and every key clears the write-scope
    safety gate before the dry run reports success, not only before `--execute` writes. The dry
    run lists, per key, every sprint it will actually leave, not only `SPRINT_ID`: a dry run that
    under-reports the blast radius is worse than none. The write is audited as soon as Jira
    accepts it -- before the post-write reread runs -- so a write that landed is never left with
    no audit record just because that reread afterwards failed. The Agile endpoint returns 204
    with no body, so after `--execute` every key's sprint field is re-read again, and `results`
    reports, per key, whether it left every sprint the dry run said it would -- not merely
    `SPRINT_ID` -- naming any of them it is still in. A key that still verifies as present in one
    of them after the write is reported as a FAILURE and the command exits non-zero, even though
    the POST itself returned success. If the reread itself fails, the write is reported as
    accepted but unverified (`verified: null`) rather than as a failed write, and the command
    exits with the verification-failure code, not the generic API-error one.
    """
    if not keys:
        _print_error("No issue keys given:", "pass at least one issue key.")
        raise typer.Exit(2)

    with _client() as client:
        try:
            safety = [
                ensure_issue_write_allowed(client.config, client, issue_key) for issue_key in keys
            ]
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)

        sprint_field = client.config.sprint_field
        try:
            precondition = sprint_removal_blast_radius(
                sprint_membership_report(client, keys, sprint_id, sprint_field=sprint_field),
                sprint_id,
            )
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

        if any(not entry["in_sprint"] for entry in precondition):
            _exit_sprint_precondition_error(
                SprintPreconditionError(sprint_id, precondition), json_mode=json_output
            )

        at_risk = [entry for entry in precondition if entry["other_sprints"]]
        if at_risk and not allow_multi_sprint_removal:
            _exit_sprint_multi_removal_error(
                SprintMultiRemovalError(sprint_id, precondition), json_mode=json_output
            )

        if not execute:
            _say(
                f"[yellow]Dry run:[/yellow] would move {len(keys)} issue(s) from sprint "
                f"{sprint_id} to the backlog",
                json_mode=json_output,
            )
            for entry in precondition:
                _say(
                    f"- {entry['key']} -> backlog, leaving "
                    f"{_sprint_locations(entry['sprints_removed'])}",
                    json_mode=json_output,
                )
            _say(f"safety: {safety}", json_mode=json_output)
            if json_output:
                _emit_json(
                    {
                        "sprint_id": sprint_id,
                        "issue_keys": list(keys),
                        "safety": safety,
                        "precondition": precondition,
                        "dry_run": True,
                    }
                )
            _say("Pass --execute to write it to Jira.", json_mode=json_output)
            return

        try:
            data = client.move_issues_to_backlog(list(keys))
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

        verification_error: JiraApiError | None = None
        results: list[dict[str, Any]] | None = None
        try:
            post_write = sprint_removal_blast_radius(
                sprint_membership_report(client, keys, sprint_id, sprint_field=sprint_field),
                sprint_id,
            )
            results = sprint_removal_verification_results(precondition, post_write)
        except JiraApiError as exc:
            verification_error = exc

        audit_write(
            client.config,
            action="sprint_remove",
            issue_key=None,
            payload={
                "sprint_id": sprint_id,
                "issue_keys": list(keys),
                "pre_write_sprints": {entry["key"]: entry["sprints"] for entry in precondition},
            },
            result=(
                {**data, "verification": results}
                if verification_error is None
                else {**data, "verified": None, "verification_error": str(verification_error)}
            ),
        )

    if verification_error is not None:
        _print_error(
            "Verification unavailable:",
            "the write was accepted by Jira but the post-write reread failed, so it could not "
            f"be confirmed: {verification_error}",
        )
        if json_output:
            _emit_json(
                {
                    "ok": False,
                    "error_type": "sprint_verification_unavailable",
                    "sprint_id": sprint_id,
                    "issue_keys": list(keys),
                    "verified": None,
                    "message": (
                        "The sprint-remove write was accepted by Jira, but the post-write "
                        "verification reread failed, so it could not be confirmed."
                    ),
                    "dry_run": False,
                }
            )
        raise typer.Exit(SPRINT_VERIFICATION_EXIT_CODE)

    assert results is not None
    failures = [entry for entry in results if not entry["verified"]]
    for entry in results:
        if entry["verified"]:
            _say(f"- {entry['key']}: verified out of sprint {sprint_id}", json_mode=json_output)
        else:
            _say(
                f"- {entry['key']}: [red]FAILURE[/red] -- still in "
                f"{_sprint_locations(entry['still_in'])} after the write",
                json_mode=json_output,
            )

    if json_output:
        _emit_json(
            {
                "sprint_id": sprint_id,
                "issue_keys": list(keys),
                "result": data,
                "results": results,
                "ok": not failures,
                "dry_run": False,
            }
        )
        if failures:
            raise typer.Exit(SPRINT_VERIFICATION_EXIT_CODE)
        return

    if failures:
        console.print(
            f"[red]{len(failures)} of {len(keys)} issue(s) did not verify out of sprint "
            f"{sprint_id} after the write.[/red]"
        )
        raise typer.Exit(SPRINT_VERIFICATION_EXIT_CODE)

    console.print(f"Moved {len(keys)} issue(s) from sprint {sprint_id} to the backlog, verified.")


# --- Issue type change ---------------------------------------------------------------------


@app.command("issue-type-options")
def issue_type_options(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Print the curated result document (issue, allowed_values), not Jira's raw "
                "editmeta response."
            ),
        ),
    ] = False,
) -> None:
    """List the issue types one issue may become. Read-only, no --execute."""
    with _client() as client:
        try:
            metadata = client.get_edit_metadata(key)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    allowed_values = issue_type_allowed_values(metadata)

    if json_output:
        _emit_json({"issue": key, "allowed_values": allowed_values})
        return

    if allowed_values:
        console.print(f"{key} may become: {', '.join(allowed_values)}")
    else:
        console.print(f"{key}: editmeta returned no allowed issue types.")


@app.command("issue-type")
def issue_type(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    new_type: Annotated[str, typer.Argument(help="Target issue type name, for example Bug.")],
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually change the issue type in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Change one issue's issue type. Defaults to dry-run.

    Refuses locally, before any write, when `new_type` is not among editmeta's allowed values for
    this issue: a doomed PUT with a Jira error is worse than a local refusal that shows the
    options. When executing, this captures `issue_version` immediately before the write and
    passes it as `expect_updated`, so a change landing between that read and the PUT surfaces as
    `JiraConcurrentEditError` here rather than silently overwriting it. That refusal goes through
    the same `_exit_concurrent_edit_error` helper, and therefore the same exit code 4 and the same
    machine-readable JSON document, as every other guarded whole-field rewrite in this file. When
    Jira's response to that same read carries no usable `updated` timestamp, `issue_version`
    raises `JiraMissingConcurrencyTokenError`, caught here the same way `human-brief` and
    `edit-issue` catch it, so the same condition produces the same
    `concurrent_edit_guard_unavailable` refusal on every guarded command rather than a generic
    exit-1 "Jira API error" on this one.
    """
    with _client() as client:
        try:
            if not client.is_issue_type_allowed(key, new_type):
                metadata = client.get_edit_metadata(key)
                allowed_values = issue_type_allowed_values(metadata)
                _print_error(
                    "Refused:",
                    f"{new_type!r} is not an allowed issue type for {key}. Allowed values: "
                    f"{', '.join(allowed_values) or 'none returned by editmeta'}.",
                )
                raise typer.Exit(2)

            safety = ensure_issue_write_allowed(client.config, client, key)

            if not execute:
                _say("[yellow]Dry run:[/yellow] would change issue type", json_mode=json_output)
                _say(f"issue: {key}", json_mode=json_output)
                _say(f"new type: {new_type}", json_mode=json_output)
                _say(f"safety: {safety}", json_mode=json_output)
                if json_output:
                    _emit_json(
                        {
                            "issue": key,
                            "requested_type": new_type,
                            "dry_run": True,
                        }
                    )
                _say("Pass --execute to write it to Jira.", json_mode=json_output)
                return

            expected_updated = client.issue_version(key)
            try:
                data = client.set_issue_type(key, new_type, expect_updated=expected_updated)
            except JiraConcurrentEditError as exc:
                _exit_concurrent_edit_error(exc, json_mode=json_output)

            audit_write(
                client.config,
                action="issue_type",
                issue_key=key,
                payload={"new_type": new_type, "expect_updated": expected_updated},
                result=data,
            )
        except JiraMissingConcurrencyTokenError as exc:
            _exit_missing_concurrency_token_error(exc, json_mode=json_output)
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc, json_mode=json_output)
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    if json_output:
        _emit_json({**data, "dry_run": False})
        return

    console.print(f"Changed {key} to issue type {new_type}.")


def _run_search(
    jql: str,
    *,
    limit: int,
    fields: str | None,
    next_page_token: str | None,
    json_output: bool,
    all_pages: bool = False,
    count: bool = False,
) -> None:
    """Run a JQL search and print or emit the one curated result shape every search-backed
    command shares: ``recent``, ``mine``, ``fc-open``, ``by-status``, ``test-scope``, and
    ``search`` itself. There is no separate raw-passthrough path any more, and no ``curated``
    flag to pick between them — a caller that trusts one page's raw ``issues`` as the whole
    answer is exactly the false-completeness defect this shape exists to rule out, and it does
    not matter which command produced that page.

    It honours ``limit`` beyond one page and ``all_pages`` via ``search_all_issues``, adds
    ``count``'s explicitly-approximate total, and always reports ``returned``/``is_last``
    instead of a total the API never gave it. This never emits a ``total`` key: none of these
    commands report a match count `/search/jql` never had in the first place.
    """
    approximate_total: int | None = None
    with _client() as client:
        sprint_field = client.config.sprint_field
        # A caller who left --fields at its shared default also gets the sprint field appended
        # to what is actually requested (see `issue`'s identical rule); an explicit --fields is
        # used exactly as given, matching the existing "explicit fields are not augmented" rule.
        if fields == _DEFAULT_FIELDS_CSV:
            requested_fields: list[str] | None = [*DEFAULT_FIELDS, sprint_field]
        else:
            requested_fields = _field_list(fields)
        try:
            if count:
                approximate_total = client.approximate_issue_count(jql)
            if next_page_token is None:
                data = client.search_all_issues(
                    jql,
                    limit=None if all_pages else limit,
                    fields=requested_fields,
                )
            else:
                data = client.search_issues(
                    jql,
                    limit=limit,
                    fields=requested_fields,
                    next_page_token=next_page_token,
                )
        except JiraApiError as exc:
            _exit_api_error(exc, json_mode=json_output)

    issues = data.get("issues") or []
    # `data` is `search_all_issues`'s own result (spelled `is_last`, snake_case like this tool's
    # other result documents) when this is the first curated page, but Jira's raw `/search/jql`
    # response (spelled `isLast`, Jira's own casing) when continuing a curated search from an
    # explicit --next-page-token. Try this method's spelling first, then Jira's raw one.
    is_last = data.get("is_last", data.get("isLast"))
    returned = len(issues) if isinstance(issues, list) else 0
    next_token = data.get("nextPageToken")

    # Same "Jira did not return requested field(s)" banner `issue` prints, reused rather than
    # rewritten: a wrong sprint field id (or any other field this site does not recognize) comes
    # back as a silent HTTP 200 with the field just missing, on every row, not only on `issue`.
    # Union across every returned row rather than just the first: Jira returns a consistent field
    # set per query in practice, but there is no reason to trust only one row's shape when the
    # whole page is in hand.
    if isinstance(issues, list) and issues:
        returned_fields: set[str] = set()
        for issue_row in issues:
            if isinstance(issue_row, dict):
                returned_fields.update((issue_row.get("fields") or {}).keys())
        _warn_missing_fields(requested_fields, returned_fields, json_mode=json_output)

    if json_output:
        payload: dict[str, Any] = {"issues": issues, "returned": returned, "is_last": is_last}
        if next_token:
            payload["nextPageToken"] = next_token
        if count:
            # Always carry the key when --count was requested, even when Jira's response left
            # `count` out and `approximate_total` is None: a caller must be able to tell "the
            # count was never asked for" (key absent) apart from "it was asked for and Jira did
            # not answer" (key present, value null), rather than reading a dropped key as the
            # latter.
            payload["approximate_total"] = approximate_total
        _emit_json(payload)
        return

    if not isinstance(issues, list):
        error_console.print("[red]Unexpected Jira response: issues is not a list[/red]")
        raise typer.Exit(1)
    # Only render a Sprint column when the sprint field was actually part of what was requested
    # -- see `issue`'s identical rule.
    render_sprint_field = (
        sprint_field if requested_fields and sprint_field in requested_fields else None
    )
    _print_issues_table(issues, sprint_field=render_sprint_field)
    if is_last is True:
        console.print(f"returned {returned} issues; complete")
    elif is_last is False:
        console.print(f"returned {returned} issues; more remain")
    else:
        console.print(f"returned {returned} issues")
    if next_token:
        console.print(f"nextPageToken: {next_token}")
    if count:
        if approximate_total is not None:
            console.print(
                f"approximate match count: ~{approximate_total} "
                "(Jira's approximate count, not exact)"
            )
        else:
            console.print(
                "approximate match count: unavailable (Jira's response did not include a count)"
            )


def _cleanup_uploaded_attachments(
    client: JiraClient, issue_key: str, attachment_ids: list[str]
) -> None:
    """Best-effort delete of attachments this command uploaded when the write that would have
    referenced them never happened — a refused concurrency guard or any other failure between
    the upload and the write. Mirrors `mcp_server._cleanup_uploaded_attachments`. Deliberately
    swallows its own `JiraApiError` so a failed cleanup attempt can never mask the original error
    that is about to be reported and cause the exit.
    """
    for attachment_id in attachment_ids:
        try:
            client.delete_attachment(attachment_id, issue_key=issue_key)
        except JiraApiError:
            pass


def _exit_api_error(exc: JiraApiError, *, json_mode: bool = False) -> None:
    """Report a Jira API failure and exit 1, on both output channels when `json_mode` is set.

    Prose always goes to stderr, exactly as before `json_mode` existed. When the caller passed
    `--json`, stdout also gets exactly one machine-readable document instead of staying empty:
    an empty stdout on a non-zero exit is indistinguishable from a hang to a `--json` caller
    piping into `jq` or `json.load`, and was read as such this week. `json_mode` defaults to
    False so a call site that genuinely has no `--json` concept (a command with no such option)
    is unaffected.
    """
    _print_error("Jira API error:", exc)
    if exc.details:
        error_console.print(json.dumps(exc.details, indent=2, ensure_ascii=False))
    if json_mode:
        _emit_json(
            {
                "ok": False,
                "error_type": "jira_api_error",
                "status_code": exc.status_code,
                "message": str(exc),
            }
        )
    raise typer.Exit(1) from exc


def _exit_safety_error(exc: JiraWriteSafetyError, *, json_mode: bool = False) -> None:
    """Report a write-safety refusal and exit 3, on both output channels when `json_mode` is set.

    See `_exit_api_error` for why an empty stdout on a `--json` invocation is itself a defect.
    """
    _print_error("Write safety error:", exc)
    if json_mode:
        _emit_json(
            {
                "ok": False,
                "error_type": "write_safety_error",
                "message": str(exc),
            }
        )
    raise typer.Exit(3) from exc


# Distinct from _exit_api_error's 1 and _exit_safety_error's 3, so a script can tell a lost-update
# refusal apart from a transport failure or a write-scope refusal.
CONCURRENT_EDIT_EXIT_CODE = 4


def _exit_concurrent_edit_error(exc: JiraConcurrentEditError, *, json_mode: bool) -> None:
    """Refuse a whole-field rewrite that lost the check-then-act race, on both output channels.

    This is reached only when the guard itself found a mismatch immediately before the write, so
    the write never happened. The caller must re-read the issue and retry with the fresh
    `updated` value; retrying blindly with the same expectation will refuse again.
    """
    _print_error(
        "Concurrent edit refused:",
        f"{exc.issue_key} changed since it was read (expected updated={exc.expected_updated!r}, "
        f"found updated={exc.found_updated!r}). Re-read the issue and retry.",
    )
    if json_mode:
        _emit_json(
            {
                "ok": False,
                "error_type": "concurrent_edit",
                "issue": exc.issue_key,
                "expected_updated": exc.expected_updated,
                "found_updated": exc.found_updated,
                "dry_run": False,
            }
        )
    raise typer.Exit(CONCURRENT_EDIT_EXIT_CODE) from exc


# Distinct from every exit code above: these are specific to the sprint commands' own
# read-then-decide checks, not a Jira API failure, a write-scope refusal, or a lost-update race.
# 5 fires before any write is attempted (sprint-remove's precondition: a key not in SPRINT_ID at
# all); 6 fires after a write that Jira accepted but a post-write reread did not confirm, OR when
# that reread itself failed and the write's outcome is therefore unknown rather than failed; 7
# fires before any write, like 5, but for a different reason -- a key IS in SPRINT_ID but ALSO in
# another non-closed sprint the write would silently clear too. A script can tell "refused, wrong
# sprint" (5) apart from "refused, would also hit an unnamed sprint" (7) apart from "wrote, but
# did not verify or could not check" (6) without parsing prose.
SPRINT_PRECONDITION_EXIT_CODE = 5
SPRINT_VERIFICATION_EXIT_CODE = 6
SPRINT_MULTI_REMOVAL_EXIT_CODE = 7


def _sprint_locations(sprints: list[dict[str, Any]]) -> str:
    """Human-readable "where a key currently is", for sprint precondition and verification
    reporting: every sprint id and name it is in, or "no sprint" when the list is empty.

    Filters non-dict entries the same way `_sprint_cell` and `client.is_in_sprint` already do:
    a legacy greenhopper sprint field can carry a plain string entry instead of an object, and
    without this guard `sprint.get(...)` on that entry raises `AttributeError` instead of just
    being skipped like everywhere else that reads this list.
    """
    sprints = [sprint for sprint in sprints if isinstance(sprint, dict)]
    if not sprints:
        return "no sprint"
    return ", ".join(f"{sprint.get('id')} {sprint.get('name')!r}" for sprint in sprints)


def _exit_sprint_precondition_error(exc: SprintPreconditionError, *, json_mode: bool) -> None:
    """Refuse the whole `sprint-remove` batch before any write, when a key is not currently in
    the named sprint. Lists every key's actual current sprint(s), not only which keys mismatched,
    because the point of this refusal is to make the caller's wrong assumption visible rather
    than just recorded.
    """
    _print_error("Refusing sprint-remove:", str(exc))
    for entry in exc.report:
        marker = (
            "in target sprint"
            if entry["in_sprint"]
            else f"NOT in target sprint (in {_sprint_locations(entry['sprints'])})"
        )
        _say(f"- {entry['key']}: {marker}", json_mode=json_mode)
    if json_mode:
        _emit_json(
            {
                "ok": False,
                "error_type": "sprint_precondition_failed",
                "sprint_id": exc.sprint_id,
                "report": exc.report,
                "dry_run": None,
            }
        )
    raise typer.Exit(SPRINT_PRECONDITION_EXIT_CODE) from exc


def _exit_sprint_multi_removal_error(exc: SprintMultiRemovalError, *, json_mode: bool) -> None:
    """Refuse the whole `sprint-remove` batch before any write, when a key is currently in
    `SPRINT_ID` (the precondition above already passed) but ALSO in another non-closed sprint.

    `POST /rest/agile/1.0/backlog/issue` removes every FUTURE and ACTIVE sprint an issue is in,
    not only the one this command was told to check, so writing without this refusal would take
    that key out of a sprint nobody named -- and report success, since the old code only checked
    the named sprint both before and after the write. Names every additional sprint each at-risk
    key would leave, not only that one exists, and points at the opt-in flag rather than just
    refusing silently.
    """
    _print_error("Refusing sprint-remove:", str(exc))
    for entry in exc.report:
        if entry["other_sprints"]:
            _say(
                f"- {entry['key']}: would ALSO leave "
                f"{_sprint_locations(entry['other_sprints'])}",
                json_mode=json_mode,
            )
    if json_mode:
        _emit_json(
            {
                "ok": False,
                "error_type": "sprint_multi_removal_refused",
                "sprint_id": exc.sprint_id,
                "report": exc.report,
                "dry_run": None,
            }
        )
    raise typer.Exit(SPRINT_MULTI_REMOVAL_EXIT_CODE) from exc


def _exit_missing_concurrency_token_error(
    exc: JiraMissingConcurrencyTokenError, *, json_mode: bool
) -> None:
    """Refuse a guarded whole-field rewrite when Jira returned no token to guard it with.

    Reached only when the guard is on (the default) and the same read that fetched the content
    did not carry a usable `updated` timestamp. Writing anyway would silently perform the
    unguarded write the caller did not ask for and report it as a guarded success — the exact
    bug this refusal exists to prevent. Shares `CONCURRENT_EDIT_EXIT_CODE` with
    `_exit_concurrent_edit_error`: both are the same family of "refused to write without the
    requested guard", distinguished by `error_type` rather than by exit code.
    """
    _print_error(
        "Concurrency guard unavailable:",
        f"{exc.issue_key}: Jira returned no usable 'updated' timestamp to guard the write with. "
        "Re-read the issue and retry, or pass the command's guard opt-out flag to proceed "
        "without the guard.",
    )
    if json_mode:
        _emit_json(
            {
                "ok": False,
                "error_type": "concurrent_edit_guard_unavailable",
                "issue": exc.issue_key,
                "dry_run": False,
            }
        )
    raise typer.Exit(CONCURRENT_EDIT_EXIT_CODE) from exc


if __name__ == "__main__":
    app()
