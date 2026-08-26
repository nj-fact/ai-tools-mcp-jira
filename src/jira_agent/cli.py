from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from jira_agent.audit import audit_write
from jira_agent.client import (
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
    drift_reconcile_comment_body,
    factory_contract_comment_body,
    read_text_source,
    update_drift_labels,
)

app = typer.Typer(
    help="Small local Jira API tool for agent workflows.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)

DEFAULT_FIELDS = ["summary", "status", "assignee", "issuetype", "project", "updated"]


def _client() -> JiraClient:
    try:
        config = load_config()
    except Exception as exc:
        error_console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    return JiraClient(config)


def _config():
    try:
        return load_config()
    except Exception as exc:
        error_console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(2) from exc


def _print_json(data: Any) -> None:
    console.print(json.dumps(data, indent=2, ensure_ascii=False))


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


def _issue_row(issue: dict[str, Any]) -> dict[str, str]:
    fields = issue.get("fields") or {}
    status = fields.get("status") or {}
    assignee = fields.get("assignee") or {}
    issue_type = fields.get("issuetype") or {}
    project = fields.get("project") or {}
    return {
        "key": str(issue.get("key") or ""),
        "summary": str(fields.get("summary") or ""),
        "status": str(status.get("name") or ""),
        "assignee": str(assignee.get("displayName") or "Unassigned"),
        "type": str(issue_type.get("name") or ""),
        "project": str(project.get("key") or ""),
        "updated": str(fields.get("updated") or ""),
    }


def _print_issues_table(issues: list[dict[str, Any]]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Assignee", no_wrap=True)
    table.add_column("Summary")
    for issue in issues:
        row = _issue_row(issue)
        table.add_row(row["key"], row["status"], row["assignee"], row["summary"])
    console.print(table)


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
    console.print(table)


def _adf_to_text(node: Any) -> str:
    return adf_to_text(node)


def _print_agent_report(report: dict[str, Any]) -> None:
    console.print(f"[bold]{report['key']}[/bold] {report.get('summary') or ''}")
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
        console.print(table)
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
    console.print(table)


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
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
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
            _exit_api_error(exc)

    if json_output:
        _print_json(users)
        return
    _print_users_table(users)


@app.command()
def issue(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    fields: Annotated[
        str | None,
        typer.Option("--fields", help="Comma-separated Jira fields to fetch."),
    ] = ",".join(DEFAULT_FIELDS),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Fetch one Jira issue."""
    with _client() as client:
        try:
            data = client.get_issue(key, fields=_field_list(fields))
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    _print_issues_table([data])


@app.command()
def comments(
    key: Annotated[str, typer.Argument(help="Jira issue key, for example ABC-123.")],
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum comments to return."),
    ] = 20,
    start_at: Annotated[
        int,
        typer.Option("--start-at", min=0, help="Pagination start offset."),
    ] = 0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """List comments for one Jira issue."""
    with _client() as client:
        try:
            data = client.list_comments(key, start_at=start_at, limit=limit)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    comments_data = data.get("comments") or []
    if not isinstance(comments_data, list):
        error_console.print("[red]Unexpected Jira response: comments is not a list[/red]")
        raise typer.Exit(1)

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
    console.print(table)


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
        console.print("[yellow]Dry run:[/yellow] would add this comment")
        console.print(f"issue: {key}")
        console.print(body)
        console.print("Pass --execute to write it to Jira.")
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
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    console.print(f"Added comment {data.get('id')} to {key}.")


@app.command()
def recent(
    days: Annotated[
        int,
        typer.Option("--days", min=1, max=365, help="Look back this many days."),
    ] = 30,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum issues to return."),
    ] = 20,
    fields: Annotated[
        str | None,
        typer.Option("--fields", help="Comma-separated Jira fields to fetch."),
    ] = ",".join(DEFAULT_FIELDS),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
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
        typer.Option("--limit", min=1, max=100, help="Maximum issues to return."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
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
        fields=",".join(DEFAULT_FIELDS),
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
        typer.Option("--limit", min=1, max=100, help="Maximum issues to return."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Alias for mine."""
    mine(project=project, limit=limit, json_output=json_output)


@app.command("fc-open")
def fc_open(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum issues to return."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Show open FC issues."""
    jql = "project = FC AND statusCategory != Done ORDER BY updated DESC"
    _run_search(
        jql,
        limit=limit,
        fields=",".join(DEFAULT_FIELDS),
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
        typer.Option("--limit", min=1, max=100, help="Maximum issues to return."),
    ] = 20,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
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
        fields=",".join(DEFAULT_FIELDS),
        next_page_token=None,
        json_output=json_output,
    )


@app.command()
def search(
    jql: Annotated[str, typer.Argument(help="JQL query.")],
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=100, help="Maximum issues to return."),
    ] = 20,
    fields: Annotated[
        str | None,
        typer.Option("--fields", help="Comma-separated Jira fields to fetch."),
    ] = ",".join(DEFAULT_FIELDS),
    next_page_token: Annotated[
        str | None,
        typer.Option("--next-page-token", help="Jira enhanced search nextPageToken."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Search Jira issues with JQL."""
    _run_search(
        jql,
        limit=limit,
        fields=fields,
        next_page_token=next_page_token,
        json_output=json_output,
    )


@app.command("test-scope")
def test_scope(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=20, help="Maximum test-scope issues to return."),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Run a read-only smoke query against the configured Jira test scope."""
    config = _config()
    console.print(f"test JQL: {config.test_jql}")
    with JiraClient(config) as client:
        try:
            data = client.search_issues(
                config.test_jql,
                limit=limit,
                fields=DEFAULT_FIELDS,
            )
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    issues = data.get("issues") or []
    if not isinstance(issues, list):
        error_console.print("[red]Unexpected Jira response: issues is not a list[/red]")
        raise typer.Exit(1)
    _print_issues_table(issues)
    console.print(f"scope issue count shown: {len(issues)}")


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
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
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
        console.print("[yellow]Dry run:[/yellow] would create this Jira test issue")
        console.print(f"project: {config.test_project_key}")
        console.print(f"issue type: {config.test_issue_type}")
        _print_json(payload)
        console.print("Pass --execute to create it in Jira.")
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
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
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
            help="Optional heading whose body is selected from the description source.",
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
        error_console.print(f"[red]Description error:[/red] {exc}")
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
        error_console.print(f"[red]Create error:[/red] {exc}")
        raise typer.Exit(2) from exc
    description_adf = payload["fields"].get("description")
    diagram_count = (
        len(mermaid_sources_from_adf(description_adf))
        if render_mermaid and isinstance(description_adf, dict)
        else 0
    )
    if not execute:
        console.print("[yellow]Dry run:[/yellow] would create this Jira issue")
        _print_json(payload)
        if diagram_count:
            console.print(f"Mermaid diagrams to render after creation: {diagram_count}")
        console.print("Pass --execute to create it in Jira.")
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
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        if mermaid_result is None:
            _print_json(data)
        else:
            _print_json({"issue": data, "mermaid": mermaid_result})
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
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Render, upload, and embed the diagrams in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the machine-readable result."),
    ] = False,
) -> None:
    """Render and embed all Mermaid blocks in an existing issue. Defaults to dry-run."""
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
                error_console.print(f"[yellow]No Mermaid code blocks found on {key}.[/yellow]")
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
                    _print_json(result)
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
            )
            audit_write(
                config,
                action="render_mermaid_diagrams",
                issue_key=key,
                payload={
                    "files": result["files"],
                    "attachment_ids": result["attachment_ids"],
                    "deleted_attachment_ids": result["deleted_attachment_ids"],
                },
                result=result["result"],
            )
    except JiraWriteSafetyError as exc:
        _exit_safety_error(exc)
    except JiraApiError as exc:
        _exit_api_error(exc)
    except (OSError, RuntimeError, ValueError) as exc:
        error_console.print(f"[red]Mermaid error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        _print_json(result)
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
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    _print_issues_table([data])


@app.command("agent-report")
def agent_report(
    key: Annotated[
        str | None,
        typer.Argument(help="Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY."),
    ] = None,
    comments_limit: Annotated[
        int,
        typer.Option("--comments", min=0, max=100, help="Maximum comments to include."),
    ] = 5,
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
                comment_limit=comments_limit,
                changelog_limit=changelog_limit,
                write_scope=config.test_issue_key,
            )
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(report)
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
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually update the Jira description."),
    ] = False,
) -> None:
    """Write a Human Brief into the Jira description. Defaults to dry-run."""
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
        error_console.print(f"[red]Human brief error:[/red] {exc}")
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

            data = client.edit_issue(issue_key, description=body)
            audit_write(
                config,
                action="human_brief",
                issue_key=issue_key,
                payload=payload,
                result=data,
            )
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
            error_console.print(f"[red]Factory Contract error:[/red] {exc}")
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
    """Set or remove drift-* labels. Defaults to dry-run."""
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
            issue = client.get_issue(issue_key, fields=["labels"])
            existing = (issue.get("fields") or {}).get("labels") or []
            labels = update_drift_labels(
                existing,
                add=_csv_list(add),
                remove=_csv_list(remove),
                clear=clear,
            )
            payload = edit_issue_payload(labels=labels)
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would update drift labels")
                console.print(f"issue: {issue_key}")
                console.print(f"safety: {safety}")
                _print_json({"before": existing, "after": labels, "payload": payload})
                console.print("Pass --execute to write it to Jira.")
                return
            result = client.edit_issue(issue_key, labels=labels)
            audit_write(
                config,
                action="drift_labels",
                issue_key=issue_key,
                payload={"before": existing, "after": labels},
                result=result,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)
        except ValueError as exc:
            error_console.print(f"[red]Drift label error:[/red] {exc}")
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
            help="Optional heading whose body is selected from the description source.",
        ),
    ] = None,
    labels: Annotated[
        str | None,
        typer.Option("--labels", help="Comma-separated labels to set."),
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
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually edit the issue in Jira."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the raw Jira JSON response."),
    ] = False,
) -> None:
    """Edit issue fields with Markdown-to-ADF rendering. Defaults to dry-run."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
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
            parent_key=parent,
            priority=priority,
            story_points=story_points,
            story_points_field=config.story_points_field,
        )
    except (OSError, ValueError) as exc:
        error_console.print(f"[red]Edit error:[/red] {exc}")
        raise typer.Exit(2) from exc

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
            if not execute:
                console.print("[yellow]Dry run:[/yellow] would edit issue")
                console.print(f"issue: {issue_key}")
                console.print(f"safety: {safety}")
                _print_json(payload)
                console.print("Pass --execute to write it to Jira.")
                return

            data = client.edit_issue(
                issue_key,
                summary=summary,
                description=body,
                labels=_csv_list(labels),
                parent_key=parent,
                priority=priority,
                story_points=story_points,
            )
            audit_write(
                config,
                action="edit_issue",
                issue_key=issue_key,
                payload=payload,
                result=data,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    console.print(f"Edited {issue_key}.")


@app.command("mcp-config")
def mcp_config() -> None:
    """Print the local MCP client configuration block."""
    _print_json(_mcp_config_payload())


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
                    error_console.print(f"[red]No transition found for:[/red] {transition_name}")
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
                console.print("[yellow]Dry run:[/yellow] would assign issue")
                console.print(f"issue: {issue_key}")
                console.print(f"target accountId: {target_account_id or 'Unassigned'}")
                if not unassigned and not account_id:
                    console.print(f"target user: {me.get('displayName')}")
                console.print("Pass --execute to write it to Jira.")
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
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
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
    outward_key: Annotated[str, typer.Argument(help="Outward issue key.")],
    inward_key: Annotated[str, typer.Argument(help="Inward issue key.")],
    link_type: Annotated[
        str,
        typer.Option("--type", help="Jira issue link type name."),
    ] = "Relates",
    comment: Annotated[
        str | None,
        typer.Option("--comment", help="Optional comment on the outward issue."),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually create the issue link in Jira."),
    ] = False,
) -> None:
    """Link two Jira issues. Defaults to dry-run."""
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
                console.print("[yellow]Dry run:[/yellow] would attach file")
                console.print(f"issue: {issue_key}")
                console.print(f"file: {file_path}")
                console.print(f"safety: {safety}")
                console.print("Pass --execute to upload it to Jira.")
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
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
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
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Upload, embed, and clean up in Jira."),
    ] = False,
) -> None:
    """Upload a PNG diagram and embed it in the Jira description. Defaults to dry-run."""
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
        error_console.print(f"[red]Diagram error:[/red] {exc}")
        raise typer.Exit(2) from exc

    delete_names = set(_csv_list(delete_filename) or [])
    delete_ids = set(_csv_list(delete_attachment_id) or [])

    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, issue_key)
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
            media_id = client.attachment_media_id(attachment_id, issue_key=issue_key)
            existing_description = issue.get("fields", {}).get("description") or {
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
            result = client.edit_issue_adf_description(issue_key, description)
            deleted: list[str] = []
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
                },
                result=result,
            )
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(
        f"Embedded {file_path.name} ({attachment_id}) into {issue_key}; "
        f"deleted {len(deleted)} old attachment(s)."
    )


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
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Actually update the Jira description."),
    ] = False,
) -> None:
    """Insert or replace a linked screenshot gallery using existing Jira attachments."""
    config = _config()
    try:
        manifest = load_image_gallery_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        error_console.print(f"[red]Gallery manifest error:[/red] {exc}")
        raise typer.Exit(2) from exc

    filenames = {str(row["file"]) for section in manifest["sections"] for row in section["rows"]}
    with JiraClient(config) as client:
        try:
            safety = ensure_issue_write_allowed(config, client, key)
            issue = client.get_issue(key, fields=["description", "attachment"])
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

            result = client.edit_issue_adf_description(key, description)
            audit_write(
                config,
                action="upsert_image_gallery",
                issue_key=key,
                payload={
                    "manifest": str(manifest_path),
                    "filenames": sorted(filenames),
                    "documentation_filename": documentation_filename,
                    "before_heading": before_heading,
                },
                result=result,
            )
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
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
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
        _print_json(records)
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
    console.print(table)


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
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
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
    console.print(table)


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
                error_console.print(f"[red]No transition found for:[/red] {transition_name_or_id}")
                raise typer.Exit(1)

            if not execute:
                target = selected.get("to") or {}
                console.print("[yellow]Dry run:[/yellow] would transition issue")
                console.print(f"issue: {key}")
                console.print(f"transition: {selected.get('name')} ({selected.get('id')})")
                console.print(f"target status: {target.get('name')}")
                console.print("Pass --execute to write it to Jira.")
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
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    console.print(f"Transitioned {key} with {selected.get('name')} ({selected.get('id')}).")


def _run_search(
    jql: str,
    *,
    limit: int,
    fields: str | None,
    next_page_token: str | None,
    json_output: bool,
) -> None:
    with _client() as client:
        try:
            data = client.search_issues(
                jql,
                limit=limit,
                fields=_field_list(fields),
                next_page_token=next_page_token,
            )
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    issues = data.get("issues") or []
    if not isinstance(issues, list):
        error_console.print("[red]Unexpected Jira response: issues is not a list[/red]")
        raise typer.Exit(1)
    _print_issues_table(issues)
    if data.get("nextPageToken"):
        console.print(f"nextPageToken: {data['nextPageToken']}")


def _exit_api_error(exc: JiraApiError) -> None:
    error_console.print(f"[red]Jira API error:[/red] {exc}")
    if exc.details:
        error_console.print(json.dumps(exc.details, indent=2, ensure_ascii=False))
    raise typer.Exit(1) from exc


def _exit_safety_error(exc: JiraWriteSafetyError) -> None:
    error_console.print(f"[red]Write safety error:[/red] {exc}")
    raise typer.Exit(3) from exc


if __name__ == "__main__":
    app()
