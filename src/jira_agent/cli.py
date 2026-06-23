from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from jira_agent.client import (
    JiraApiError,
    JiraClient,
    create_issue_payload,
    edit_issue_payload,
    select_transition,
)
from jira_agent.config import load_config
from jira_agent.reporting import adf_to_text, agent_plan_comment_body, fetch_agent_report
from jira_agent.safety import (
    JiraWriteSafetyError,
    ensure_issue_write_allowed,
    ensure_project_write_allowed,
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


def _mcp_config_payload() -> dict[str, Any]:
    return {
        "mcpServers": {
            "jira-agent": {
                "command": "uv",
                "args": ["run", "jira-agent-mcp"],
                "cwd": str(Path.cwd()),
            }
        }
    }


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
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    console.print(f"Created Jira test issue {data.get('key')} ({data.get('id')}).")


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
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    console.print(f"Added agent plan comment {data.get('id')} to {issue_key}.")


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
        typer.Option("--description", help="New issue description."),
    ] = None,
    labels: Annotated[
        str | None,
        typer.Option("--labels", help="Comma-separated labels to set."),
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
    """Edit summary, description, or labels. Defaults to dry-run."""
    config = _config()
    issue_key = key or config.test_issue_key
    if not issue_key:
        error_console.print(
            "[red]No issue key provided and JIRA_TEST_ISSUE_KEY is not configured.[/red]"
        )
        raise typer.Exit(2)

    try:
        payload = edit_issue_payload(
            summary=summary,
            description=description,
            labels=_csv_list(labels),
        )
    except ValueError as exc:
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
                description=description,
                labels=_csv_list(labels),
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
        except JiraWriteSafetyError as exc:
            _exit_safety_error(exc)
        except JiraApiError as exc:
            _exit_api_error(exc)

    if json_output:
        _print_json(data)
        return

    console.print(f"Assigned {issue_key} to {target_account_id or 'Unassigned'}.")


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
