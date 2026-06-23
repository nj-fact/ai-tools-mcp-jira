# Jira Agent

Small local Jira API tool for Codex, Claude Code, and a future MCP wrapper.

## Setup

Fill `.env`:

```env
JIRA_SITE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=your-api-token
JIRA_TEST_JQL=project = FC AND updated >= -365d ORDER BY updated DESC
JIRA_TEST_PROJECT_KEY=FC
JIRA_TEST_ISSUE_TYPE=Task
JIRA_TEST_ISSUE_KEY=FC-123
JIRA_WRITE_PROJECT_KEYS=FC
JIRA_WRITE_ISSUE_KEYS=
```

Install and run:

```bash
uv sync
uv run jira-agent myself
uv run jira-agent mcp-config
uv run jira-agent test-scope --limit 5
uv run jira-agent mine --project FC --limit 5
uv run jira-agent fc-open --limit 5
uv run jira-agent by-status Open --project FC --limit 5
uv run jira-agent create-meta --project FC
uv run jira-agent create-test-issue "jira-agent smoke test" --description "Dry-run by default."
uv run jira-agent test-issue
uv run jira-agent agent-report
uv run jira-agent agent-report FC-107 --json
uv run jira-agent agent-plan-comment --next-step "Run implementation checks."
uv run jira-agent edit-issue --labels jira-agent,test
uv run jira-agent smoke-write
uv run jira-agent assign
uv run jira-agent assign --execute
uv run jira-agent recent --limit 5
uv run jira-agent search "updated >= -30d ORDER BY updated DESC" --limit 5
uv run jira-agent issue ABC-123
uv run jira-agent comments ABC-123 --limit 5
uv run jira-agent transitions ABC-123
uv run jira-agent add-comment ABC-123 "Investigated by local agent." --execute
uv run jira-agent transition ABC-123 "In Progress" --execute
uv run python scripts/smoke_mcp.py
```

## Current Scope

- Read current Jira user with `myself`
- Fetch one issue with `issue`
- Search issues with JQL using Jira Cloud REST API v3 enhanced search
- Fetch recently updated issues with `recent`
- Use friendly searches with `mine`, `assigned-to-me`, `fc-open`, and `by-status`
- Run read-only smoke checks inside the configured FC test scope with `test-scope`
- Inspect create metadata and create dry-run/real test issues in the configured FC project
- Fetch and assign the configured test issue
- Build compact agent-friendly issue reports with `agent-report`
- Prepare standardized agent comments with `agent-plan-comment`
- Edit summary, description, and labels with `edit-issue`
- Run bundled write smoke checks against the configured test issue
- List and add comments
- List and execute issue transitions

Write actions default to dry-run. Pass `--execute` to actually modify Jira.
Write actions are blocked unless the issue belongs to `JIRA_WRITE_PROJECT_KEYS`
or the issue key is listed in `JIRA_WRITE_ISSUE_KEYS`.

## MCP Server

The same core is exposed as an MCP server over stdio:

```bash
uv run jira-agent-mcp
```

Example MCP server command for local clients:

```json
{
  "command": "uv",
  "args": ["run", "jira-agent-mcp"],
  "cwd": "/Users/norman/Fact/jira"
}
```

Available MCP tools:

- `jira_myself`
- `jira_get_issue`
- `jira_search_issues`
- `jira_recent_issues`
- `jira_mine`
- `jira_fc_open`
- `jira_by_status`
- `jira_test_scope_issues`
- `jira_create_test_issue`
- `jira_get_test_issue`
- `jira_assign_issue`
- `jira_agent_report`
- `jira_edit_issue`
- `jira_agent_plan_comment`
- `jira_list_comments`
- `jira_add_comment`
- `jira_list_transitions`
- `jira_transition_issue`

Run an end-to-end MCP smoke test:

```bash
uv run python scripts/smoke_mcp.py
uv run python scripts/smoke_mcp.py --issue FC-107
```

The smoke test starts `jira-agent-mcp` over stdio, verifies the expected tool list, and calls `jira_agent_report`.
