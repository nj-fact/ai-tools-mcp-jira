from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client

EXPECTED_TOOLS = {
    "jira_myself",
    "jira_get_issue",
    "jira_search_issues",
    "jira_recent_issues",
    "jira_mine",
    "jira_fc_open",
    "jira_by_status",
    "jira_test_scope_issues",
    "jira_create_issue",
    "jira_create_test_issue",
    "jira_get_test_issue",
    "jira_assign_issue",
    "jira_agent_report",
    "jira_edit_issue",
    "jira_clear_issue_parent",
    "jira_patch_description_section",
    "jira_agent_plan_comment",
    "jira_write_human_brief",
    "jira_store_factory_contract",
    "jira_post_drift_reconcile_comment",
    "jira_update_drift_labels",
    "jira_user_search",
    "jira_assign_user",
    "jira_list_comments",
    "jira_add_comment",
    "jira_list_transitions",
    "jira_link_issue",
    "jira_delete_issue_link",
    "jira_attach_file",
    "jira_render_mermaid_diagrams",
    "jira_changelog",
    "jira_transition_issue",
    "jira_get_issue_sprints",
    "jira_add_issues_to_sprint",
    "jira_move_issues_to_backlog",
    "jira_config_fields",
    "jira_list_components",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the local Jira MCP server.")
    parser.add_argument(
        "--issue",
        default=None,
        help="Optional Jira issue key. Defaults to JIRA_TEST_ISSUE_KEY via the MCP server.",
    )
    parser.add_argument(
        "--comments",
        type=int,
        default=1,
        help="Number of comments to request in jira_agent_report.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "mcpServers": {
            "jira-agent": {
                "command": sys.executable,
                "args": ["-m", "jira_agent.mcp_server"],
                "cwd": str(repo_root),
            }
        }
    }

    async with Client(config) as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}
        missing_tools = EXPECTED_TOOLS - tool_names
        if missing_tools:
            raise SystemExit(f"Missing MCP tools: {', '.join(sorted(missing_tools))}")

        create_result = await client.call_tool(
            "jira_create_issue",
            {
                "project_key": "FC",
                "issue_type_name": "Task",
                "summary": "MCP formatting dry run",
                "description": "## Acceptance criteria\n\n- [ ] Native Jira checkbox",
                "labels": ["MVP", "risk-high"],
                "parent_key": "FC-123",
                "priority": "High",
            },
        )
        create_data = create_result.data
        if not isinstance(create_data, dict) or not create_data.get("dry_run"):
            raise SystemExit(f"jira_create_issue dry run failed: {create_data!r}")
        create_fields = (create_data.get("payload") or {}).get("fields") or {}
        description = create_fields.get("description") or {}
        if not any(block.get("type") == "taskList" for block in description.get("content") or []):
            raise SystemExit("jira_create_issue did not render a native ADF taskList")
        if create_fields.get("parent") != {"key": "FC-123"}:
            raise SystemExit("jira_create_issue omitted the parent field")

        arguments: dict[str, Any] = {"comments_limit": args.comments}
        if args.issue:
            arguments["key"] = args.issue

        result = await client.call_tool("jira_agent_report", arguments)
        data = result.data
        if not isinstance(data, dict) or not data.get("ok"):
            raise SystemExit(f"jira_agent_report failed: {data!r}")

        report = data.get("report") or {}
        key = report.get("key")
        url = report.get("url")
        status = (report.get("status") or {}).get("name")
        transition_count = len(report.get("transitions") or [])
        comment_count = len(report.get("comments") or [])

    print("MCP smoke test passed")
    print(f"tools: {len(tool_names)}")
    print("generic create dry run: passed")
    print(f"issue: {key}")
    print(f"url: {url}")
    print(f"status: {status}")
    print(f"comments returned: {comment_count}")
    print(f"transitions returned: {transition_count}")


if __name__ == "__main__":
    asyncio.run(main())
