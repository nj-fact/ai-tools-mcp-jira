from __future__ import annotations

from typing import Any

from jira_agent.client import JiraClient

AGENT_REPORT_FIELDS = [
    "summary",
    "description",
    "status",
    "assignee",
    "reporter",
    "issuetype",
    "project",
    "labels",
    "updated",
    "created",
]


def adf_to_text(node: Any) -> str:
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return str(node.get("text") or "")
    if node.get("type") == "hardBreak":
        return "\n"

    child_text = [adf_to_text(child) for child in node.get("content") or []]
    parts = [part for part in child_text if part]
    if node.get("type") == "doc":
        return "\n\n".join(parts)
    if node.get("type") == "paragraph":
        return "".join(parts)

    separator = "\n" if node.get("type") in {"bulletList", "orderedList", "listItem"} else ""
    return separator.join(parts)


def issue_url(site_url: str, key: str) -> str:
    return f"{site_url.rstrip('/')}/browse/{key}"


def build_agent_report(
    *,
    site_url: str,
    issue: dict[str, Any],
    comments: dict[str, Any],
    transitions: dict[str, Any],
    write_scope: str | None = None,
) -> dict[str, Any]:
    fields = issue.get("fields") or {}
    status = fields.get("status") or {}
    assignee = fields.get("assignee") or {}
    reporter = fields.get("reporter") or {}
    issue_type = fields.get("issuetype") or {}
    project = fields.get("project") or {}
    key = str(issue.get("key") or "")

    return {
        "key": key,
        "url": issue_url(site_url, key),
        "summary": fields.get("summary"),
        "description": adf_to_text(fields.get("description")).strip(),
        "status": {
            "name": status.get("name"),
            "category": (status.get("statusCategory") or {}).get("name"),
        },
        "assignee": _user_summary(assignee),
        "reporter": _user_summary(reporter),
        "issue_type": issue_type.get("name"),
        "project": {
            "key": project.get("key"),
            "name": project.get("name"),
        },
        "labels": fields.get("labels") or [],
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "comments": [_comment_summary(comment) for comment in comments.get("comments") or []],
        "comment_count": comments.get("total"),
        "transitions": [
            _transition_summary(transition) for transition in transitions.get("transitions") or []
        ],
        "write_scope": write_scope,
    }


def fetch_agent_report(
    client: JiraClient,
    *,
    issue_key: str,
    site_url: str,
    comment_limit: int = 5,
    write_scope: str | None = None,
) -> dict[str, Any]:
    issue = client.get_issue(issue_key, fields=AGENT_REPORT_FIELDS)
    comments = client.list_comments(issue_key, limit=comment_limit)
    transitions = client.get_transitions(issue_key)
    return build_agent_report(
        site_url=site_url,
        issue=issue,
        comments=comments,
        transitions=transitions,
        write_scope=write_scope,
    )


def _user_summary(user: dict[str, Any]) -> dict[str, Any] | None:
    if not user:
        return None
    return {
        "accountId": user.get("accountId"),
        "displayName": user.get("displayName"),
        "emailAddress": user.get("emailAddress"),
        "active": user.get("active"),
    }


def _comment_summary(comment: dict[str, Any]) -> dict[str, Any]:
    author = comment.get("author") or {}
    return {
        "id": comment.get("id"),
        "author": _user_summary(author),
        "created": comment.get("created"),
        "updated": comment.get("updated"),
        "body": adf_to_text(comment.get("body")).strip(),
    }


def _transition_summary(transition: dict[str, Any]) -> dict[str, Any]:
    target = transition.get("to") or {}
    return {
        "id": transition.get("id"),
        "name": transition.get("name"),
        "to": {
            "id": target.get("id"),
            "name": target.get("name"),
        },
    }
