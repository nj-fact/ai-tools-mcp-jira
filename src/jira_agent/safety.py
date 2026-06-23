from __future__ import annotations

from typing import Any

from jira_agent.client import JiraClient
from jira_agent.config import JiraConfig


class JiraWriteSafetyError(RuntimeError):
    pass


def ensure_project_write_allowed(config: JiraConfig, project_key: str) -> None:
    if project_key in config.write_project_keys:
        return
    raise JiraWriteSafetyError(
        f"Project {project_key} is not in JIRA_WRITE_PROJECT_KEYS "
        f"({', '.join(config.write_project_keys) or 'none'})."
    )


def ensure_issue_write_allowed(
    config: JiraConfig,
    client: JiraClient,
    issue_key: str,
) -> dict[str, Any]:
    if issue_key in config.write_issue_keys:
        return {"key": issue_key, "allowed_by": "issue allowlist"}

    issue = client.get_issue(issue_key, fields=["project"])
    project = (issue.get("fields") or {}).get("project") or {}
    project_key = str(project.get("key") or "")
    ensure_project_write_allowed(config, project_key)
    return {
        "key": issue_key,
        "project": project_key,
        "allowed_by": "project allowlist",
    }


def write_scope_summary(config: JiraConfig) -> dict[str, list[str]]:
    return {
        "project_keys": config.write_project_keys,
        "issue_keys": config.write_issue_keys,
    }
