from __future__ import annotations

import json

from jira_agent.audit import audit_write
from jira_agent.config import JiraConfig


def test_audit_write_appends_jsonl_record(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_AUDIT_LOG_PATH": str(audit_path),
        }
    )

    audit_write(
        config,
        action="edit_issue",
        issue_key="FC-107",
        payload={"fields": {"summary": "Updated"}},
        result={},
    )

    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["action"] == "edit_issue"
    assert records[0]["issue_key"] == "FC-107"
    assert records[0]["payload"] == {"fields": {"summary": "Updated"}}
    assert records[0]["timestamp"]
