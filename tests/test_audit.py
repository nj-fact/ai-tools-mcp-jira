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
    assert records[0]["credential_source"] == "default"
    assert records[0]["payload"] == {"fields": {"summary": "Updated"}}
    assert records[0]["timestamp"]


def test_audit_write_records_credential_source_without_secrets(tmp_path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_AUDIT_LOG_PATH": str(audit_path),
            "JIRA_PROJECT__FINRP__EMAIL": "finrp@example.com",
            "JIRA_PROJECT__FINRP__API_TOKEN": "finrp-secret-token",
        }
    )

    audit_write(config, action="add_comment", issue_key="FINRP-3", payload={}, result={})
    audit_write(config, action="add_comment", issue_key="FC-1", payload={}, result={})

    raw = audit_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    assert records[0]["credential_source"] == "FINRP"
    assert records[1]["credential_source"] == "default"
    assert "finrp-secret-token" not in raw
