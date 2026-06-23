from __future__ import annotations

import httpx

from jira_agent.client import JiraClient
from jira_agent.config import JiraConfig
from jira_agent.safety import (
    JiraWriteSafetyError,
    ensure_issue_write_allowed,
    ensure_project_write_allowed,
    write_scope_summary,
)


def make_config(**overrides: object) -> JiraConfig:
    values = {
        "JIRA_SITE_URL": "https://example.atlassian.net",
        "JIRA_EMAIL": "user@example.com",
        "JIRA_API_TOKEN": "token",
        "JIRA_WRITE_PROJECT_KEYS": "FC,ABC",
        "JIRA_WRITE_ISSUE_KEYS": "SAFE-1",
    }
    values.update(overrides)
    return JiraConfig.model_validate(values)


def test_write_project_allowlist_accepts_configured_project() -> None:
    ensure_project_write_allowed(make_config(), "FC")


def test_write_project_allowlist_blocks_unknown_project() -> None:
    try:
        ensure_project_write_allowed(make_config(), "NOPE")
    except JiraWriteSafetyError as exc:
        assert "NOPE" in str(exc)
    else:
        raise AssertionError("Expected JiraWriteSafetyError")


def test_issue_allowlist_skips_project_lookup() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Should not fetch issue for exact issue allowlist")

    config = make_config()
    client = JiraClient(
        config,
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert ensure_issue_write_allowed(config, client, "SAFE-1") == {
        "key": "SAFE-1",
        "allowed_by": "issue allowlist",
    }


def test_issue_project_allowlist_fetches_project() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": {"project": {"key": "FC"}}})

    config = make_config()
    client = JiraClient(
        config,
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert ensure_issue_write_allowed(config, client, "FC-107") == {
        "key": "FC-107",
        "project": "FC",
        "allowed_by": "project allowlist",
    }


def test_write_scope_summary_returns_lists() -> None:
    assert write_scope_summary(make_config()) == {
        "project_keys": ["FC", "ABC"],
        "issue_keys": ["SAFE-1"],
    }
