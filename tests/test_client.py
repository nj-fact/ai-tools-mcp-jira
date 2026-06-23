from __future__ import annotations

import httpx

from jira_agent.client import (
    JiraApiError,
    JiraClient,
    create_issue_payload,
    edit_issue_payload,
    text_to_adf,
)
from jira_agent.config import JiraConfig


def make_config() -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
        }
    )


def test_myself_uses_rest_api_base_url() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"displayName": "Test User"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.myself()["displayName"] == "Test User"
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/myself"


def test_search_uses_enhanced_jql_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"issues": []})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.search_issues("project = ABC", limit=5, fields=["summary", "status"])

    assert seen["url"].startswith("https://example.atlassian.net/rest/api/3/search/jql?")
    assert "jql=project+%3D+ABC" in seen["url"]
    assert "maxResults=5" in seen["url"]
    assert "fields=summary" in seen["url"]
    assert "fields=status" in seen["url"]


def test_api_error_contains_status_and_details() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"errorMessages": ["Unauthorized"]})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    try:
        client.myself()
    except JiraApiError as exc:
        assert exc.status_code == 401
        assert exc.details == {"errorMessages": ["Unauthorized"]}
    else:
        raise AssertionError("Expected JiraApiError")


def test_list_comments_uses_paginated_comment_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"comments": []})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.list_comments("ABC-123", start_at=10, limit=5)

    assert seen["url"] == (
        "https://example.atlassian.net/rest/api/3/issue/ABC-123/comment?startAt=10&maxResults=5"
    )


def test_add_comment_posts_adf_body() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(201, json={"id": "10001"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.add_comment("ABC-123", "Hello\nworld")

    assert result == {"id": "10001"}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/ABC-123/comment"
    assert seen["json"] == (
        '{"body":{"type":"doc","version":1,"content":['
        '{"type":"paragraph","content":[{"type":"text","text":"Hello"}]},'
        '{"type":"paragraph","content":[{"type":"text","text":"world"}]}'
        "]}}"
    )


def test_get_transitions_uses_issue_transitions_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"transitions": []})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.get_transitions("ABC-123")

    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/ABC-123/transitions"


def test_transition_issue_posts_transition_id() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.transition_issue("ABC-123", "31") == {}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/ABC-123/transitions"
    assert seen["json"] == '{"transition":{"id":"31"}}'


def test_create_metadata_uses_project_and_issue_type_filters() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"projects": []})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.create_metadata(project_key="FC", issue_type_name="Task")

    assert seen["url"] == (
        "https://example.atlassian.net/rest/api/3/issue/createmeta?"
        "projectKeys=FC&expand=projects.issuetypes.fields&issuetypeNames=Task"
    )


def test_create_issue_posts_minimal_issue_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(201, json={"id": "10001", "key": "FC-999"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.create_issue(
        project_key="FC",
        issue_type_name="Task",
        summary="Smoke test",
    )

    assert result == {"id": "10001", "key": "FC-999"}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue"
    assert seen["json"] == (
        '{"fields":{"project":{"key":"FC"},"issuetype":{"name":"Task"},"summary":"Smoke test"}}'
    )


def test_create_issue_payload_adds_adf_description_when_present() -> None:
    assert create_issue_payload(
        project_key="FC",
        issue_type_name="Task",
        summary="Smoke test",
        description="Hello",
    ) == {
        "fields": {
            "project": {"key": "FC"},
            "issuetype": {"name": "Task"},
            "summary": "Smoke test",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            },
        }
    }


def test_assign_issue_puts_account_id() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.assign_issue("FC-107", "abc-123") == {}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-107/assignee"
    assert seen["json"] == '{"accountId":"abc-123"}'


def test_assign_issue_can_unassign_with_null_account_id() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = request.read().decode()
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.assign_issue("FC-107", None) == {}
    assert seen["json"] == '{"accountId":null}'


def test_edit_issue_posts_edit_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.edit_issue("FC-107", summary="Updated", labels=["jira-agent"]) == {}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-107"
    assert seen["json"] == '{"fields":{"summary":"Updated","labels":["jira-agent"]}}'


def test_edit_issue_payload_requires_at_least_one_field() -> None:
    try:
        edit_issue_payload()
    except ValueError as exc:
        assert str(exc) == "At least one editable field must be provided"
    else:
        raise AssertionError("Expected ValueError")


def test_edit_issue_payload_converts_description_to_adf() -> None:
    assert edit_issue_payload(description="Hello") == {
        "fields": {
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Hello"}],
                    }
                ],
            }
        }
    }


def test_text_to_adf_preserves_blank_lines() -> None:
    assert text_to_adf("Hello\n\nworld") == {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "paragraph"},
            {"type": "paragraph", "content": [{"type": "text", "text": "world"}]},
        ],
    }
