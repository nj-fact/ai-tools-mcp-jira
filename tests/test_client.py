from __future__ import annotations

import base64
import json

import httpx

from jira_agent.client import (
    JiraApiError,
    JiraClient,
    create_issue_payload,
    edit_issue_payload,
    issue_link_payload,
    png_dimensions,
    replace_diagram_with_media,
    text_to_adf,
)
from jira_agent.config import JiraConfig
from jira_agent.image_gallery import (
    image_gallery_blocks,
    newest_attachments_by_filename,
    upsert_top_level_section,
)


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
        '{"type":"paragraph","content":[{"type":"text","text":"Hello world"}]}'
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


def test_create_issue_payload_adds_filing_fields() -> None:
    payload = create_issue_payload(
        project_key="FC",
        issue_type_name="Task",
        summary="Foundation",
        labels=["FA101", "MVP", "risk-high"],
        parent_key="FC-900",
        priority="High",
    )

    assert payload["fields"]["labels"] == ["FA101", "MVP", "risk-high"]
    assert payload["fields"]["parent"] == {"key": "FC-900"}
    assert payload["fields"]["priority"] == {"name": "High"}


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


def test_search_assignable_users_uses_issue_or_project_scope() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=[{"accountId": "a1"}])

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.search_assignable_users("Norman", issue_key="FC-107", limit=5) == [
        {"accountId": "a1"}
    ]
    assert seen["url"] == (
        "https://example.atlassian.net/rest/api/3/user/assignable/search?"
        "query=Norman&maxResults=5&issueKey=FC-107"
    )


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


def test_edit_issue_adf_description_posts_raw_adf() -> None:
    seen: dict[str, object] = {}
    adf = {"type": "doc", "version": 1, "content": []}

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

    assert client.edit_issue_adf_description("FC-107", adf) == {}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-107"
    assert seen["json"] == '{"fields":{"description":{"type":"doc","version":1,"content":[]}}}'


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


def test_edit_issue_payload_supports_parent_and_priority() -> None:
    assert edit_issue_payload(parent_key="FC-900", priority="High") == {
        "fields": {
            "parent": {"key": "FC-900"},
            "priority": {"name": "High"},
        }
    }


def test_create_issue_payload_sets_one_fix_version() -> None:
    payload = create_issue_payload(
        project_key="FC",
        issue_type_name="Task",
        summary="Ship it",
        fix_versions=["FCL-MVP"],
    )

    assert payload["fields"]["fixVersions"] == [{"name": "FCL-MVP"}]


def test_edit_issue_payload_sets_several_fix_versions() -> None:
    payload = edit_issue_payload(fix_versions=["FCL-MVP", "FCL-1.1"])

    assert payload["fields"]["fixVersions"] == [
        {"name": "FCL-MVP"},
        {"name": "FCL-1.1"},
    ]


def test_edit_issue_payload_omits_fix_versions_key_by_default() -> None:
    payload = edit_issue_payload(summary="Updated")

    assert "fixVersions" not in payload["fields"]


def test_edit_issue_payload_clears_fix_versions_with_an_empty_list() -> None:
    payload = edit_issue_payload(fix_versions=[])

    assert payload["fields"]["fixVersions"] == []


def test_edit_issue_sends_fix_versions_in_the_jira_shape() -> None:
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

    client.edit_issue("FC-107", fix_versions=["FCL-MVP"])

    assert seen["json"] == '{"fields":{"fixVersions":[{"name":"FCL-MVP"}]}}'


def test_create_issue_payload_writes_story_points_to_the_configured_field() -> None:
    payload = create_issue_payload(
        project_key="FC",
        issue_type_name="Story",
        summary="Estimated work",
        story_points=8,
    )

    assert payload["fields"]["customfield_10028"] == 8


def test_story_points_use_the_site_specific_field_id() -> None:
    payload = edit_issue_payload(story_points=5, story_points_field="customfield_10016")

    assert payload["fields"] == {"customfield_10016": 5}


def test_whole_story_points_stay_integral_and_halves_survive() -> None:
    assert edit_issue_payload(story_points=13.0)["fields"]["customfield_10028"] == 13
    assert edit_issue_payload(story_points=0.5)["fields"]["customfield_10028"] == 0.5


def test_zero_story_points_are_written_rather_than_dropped() -> None:
    assert edit_issue_payload(story_points=0)["fields"] == {"customfield_10028": 0}


def test_negative_story_points_are_rejected() -> None:
    try:
        edit_issue_payload(story_points=-1)
    except ValueError as exc:
        assert str(exc) == "Story points must not be negative"
    else:
        raise AssertionError("Expected ValueError")


def test_a_malformed_story_points_field_id_is_rejected_at_config_time() -> None:
    try:
        JiraConfig.model_validate(
            {
                "JIRA_SITE_URL": "https://example.atlassian.net",
                "JIRA_EMAIL": "user@example.com",
                "JIRA_API_TOKEN": "token",
                "JIRA_STORY_POINTS_FIELD": "Story Points",
            }
        )
    except ValueError as exc:
        assert "customfield_10028" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_edit_issue_sends_story_points_using_the_client_config() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = request.read().decode()
        return httpx.Response(200, json={})

    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_STORY_POINTS_FIELD": "customfield_11111",
        }
    )
    client = JiraClient(
        config,
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.edit_issue("FC-308", story_points=21)

    assert seen["json"] == '{"fields":{"customfield_11111":21}}'


def test_issue_link_payload_supports_optional_comment() -> None:
    assert issue_link_payload(
        outward_issue_key="FC-107",
        inward_issue_key="FC-101",
        link_type="Relates",
        comment="Related work",
    ) == {
        "type": {"name": "Relates"},
        "outwardIssue": {"key": "FC-107"},
        "inwardIssue": {"key": "FC-101"},
        "comment": {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Related work"}],
                    }
                ],
            }
        },
    }


def test_link_issues_posts_issue_link_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(201)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.link_issues(outward_issue_key="FC-107", inward_issue_key="FC-101") == {}
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issueLink"
    assert seen["json"] == (
        '{"type":{"name":"Relates"},"outwardIssue":{"key":"FC-107"},"inwardIssue":{"key":"FC-101"}}'
    )


def test_get_changelog_uses_paginated_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"values": []})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.get_changelog("FC-107", start_at=2, limit=3) == {"values": []}
    assert seen["url"] == (
        "https://example.atlassian.net/rest/api/3/issue/FC-107/changelog?startAt=2&maxResults=3"
    )


def test_add_attachment_posts_multipart_without_json_content_type(tmp_path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("X-Atlassian-Token", "")
        seen["content_type"] = request.headers.get("Content-Type", "")
        return httpx.Response(200, json=[{"id": "1", "filename": "note.txt"}])

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.add_attachment("FC-107", file_path) == [{"id": "1", "filename": "note.txt"}]
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-107/attachments"
    assert seen["token"] == "no-check"
    assert seen["content_type"].startswith("multipart/form-data; boundary=")


def test_delete_attachment_uses_attachment_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.delete_attachment("123") == {}
    assert seen == {
        "method": "DELETE",
        "url": "https://example.atlassian.net/rest/api/3/attachment/123",
    }


def test_attachment_media_id_reads_redirect_location() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.atlassian.net/rest/api/3/attachment/content/123"
        return httpx.Response(
            303,
            headers={"location": "https://api.media.atlassian.com/file/media-abc/binary"},
        )

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.attachment_media_id("123") == "media-abc"


def test_text_to_adf_omits_layout_only_blank_paragraphs() -> None:
    assert text_to_adf("Hello\n\nworld") == {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "world"}]},
        ],
    }


def test_text_to_adf_renders_human_brief_markdown_blocks() -> None:
    adf = text_to_adf(
        "# FC-121 Human Brief\n"
        "\n"
        "## Brief\n"
        "**Risk:** high — `UNBEKANNT` fail-**closed**.\n"
        "\n"
        "- [ ] **AC-1** — API guard runs before upgrade.\n"
        "- [ ] **AC-2** — UI filter hides impossible roles.\n"
        "\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  A --> B\n"
        "```\n"
        "\n"
        "| Field | Content |\n"
        "|-------|---------|\n"
        "| Operator question | Which partners are eligible? |\n"
    )

    content = adf["content"]
    assert content[0] == {
        "type": "heading",
        "attrs": {"level": 1},
        "content": [{"type": "text", "text": "FC-121 Human Brief"}],
    }
    assert content[1] == {
        "type": "heading",
        "attrs": {"level": 2},
        "content": [{"type": "text", "text": "Brief"}],
    }
    assert content[2]["type"] == "paragraph"
    assert content[2]["content"][0] == {
        "type": "text",
        "text": "Risk:",
        "marks": [{"type": "strong"}],
    }
    assert content[2]["content"][2] == {
        "type": "text",
        "text": "UNBEKANNT",
        "marks": [{"type": "code"}],
    }
    assert content[3]["type"] == "taskList"
    assert len(content[3]["content"]) == 2
    assert content[3]["content"][0]["type"] == "taskItem"
    assert content[3]["content"][0]["attrs"]["state"] == "TODO"
    assert content[3]["content"][0]["content"][0] == {
        "type": "text",
        "text": "AC-1",
        "marks": [{"type": "strong"}],
    }
    assert content[4] == {
        "type": "codeBlock",
        "content": [{"type": "text", "text": "flowchart LR\n  A --> B"}],
        "attrs": {"language": "mermaid"},
    }
    assert content[5]["type"] == "table"
    assert content[5]["content"][0]["content"][0]["type"] == "tableHeader"
    assert content[5]["content"][1]["content"][0]["type"] == "tableCell"


def test_text_to_adf_renders_filing_markdown_semantically() -> None:
    adf = text_to_adf(
        "A wrapped paragraph that continues\n"
        "on the next source line.\n\n"
        "1. First contract rule that continues\n"
        "   on another line.\n"
        "2. [Linked rule](https://actico.atlassian.net/browse/FC-1)\n\n"
        "- [ ] **AC-1 — Complete behavior.** The result remains\n"
        "  atomic when validation fails.\n"
        "- [x] Existing proof remains valid.\n\n"
        "| Issue | Summary |\n"
        "| --- | --- |\n"
        "| [Foundation](https://actico.atlassian.net/browse/FC-2) | Shared model |\n"
    )

    content = adf["content"]
    assert content[0] == {
        "type": "paragraph",
        "content": [
            {
                "type": "text",
                "text": "A wrapped paragraph that continues on the next source line.",
            }
        ],
    }
    ordered = content[1]
    assert ordered["type"] == "orderedList"
    assert ordered["content"][0]["content"][0]["content"][0]["text"] == (
        "First contract rule that continues on another line."
    )
    linked_rule = ordered["content"][1]["content"][0]["content"][0]
    assert linked_rule["text"] == "Linked rule"
    assert linked_rule["marks"] == [
        {"type": "link", "attrs": {"href": "https://actico.atlassian.net/browse/FC-1"}}
    ]
    tasks = content[2]
    assert tasks["type"] == "taskList"
    assert len(tasks["content"]) == 2
    assert tasks["content"][0]["attrs"]["state"] == "TODO"
    assert tasks["content"][0]["content"][-1]["text"].endswith(
        "The result remains atomic when validation fails."
    )
    assert tasks["content"][1]["attrs"]["state"] == "DONE"
    table_link = content[3]["content"][1]["content"][0]["content"][0]["content"][0]
    assert table_link["text"] == "Foundation"
    assert table_link["marks"][0]["type"] == "link"


def test_png_dimensions_reads_png_header(tmp_path) -> None:
    png = tmp_path / "diagram.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x03\x06\x00\x00\x00F\x08\x02\x00\x00\x00"
    )

    assert png_dimensions(png) == (774, 70)


def test_replace_diagram_with_media_replaces_mermaid_block() -> None:
    adf = text_to_adf("## Architecture\n\n```mermaid\nflowchart LR\n  A --> B\n```")

    replaced = replace_diagram_with_media(
        adf,
        media_id="media-1",
        alt="Architecture diagram",
        width=774,
        height=70,
    )

    assert replaced["content"][1] == {
        "type": "mediaSingle",
        "attrs": {"layout": "center"},
        "content": [
            {
                "type": "media",
                "attrs": {
                    "id": "media-1",
                    "type": "file",
                    "collection": "",
                    "alt": "Architecture diagram",
                    "width": 774,
                    "height": 70,
                },
            }
        ],
    }


def test_replace_diagram_with_media_handles_multiple_mermaid_blocks_in_order() -> None:
    adf = text_to_adf(
        "```mermaid\nflowchart LR\nA --> B\n```\n\n```mermaid\nsequenceDiagram\nA->>B: Next\n```"
    )

    first = replace_diagram_with_media(
        adf,
        media_id="media-1",
        alt="Flow",
        width=100,
        height=50,
    )
    second = replace_diagram_with_media(
        first,
        media_id="media-2",
        alt="Sequence",
        width=120,
        height=60,
    )

    media = [block for block in second["content"] if block["type"] == "mediaSingle"]
    assert [block["content"][0]["attrs"]["id"] for block in media] == ["media-1", "media-2"]
    assert not any(block["type"] == "codeBlock" for block in second["content"])


def test_image_gallery_blocks_build_linked_tables() -> None:
    manifest = {
        "title": "UI/UX Reference",
        "intro": "Review reference.",
        "sections": [
            {
                "heading": "Core flow",
                "rows": [
                    {
                        "file": "01-list.png",
                        "title": "Start",
                        "description": "Open the list.",
                        "alt": "List",
                    }
                ],
            }
        ],
    }

    blocks = image_gallery_blocks(
        manifest,
        attachment_urls={"01-list.png": "https://example.atlassian.net/attachment/1"},
        documentation_url="https://example.atlassian.net/attachment/2",
    )

    assert blocks[0]["content"][0]["text"] == "UI/UX Reference"
    assert blocks[-1]["type"] == "table"
    screenshot = blocks[-1]["content"][1]["content"][2]["content"][0]["content"][0]
    assert screenshot["marks"][0]["attrs"]["href"].endswith("/attachment/1")


def test_newest_attachments_by_filename_prefers_highest_numeric_id() -> None:
    selected = newest_attachments_by_filename(
        [
            {"id": "286187", "filename": "03-decision.png"},
            {"id": "286240", "filename": "03-decision.png"},
            {"id": "286188", "filename": "04-contract.png"},
        ]
    )

    assert selected["03-decision.png"]["id"] == "286240"
    assert selected["04-contract.png"]["id"] == "286188"


def test_upsert_top_level_section_inserts_before_named_heading() -> None:
    adf = text_to_adf("### Outcome\n\nExisting.\n\n### Child Issues\n\nChildren.")
    gallery = [
        {
            "type": "heading",
            "attrs": {"level": 3},
            "content": [{"type": "text", "text": "UI/UX Reference"}],
        }
    ]

    updated = upsert_top_level_section(
        adf,
        title="UI/UX Reference",
        blocks=gallery,
        before_heading="Child Issues",
    )

    headings = [
        block["content"][0]["text"] for block in updated["content"] if block["type"] == "heading"
    ]
    assert headings == ["Outcome", "UI/UX Reference", "Child Issues"]


def test_update_comment_puts_adf_body_to_the_comment_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"id": "667040", "updated": "2026-08-21T09:30:00.000+0200"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.update_comment("FC-264", "667040", "Corrected text")

    assert result["id"] == "667040"
    assert seen["method"] == "PUT"
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-264/comment/667040"
    assert seen["json"] == {"body": text_to_adf("Corrected text")}


def test_get_comment_reads_one_comment_by_id() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"id": "667040"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.get_comment("FC-264", "667040")["id"] == "667040"
    assert seen["method"] == "GET"
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-264/comment/667040"


def make_override_config() -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_PROJECT__FINRP__EMAIL": "finrp@example.com",
            "JIRA_PROJECT__FINRP__API_TOKEN": "finrp-token",
        }
    )


def basic_auth(email: str, token: str) -> str:
    return "Basic " + base64.b64encode(f"{email}:{token}".encode()).decode()


def test_client_selects_credentials_per_issue_key() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["Authorization"]))
        return httpx.Response(200, json={"key": "stub"})

    with JiraClient(make_override_config(), transport=httpx.MockTransport(handler)) as client:
        client.get_issue("FINRP-123")
        client.get_issue("FC-1")
        client.myself()

    assert seen[0] == (
        "https://example.atlassian.net/rest/api/3/issue/FINRP-123",
        basic_auth("finrp@example.com", "finrp-token"),
    )
    assert seen[1][1] == basic_auth("user@example.com", "token")
    assert seen[2][1] == basic_auth("user@example.com", "token")


def test_client_uses_override_site_url_for_that_project_only() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["Authorization"]))
        return httpx.Response(200, json={"key": "stub"})

    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_PROJECT__FINRP__EMAIL": "finrp@example.com",
            "JIRA_PROJECT__FINRP__API_TOKEN": "finrp-token",
            "JIRA_PROJECT__FINRP__SITE_URL": "https://other.atlassian.net",
        }
    )
    with JiraClient(config, transport=httpx.MockTransport(handler)) as client:
        client.get_issue("FINRP-123")
        client.myself()

    assert seen[0][0] == "https://other.atlassian.net/rest/api/3/issue/FINRP-123"
    assert seen[1][0] == "https://example.atlassian.net/rest/api/3/myself"


def test_create_issue_uses_the_target_project_credentials() -> None:
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), request.headers["Authorization"]))
        return httpx.Response(201, json={"key": "FINRP-1"})

    with JiraClient(make_override_config(), transport=httpx.MockTransport(handler)) as client:
        client.create_issue(project_key="FINRP", issue_type_name="Task", summary="Summary")
        client.create_issue(project_key="FC", issue_type_name="Task", summary="Summary")

    assert seen[0] == (
        "POST",
        "https://example.atlassian.net/rest/api/3/issue",
        basic_auth("finrp@example.com", "finrp-token"),
    )
    assert seen[1][2] == basic_auth("user@example.com", "token")


def test_myself_can_target_a_project_credential_set() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Authorization"])
        return httpx.Response(200, json={"accountId": "1"})

    with JiraClient(make_override_config(), transport=httpx.MockTransport(handler)) as client:
        client.myself(key="FINRP-42")
        client.myself()

    assert seen[0] == basic_auth("finrp@example.com", "finrp-token")
    assert seen[1] == basic_auth("user@example.com", "token")


def test_injected_http_client_serves_the_default_credentials() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"key": "FC-1"})

    injected = httpx.Client(
        base_url="https://example.atlassian.net/rest/api/3",
        transport=httpx.MockTransport(handler),
    )
    with JiraClient(make_override_config(), http_client=injected) as client:
        client.get_issue("FC-1")

    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-1"
    assert not injected.is_closed
