from __future__ import annotations

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
    assert content[2] == {
        "type": "heading",
        "attrs": {"level": 2},
        "content": [{"type": "text", "text": "Brief"}],
    }
    assert content[3]["type"] == "paragraph"
    assert content[3]["content"][0] == {
        "type": "text",
        "text": "Risk:",
        "marks": [{"type": "strong"}],
    }
    assert content[3]["content"][2] == {
        "type": "text",
        "text": "UNBEKANNT",
        "marks": [{"type": "code"}],
    }
    assert content[5]["type"] == "taskList"
    assert len(content[5]["content"]) == 2
    assert content[5]["content"][0]["type"] == "taskItem"
    assert content[5]["content"][0]["attrs"]["state"] == "TODO"
    assert content[5]["content"][0]["content"][0] == {
        "type": "text",
        "text": "AC-1",
        "marks": [{"type": "strong"}],
    }
    assert content[7] == {
        "type": "codeBlock",
        "content": [{"type": "text", "text": "flowchart LR\n  A --> B"}],
        "attrs": {"language": "mermaid"},
    }
    assert content[9]["type"] == "table"
    assert content[9]["content"][0]["content"][0]["type"] == "tableHeader"
    assert content[9]["content"][1]["content"][0]["type"] == "tableCell"


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
    ordered = content[2]
    assert ordered["type"] == "orderedList"
    assert ordered["content"][0]["content"][0]["content"][0]["text"] == (
        "First contract rule that continues on another line."
    )
    linked_rule = ordered["content"][1]["content"][0]["content"][0]
    assert linked_rule["text"] == "Linked rule"
    assert linked_rule["marks"] == [
        {"type": "link", "attrs": {"href": "https://actico.atlassian.net/browse/FC-1"}}
    ]
    tasks = content[4]
    assert tasks["type"] == "taskList"
    assert len(tasks["content"]) == 2
    assert tasks["content"][0]["attrs"]["state"] == "TODO"
    assert tasks["content"][0]["content"][-1]["text"].endswith(
        "The result remains atomic when validation fails."
    )
    assert tasks["content"][1]["attrs"]["state"] == "DONE"
    table_link = content[6]["content"][1]["content"][0]["content"][0]["content"][0]
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

    assert replaced["content"][2]["type"] == "paragraph"
    assert replaced["content"][2]["content"][0]["text"] == "Rendered architecture diagram:"
    assert replaced["content"][3] == {
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
