from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from jira_agent import client as client_module
from jira_agent.client import (
    AdfPatchInvariantError,
    AdfSectionPatchRefused,
    JiraApiError,
    JiraClient,
    JiraConcurrentEditError,
    SprintPreconditionError,
    adf_contains_media_id,
    adf_heading_text,
    create_issue_payload,
    edit_issue_payload,
    is_in_sprint,
    issue_link_payload,
    patch_adf_description_section,
    png_dimensions,
    replace_diagram_with_media,
    sprint_membership_report,
    sprint_verification_results,
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


def test_edit_issue_payload_omits_parent_unless_set_or_explicitly_cleared() -> None:
    assert "parent" not in edit_issue_payload(summary="Updated")["fields"]
    assert edit_issue_payload(clear_parent=True)["fields"] == {"parent": None}
    try:
        edit_issue_payload(parent_key="FC-900", clear_parent=True)
    except ValueError as exc:
        assert str(exc) == "Use either parent_key or clear_parent, not both"
    else:
        raise AssertionError("Expected ValueError")


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


def test_sprint_field_defaults_to_the_probed_id() -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
        }
    )

    assert config.sprint_field == "customfield_10020"


def test_sprint_field_honours_its_env_alias() -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_SPRINT_FIELD": "customfield_12345",
        }
    )

    assert config.sprint_field == "customfield_12345"


def test_a_malformed_sprint_field_id_is_rejected_at_config_time() -> None:
    try:
        JiraConfig.model_validate(
            {
                "JIRA_SITE_URL": "https://example.atlassian.net",
                "JIRA_EMAIL": "user@example.com",
                "JIRA_API_TOKEN": "token",
                "JIRA_SPRINT_FIELD": "Sprint",
            }
        )
    except ValueError as exc:
        assert "customfield_10020" in str(exc)
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
        "inwardIssue": {"key": "FC-107"},
        "outwardIssue": {"key": "FC-101"},
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
        '{"type":{"name":"Relates"},"inwardIssue":{"key":"FC-107"},"outwardIssue":{"key":"FC-101"}}'
    )


def test_issue_link_payload_maps_fc_350_blocks_fc_352_to_jira_rest_direction() -> None:
    assert issue_link_payload(
        outward_issue_key="FC-350",
        inward_issue_key="FC-352",
        link_type="Blocks",
    ) == {
        "type": {"name": "Blocks"},
        "inwardIssue": {"key": "FC-350"},
        "outwardIssue": {"key": "FC-352"},
    }


def test_delete_issue_link_uses_issue_link_endpoint() -> None:
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

    assert client.delete_issue_link("10001", issue_key="FC-107") == {}
    assert seen == {
        "method": "DELETE",
        "url": "https://example.atlassian.net/rest/api/3/issueLink/10001",
    }


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


def test_replace_diagram_with_media_can_target_one_existing_media_id() -> None:
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "mediaSingle",
                "attrs": {"layout": "center"},
                "content": [{"type": "media", "attrs": {"id": "old-media"}}],
            },
            {
                "type": "mediaSingle",
                "attrs": {"layout": "center"},
                "content": [{"type": "media", "attrs": {"id": "keep-media"}}],
            },
        ],
    }

    replaced = replace_diagram_with_media(
        adf,
        media_id="new-media",
        alt="Updated diagram",
        width=1600,
        height=900,
        replace_media_id="old-media",
    )

    assert adf_contains_media_id(replaced, "new-media")
    assert adf_contains_media_id(replaced, "keep-media")
    assert not adf_contains_media_id(replaced, "old-media")


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


def _heading(level: int, text: str) -> dict[str, Any]:
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": text}],
    }


def _paragraph(text: str) -> dict[str, Any]:
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def section_patch_document() -> dict[str, Any]:
    """A description shaped like the FC-385 issue a section patch must not flatten.

    Top-level indices: 0 h1, 1 paragraph, 2 h2 Observed, 3 paragraph, 4 h2 Scope, 5 h3 In scope,
    6 bulletList, 7 table, 8 h2 Acceptance criteria, 9 taskList, 10 h2 Test approach, 11 paragraph.
    """
    return {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(1, "Dry run prevented a description wipe"),
            _paragraph("Reported after an edit-issue dry run."),
            _heading(2, "Observed"),
            _paragraph("The payload carried one bullet list and nothing else."),
            _heading(2, "Scope"),
            _heading(3, "In scope"),
            {
                "type": "bulletList",
                "content": [
                    {"type": "listItem", "content": [_paragraph("Description writes")]},
                    {"type": "listItem", "content": [_paragraph("Section patches")]},
                ],
            },
            {
                "type": "table",
                "attrs": {"isNumberColumnEnabled": False, "layout": "default"},
                "content": [
                    {
                        "type": "tableRow",
                        "content": [
                            {
                                "type": "tableHeader",
                                "attrs": {},
                                "content": [_paragraph("Field")],
                            },
                            {
                                "type": "tableCell",
                                "attrs": {},
                                "content": [_paragraph("description")],
                            },
                        ],
                    }
                ],
            },
            _heading(2, "Acceptance criteria"),
            {
                "type": "taskList",
                "attrs": {"localId": "task-list-1"},
                "content": [
                    {
                        "type": "taskItem",
                        "attrs": {"localId": "task-1", "state": "DONE"},
                        "content": [{"type": "text", "text": "A dry run shows the node counts"}],
                    },
                    {
                        "type": "taskItem",
                        "attrs": {"localId": "task-2", "state": "TODO"},
                        "content": [{"type": "text", "text": "A patch never wipes the field"}],
                    },
                ],
            },
            _heading(2, "Test approach"),
            _paragraph("Unit tests over the ADF document."),
        ],
    }


def ambiguous_section_document() -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(2, "Scope"),
            _paragraph("First."),
            _heading(2, "Scope"),
            _paragraph("Second."),
        ],
    }


def test_patch_replaces_a_middle_section_and_leaves_every_other_node_deep_equal() -> None:
    original = section_patch_document()
    reference = section_patch_document()

    patch = patch_adf_description_section(
        original,
        section="Scope",
        content="## Scope\n\nOnly the description field.",
        expect_swallowed_headings=["In scope"],
    )

    assert (patch["heading"], patch["level"], patch["mode"]) == ("Scope", 2, "replace")
    assert (patch["start_index"], patch["end_index"]) == (4, 8)
    assert patch["replaced_node_count"] == 4
    assert patch["new_node_count"] == 2
    assert (patch["preserved_before"], patch["preserved_after"]) == (4, 4)
    assert (patch["original_node_count"], patch["patched_node_count"]) == (12, 10)

    content = patch["document"]["content"]
    assert content[:4] == reference["content"][:4]
    assert content[6:] == reference["content"][8:]
    assert content[4:6] == patch["new_nodes"]


def test_patch_accepts_a_heading_written_with_hashes_and_mixed_case() -> None:
    patch = patch_adf_description_section(
        section_patch_document(),
        section="##   scope  ",
        content="## Scope\n\nMatched anyway.",
        expect_swallowed_headings=["In scope"],
    )

    assert (patch["start_index"], patch["end_index"]) == (4, 8)


def test_patch_replaces_the_first_body_section_and_keeps_the_title() -> None:
    reference = section_patch_document()

    patch = patch_adf_description_section(
        section_patch_document(),
        section="Observed",
        content="## Observed\n\nThe payload dropped five acceptance criteria.",
    )

    content = patch["document"]["content"]
    assert (patch["start_index"], patch["end_index"]) == (2, 4)
    assert content[:2] == reference["content"][:2]
    assert content[4:] == reference["content"][4:]


def test_append_to_a_whole_document_span_needs_no_opt_in_because_it_removes_nothing() -> None:
    patch = patch_adf_description_section(
        section_patch_document(),
        section="Dry run prevented a description wipe",
        mode="append",
        content="- One more note",
    )

    assert patch["whole_document"] is True
    assert patch["patched_node_count"] == 13
    assert patch["document"]["content"][:12] == section_patch_document()["content"]


def test_patch_replaces_the_last_section() -> None:
    reference = section_patch_document()

    patch = patch_adf_description_section(
        section_patch_document(),
        section="Test approach",
        content="## Test approach\n\nUnit tests plus one regression test.",
    )

    content = patch["document"]["content"]
    assert (patch["start_index"], patch["end_index"]) == (10, 12)
    assert patch["preserved_after"] == 0
    assert content[:10] == reference["content"][:10]
    assert len(content) == 12


def test_patch_delete_removes_exactly_the_matched_slice() -> None:
    reference = section_patch_document()

    patch = patch_adf_description_section(
        section_patch_document(),
        section="Scope",
        mode="delete",
        expect_swallowed_headings=["In scope"],
    )

    content = patch["document"]["content"]
    assert patch["new_nodes"] == []
    assert (patch["replaced_node_count"], patch["patched_node_count"]) == (4, 8)
    assert content == reference["content"][:4] + reference["content"][8:]
    assert not [node for node in content if node.get("type") in {"table", "bulletList"}]


def test_patch_append_keeps_the_existing_section_body() -> None:
    reference = section_patch_document()

    patch = patch_adf_description_section(
        section_patch_document(),
        section="Acceptance criteria",
        mode="append",
        content="- [ ] The invariant check refuses a wipe",
    )

    content = patch["document"]["content"]
    assert (patch["start_index"], patch["end_index"]) == (8, 10)
    assert content[8] == reference["content"][8]
    assert content[9] == reference["content"][9]
    assert content[10]["type"] == "taskList"
    assert content[11:] == reference["content"][10:]


def test_patch_prepend_inserts_below_the_heading_and_keeps_the_body() -> None:
    reference = section_patch_document()

    patch = patch_adf_description_section(
        section_patch_document(),
        section="Acceptance criteria",
        mode="prepend",
        content="The criteria below are binding.",
    )

    content = patch["document"]["content"]
    assert content[8] == reference["content"][8]
    assert content[9]["type"] == "paragraph"
    assert content[10] == reference["content"][9]
    assert content[11:] == reference["content"][10:]


def test_patch_without_include_heading_preserves_the_original_heading_node() -> None:
    original = section_patch_document()
    heading_node = original["content"][2]

    patch = patch_adf_description_section(
        original,
        section="Observed",
        include_heading=False,
        content="The payload reduced the description to one bullet list.",
    )

    content = patch["document"]["content"]
    assert content[2] is heading_node
    assert patch["new_nodes"][0] is heading_node
    assert content[3]["type"] == "paragraph"


def test_patch_covers_a_subsection_and_stops_at_the_next_peer_heading() -> None:
    patch = patch_adf_description_section(
        section_patch_document(),
        section="Scope",
        content="## Scope\n\nRewritten.",
        expect_swallowed_headings=["In scope"],
    )

    replaced_types = [node["type"] for node in section_patch_document()["content"][4:8]]
    assert replaced_types == ["heading", "heading", "bulletList", "table"]
    assert patch["document"]["content"][6] == section_patch_document()["content"][8]


def test_patch_leaves_task_item_ids_and_states_untouched() -> None:
    original = section_patch_document()
    task_list = original["content"][9]

    patch = patch_adf_description_section(
        original,
        section="Observed",
        content="## Observed\n\nRewritten.",
    )

    patched_task_list = patch["document"]["content"][9]
    assert patched_task_list is task_list
    assert [item["attrs"] for item in patched_task_list["content"]] == [
        {"localId": "task-1", "state": "DONE"},
        {"localId": "task-2", "state": "TODO"},
    ]


def test_patch_reports_the_existing_headings_when_the_section_is_unknown() -> None:
    try:
        patch_adf_description_section(
            section_patch_document(),
            section="Reproduction",
            content="## Reproduction\n\nRun the dry run.",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Description section not found: Reproduction. Existing headings: "
            "Dry run prevented a description wipe, Observed, Scope, In scope, "
            "Acceptance criteria, Test approach"
        )
    else:
        raise AssertionError("Expected ValueError")


def test_patch_refuses_an_ambiguous_heading() -> None:
    try:
        patch_adf_description_section(
            ambiguous_section_document(),
            section="Scope",
            content="## Scope\n\nRewritten.",
        )
    except ValueError as exc:
        assert str(exc) == "Description section is ambiguous: Scope"
    else:
        raise AssertionError("Expected ValueError")


def test_patch_refuses_a_missing_or_non_adf_description() -> None:
    for description, message in [
        (None, "The issue has no description, so no section can be patched."),
        (
            "plain text",
            "The issue description is not an ADF doc node, so no section can be patched.",
        ),
        (
            {"type": "paragraph", "content": []},
            "The issue description is not an ADF doc node, so no section can be patched.",
        ),
        (
            {"type": "doc", "version": 1, "content": []},
            "The issue description is empty, so no section can be patched.",
        ),
    ]:
        try:
            patch_adf_description_section(description, section="Scope", content="## Scope\n\nX")
        except ValueError as exc:
            assert str(exc) == message
        else:
            raise AssertionError("Expected ValueError")


def test_patch_refuses_content_that_does_not_match_the_mode() -> None:
    for kwargs, message in [
        (
            {"mode": "delete", "content": "## Scope\n\nX"},
            "Mode delete removes the section and takes no content.",
        ),
        ({"mode": "replace"}, "Mode replace requires the Markdown content of the new section."),
        (
            {"mode": "append", "content": "   "},
            "Mode append requires the Markdown content of the new section.",
        ),
        (
            {"mode": "overwrite", "content": "X"},
            "Unknown section patch mode: overwrite. Use one of replace, append, prepend, delete.",
        ),
    ]:
        try:
            patch_adf_description_section(section_patch_document(), section="Scope", **kwargs)
        except ValueError as exc:
            assert str(exc) == message
        else:
            raise AssertionError("Expected ValueError")


def test_a_rebuild_that_returns_only_the_new_content_is_refused(monkeypatch) -> None:
    """The FC-385 defect class: the operation reimplemented as a whole-field replacement."""
    content = "## Acceptance criteria\n\n- [ ] Only this bullet survives"
    monkeypatch.setattr(
        client_module,
        "_rebuild_with_section",
        lambda source, *, start, end, new_nodes: text_to_adf(content)["content"],
    )

    try:
        patch_adf_description_section(
            section_patch_document(),
            section="Acceptance criteria",
            content=content,
        )
    except AdfPatchInvariantError as exc:
        assert str(exc) == "Slice replacement produced 2 top-level nodes instead of 12."
    else:
        raise AssertionError("Expected AdfPatchInvariantError")


def test_a_rebuild_that_re_renders_an_untouched_node_is_refused(monkeypatch) -> None:
    """Equal-looking nodes are not enough: an untouched node has to be the same object."""

    def rerender(source, *, start, end, new_nodes):
        rebuilt = [json.loads(json.dumps(node)) for node in source[:start]]
        return [*rebuilt, *new_nodes, *source[end:]]

    monkeypatch.setattr(client_module, "_rebuild_with_section", rerender)

    try:
        patch_adf_description_section(
            section_patch_document(),
            section="Acceptance criteria",
            content="## Acceptance criteria\n\n- [ ] Rewritten",
        )
    except AdfPatchInvariantError as exc:
        assert str(exc) == (
            "Slice replacement rebuilt one of the 8 top-level node(s) before the slice."
        )
    else:
        raise AssertionError("Expected AdfPatchInvariantError")


def test_a_rebuild_that_mutates_the_caller_document_is_refused(monkeypatch) -> None:
    document = section_patch_document()

    def mutate(source, *, start, end, new_nodes):
        document["content"][start:end] = new_nodes
        return list(document["content"])

    monkeypatch.setattr(client_module, "_rebuild_with_section", mutate)

    try:
        patch_adf_description_section(
            document,
            section="Acceptance criteria",
            content="## Acceptance criteria\n\n- [ ] Rewritten",
        )
    except AdfPatchInvariantError as exc:
        assert str(exc) == (
            "Slice replacement changed the document it was given instead of building a new one."
        )
    else:
        raise AssertionError("Expected AdfPatchInvariantError")


def test_bounds_that_are_not_a_slice_of_the_document_are_refused(monkeypatch) -> None:
    monkeypatch.setattr(client_module, "_adf_section_bounds", lambda content, section: (8, 4))

    try:
        patch_adf_description_section(
            section_patch_document(),
            section="Acceptance criteria",
            content="## Acceptance criteria\n\n- [ ] Rewritten",
        )
    except AdfPatchInvariantError as exc:
        assert str(exc) == "Slice bounds 8..4 are not a slice of 12 top-level nodes."
    else:
        raise AssertionError("Expected AdfPatchInvariantError")


def test_a_section_patch_never_reduces_the_description_to_the_new_content_alone() -> None:
    """FC-385 regression: a section patch must not behave like a whole-field replacement.

    A reimplementation that renders the caller's Markdown into the description field would make
    the patched document equal to text_to_adf(content) and would drop every other section.
    """
    content = "## Acceptance criteria\n\n- [ ] Only this bullet survives"
    reference = section_patch_document()

    patch = patch_adf_description_section(
        section_patch_document(),
        section="Acceptance criteria",
        content=content,
    )
    document = patch["document"]

    assert document != text_to_adf(content)
    assert document["content"] != text_to_adf(content)["content"]
    assert len(document["content"]) > len(text_to_adf(content)["content"])
    for index in list(range(8)) + [10, 11]:
        assert reference["content"][index] in document["content"]
    assert patch["preserved_before"] == 8
    assert patch["preserved_after"] == 2


def malformed_heading_document(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """The reviewer's repro: Observed carries a broken level, Acceptance criteria follows it."""
    heading: dict[str, Any] = {
        "type": "heading",
        "content": [{"type": "text", "text": "Observed"}],
    }
    if attrs is not None:
        heading["attrs"] = attrs
    return {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(1, "Title"),
            _paragraph("Intro."),
            heading,
            _paragraph("The payload replaced the whole field."),
            _heading(2, "Acceptance criteria"),
            {
                "type": "taskList",
                "attrs": {"localId": "task-list-1"},
                "content": [
                    {
                        "type": "taskItem",
                        "attrs": {"localId": "task-1", "state": "TODO"},
                        "content": [{"type": "text", "text": "Survive the patch"}],
                    }
                ],
            },
            _heading(2, "Test approach"),
            _paragraph("Unit tests."),
        ],
    }


def test_a_matched_heading_without_a_usable_level_is_refused_not_guessed() -> None:
    for attrs in [
        None,
        {},
        {"level": None},
        {"level": "two"},
        {"level": []},
        {"level": True},
        {"level": 0},
        {"level": 7},
        {"level": 2.0},
    ]:
        try:
            patch_adf_description_section(
                malformed_heading_document(attrs),
                section="Observed",
                content="## Observed\n\nRewritten.",
            )
        except ValueError as exc:
            assert str(exc) == (
                "Heading 'Observed' has no usable attrs.level between 1 and 6, so the extent "
                "of its section cannot be determined."
            )
        else:
            raise AssertionError(f"Expected ValueError for attrs={attrs!r}")


def test_an_unreadable_heading_below_the_target_ends_the_section() -> None:
    document = malformed_heading_document({"level": 2})
    document["content"][4] = {
        "type": "heading",
        "content": [{"type": "text", "text": "Acceptance criteria"}],
    }
    reference = [node for node in document["content"]]

    patch = patch_adf_description_section(
        document,
        section="Observed",
        content="## Observed\n\nRewritten.",
    )

    assert (patch["start_index"], patch["end_index"]) == (2, 4)
    assert patch["document"]["content"][4:] == reference[4:]
    assert patch["document"]["content"][5]["type"] == "taskList"


def test_replace_with_include_heading_requires_content_that_carries_a_heading() -> None:
    try:
        patch_adf_description_section(
            section_patch_document(),
            section="Observed",
            content="Just a paragraph, and the heading would be gone.",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Mode replace with include_heading replaces the 'Observed' heading node, so the "
            "content has to start with its own heading. Add the heading to the content, or "
            "pass include_heading=false to keep the existing heading and change only the body."
        )
    else:
        raise AssertionError("Expected ValueError")


def test_heading_text_reads_emoji_mention_status_and_date_nodes() -> None:
    heading = {
        "type": "heading",
        "attrs": {"level": 2},
        "content": [
            {"type": "emoji", "attrs": {"shortName": ":warning:", "text": "⚠"}},
            {"type": "text", "text": " Blocked for "},
            {"type": "mention", "attrs": {"id": "557", "text": "@Sabine"}},
            {"type": "text", "text": " at "},
            {"type": "status", "attrs": {"text": "In progress", "color": "yellow"}},
        ],
    }
    document = {
        "type": "doc",
        "version": 1,
        "content": [heading, _paragraph("Body."), _heading(2, "Next"), _paragraph("More.")],
    }

    patch = patch_adf_description_section(
        document,
        section="⚠ blocked for @sabine at in progress",
        content="## Unblocked\n\nDone.",
    )

    assert patch["heading"] == "⚠ Blocked for @Sabine at In progress"
    assert (patch["start_index"], patch["end_index"]) == (0, 2)
    assert adf_heading_text({"type": "date", "attrs": {"timestamp": "1758067200000"}}) == (
        "1758067200000"
    )
    assert adf_heading_text({"type": "emoji", "attrs": {"shortName": ":smile:"}}) == ":smile:"


def test_a_heading_split_across_text_nodes_and_marks_still_matches() -> None:
    heading = {
        "type": "heading",
        "attrs": {"level": 2},
        "content": [
            {"type": "text", "text": "Acceptance "},
            {"type": "text", "text": "criteria", "marks": [{"type": "strong"}]},
        ],
    }
    document = {
        "type": "doc",
        "version": 1,
        "content": [_heading(1, "Title"), heading, _paragraph("Body.")],
    }

    patch = patch_adf_description_section(
        document,
        section="acceptance criteria",
        content="## Acceptance criteria\n\nRewritten.",
    )

    assert patch["heading"] == "Acceptance criteria"
    assert (patch["start_index"], patch["end_index"]) == (1, 3)


def test_the_patch_keeps_document_metadata_and_leaves_the_input_untouched() -> None:
    document = section_patch_document()
    document["version"] = 1
    snapshot = json.loads(json.dumps(document))
    nodes = document["content"]

    patch = patch_adf_description_section(
        document,
        section="Observed",
        content="## Observed\n\nRewritten.",
    )

    assert patch["document"]["type"] == "doc"
    assert patch["document"]["version"] == 1
    assert patch["document"] is not document
    assert patch["document"]["content"] is not nodes
    assert document == snapshot
    assert document["content"] is nodes


def test_upsert_top_level_section_refuses_an_ambiguous_title() -> None:
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(3, "Screens"),
            _paragraph("One."),
            _heading(3, "Screens"),
            _paragraph("Two."),
        ],
    }

    try:
        upsert_top_level_section(adf, title="Screens", blocks=[_heading(3, "Screens")])
    except ValueError as exc:
        assert str(exc) == "Description section is ambiguous: Screens"
    else:
        raise AssertionError("Expected ValueError")


def test_upsert_top_level_section_replaces_the_section_with_its_subsections() -> None:
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(3, "Screens"),
            _heading(4, "Desktop"),
            _paragraph("Old."),
            _heading(3, "Child Issues"),
            _paragraph("Keep."),
        ],
    }

    result = upsert_top_level_section(
        adf,
        title="screens",
        blocks=[_heading(3, "Screens"), _paragraph("New.")],
    )

    assert result["content"] == [
        _heading(3, "Screens"),
        _paragraph("New."),
        _heading(3, "Child Issues"),
        _paragraph("Keep."),
    ]


def test_include_heading_false_is_refused_outside_replace_mode() -> None:
    for mode, extra in [
        ("append", {"content": "- One more"}),
        ("prepend", {"content": "Intro."}),
        ("delete", {}),
    ]:
        try:
            patch_adf_description_section(
                section_patch_document(),
                section="Acceptance criteria",
                mode=mode,
                include_heading=False,
                **extra,
            )
        except ValueError as exc:
            assert str(exc) == (
                f"include_heading applies to mode replace only; mode {mode} always keeps the "
                "existing heading node."
            )
        else:
            raise AssertionError(f"Expected ValueError for mode {mode}")


def preamble_document() -> dict[str, object]:
    """A description whose title is not node 0, which is how the whole-field guard was bypassed.

    Indices: 0 panel, 1 h1 Title, 2 h2 Observed, 3 paragraph, 4 h2 Acceptance criteria,
    5 taskList, 6 h2 Test approach, 7 paragraph.
    """
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "panel",
                "attrs": {"panelType": "info"},
                "content": [_paragraph("Read the linked decision first.")],
            },
            _heading(1, "Title"),
            _heading(2, "Observed"),
            _paragraph("The payload replaced the whole field."),
            _heading(2, "Acceptance criteria"),
            {
                "type": "taskList",
                "attrs": {"localId": "task-list-1"},
                "content": [
                    {
                        "type": "taskItem",
                        "attrs": {"localId": "task-1", "state": "TODO"},
                        "content": [{"type": "text", "text": "Survive the patch"}],
                    }
                ],
            },
            _heading(2, "Test approach"),
            _paragraph("Unit tests."),
        ],
    }


def test_a_section_ends_at_the_next_heading_even_with_no_body_between_them() -> None:
    """Kills the `end = start + 1` off-by-one: 1..2 and 1..3 are both valid slices."""
    document = {
        "type": "doc",
        "version": 1,
        "content": [_heading(1, "Title"), _heading(2, "A"), _heading(2, "B")],
    }
    reference = {
        "type": "doc",
        "version": 1,
        "content": [_heading(1, "Title"), _heading(2, "A"), _heading(2, "B")],
    }

    patch = patch_adf_description_section(
        document,
        section="A",
        content="## A\n\nBody for A.",
    )

    assert (patch["start_index"], patch["end_index"]) == (1, 2)
    assert patch["replaced_node_count"] == 1
    assert patch["document"]["content"][0] == reference["content"][0]
    assert patch["document"]["content"][-1] == reference["content"][2]


def test_consecutive_headings_before_a_body_do_not_widen_the_slice() -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [_heading(1, "Title"), _heading(2, "A"), _heading(2, "B"), _paragraph("B.")],
    }

    patch = patch_adf_description_section(document, section="A", content="## A\n\nBody for A.")

    assert (patch["start_index"], patch["end_index"]) == (1, 2)
    assert [node.get("type") for node in patch["document"]["content"]] == [
        "heading",
        "heading",
        "paragraph",
        "heading",
        "paragraph",
    ]


def test_a_heading_as_the_last_node_is_a_section_with_an_empty_body() -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [_heading(1, "Title"), _paragraph("Intro."), _heading(2, "A")],
    }

    patch = patch_adf_description_section(document, section="A", content="## A\n\nBody for A.")

    assert (patch["start_index"], patch["end_index"]) == (2, 3)
    assert patch["preserved_after"] == 0
    assert patch["document"]["content"][:2] == document["content"][:2]


def test_a_rebuild_that_corrupts_a_node_after_the_section_is_refused(monkeypatch) -> None:
    def copy_tail(source, *, start, end, new_nodes):
        return [
            *source[:start],
            *new_nodes,
            *[json.loads(json.dumps(node)) for node in source[end:]],
        ]

    monkeypatch.setattr(client_module, "_rebuild_with_section", copy_tail)

    try:
        patch_adf_description_section(
            section_patch_document(),
            section="Observed",
            content="## Observed\n\nRewritten.",
        )
    except AdfPatchInvariantError as exc:
        assert str(exc) == (
            "Slice replacement rebuilt one of the 8 top-level node(s) after the slice."
        )
    else:
        raise AssertionError("Expected AdfPatchInvariantError")


def test_a_rebuild_that_reorders_the_new_slice_is_refused(monkeypatch) -> None:
    def reverse_new(source, *, start, end, new_nodes):
        return [*source[:start], *reversed(new_nodes), *source[end:]]

    monkeypatch.setattr(client_module, "_rebuild_with_section", reverse_new)

    try:
        patch_adf_description_section(
            section_patch_document(),
            section="Observed",
            content="## Observed\n\nRewritten.",
        )
    except AdfPatchInvariantError as exc:
        assert str(exc) == "Slice replacement did not place the new nodes in the matched slice."
    else:
        raise AssertionError("Expected AdfPatchInvariantError")


def test_new_task_items_do_not_reuse_a_local_id_already_in_the_description() -> None:
    patch = patch_adf_description_section(
        section_patch_document(),
        section="Observed",
        content="## Observed\n\n- [ ] First new item\n- [x] Second new item",
    )
    local_ids = [
        node["attrs"]["localId"]
        for node in _walk(patch["document"])
        if isinstance(node.get("attrs"), dict) and "localId" in node["attrs"]
    ]

    assert len(local_ids) == len(set(local_ids))
    assert "task-1" in local_ids and "task-2" in local_ids
    assert {"task-3", "task-4", "task-list-3"} <= set(local_ids)


def test_the_task_id_seed_looks_at_the_whole_document_not_only_the_top_level() -> None:
    nested = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(2, "Observed"),
            {
                "type": "panel",
                "attrs": {"panelType": "note"},
                "content": [
                    {
                        "type": "taskList",
                        "attrs": {"localId": "task-list-9"},
                        "content": [
                            {
                                "type": "taskItem",
                                "attrs": {"localId": "task-9", "state": "TODO"},
                                "content": [{"type": "text", "text": "Nested"}],
                            }
                        ],
                    }
                ],
            },
            _heading(2, "Acceptance criteria"),
            _paragraph("Body."),
        ],
    }

    patch = patch_adf_description_section(
        nested,
        section="Acceptance criteria",
        content="## Acceptance criteria\n\n- [ ] New item",
    )
    local_ids = [
        node["attrs"]["localId"]
        for node in _walk(patch["document"])
        if isinstance(node.get("attrs"), dict) and "localId" in node["attrs"]
    ]

    assert len(local_ids) == len(set(local_ids))
    assert {"task-10", "task-list-10"} <= set(local_ids)


def test_upsert_top_level_section_keeps_untouched_nodes_as_the_same_objects() -> None:
    keep_before = _paragraph("Before.")
    keep_after = _heading(3, "Child Issues")
    adf = {
        "type": "doc",
        "version": 1,
        "content": [keep_before, _heading(3, "Screens"), _paragraph("Old."), keep_after],
    }

    result = upsert_top_level_section(
        adf,
        title="Screens",
        blocks=[_heading(3, "Screens"), _paragraph("New.")],
    )

    assert result["content"][0] is keep_before
    assert result["content"][3] is keep_after
    assert adf["content"][1]["type"] == "heading"
    assert len(adf["content"]) == 4


def test_upsert_top_level_section_appends_into_an_empty_document() -> None:
    blocks = [_heading(3, "Screens"), _paragraph("New.")]

    result = upsert_top_level_section({"type": "doc", "version": 1}, title="Screens", blocks=blocks)

    assert result["content"] == blocks


def _walk(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        found.append(current)
        children = current.get("content")
        if isinstance(children, list):
            stack.extend(children)
    return found


EMPTY_HEADING = {"type": "heading", "attrs": {"level": 1}, "content": []}
BLANK_HEADING = {
    "type": "heading",
    "attrs": {"level": 1},
    "content": [{"type": "text", "text": "   "}],
}
UNLEVELLED_HEADING = {"type": "heading", "content": [{"type": "text", "text": "Stray"}]}


def _stray_heading(level: object) -> dict[str, Any]:
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": "Stray"}],
    }


def _titled_document(prefix: list[dict[str, Any]], suffix: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [*prefix, *section_patch_document()["content"], *suffix],
    }


def _panel() -> dict[str, Any]:
    return {
        "type": "panel",
        "attrs": {"panelType": "info"},
        "content": [_paragraph("Read the linked decision first.")],
    }


def _padded_title_document(padding: list[dict[str, Any]]) -> dict[str, Any]:
    """A titled description behind `padding`. Round 4 used padding to switch the guard off."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            *padding,
            _heading(1, "Title"),
            _heading(2, "A"),
            {
                "type": "taskList",
                "attrs": {"localId": "task-list-1"},
                "content": [
                    {
                        "type": "taskItem",
                        "attrs": {"localId": "task-1", "state": "DONE"},
                        "content": [{"type": "text", "text": "Survive"}],
                    }
                ],
            },
            _heading(2, "B"),
            _paragraph("z"),
        ],
    }


EMPTY_HEADING = {"type": "heading", "attrs": {"level": 1}, "content": []}
BLANK_HEADING = {
    "type": "heading",
    "attrs": {"level": 1},
    "content": [{"type": "text", "text": "   "}],
}
UNLEVELLED_HEADING = {"type": "heading", "content": [{"type": "text", "text": "Stray"}]}
TITLE = "Dry run prevented a description wipe"
TITLE_SUBSECTIONS = ["Observed", "Scope", "In scope", "Acceptance criteria", "Test approach"]


def _panel() -> dict[str, Any]:
    return {
        "type": "panel",
        "attrs": {"panelType": "info"},
        "content": [_paragraph("Read the linked decision first.")],
    }


def _stray_heading(level: object) -> dict[str, Any]:
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": "Stray"}],
    }


def _titled_document(prefix: list[Any], suffix: list[Any]) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [*prefix, *section_patch_document()["content"], *suffix],
    }


def _refuse(document: dict[str, Any], section: str, **kwargs: Any) -> str:
    kwargs.setdefault("mode", "delete")
    try:
        patch_adf_description_section(document, section=section, **kwargs)
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"Expected ValueError patching {section!r}")


# --- control 1: nothing outside the slice would survive ----------------------


def test_control_1_refuses_a_slice_that_is_the_whole_content_list() -> None:
    for mode, extra in [("delete", {}), ("replace", {"content": "# T\n\nNew."})]:
        message = _refuse(section_patch_document(), TITLE, mode=mode, **extra)
        assert "Nothing outside the slice would survive" in message
        assert "covers all 12 top-level nodes" in message
        assert "allow_wide_rewrite" in message


def test_control_1_is_pure_arithmetic_and_inspects_no_node() -> None:
    """Round 2's panel and round 3's content-free heading move the bounds, nothing else."""
    for prefix, suffix in [([_panel()], []), ([], [EMPTY_HEADING]), ([UNLEVELLED_HEADING], [])]:
        patch = patch_adf_description_section(
            _titled_document(prefix, suffix),
            section=TITLE,
            mode="delete",
            expect_swallowed_headings=TITLE_SUBSECTIONS,
        )
        assert patch["whole_document"] is False


def test_control_1_makes_replace_no_more_permissive_than_delete() -> None:
    """Round 6 F2: replace used to rewrite a 21-node field to one node without a flag."""
    document = {
        "type": "doc",
        "version": 1,
        "content": [_heading(1, "Title"), *[_paragraph(str(index)) for index in range(20)]],
    }
    for mode, extra in [("replace", {"content": "# Title\n\nNew."}), ("delete", {})]:
        assert "Nothing outside the slice would survive" in _refuse(
            document, "Title", mode=mode, **extra
        )

    patch = patch_adf_description_section(
        document, section="Title", content="# Title\n\nNew.", allow_wide_rewrite=True
    )
    assert patch["patched_node_count"] == 2


# --- control 2: the naming duty ----------------------------------------------


def test_control_2_refuses_a_slice_whose_swallowed_sections_were_not_named() -> None:
    message = _refuse(_titled_document([_panel()], []), TITLE)
    assert "Swallowed-section mismatch" in message
    assert (
        'the slice removes ["Observed", "Scope", "In scope", "Acceptance criteria", '
        '"Test approach"]'
    ) in message
    assert "the call named []" in message
    assert "Nothing outside the slice would survive" not in message


def test_control_2_accepts_the_exact_ordered_list_the_refusal_printed() -> None:
    patch = patch_adf_description_section(
        _titled_document([_panel()], []),
        section=TITLE,
        mode="delete",
        expect_swallowed_headings=TITLE_SUBSECTIONS,
    )

    assert patch["swallowed_headings"] == TITLE_SUBSECTIONS
    assert patch["document"]["content"] == [_panel()]


def test_control_2_compares_an_ordered_list_and_not_a_set() -> None:
    swapped = [
        "Observed",
        "Scope",
        "Acceptance criteria",
        "In scope",
        "Test approach",
    ]
    message = _refuse(_titled_document([_panel()], []), TITLE, expect_swallowed_headings=swapped)
    assert "Swallowed-section mismatch" in message


def test_control_2_refuses_a_short_a_long_and_a_spurious_expectation() -> None:
    document = _titled_document([_panel()], [])
    for named in [TITLE_SUBSECTIONS[:2], [*TITLE_SUBSECTIONS, "Extra"]]:
        assert "Swallowed-section mismatch" in _refuse(
            document, TITLE, expect_swallowed_headings=named
        )

    # A caller with a wrong model of the document must not proceed either.
    assert "Swallowed-section mismatch" in _refuse(
        section_patch_document(), "Test approach", expect_swallowed_headings=["Nope"]
    )


def test_control_2_names_a_heading_that_has_no_text() -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(1, "Title"),
            _heading(2, "A"),
            {"type": "heading", "attrs": {"level": 3}, "content": []},
            {"type": "heading", "attrs": {"level": 3}, "content": [{"type": "text", "text": " "}]},
            _paragraph("body"),
            _heading(1, "Notes"),
        ],
    }

    message = _refuse(document, "A")
    assert 'the slice removes ["<untitled>", "<untitled>"]' in message

    patch = patch_adf_description_section(
        document,
        section="A",
        mode="delete",
        expect_swallowed_headings=["<untitled>", "<untitled>"],
    )
    assert patch["swallowed_headings"] == ["<untitled>", "<untitled>"]


def test_control_2_handles_duplicate_heading_text_inside_one_slice() -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(1, "Title"),
            _heading(2, "A"),
            _heading(3, "Detail"),
            _paragraph("one"),
            _heading(3, "Detail"),
            _paragraph("two"),
            _heading(2, "B"),
        ],
    }

    assert "Swallowed-section mismatch" in _refuse(
        document, "A", expect_swallowed_headings=["Detail"]
    )
    patch = patch_adf_description_section(
        document, section="A", mode="delete", expect_swallowed_headings=["Detail", "Detail"]
    )
    assert patch["swallowed_headings"] == ["Detail", "Detail"]


def test_control_2_prints_the_whole_list_and_never_truncates_it() -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(1, "Title"),
            *[_heading(2, f"S{index}") for index in range(8)],
            _paragraph("tail"),
            _heading(1, "Notes"),
        ],
    }

    message = _refuse(document, "Title")
    assert '"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"' in message
    assert "more" not in message


# --- the two controls are independent ----------------------------------------


def test_allow_wide_rewrite_does_not_satisfy_the_naming_duty() -> None:
    message = _refuse(section_patch_document(), TITLE, allow_wide_rewrite=True)

    assert "Swallowed-section mismatch" in message
    assert "Nothing outside the slice would survive" not in message


def test_naming_does_not_satisfy_the_whole_content_control() -> None:
    message = _refuse(section_patch_document(), TITLE, expect_swallowed_headings=TITLE_SUBSECTIONS)

    assert "Nothing outside the slice would survive" in message
    assert "Swallowed-section mismatch" not in message


def test_a_slice_that_trips_both_controls_reports_both() -> None:
    message = _refuse(section_patch_document(), TITLE)

    assert "Nothing outside the slice would survive" in message
    assert "Swallowed-section mismatch" in message


def test_both_controls_together_let_the_patch_through() -> None:
    patch = patch_adf_description_section(
        section_patch_document(),
        section=TITLE,
        content="# Title\n\nNew.",
        expect_swallowed_headings=TITLE_SUBSECTIONS,
        allow_wide_rewrite=True,
    )

    assert patch["whole_document"] is True
    assert patch["patched_node_count"] == 2


# --- neither control applies to append or prepend ----------------------------


def test_append_and_prepend_need_neither_control_because_they_remove_nothing() -> None:
    reference = section_patch_document()
    for mode in ("append", "prepend"):
        patch = patch_adf_description_section(
            section_patch_document(), section=TITLE, mode=mode, content="- One more note"
        )
        assert patch["whole_document"] is True
        assert patch["swallowed_headings"] == TITLE_SUBSECTIONS
        assert patch["patched_node_count"] == 13
        assert all(node in patch["document"]["content"] for node in reference["content"])


def _controls_that_fire(message: str) -> set[str]:
    fired = set()
    if "Nothing outside the slice would survive" in message:
        fired.add("whole-content")
    if "Swallowed-section mismatch" in message:
        fired.add("naming")
    return fired


def test_every_historical_defect_shape_refuses_and_names_which_control_fired() -> None:
    """Rounds 1 to 6. A shape refused by the wrong control would be a latent defect."""
    padded = lambda pad: {  # noqa: E731 - a table of shapes reads better than six defs
        "type": "doc",
        "version": 1,
        "content": [
            *pad,
            _heading(1, "Title"),
            _heading(2, "A"),
            _paragraph("x"),
            _heading(2, "B"),
            _paragraph("z"),
        ],
    }
    shapes = [
        # round 1: the title is node 0, so the slice is the whole content list.
        ("r1 title at index 0", section_patch_document(), TITLE, {"whole-content", "naming"}),
        # round 2: a leading panel moves the bounds; only the naming duty applies.
        ("r2 panel preamble", _titled_document([_panel()], []), TITLE, {"naming"}),
        # round 3: content-free and malformed headings outside the slice.
        ("r3 empty heading appended", _titled_document([], [EMPTY_HEADING]), TITLE, {"naming"}),
        ("r3 empty heading prepended", _titled_document([EMPTY_HEADING], []), TITLE, {"naming"}),
        ("r3 blank-text heading", _titled_document([], [BLANK_HEADING]), TITLE, {"naming"}),
        (
            "r3 real heading appended",
            _titled_document([], [_heading(1, "Appendix")]),
            TITLE,
            {"naming"},
        ),
        ("r3 heading with no level", _titled_document([UNLEVELLED_HEADING], []), TITLE, {"naming"}),
        ("r3 heading at level 0", _titled_document([_stray_heading(0)], []), TITLE, {"naming"}),
        ("r3 heading at level 9", _titled_document([_stray_heading(9)], []), TITLE, {"naming"}),
        ("r3 heading at level '1'", _titled_document([_stray_heading("1")], []), TITLE, {"naming"}),
        # round 4: padding outside the slice used to buy a wipe through the size term.
        (
            "r4 panel and four paragraphs",
            padded([_panel(), *[_paragraph(c) for c in "abcd"]]),
            "Title",
            {"naming"},
        ),
        ("r4 five rule nodes", padded([{"type": "rule"} for _ in range(5)]), "Title", {"naming"}),
        (
            "r4 five empty paragraphs",
            padded([{"type": "paragraph"} for _ in range(5)]),
            "Title",
            {"naming"},
        ),
        # round 5: an unreadable level outside the slice used to lower the rank bar.
        (
            "r5 degenerate heading outside",
            padded([UNLEVELLED_HEADING, *[_paragraph(str(i)) for i in range(5)]]),
            "Title",
            {"naming"},
        ),
    ]
    for name, document, section, expected in shapes:
        fired = _controls_that_fire(_refuse(document, section))
        assert fired == expected, f"{name}: {fired} fired, expected {expected}"


def test_round_6_f1_a_content_free_heading_no_longer_buys_a_subsection_wipe() -> None:
    """F1: one empty h1 outside the slice set the rank bar to 1 and allowed 40 nodes to 22."""
    body = [
        node
        for index in range(6)
        for node in (_heading(3, f"D{index}"), _paragraph("x"), {"type": "rule"})
    ]
    document = {
        "type": "doc",
        "version": 1,
        "content": [
            EMPTY_HEADING,
            *[{"type": "paragraph"} for _ in range(20)],
            _heading(2, "Scope"),
            *body,
        ],
    }
    names = [f"D{index}" for index in range(6)]

    assert _controls_that_fire(
        _refuse(document, "Scope", mode="replace", content="## Scope\n\nnew")
    ) == {"naming"}

    patch = patch_adf_description_section(
        document,
        section="Scope",
        content="## Scope\n\nnew",
        expect_swallowed_headings=names,
    )
    assert patch["swallowed_headings"] == names
    assert patch["document"]["content"][0] is document["content"][0]


def test_round_6_f3_a_trailing_content_free_heading_no_longer_flips_the_decision() -> None:
    siblings = lambda suffix: {  # noqa: E731
        "type": "doc",
        "version": 1,
        "content": [
            _heading(2, "Observed"),
            _paragraph("a"),
            _heading(2, "Scope"),
            _paragraph("b"),
            _heading(3, "In scope"),
            _paragraph("c"),
            _heading(2, "AC"),
            _paragraph("d"),
            *suffix,
        ],
    }
    for suffix in ([], [EMPTY_HEADING]):
        assert _controls_that_fire(
            _refuse(siblings(suffix), "Scope", mode="replace", content="## Scope\n\nnew")
        ) == {"naming"}

    patch = patch_adf_description_section(
        siblings([EMPTY_HEADING]),
        section="Scope",
        content="## Scope\n\nnew",
        expect_swallowed_headings=["In scope"],
    )
    assert patch["patched_node_count"] == 7


def test_round_6_f4_the_trailing_heading_witness() -> None:
    """As stated the slice swallows nothing, so it is allowed and always was correct.

    The witness that carried the defect is the same shape with A owning a subsection: a trailing
    h1 used to quiet the rank term there. The naming duty catches it whatever sits around it.
    """
    as_stated = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(2, "A"),
            _paragraph("a"),
            _heading(2, "B"),
            _paragraph("b"),
            _heading(1, "Notes"),
        ],
    }
    patch = patch_adf_description_section(as_stated, section="A", content="## A\n\nnew")
    assert patch["swallowed_headings"] == []
    assert patch["document"]["content"][2:] == as_stated["content"][2:]

    carrying = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(2, "A"),
            _paragraph("a"),
            _heading(3, "A1"),
            _paragraph("x"),
            _heading(2, "B"),
            *[_paragraph(str(index)) for index in range(4)],
            _heading(1, "Notes"),
        ],
    }
    assert _controls_that_fire(_refuse(carrying, "A", mode="replace", content="## A\n\nnew")) == {
        "naming"
    }


def test_the_number_of_surviving_nodes_does_not_enter_the_decision() -> None:
    """Rounds 4 and 6 both turned on counting survivors. Padding must change nothing now."""

    def document(padding: int) -> dict[str, Any]:
        return {
            "type": "doc",
            "version": 1,
            "content": [
                *[{"type": "paragraph"} for _ in range(padding)],
                _heading(2, "A"),
                _heading(3, "A1"),
                _paragraph("x"),
            ],
        }

    for padding in (1, 5, 40):
        assert _controls_that_fire(_refuse(document(padding), "A")) == {"naming"}
        patch = patch_adf_description_section(
            document(padding), section="A", mode="delete", expect_swallowed_headings=["A1"]
        )
        assert patch["patched_node_count"] == padding


def test_the_empty_description_rule_still_applies_behind_the_flag() -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [_heading(2, "Only section"), _paragraph("Body.")],
    }

    assert _controls_that_fire(_refuse(document, "Only section")) == {"whole-content"}

    try:
        patch_adf_description_section(
            document, section="Only section", mode="delete", allow_wide_rewrite=True
        )
    except ValueError as exc:
        assert "would leave the description with no content" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_naming_the_only_section_of_a_description_is_not_refused() -> None:
    """Documented residual: nothing unnamed can be lost, so neither control applies."""
    document = {
        "type": "doc",
        "version": 1,
        "content": [_panel(), _heading(1, "Title"), *[_paragraph(str(i)) for i in range(50)]],
    }

    patch = patch_adf_description_section(document, section="Title", mode="delete")

    assert patch["swallowed_headings"] == []
    assert patch["whole_document"] is False
    assert patch["document"]["content"] == [document["content"][0]]


def test_the_image_gallery_splice_is_untouched_by_the_two_controls() -> None:
    """It reaches find_adf_section_bounds only, so it carries no naming duty."""
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(3, "Screens"),
            _heading(4, "Desktop"),
            _paragraph("Old."),
            _heading(3, "Child Issues"),
        ],
    }

    result = upsert_top_level_section(
        adf, title="Screens", blocks=[_heading(3, "Screens"), _paragraph("New.")]
    )

    assert result["content"] == [
        _heading(3, "Screens"),
        _paragraph("New."),
        _heading(3, "Child Issues"),
    ]


def _copy_back(message: str) -> list[str]:
    """Recover the swallowed names from a refusal the way a caller would: parse the array."""
    start = message.index("the slice removes ") + len("the slice removes ")
    end = message.index("] on top of", start) + 1
    return json.loads(message[start:end])


def _awkward_document(name: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            _paragraph("Preamble so control 1 stays out of this."),
            _heading(2, "Scope"),
            _heading(3, name),
            _paragraph("body"),
            _heading(2, "Next"),
        ],
    }


def test_the_names_a_refusal_prints_authorise_the_patch_when_copied_back() -> None:
    """The naming duty is only a control if the refusal itself carries the correction."""
    for name in [
        "In scope, Out of scope",
        "nothing",
        "@Ada, Lovelace",
        "In  scope",
        '"quoted"',
        "<untitled>",
    ]:
        document = _awkward_document(name)
        reference = _awkward_document(name)
        message = _refuse(document, "Scope", mode="replace", content="## Scope\n\nnew")
        recovered = _copy_back(message)

        assert recovered == [name], f"{name!r} came back as {recovered!r}"

        patch = patch_adf_description_section(
            document,
            section="Scope",
            content="## Scope\n\nnew",
            expect_swallowed_headings=recovered,
        )
        assert patch["swallowed_headings"] == [name]
        assert patch["document"]["content"][0] == reference["content"][0]
        assert patch["document"]["content"][-1] == reference["content"][-1]


def test_a_comma_in_a_name_is_one_name_and_not_two() -> None:
    document = _awkward_document("In scope, Out of scope")

    assert "Swallowed-section mismatch" in _refuse(
        document,
        "Scope",
        mode="replace",
        content="## Scope\n\nnew",
        expect_swallowed_headings=["In scope", "Out of scope"],
    )


def test_an_empty_list_is_not_the_word_nothing() -> None:
    """Two opposite situations must not produce the same sentence."""
    swallows_nothing = _refuse(
        _awkward_document("Detail"),
        "Next",
        mode="replace",
        content="## Next\n\nnew",
        expect_swallowed_headings=["nothing"],
    )
    swallows_a_section_called_nothing = _refuse(
        _awkward_document("nothing"), "Scope", mode="replace", content="## Scope\n\nnew"
    )

    assert "the slice removes [] on top of" in swallows_nothing
    assert 'the call named ["nothing"]' in swallows_nothing
    assert 'the slice removes ["nothing"] on top of' in swallows_a_section_called_nothing
    assert "the call named []" in swallows_a_section_called_nothing
    assert swallows_nothing != swallows_a_section_called_nothing


def test_names_differing_only_by_case_do_not_match() -> None:
    assert "Swallowed-section mismatch" in _refuse(
        _awkward_document("In scope"),
        "Scope",
        mode="replace",
        content="## Scope\n\nnew",
        expect_swallowed_headings=["in scope"],
    )


def test_internal_whitespace_is_significant_and_surrounding_whitespace_is_not() -> None:
    document = _awkward_document("In  scope")

    assert "Swallowed-section mismatch" in _refuse(
        document,
        "Scope",
        mode="replace",
        content="## Scope\n\nnew",
        expect_swallowed_headings=["In scope"],
    )
    patch = patch_adf_description_section(
        document,
        section="Scope",
        content="## Scope\n\nnew",
        expect_swallowed_headings=["  In  scope  "],
    )
    assert patch["swallowed_headings"] == ["In  scope"]


def test_a_refusal_carries_the_lists_as_data() -> None:
    try:
        patch_adf_description_section(
            _awkward_document("In scope, Out of scope"),
            section="Scope",
            content="## Scope\n\nnew",
            expect_swallowed_headings=["wrong"],
        )
    except AdfSectionPatchRefused as exc:
        assert exc.swallowed_headings == ["In scope, Out of scope"]
        assert exc.named_headings == ["wrong"]
        assert exc.whole_document is False
        assert isinstance(exc, ValueError)
    else:
        raise AssertionError("Expected AdfSectionPatchRefused")


def test_a_non_ascii_name_is_printed_readably_and_still_copies_back() -> None:
    document = _awkward_document("Prüfung ⚠ Sicherheit")

    message = _refuse(document, "Scope", mode="replace", content="## Scope\n\nnew")

    assert 'the slice removes ["Prüfung ⚠ Sicherheit"]' in message
    assert "\\u" not in message
    patch = patch_adf_description_section(
        document,
        section="Scope",
        content="## Scope\n\nnew",
        expect_swallowed_headings=_copy_back(message),
    )
    assert patch["swallowed_headings"] == ["Prüfung ⚠ Sicherheit"]


# --- Optimistic concurrency guard (issue_version, expect_updated) --------------------------


def test_issue_version_reads_updated_field() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"fields": {"updated": "2026-09-01T10:00:00.000+0200"}})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.issue_version("FC-1") == "2026-09-01T10:00:00.000+0200"
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-1?fields=updated"


def test_edit_issue_adf_description_guard_blocks_stale_write_without_putting() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"fields": {"updated": "T1"}})
        return httpx.Response(200, json={"id": "FC-1"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    adf = {"type": "doc", "version": 1, "content": []}
    try:
        client.edit_issue_adf_description("FC-1", adf, expect_updated="T0")
    except JiraConcurrentEditError as exc:
        assert exc.issue_key == "FC-1"
        assert exc.expected_updated == "T0"
        assert exc.found_updated == "T1"
    else:
        raise AssertionError("Expected JiraConcurrentEditError")

    # The mismatch must be caught before the PUT is ever issued.
    assert calls == ["GET"]


def test_edit_issue_adf_description_guard_allows_matching_update() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"fields": {"updated": "T1"}})
        return httpx.Response(200, json={"id": "FC-1"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    adf = {"type": "doc", "version": 1, "content": []}
    result = client.edit_issue_adf_description("FC-1", adf, expect_updated="T1")

    assert result == {"id": "FC-1"}
    assert calls == ["GET", "PUT"]


def test_edit_issue_adf_description_without_guard_sends_no_extra_read() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.edit_issue_adf_description("FC-1", {"type": "doc", "version": 1, "content": []})

    assert calls == ["PUT"]


def test_edit_issue_guard_blocks_stale_write_without_putting() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"fields": {"updated": "T1"}})
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    try:
        client.edit_issue("FC-1", summary="Updated", expect_updated="T0")
    except JiraConcurrentEditError:
        pass
    else:
        raise AssertionError("Expected JiraConcurrentEditError")

    assert calls == ["GET"]


def test_set_issue_type_puts_issuetype_name() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.set_issue_type("FC-1", "Bug") == {}
    assert seen["method"] == "PUT"
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-1"
    assert seen["json"] == {"fields": {"issuetype": {"name": "Bug"}}}


def test_set_issue_type_guard_blocks_stale_write_without_putting() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"fields": {"updated": "T1"}})
        return httpx.Response(204)

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    try:
        client.set_issue_type("FC-1", "Bug", expect_updated="T0")
    except JiraConcurrentEditError:
        pass
    else:
        raise AssertionError("Expected JiraConcurrentEditError")

    assert calls == ["GET"]


def test_get_edit_metadata_uses_editmeta_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"fields": {}})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.get_edit_metadata("FC-1")

    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-1/editmeta"


def test_is_issue_type_allowed_matches_allowed_values_case_insensitively() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "fields": {
                    "issuetype": {
                        "allowedValues": [
                            {"id": "10001", "name": "Story"},
                            {"id": "10002", "name": "Bug"},
                        ]
                    }
                }
            },
        )

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.is_issue_type_allowed("FC-1", "bug") is True
    assert client.is_issue_type_allowed("FC-1", "Epic") is False


# --- Comment delete, and update_comment's ADF rendering vs add_comment ----------------------


def test_delete_comment_uses_delete_verb_and_path() -> None:
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

    assert client.delete_comment("FC-1", "667040") == {}
    assert seen["method"] == "DELETE"
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/issue/FC-1/comment/667040"


def test_update_comment_renders_markdown_to_adf_the_same_way_as_add_comment() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode())["body"])
        return httpx.Response(200, json={"id": "1"})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    markdown = "**Bold** and a [link](https://example.com)\n\n- one\n- two"
    client.add_comment("FC-1", markdown)
    client.update_comment("FC-1", "1", markdown)

    assert bodies[0] == bodies[1] == text_to_adf(markdown)


# --- Comment paging and ordering ------------------------------------------------------------


def test_list_comments_passes_order_by_only_when_set() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"comments": [], "total": 0})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    client.list_comments("FC-1", limit=5)
    client.list_comments("FC-1", limit=5, order_by="-created")

    assert "orderBy" not in seen[0]
    assert seen[1].endswith("orderBy=-created")


def test_list_all_comments_pages_until_total_is_reached() -> None:
    pages = [
        {"comments": [{"id": "1"}, {"id": "2"}], "total": 5},
        {"comments": [{"id": "3"}, {"id": "4"}], "total": 5},
        {"comments": [{"id": "5"}], "total": 5},
    ]
    seen_start_at: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen_start_at.append(int(params["startAt"]))
        return httpx.Response(200, json=pages[len(seen_start_at) - 1])

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.list_all_comments("FC-1", page_size=2)

    assert [comment["id"] for comment in result["comments"]] == ["1", "2", "3", "4", "5"]
    assert result["total"] == 5
    assert seen_start_at == [0, 2, 4]


def test_list_all_comments_stops_on_an_empty_page() -> None:
    pages = [
        {"comments": [{"id": "1"}], "total": 5},
        {"comments": [], "total": 5},
    ]
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, json=pages[calls["count"]])
        calls["count"] += 1
        return response

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.list_all_comments("FC-1", page_size=1)

    assert [comment["id"] for comment in result["comments"]] == ["1"]
    assert result["total"] == 5
    assert calls["count"] == 2


def test_list_all_comments_keeps_the_first_pages_total_not_the_last() -> None:
    """F6 regression: the old code kept overwriting `total` with each page's own `total`, so the
    LAST page's value won, not the first's. A deletion between pages can make that last value
    smaller than the first: `comment_count: 90` alongside `comments_returned: 100` reads as
    `comments_truncated: false` while claiming fewer comments exist than were just returned.
    """
    pages = [
        {"comments": [{"id": str(i)} for i in range(1, 51)], "total": 100},
        {"comments": [{"id": str(i)} for i in range(51, 101)], "total": 90},
    ]
    calls = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, json=pages[calls["count"]])
        calls["count"] += 1
        return response

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.list_all_comments("FC-1", page_size=50)

    assert len(result["comments"]) == 100
    assert result["total"] == 100
    assert result["total"] >= len(result["comments"])


def test_list_all_comments_total_is_never_less_than_comments_returned() -> None:
    """F6 regression, the exact synthetic case from the finding: a page reporting `total: 0`
    alongside one actual comment used to report `total: 0` with `comments_returned: 1` — fewer
    comments claimed to exist than were just handed back in the same response.
    """
    pages = [{"comments": [{"id": "1"}], "total": 0}]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[0])

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.list_all_comments("FC-1", page_size=50)

    assert len(result["comments"]) == 1
    assert result["total"] == 1


# --- Search paging with no fabricated total, and the approximate-count endpoint -------------


def test_search_all_issues_follows_next_page_token_until_is_last() -> None:
    pages = [
        {"issues": [{"key": "FC-1"}, {"key": "FC-2"}], "isLast": False, "nextPageToken": "tok-1"},
        {"issues": [{"key": "FC-3"}, {"key": "FC-4"}], "isLast": False, "nextPageToken": "tok-2"},
        {"issues": [{"key": "FC-5"}], "isLast": True},
    ]
    seen_tokens: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen_tokens.append(params.get("nextPageToken"))
        return httpx.Response(200, json=pages[len(seen_tokens) - 1])

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.search_all_issues("project = FC ORDER BY created ASC", page_size=2)

    assert [issue["key"] for issue in result["issues"]] == ["FC-1", "FC-2", "FC-3", "FC-4", "FC-5"]
    assert result["returned"] == 5
    # This method's own result spells it `is_last`, snake_case like the rest of this tool's
    # result documents; only the raw Jira pages above keep Jira's own `isLast` casing.
    assert result["is_last"] is True
    assert "isLast" not in result
    assert "nextPageToken" not in result
    assert "total" not in result
    assert seen_tokens == [None, "tok-1", "tok-2"]


def test_search_all_issues_limit_truncates_and_keeps_next_page_token() -> None:
    pages = [
        {"issues": [{"key": "FC-1"}, {"key": "FC-2"}], "isLast": False, "nextPageToken": "tok-1"},
        {"issues": [{"key": "FC-3"}], "isLast": False, "nextPageToken": "tok-2"},
    ]
    seen_max_results: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen_max_results.append(int(params["maxResults"]))
        return httpx.Response(200, json=pages[len(seen_max_results) - 1])

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.search_all_issues("project = FC", limit=3, page_size=2)

    assert [issue["key"] for issue in result["issues"]] == ["FC-1", "FC-2", "FC-3"]
    assert result["returned"] == 3
    assert result["nextPageToken"] == "tok-2"
    assert result["is_last"] is False
    assert "isLast" not in result
    assert seen_max_results == [2, 1]
    assert "total" not in result


def test_search_all_issues_missing_is_last_stops_without_looping_and_reports_unknown() -> None:
    """R1 regression: a page with no `isLast` and no `nextPageToken` must not be read as complete.

    Before the fix, `is_last = bool(page.get("isLast", True))` defaulted the missing signal to
    `True`, which happened to terminate the loop correctly here but reported false completeness.
    A more dangerous variant of the same bug is covered by the next test: `if is_last: break` used
    to be the *only* termination check, and `None` is falsy, so a missing signal with no token to
    resume with would have looped again with `next_page_token=None`, re-fetching the same first
    page up to `_MAX_SEARCH_PAGES` times instead of stopping after one request.
    """
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"issues": [{"key": "FC-1"}]})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.search_all_issues("project = FC")

    assert request_count == 1
    assert [issue["key"] for issue in result["issues"]] == ["FC-1"]
    assert result["returned"] == 1
    assert result["is_last"] is None
    assert "isLast" not in result
    assert "nextPageToken" not in result
    assert "total" not in result


def test_search_all_issues_missing_is_last_with_token_keeps_paging_to_explicit_signal() -> None:
    """A missing `isLast` with a `nextPageToken` present still has something to resume with, so
    the walk keeps going (per the documented termination rule) until a page gives an explicit
    signal — here `isLast: True` on the second page.
    """
    pages = [
        {"issues": [{"key": "FC-1"}], "nextPageToken": "tok-1"},  # isLast absent
        {"issues": [{"key": "FC-2"}], "isLast": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        index = 0 if params.get("nextPageToken") is None else 1
        return httpx.Response(200, json=pages[index])

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    result = client.search_all_issues("project = FC", page_size=1)

    assert [issue["key"] for issue in result["issues"]] == ["FC-1", "FC-2"]
    assert result["is_last"] is True
    assert "nextPageToken" not in result


def test_approximate_issue_count_posts_jql_and_returns_raw_count() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"count": 812})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.approximate_issue_count("project = FC") == 812
    assert seen["method"] == "POST"
    assert seen["url"] == "https://example.atlassian.net/rest/api/3/search/approximate-count"
    assert seen["json"] == {"jql": "project = FC"}


# --- Agile API: boards, sprints, backlog -----------------------------------------------------


def test_list_boards_uses_agile_board_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"values": []})

    with JiraClient(make_config(), transport=httpx.MockTransport(handler)) as client:
        client.list_boards(board_type="scrum", name="FC board", max_results=25)

    assert seen["url"] == (
        "https://example.atlassian.net/rest/agile/1.0/board?maxResults=25&type=scrum&name=FC+board"
    )


def test_list_board_sprints_uses_agile_sprint_endpoint() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"values": []})

    with JiraClient(make_config(), transport=httpx.MockTransport(handler)) as client:
        client.list_board_sprints(814, state="active")

    assert seen["url"] == (
        "https://example.atlassian.net/rest/agile/1.0/board/814/sprint?maxResults=50&state=active"
    )


def test_get_issue_sprints_parses_the_sprint_field() -> None:
    seen: dict[str, str] = {}
    sprint = {
        "id": 7340,
        "name": "10 SEPA-Mandate, Disbursement",
        "state": "active",
        "boardId": 814,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"fields": {"customfield_10020": [sprint]}})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.get_issue_sprints("FC-1") == [sprint]
    assert seen["url"] == (
        "https://example.atlassian.net/rest/api/3/issue/FC-1?fields=customfield_10020"
    )


def test_get_issue_sprints_returns_empty_list_when_field_absent() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": {}})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.get_issue_sprints("FC-1") == []


def test_get_issue_sprints_returns_empty_list_when_field_is_null() -> None:
    """Verified fact: an issue in no sprint on this Jira site returns the sprint field PRESENT
    with the value `null` -- not `[]`, not absent -- confirmed via `jql=project = FC AND Sprint
    IS EMPTY`. This already passes against the current code (`isinstance(None, list)` is False,
    so it falls through to `[]` exactly like an absent field or an empty list), so this is
    recorded as a regression test rather than left unverified, per the hardening brief's own
    instruction to check what the current code does with `null` before assuming it needs a fix.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"fields": {"customfield_10020": None}})

    client = JiraClient(
        make_config(),
        http_client=httpx.Client(
            base_url="https://example.atlassian.net/rest/api/3",
            transport=httpx.MockTransport(handler),
        ),
    )

    assert client.get_issue_sprints("FC-1") == []


def test_is_in_sprint_matches_by_id_regardless_of_int_or_str() -> None:
    sprints = [{"id": 7340, "name": "10 SEPA-Mandate, Disbursement", "state": "active"}]

    assert is_in_sprint(sprints, 7340) is True
    assert is_in_sprint(sprints, "7340") is True
    assert is_in_sprint(sprints, 9001) is False
    assert is_in_sprint([], 7340) is False


def test_sprint_membership_report_reads_each_key_and_reports_membership() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def get_issue_sprints(self, issue_key, *, sprint_field):
            self.calls.append((issue_key, sprint_field))
            if issue_key == "FC-1":
                return [{"id": 7340, "name": "Sprint 7340", "state": "active"}]
            return []

    client = FakeClient()

    report = sprint_membership_report(
        client, ["FC-1", "FC-2"], 7340, sprint_field="customfield_10020"
    )

    assert report == [
        {
            "key": "FC-1",
            "sprint_id": 7340,
            "sprints": [{"id": 7340, "name": "Sprint 7340", "state": "active"}],
            "in_sprint": True,
        },
        {"key": "FC-2", "sprint_id": 7340, "sprints": [], "in_sprint": False},
    ]
    assert client.calls == [("FC-1", "customfield_10020"), ("FC-2", "customfield_10020")]


def test_sprint_verification_results_marks_verified_against_expected_polarity() -> None:
    report = [
        {"key": "FC-1", "sprint_id": 7340, "sprints": [{"id": 7340}], "in_sprint": True},
        {"key": "FC-2", "sprint_id": 7340, "sprints": [], "in_sprint": False},
    ]

    added = sprint_verification_results(report, expected_in_sprint=True)
    assert [entry["verified"] for entry in added] == [True, False]
    assert [entry["expected_in_sprint"] for entry in added] == [True, True]

    removed = sprint_verification_results(report, expected_in_sprint=False)
    assert [entry["verified"] for entry in removed] == [False, True]


def test_sprint_precondition_error_names_the_mismatched_keys() -> None:
    report = [
        {"key": "FC-1", "sprint_id": 7340, "sprints": [{"id": 7340}], "in_sprint": True},
        {"key": "FC-2", "sprint_id": 7340, "sprints": [], "in_sprint": False},
    ]

    error = SprintPreconditionError(7340, report)

    assert "FC-2" in str(error)
    assert "FC-1" not in str(error)
    assert error.sprint_id == 7340
    assert error.report == report


def test_add_issues_to_sprint_posts_issue_keys() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(204)

    with JiraClient(make_config(), transport=httpx.MockTransport(handler)) as client:
        assert client.add_issues_to_sprint(7340, ["FC-1", "FC-2"]) == {}

    assert seen["method"] == "POST"
    assert seen["url"] == "https://example.atlassian.net/rest/agile/1.0/sprint/7340/issue"
    assert seen["json"] == {"issues": ["FC-1", "FC-2"]}


def test_move_issues_to_backlog_posts_issue_keys() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content.decode())
        return httpx.Response(204)

    with JiraClient(make_config(), transport=httpx.MockTransport(handler)) as client:
        assert client.move_issues_to_backlog(["FC-1"]) == {}

    assert seen["method"] == "POST"
    assert seen["url"] == "https://example.atlassian.net/rest/agile/1.0/backlog/issue"
    assert seen["json"] == {"issues": ["FC-1"]}


def test_agile_calls_use_the_agile_base_and_still_honour_project_credential_overrides() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["Authorization"]))
        return httpx.Response(200, json={"values": []})

    with JiraClient(make_override_config(), transport=httpx.MockTransport(handler)) as client:
        client.list_boards()
        client.add_issues_to_sprint(1, ["FINRP-9"])
        client.get_issue("FC-1")

    assert seen[0] == (
        "https://example.atlassian.net/rest/agile/1.0/board?maxResults=50",
        basic_auth("user@example.com", "token"),
    )
    assert seen[1] == (
        "https://example.atlassian.net/rest/agile/1.0/sprint/1/issue",
        basic_auth("finrp@example.com", "finrp-token"),
    )
    assert seen[2] == (
        "https://example.atlassian.net/rest/api/3/issue/FC-1",
        basic_auth("user@example.com", "token"),
    )


def test_agile_call_uses_per_project_site_url_override() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"issues": []})

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
        client.add_issues_to_sprint(1, ["FINRP-9"])

    assert seen["url"] == "https://other.atlassian.net/rest/agile/1.0/sprint/1/issue"
