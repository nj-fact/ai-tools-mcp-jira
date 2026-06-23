from __future__ import annotations

from jira_agent.reporting import adf_to_text, build_agent_report, issue_url


def test_issue_url_uses_browse_url() -> None:
    assert issue_url("https://example.atlassian.net/", "FC-107") == (
        "https://example.atlassian.net/browse/FC-107"
    )


def test_adf_to_text_formats_paragraphs_and_hard_breaks() -> None:
    assert (
        adf_to_text(
            {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "hardBreak"},
                            {"type": "text", "text": "world"},
                        ],
                    },
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Second paragraph"}],
                    },
                ],
            }
        )
        == "Hello\nworld\n\nSecond paragraph"
    )


def test_build_agent_report_normalizes_issue_context() -> None:
    report = build_agent_report(
        site_url="https://example.atlassian.net",
        issue={
            "key": "FC-107",
            "fields": {
                "summary": "Test subject",
                "description": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Description"}],
                        }
                    ],
                },
                "status": {
                    "name": "Open",
                    "statusCategory": {"name": "To Do"},
                },
                "assignee": {"accountId": "a1", "displayName": "Norman Janert"},
                "reporter": {"accountId": "a1", "displayName": "Norman Janert"},
                "issuetype": {"name": "Task"},
                "project": {"key": "FC", "name": "First Cloud"},
                "labels": ["jira-agent"],
                "created": "2026-06-23T14:00:00.000+0200",
                "updated": "2026-06-23T14:30:00.000+0200",
            },
        },
        comments={
            "total": 1,
            "comments": [
                {
                    "id": "1",
                    "author": {"accountId": "a1", "displayName": "Norman Janert"},
                    "created": "2026-06-23T14:20:00.000+0200",
                    "body": {
                        "type": "doc",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "Comment"}],
                            }
                        ],
                    },
                }
            ],
        },
        transitions={"transitions": [{"id": "4", "name": "Start", "to": {"name": "Doing"}}]},
        write_scope="FC-107",
    )

    assert report["key"] == "FC-107"
    assert report["url"] == "https://example.atlassian.net/browse/FC-107"
    assert report["description"] == "Description"
    assert report["status"] == {"name": "Open", "category": "To Do"}
    assert report["assignee"]["displayName"] == "Norman Janert"
    assert report["comments"][0]["body"] == "Comment"
    assert report["transitions"] == [
        {"id": "4", "name": "Start", "to": {"id": None, "name": "Doing"}}
    ]
    assert report["write_scope"] == "FC-107"
