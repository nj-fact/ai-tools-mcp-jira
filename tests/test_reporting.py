from __future__ import annotations

from jira_agent.reporting import adf_to_text, agent_plan_comment_body, build_agent_report, issue_url


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


def _report_comment(n: int) -> dict[str, object]:
    return {
        "id": str(n),
        "author": {"displayName": "Norman Janert"},
        "created": f"2026-01-{n:02d}T00:00:00.000+0000",
        "body": {
            "type": "doc",
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": f"c{n}"}]}],
        },
    }


def _report_issue() -> dict[str, object]:
    return {
        "key": "FC-1",
        "fields": {
            "summary": "Subject",
            "status": {"name": "Open"},
            "issuetype": {"name": "Task"},
            "project": {"key": "FC", "name": "First Cloud"},
        },
    }


def test_build_agent_report_flags_truncation_on_partial_comment_page() -> None:
    report = build_agent_report(
        site_url="https://example.atlassian.net",
        issue=_report_issue(),
        comments={"total": 18, "comments": [_report_comment(n) for n in range(1, 6)]},
        transitions={"transitions": []},
    )

    assert report["comment_count"] == 18
    assert report["comments_returned"] == 5
    assert report["comments_truncated"] is True


def test_build_agent_report_is_not_truncated_when_every_comment_is_present() -> None:
    report = build_agent_report(
        site_url="https://example.atlassian.net",
        issue=_report_issue(),
        comments={"total": 18, "comments": [_report_comment(n) for n in range(1, 19)]},
        transitions={"transitions": []},
    )

    assert report["comment_count"] == 18
    assert report["comments_returned"] == 18
    assert report["comments_truncated"] is False


def test_fetch_agent_report_int_limit_fetches_newest_first() -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        def get_issue(self, _issue_key: str, *, fields: list[str]) -> dict[str, object]:
            return _report_issue()

        def list_comments(
            self, _issue_key: str, *, limit: int, order_by: str | None = None
        ) -> dict[str, object]:
            calls["limit"] = limit
            calls["order_by"] = order_by
            return {"comments": [], "total": 18}

        def get_transitions(self, _issue_key: str) -> dict[str, object]:
            return {"transitions": []}

    from jira_agent.reporting import fetch_agent_report

    report = fetch_agent_report(
        FakeClient(),
        issue_key="FC-1",
        site_url="https://example.atlassian.net",
        comment_limit=5,
    )

    assert calls == {"limit": 5, "order_by": "-created"}
    assert report["comment_count"] == 18
    assert report["comments_returned"] == 0
    assert report["comments_truncated"] is True


def test_fetch_agent_report_all_pages_through_list_all_comments() -> None:
    calls: dict[str, object] = {}

    class FakeClient:
        def get_issue(self, _issue_key: str, *, fields: list[str]) -> dict[str, object]:
            return _report_issue()

        def list_all_comments(self, issue_key: str) -> dict[str, object]:
            calls["issue_key"] = issue_key
            return {"comments": [_report_comment(n) for n in range(1, 19)], "total": 18}

        def get_transitions(self, _issue_key: str) -> dict[str, object]:
            return {"transitions": []}

    from jira_agent.reporting import fetch_agent_report

    report = fetch_agent_report(
        FakeClient(),
        issue_key="FC-1",
        site_url="https://example.atlassian.net",
        comment_limit="all",
    )

    assert calls == {"issue_key": "FC-1"}
    assert report["comment_count"] == 18
    assert report["comments_returned"] == 18
    assert report["comments_truncated"] is False


def test_build_agent_report_can_include_changelog_via_fetch_mock() -> None:
    class FakeClient:
        def get_issue(self, _issue_key: str, *, fields: list[str]) -> dict[str, object]:
            return {
                "key": "FC-107",
                "fields": {
                    "summary": "Test",
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Task"},
                    "project": {"key": "FC", "name": "First Cloud"},
                },
            }

        def list_comments(
            self, _issue_key: str, *, limit: int, order_by: str | None = None
        ) -> dict[str, object]:
            return {"comments": [], "total": 0}

        def get_transitions(self, _issue_key: str) -> dict[str, object]:
            return {"transitions": []}

        def get_changelog(self, _issue_key: str, *, limit: int) -> dict[str, object]:
            return {
                "values": [
                    {
                        "id": "1",
                        "created": "2026-06-23T15:00:00.000+0200",
                        "author": {"displayName": "Norman Janert"},
                        "items": [{"field": "labels", "fromString": "", "toString": "jira-agent"}],
                    }
                ]
            }

    from jira_agent.reporting import fetch_agent_report

    report = fetch_agent_report(
        FakeClient(),
        issue_key="FC-107",
        site_url="https://example.atlassian.net",
        changelog_limit=1,
    )

    assert report["changelog"] == [
        {
            "id": "1",
            "created": "2026-06-23T15:00:00.000+0200",
            "author": {
                "accountId": None,
                "displayName": "Norman Janert",
                "emailAddress": None,
                "active": None,
            },
            "items": [{"field": "labels", "from": "", "to": "jira-agent"}],
        }
    ]


def test_agent_plan_comment_body_contains_context_and_options() -> None:
    body = agent_plan_comment_body(
        {
            "key": "FC-107",
            "summary": "Test subject",
            "status": {"name": "Open"},
            "assignee": {"displayName": "Norman Janert"},
        },
        plan="Check implementation.",
        next_step="Run tests.",
        risk="None.",
        result="Pending.",
    )

    assert "Agent plan update for FC-107" in body
    assert "Summary: Test subject" in body
    assert "Status: Open" in body
    assert "Plan: Check implementation." in body
    assert "Next step: Run tests." in body
    assert "Risk: None." in body
    assert "Result: Pending." in body
