from __future__ import annotations

from jira_agent import mcp_server
from jira_agent.client import text_to_adf
from jira_agent.config import JiraConfig


def make_config(tmp_path) -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


def test_jira_create_issue_dry_run_reports_mermaid_diagrams() -> None:
    result = mcp_server.jira_create_issue(
        project_key="FC",
        issue_type_name="Task",
        summary="Diagram",
        description="```mermaid\nflowchart LR\nA --> B\n```",
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["mermaid_diagrams"] == 1


def test_jira_create_issue_embeds_mermaid_after_creation(tmp_path, monkeypatch) -> None:
    config = make_config(tmp_path)
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def create_issue(self, **kwargs):
            calls["create"] = kwargs
            return {"id": "10001", "key": "FC-123"}

    def fake_embed(client, **kwargs):
        calls["embed_client"] = client
        calls["embed"] = kwargs
        return {
            "ok": True,
            "diagram_count": 1,
            "files": [str(tmp_path / "fc-123-mermaid-1.png")],
            "attachment_ids": ["42"],
            "deleted_attachment_ids": [],
            "result": {"updated": True},
        }

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)
    monkeypatch.setattr(mcp_server, "embed_mermaid_diagrams", fake_embed)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    result = mcp_server.jira_create_issue(
        project_key="FC",
        issue_type_name="Task",
        summary="Diagram",
        description="```mermaid\nflowchart LR\nA --> B\n```",
        execute=True,
    )

    assert result["ok"] is True
    assert result["issue"]["key"] == "FC-123"
    assert result["mermaid"]["diagram_count"] == 1
    assert calls["embed"]["issue_key"] == "FC-123"
    assert calls["embed"]["description"] == text_to_adf("```mermaid\nflowchart LR\nA --> B\n```")


def test_jira_create_issue_can_disable_mermaid_embedding(tmp_path, monkeypatch) -> None:
    config = make_config(tmp_path)

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def create_issue(self, **_kwargs):
            return {"id": "10001", "key": "FC-123"}

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)
    monkeypatch.setattr(
        mcp_server,
        "embed_mermaid_diagrams",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not embed")),
    )
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    result = mcp_server.jira_create_issue(
        project_key="FC",
        issue_type_name="Task",
        summary="Diagram",
        description="```mermaid\nflowchart LR\nA --> B\n```",
        execute=True,
        render_mermaid=False,
    )

    assert result == {"ok": True, "issue": {"id": "10001", "key": "FC-123"}}


def _heading(level: int, text: str) -> dict[str, object]:
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": text}],
    }


def _paragraph(text: str) -> dict[str, object]:
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def description_document(*, duplicate_heading: bool = False) -> dict[str, object]:
    """Indices: 0 h1, 1 h2 Observed, 2 paragraph, 3 h2 Acceptance criteria, 4 taskList."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            _heading(1, "Description wipe"),
            _heading(2, "Observed"),
            _paragraph("The payload replaced the whole field."),
            _heading(2, "Observed" if duplicate_heading else "Acceptance criteria"),
            {
                "type": "taskList",
                "attrs": {"localId": "task-list-1"},
                "content": [
                    {
                        "type": "taskItem",
                        "attrs": {"localId": "task-1", "state": "TODO"},
                        "content": [{"type": "text", "text": "Keep every other section"}],
                    }
                ],
            },
        ],
    }


class PatchFakeClient:
    """A Jira client that records writes and fails the test if a dry run writes."""

    def __init__(self, _config, *, duplicate_heading: bool = False) -> None:
        self.duplicate_heading = duplicate_heading
        self.writes: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key: str, *, fields=None, expand=None) -> dict[str, object]:
        return {
            "key": issue_key,
            "fields": {
                "project": {"key": "FC"},
                "description": description_document(duplicate_heading=self.duplicate_heading),
            },
        }

    def edit_issue_adf_description(self, issue_key: str, description) -> dict[str, object]:
        self.writes.append({"issue": issue_key, "description": description})
        return {"updated": True}


def _install_patch_client(monkeypatch, tmp_path, **kwargs) -> PatchFakeClient:
    config = make_config(tmp_path)
    client = PatchFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_patch_description_section_dry_run_reports_counts_without_writing(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Acceptance criteria",
        content="## Acceptance criteria\n\n- [ ] Only the section changes",
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert (result["heading"], result["level"]) == ("Acceptance criteria", 2)
    assert (result["start_index"], result["end_index"]) == (3, 5)
    assert (result["preserved_before"], result["preserved_after"]) == (3, 0)
    assert (result["original_node_count"], result["patched_node_count"]) == (5, 5)
    assert [node["type"] for node in result["new_nodes"]] == ["heading", "taskList"]
    assert "document" not in result
    assert client.writes == []


def test_jira_patch_description_section_writes_only_the_matched_section(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)
    reference = description_document()

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nThe dry run showed a one-bullet payload.",
        execute=True,
    )

    assert result["ok"] is True
    assert result["replaced_node_count"] == 2
    written = client.writes[0]["description"]
    assert written["content"][0] == reference["content"][0]
    assert written["content"][3:] == reference["content"][3:]
    assert written["content"][4]["content"][0]["attrs"] == {"localId": "task-1", "state": "TODO"}


def test_jira_patch_description_section_refuses_an_unknown_heading_without_writing(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Reproduction",
        content="## Reproduction\n\nRun the dry run.",
        execute=True,
    )

    assert result["ok"] is False
    assert result["message"] == (
        "Description section not found: Reproduction. Existing headings: "
        "Description wipe, Observed, Acceptance criteria"
    )
    assert client.writes == []


def test_jira_patch_description_section_refuses_an_ambiguous_heading_without_writing(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path, duplicate_heading=True)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nRewritten.",
        execute=True,
    )

    assert result == {"ok": False, "message": "Description section is ambiguous: Observed"}
    assert client.writes == []


def test_jira_patch_description_section_delete_removes_the_slice_and_rejects_content(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    rejected = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Acceptance criteria",
        mode="delete",
        content="## Acceptance criteria\n\nX",
        execute=True,
    )
    assert rejected == {
        "ok": False,
        "message": "Mode delete removes the section and takes no content.",
    }
    assert client.writes == []

    deleted = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Acceptance criteria",
        mode="delete",
        execute=True,
    )

    assert deleted["ok"] is True
    assert deleted["patched_node_count"] == 3
    assert client.writes[0]["description"]["content"] == description_document()["content"][:3]


class LinkParentFakeClient:
    def __init__(
        self, _config, *, duplicate_link: bool = False, verify_delete: bool = True
    ) -> None:
        self.deleted_link_ids: list[str] = []
        self.clear_parent_calls = 0
        self.parent = {"key": "FC-300"}
        self.verify_delete = verify_delete
        self.outward_links = [
            {
                "id": "10001",
                "type": {"name": "Relates"},
                "outwardIssue": {"key": "FC-356"},
            }
        ]
        if duplicate_link:
            self.outward_links.append(
                {
                    "id": "10002",
                    "type": {"name": "Relates"},
                    "outwardIssue": {"key": "FC-356"},
                }
            )
        self.inward_links = [
            {
                "id": "10001",
                "type": {"name": "Relates"},
                "inwardIssue": {"key": "FC-354"},
            }
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        if fields == ["parent"]:
            return {"key": issue_key, "fields": {"parent": self.parent}}
        if fields == ["issuelinks"]:
            links = self.outward_links if issue_key == "FC-354" else self.inward_links
            if self.deleted_link_ids and self.verify_delete:
                links = [link for link in links if link["id"] not in self.deleted_link_ids]
            return {"key": issue_key, "fields": {"issuelinks": links}}
        raise AssertionError(f"Unexpected fields: {fields}")

    def edit_issue(self, issue_key, *, clear_parent=False, **_kwargs):
        assert issue_key == "FC-359"
        assert clear_parent is True
        self.clear_parent_calls += 1
        self.parent = None
        return {}

    def delete_issue_link(self, link_id, *, issue_key):
        assert issue_key == "FC-354"
        self.deleted_link_ids.append(link_id)
        return {}


def _install_link_parent_client(monkeypatch, tmp_path, **kwargs) -> LinkParentFakeClient:
    config = make_config(tmp_path)
    client = LinkParentFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_delete_issue_link_dry_run_resolves_but_does_not_delete(tmp_path, monkeypatch) -> None:
    client = _install_link_parent_client(monkeypatch, tmp_path)

    result = mcp_server.jira_delete_issue_link("FC-354", "FC-356")

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["resolved_link"]["id"] == "10001"
    assert client.deleted_link_ids == []


def test_jira_delete_issue_link_execute_deletes_and_verifies_both_endpoints(
    tmp_path, monkeypatch
) -> None:
    client = _install_link_parent_client(monkeypatch, tmp_path)

    result = mcp_server.jira_delete_issue_link("FC-354", "FC-356", execute=True)

    assert result == {
        "ok": True,
        "resolved_link": {
            "id": "10001",
            "type": "Relates",
            "outward_key": "FC-354",
            "inward_key": "FC-356",
        },
        "still_present": {"FC-354": False, "FC-356": False},
    }
    assert client.deleted_link_ids == ["10001"]


def test_jira_delete_issue_link_refuses_ambiguous_or_mismatched_target_without_delete(
    tmp_path, monkeypatch
) -> None:
    client = _install_link_parent_client(monkeypatch, tmp_path, duplicate_link=True)

    ambiguous = mcp_server.jira_delete_issue_link("FC-354", "FC-356", execute=True)
    mismatch = mcp_server.jira_delete_issue_link(
        "FC-354", "FC-356", link_id="10002", execute=True
    )

    assert ambiguous["ok"] is False
    assert "Expected exactly one directed" in ambiguous["message"]
    assert mismatch["ok"] is False
    assert "Expected exactly one directed" in mismatch["message"]
    assert client.deleted_link_ids == []


def test_jira_delete_issue_link_refuses_success_when_the_reread_still_has_the_link(
    tmp_path, monkeypatch
) -> None:
    client = _install_link_parent_client(monkeypatch, tmp_path, verify_delete=False)

    result = mcp_server.jira_delete_issue_link("FC-354", "FC-356", execute=True)

    assert result["ok"] is False
    assert result["still_present"] == {"FC-354": True, "FC-356": True}
    assert client.deleted_link_ids == ["10001"]


def test_jira_clear_issue_parent_is_explicit_dry_run_then_verified_execute(
    tmp_path, monkeypatch
) -> None:
    client = _install_link_parent_client(monkeypatch, tmp_path)

    dry_run = mcp_server.jira_clear_issue_parent("FC-359")
    executed = mcp_server.jira_clear_issue_parent("FC-359", execute=True)

    assert dry_run["dry_run"] is True
    assert dry_run["before"] == {"key": "FC-300"}
    assert dry_run["payload"] == {"fields": {"parent": None}}
    assert client.clear_parent_calls == 1
    assert executed == {"ok": True, "issue": "FC-359", "verified_parent": None}


def test_jira_patch_description_section_refuses_an_unnamed_swallowed_section(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Description wipe",
        content="# Description wipe\n\n- Only this bullet survives",
        execute=True,
    )

    assert result["ok"] is False
    assert "Swallowed-section mismatch" in result["message"]
    assert 'the slice removes ["Observed", "Acceptance criteria"]' in result["message"]
    assert result["error_type"] == "section_patch_refused"
    assert result["swallowed_headings"] == ["Observed", "Acceptance criteria"]
    assert result["expect_swallowed_headings"] == []
    assert result["whole_document"] is True
    assert "Nothing outside the slice would survive" in result["message"]
    assert "allow_wide_rewrite" in result["message"]
    assert client.writes == []


def test_jira_patch_description_section_writes_a_named_wide_rewrite_when_asked(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Description wipe",
        content="# Description wipe\n\n- Only this bullet survives",
        expect_swallowed_headings=["Observed", "Acceptance criteria"],
        allow_wide_rewrite=True,
        execute=True,
    )

    assert result["ok"] is True
    assert result["whole_document"] is True
    assert result["swallowed_headings"] == ["Observed", "Acceptance criteria"]
    assert (result["preserved_before"], result["preserved_after"]) == (0, 0)
    assert len(client.writes[0]["description"]["content"]) == 2


def test_jira_patch_description_section_dry_run_reports_the_swallowed_sections(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nRewritten.",
    )

    assert result["whole_document"] is False
    assert result["swallowed_headings"] == []
    assert client.writes == []
