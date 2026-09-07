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
