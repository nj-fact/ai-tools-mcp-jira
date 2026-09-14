from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from jira_agent import mcp_server
from jira_agent.client import (
    JiraApiError,
    JiraConcurrentEditError,
    JiraMissingConcurrencyTokenError,
    text_to_adf,
)
from jira_agent.config import JiraConfig

# A minimal valid PNG header (signature + IHDR), reused from tests/test_client.py's own
# png_dimensions fixture: decodes to (774, 70).
_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x03\x06\x00\x00\x00F\x08\x02\x00\x00\x00"


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
    """A Jira client that records writes and fails the test if a dry run writes.

    `updated` is what get_issue reports having read; `server_updated` is what the write's guard
    is checked against. They default equal (no race). A test sets `server_updated` to a
    different value to simulate an edit landing between the read and the write.
    """

    def __init__(
        self,
        _config,
        *,
        duplicate_heading: bool = False,
        updated: str = "2026-09-01T00:00:00.000+0000",
        server_updated: str | None = None,
        omit_updated: bool = False,
    ) -> None:
        self.duplicate_heading = duplicate_heading
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated
        self.omit_updated = omit_updated
        self.writes: list[dict[str, object]] = []
        self.expect_updated_calls: list[str | None] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key: str, *, fields=None, expand=None) -> dict[str, object]:
        fields_dict: dict[str, object] = {
            "project": {"key": "FC"},
            "description": description_document(duplicate_heading=self.duplicate_heading),
        }
        if not self.omit_updated:
            fields_dict["updated"] = self.updated
        return {"key": issue_key, "fields": fields_dict}

    def edit_issue_adf_description(
        self, issue_key: str, description, *, expect_updated: str | None = None
    ) -> dict[str, object]:
        self.expect_updated_calls.append(expect_updated)
        if expect_updated is not None and expect_updated != self.server_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.server_updated!r}. Re-read the issue and "
                "retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.server_updated,
            )
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


class LinkCreateFakeClient:
    def __init__(self, _config) -> None:
        self.created: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        assert fields == ["project"]
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def link_issues(self, **kwargs):
        self.created.append(kwargs)
        return {}


def test_jira_link_issue_keeps_fc_350_blocks_fc_352_business_direction(
    tmp_path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    client = LinkCreateFakeClient(config)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    dry_run = mcp_server.jira_link_issue("FC-350", "FC-352", link_type="Blocks")
    executed = mcp_server.jira_link_issue("FC-350", "FC-352", link_type="Blocks", execute=True)

    assert dry_run["payload"] == {
        "type": {"name": "Blocks"},
        "inwardIssue": {"key": "FC-350"},
        "outwardIssue": {"key": "FC-352"},
    }
    assert client.created == [
        {
            "outward_issue_key": "FC-350",
            "inward_issue_key": "FC-352",
            "link_type": "Blocks",
            "comment": None,
        }
    ]
    assert executed == {"ok": True, "outward_key": "FC-350", "inward_key": "FC-352"}


class ExistingPngFakeClient:
    def __init__(self, _config, *, updated="2026-01-01T00:00:00.000+0000", server_updated=None):
        self.description = {"type": "doc", "version": 1, "content": []}
        self.edits: list[dict[str, object]] = []
        self.expect_updated_calls: list[str | None] = []
        self.attachment = {
            "id": "170501",
            "filename": "fc349-disbursement-flow.png",
            "mimeType": "image/png",
        }
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        # The initial read includes "updated" (the guard token); the post-write verification
        # reread does not need it.
        assert fields is not None and "description" in fields and "attachment" in fields, fields
        return {
            "key": issue_key,
            "fields": {
                "description": self.description,
                "attachment": [self.attachment],
                "updated": self.updated,
            },
        }

    def attachment_media_id(self, attachment_id, *, issue_key):
        assert attachment_id == "170501"
        assert issue_key == "FC-349"
        return "media-fc349"

    def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
        assert issue_key == "FC-349"
        self.expect_updated_calls.append(expect_updated)
        if expect_updated is not None and expect_updated != self.server_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.server_updated!r}. Re-read the issue and "
                "retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.server_updated,
            )
        self.edits.append(description)
        self.description = description
        return {}


def test_jira_embed_existing_png_attachment_dry_run_then_reuses_existing_attachment(
    tmp_path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    client = ExistingPngFakeClient(config)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    dry_run = mcp_server.jira_embed_existing_png_attachment(
        "fc349-disbursement-flow.png", key="FC-349", width=1600, height=900
    )
    executed = mcp_server.jira_embed_existing_png_attachment(
        "fc349-disbursement-flow.png", key="FC-349", width=1600, height=900, execute=True
    )

    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert dry_run["payload"]["attachment_id"] == "170501"
    assert dry_run["payload"]["media_id"] == "media-fc349"
    assert client.edits == [
        {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "mediaSingle",
                    "attrs": {"layout": "center"},
                    "content": [
                        {
                            "type": "media",
                            "attrs": {
                                "id": "media-fc349",
                                "type": "file",
                                "collection": "",
                                "alt": "Architecture diagram",
                                "width": 1600,
                                "height": 900,
                            },
                        }
                    ],
                }
            ],
        }
    ]
    assert executed["ok"] is True
    assert executed["media_id"] == "media-fc349"


def test_jira_embed_existing_png_attachment_refuses_on_concurrent_edit(
    tmp_path, monkeypatch
) -> None:
    """G2 regression: this tool had no guard at all before this fix, so a concurrent edit
    landing between the read and the write was silently reverted."""
    config = make_config(tmp_path)
    client = ExistingPngFakeClient(
        config,
        updated="2026-01-01T00:00:00.000+0000",
        server_updated="2026-01-01T00:05:00.000+0000",
    )
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    result = mcp_server.jira_embed_existing_png_attachment(
        "fc349-disbursement-flow.png", key="FC-349", width=1600, height=900, execute=True
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.edits == []


def test_jira_embed_existing_png_attachment_refuses_when_updated_missing(
    tmp_path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    client = ExistingPngFakeClient(config, updated=None)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    result = mcp_server.jira_embed_existing_png_attachment(
        "fc349-disbursement-flow.png", key="FC-349", width=1600, height=900, execute=True
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.edits == []
    assert client.expect_updated_calls == []


def test_jira_embed_existing_png_attachment_guard_false_ignores_missing_updated(
    tmp_path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    client = ExistingPngFakeClient(config, updated=None)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    result = mcp_server.jira_embed_existing_png_attachment(
        "fc349-disbursement-flow.png",
        key="FC-349",
        width=1600,
        height=900,
        guard_concurrent_edit=False,
        execute=True,
    )

    assert result["ok"] is True
    assert client.expect_updated_calls == [None]


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


def test_jira_patch_description_section_dry_run_reports_expect_updated(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nRewritten.",
    )

    assert result["dry_run"] is True
    assert result["expect_updated"] == client.updated
    assert client.writes == []


def test_jira_patch_description_section_execute_passes_the_same_read_updated_as_guard(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nGuarded write.",
        execute=True,
    )

    assert result["ok"] is True
    assert client.expect_updated_calls == [client.updated]


def test_jira_patch_description_section_refuses_a_concurrent_edit(tmp_path, monkeypatch) -> None:
    client = _install_patch_client(
        monkeypatch,
        tmp_path,
        updated="2026-09-01T00:00:00.000+0000",
        server_updated="2026-09-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nGuarded write.",
        execute=True,
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert result["issue"] == "FC-385"
    assert result["expected_updated"] == "2026-09-01T00:00:00.000+0000"
    assert result["found_updated"] == "2026-09-01T00:05:00.000+0000"
    assert "Re-read the issue and retry" in result["message"]
    assert client.writes == []


def test_jira_patch_description_section_guard_concurrent_edit_false_skips_the_guard(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(
        monkeypatch,
        tmp_path,
        updated="2026-09-01T00:00:00.000+0000",
        server_updated="2026-09-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nUnguarded write.",
        guard_concurrent_edit=False,
        execute=True,
    )

    assert result["ok"] is True
    assert client.expect_updated_calls == [None]
    assert len(client.writes) == 1


def test_jira_patch_description_section_refuses_when_guard_requested_but_updated_is_missing(
    tmp_path, monkeypatch
) -> None:
    """F2-adjacent: the CLI's patch-description-section has this same defect (a same-request
    read that silently passes expect_updated=None when Jira's response omits `updated`, leaving
    the write completely unguarded while the caller believes it is protected). This must refuse
    instead, and must not issue a second read to try to recover a token.

    The message now comes from the shared `JiraMissingConcurrencyTokenError`, the same one
    `require_concurrency_token` raises for every other guarded command's read, rather than a
    copy this tool used to build inline -- see G9: one condition, one outcome, on both surfaces.
    """
    client = _install_patch_client(monkeypatch, tmp_path, omit_updated=True)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nGuarded write.",
        execute=True,
    )

    assert result == {
        "ok": False,
        "error_type": "concurrent_edit_guard_unavailable",
        "issue": "FC-385",
        "message": result["message"],
    }
    assert "usable" in result["message"] and "updated" in result["message"]
    assert "guard" in result["message"]
    assert client.writes == []
    assert client.expect_updated_calls == []


def test_jira_patch_description_section_refuses_at_dry_run_too_when_guard_unavailable(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path, omit_updated=True)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nGuarded write.",
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.writes == []


def test_jira_patch_description_section_refuses_when_updated_is_empty_string(
    tmp_path, monkeypatch
) -> None:
    """G9 regression: `updated` can be present but an empty string, not merely absent. The old
    inline check only tested `read_updated is None`, so `updated: ""` slipped through as
    `expect_updated=""`, which then compared unequal to whatever the write's guard found and
    raised a misleading "concurrent_edit" refusal (blaming a concurrent editor that does not
    exist) instead of "concurrent_edit_guard_unavailable". require_concurrency_token treats an
    empty string the same as a missing one; this must produce the same guard-unavailable
    refusal `omit_updated=True` does, not a phantom concurrent-edit refusal.
    """
    client = _install_patch_client(
        monkeypatch, tmp_path, updated="", server_updated="2026-09-01T00:00:00.000+0000"
    )

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nGuarded write.",
        execute=True,
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.writes == []
    assert client.expect_updated_calls == []


def test_jira_patch_description_section_guard_false_ignores_missing_updated(
    tmp_path, monkeypatch
) -> None:
    client = _install_patch_client(monkeypatch, tmp_path, omit_updated=True)

    result = mcp_server.jira_patch_description_section(
        key="FC-385",
        section="Observed",
        content="## Observed\n\nUnguarded write.",
        guard_concurrent_edit=False,
        execute=True,
    )

    assert result["ok"] is True
    assert client.expect_updated_calls == [None]
    assert len(client.writes) == 1


class EditIssueFakeClient:
    """A Jira client for jira_edit_issue's concurrency guard, mirroring PatchFakeClient."""

    def __init__(
        self,
        _config,
        *,
        updated: str = "2026-09-01T00:00:00.000+0000",
        server_updated: str | None = None,
    ) -> None:
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated
        self.edits: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        assert fields == ["project"]
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def issue_version(self, issue_key):
        return self.updated

    def edit_issue(self, issue_key, *, expect_updated=None, **kwargs):
        if expect_updated is not None and expect_updated != self.server_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.server_updated!r}. Re-read the issue and "
                "retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.server_updated,
            )
        self.edits.append({"issue": issue_key, "expect_updated": expect_updated, **kwargs})
        return {}


def _install_edit_issue_client(monkeypatch, tmp_path, **kwargs) -> EditIssueFakeClient:
    config = make_config(tmp_path)
    client = EditIssueFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_edit_issue_dry_run_reports_expect_updated_without_writing(
    tmp_path, monkeypatch
) -> None:
    client = _install_edit_issue_client(monkeypatch, tmp_path)

    result = mcp_server.jira_edit_issue(key="FC-400", summary="New summary")

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["expect_updated"] == client.updated
    assert client.edits == []


def test_jira_edit_issue_execute_passes_the_read_updated_as_guard(tmp_path, monkeypatch) -> None:
    client = _install_edit_issue_client(monkeypatch, tmp_path)

    result = mcp_server.jira_edit_issue(key="FC-400", summary="New summary", execute=True)

    assert result == {"ok": True, "issue": "FC-400"}
    assert client.edits[0]["expect_updated"] == client.updated


def test_jira_edit_issue_refuses_a_concurrent_edit(tmp_path, monkeypatch) -> None:
    client = _install_edit_issue_client(
        monkeypatch,
        tmp_path,
        updated="2026-09-01T00:00:00.000+0000",
        server_updated="2026-09-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_edit_issue(key="FC-400", summary="New summary", execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert result["issue"] == "FC-400"
    assert result["expected_updated"] == "2026-09-01T00:00:00.000+0000"
    assert result["found_updated"] == "2026-09-01T00:05:00.000+0000"
    assert client.edits == []


def test_jira_edit_issue_guard_concurrent_edit_false_skips_the_guard(tmp_path, monkeypatch) -> None:
    client = _install_edit_issue_client(
        monkeypatch,
        tmp_path,
        updated="2026-09-01T00:00:00.000+0000",
        server_updated="2026-09-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_edit_issue(
        key="FC-400", summary="New summary", guard_concurrent_edit=False, execute=True
    )

    assert result == {"ok": True, "issue": "FC-400"}
    assert client.edits[0]["expect_updated"] is None


def test_jira_agent_report_accepts_all_as_comment_limit_and_returns_true_counts(
    tmp_path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_fetch_agent_report(client, **kwargs):
        calls["kwargs"] = kwargs
        return {
            "key": "FC-406",
            "comment_count": 18,
            "comments_returned": 18,
            "comments_truncated": False,
        }

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)
    monkeypatch.setattr(mcp_server, "fetch_agent_report", fake_fetch_agent_report)

    result = mcp_server.jira_agent_report(key="FC-406", comments_limit="all")

    assert result["ok"] is True
    assert calls["kwargs"]["comment_limit"] == "all"
    assert result["report"]["comment_count"] == 18
    assert result["report"]["comments_returned"] == 18
    assert result["report"]["comments_truncated"] is False


def test_jira_agent_report_refuses_a_negative_comments_limit(tmp_path, monkeypatch) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(
        mcp_server,
        "fetch_agent_report",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not reach Jira")),
    )

    result = mcp_server.jira_agent_report(key="FC-406", comments_limit=-1)

    assert result == {
        "ok": False,
        "message": "comments_limit must be a non-negative integer or 'all', got -1.",
    }


def test_jira_agent_report_docstring_claims_newest_first_for_both_paths() -> None:
    """F1: the docstring is what a caller reads to decide how to interpret the comment list."""
    doc = mcp_server.jira_agent_report.__doc__ or ""
    assert "newest first" in doc.lower()
    assert "comments_order" in doc


def test_jira_agent_report_all_path_returns_comments_newest_first(tmp_path, monkeypatch) -> None:
    config = make_config(tmp_path)

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, *, fields=None, expand=None):
            return {
                "key": issue_key,
                "fields": {
                    "summary": "S",
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Task"},
                    "project": {"key": "FC"},
                },
            }

        def list_all_comments(self, issue_key, *, order_by=None):
            # A faithful fake of the real client: without an explicit order_by, Jira's own
            # order is oldest-first, so a fake that ignores order_by reproduces that honestly.
            oldest_first = [
                {"id": "1", "created": "2026-01-01T00:00:00.000+0000"},
                {"id": "2", "created": "2026-01-02T00:00:00.000+0000"},
            ]
            comments = list(reversed(oldest_first)) if order_by == "-created" else oldest_first
            return {"comments": comments, "total": 2}

        def get_transitions(self, issue_key):
            return {"transitions": []}

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)

    result = mcp_server.jira_agent_report(key="FC-1", comments_limit="all")

    assert result["ok"] is True
    ids = [comment["id"] for comment in result["report"]["comments"]]
    assert ids == ["2", "1"], "the 'all' path must return comments newest-first"
    assert result["report"].get("comments_order") == "newest-first"


def test_jira_get_issue_default_fields_include_parent_and_the_configured_sprint_field(
    tmp_path, monkeypatch
) -> None:
    """R2: `parent` lives in the one shared `DEFAULT_FIELDS` (see `client.DEFAULT_FIELDS`), not
    bolted on separately for `jira_get_issue` alone, so this asserts against the shared list
    directly rather than `[*DEFAULT_FIELDS, "parent"]` — asserting the old shape here would
    require `parent` twice under the single-list contract.

    Sprint hardening item 4: the configured sprint field is appended on top of `DEFAULT_FIELDS`
    for this default-fields case, resolved from config at call time (`_sprint_field`) rather than
    folded into the shared `DEFAULT_FIELDS` constant itself -- see `_fields_or_default`.
    """
    config = make_config(tmp_path)
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, *, fields=None, expand=None):
            calls["fields"] = fields
            return {"key": issue_key, "fields": {}}

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)

    result = mcp_server.jira_get_issue("FC-401")

    assert result["ok"] is True
    assert calls["fields"] == [*mcp_server.DEFAULT_FIELDS, "customfield_10020"]
    assert "parent" in calls["fields"]


def test_jira_get_issue_explicit_fields_are_not_augmented_with_parent(
    tmp_path, monkeypatch
) -> None:
    config = make_config(tmp_path)
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, *, fields=None, expand=None):
            calls["fields"] = fields
            return {"key": issue_key, "fields": {}}

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)

    mcp_server.jira_get_issue("FC-401", fields=["summary"])

    assert calls["fields"] == ["summary"]


class SearchFakeClient:
    def __init__(self, _config) -> None:
        self.search_issues_calls: list[dict[str, object]] = []
        self.search_all_issues_calls: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def search_issues(self, jql, *, limit, fields, next_page_token=None, project_key=None):
        self.search_issues_calls.append(
            {
                "jql": jql,
                "limit": limit,
                "fields": fields,
                "next_page_token": next_page_token,
                "project_key": project_key,
            }
        )
        # This is the raw Jira response shape, so it keeps Jira's own "isLast" spelling.
        return {"issues": [{"key": "FC-1"}, {"key": "FC-2"}], "isLast": True}

    def search_all_issues(self, jql, *, limit=None, fields=None, page_size=100, project_key=None):
        self.search_all_issues_calls.append(
            {
                "jql": jql,
                "limit": limit,
                "fields": fields,
                "page_size": page_size,
                "project_key": project_key,
            }
        )
        # Simulates today's `client.search_all_issues`, which still returns `isLast`.
        return {
            "issues": [{"key": f"FC-{n}"} for n in range(1, 151)],
            "returned": 150,
            "isLast": False,
            "nextPageToken": "tok-2",
        }


def _install_search_client(monkeypatch, tmp_path) -> SearchFakeClient:
    config = make_config(tmp_path)
    client = SearchFakeClient(config)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    return client


def test_jira_search_issues_never_invents_a_total_and_reports_returned_and_is_last(
    tmp_path, monkeypatch
) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_search_issues("project = FC", limit=20)

    assert result["ok"] is True
    assert "total" not in result["result"]
    assert "isLast" not in result["result"]
    assert result["result"]["returned"] == 2
    assert result["result"]["is_last"] is True
    assert client.search_issues_calls[0]["limit"] == 20
    assert client.search_all_issues_calls == []


def test_jira_search_issues_default_fields_include_the_configured_sprint_field(
    tmp_path, monkeypatch
) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    mcp_server.jira_search_issues("project = FC", limit=20)

    expected_fields = [*mcp_server.DEFAULT_FIELDS, "customfield_10020"]
    assert client.search_issues_calls[0]["fields"] == expected_fields


def test_jira_search_issues_explicit_fields_are_not_augmented_with_sprint(
    tmp_path, monkeypatch
) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    mcp_server.jira_search_issues("project = FC", limit=20, fields=["summary"])

    assert client.search_issues_calls[0]["fields"] == ["summary"]


def test_jira_recent_issues_default_fields_include_the_configured_sprint_field(
    tmp_path, monkeypatch
) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    mcp_server.jira_recent_issues(days=7, limit=20)

    expected_fields = [*mcp_server.DEFAULT_FIELDS, "customfield_10020"]
    assert client.search_issues_calls[0]["fields"] == expected_fields


def test_jira_test_scope_issues_default_fields_include_the_configured_sprint_field(
    tmp_path, monkeypatch
) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    mcp_server.jira_test_scope_issues(limit=5)

    expected_fields = [*mcp_server.DEFAULT_FIELDS, "customfield_10020"]
    assert client.search_issues_calls[0]["fields"] == expected_fields


def test_jira_search_issues_single_page_reports_is_last_none_when_signal_is_absent(
    tmp_path, monkeypatch
) -> None:
    """F3b: a page with no completeness signal at all must read as unknown, not complete."""
    config = make_config(tmp_path)

    class NoSignalClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def search_issues(self, jql, *, limit, fields, next_page_token=None, project_key=None):
            return {"issues": [{"key": "FC-1"}]}

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: NoSignalClient())

    result = mcp_server.jira_search_issues("project = FC", limit=20)

    assert result["result"]["returned"] == 1
    assert result["result"]["is_last"] is None


def test_jira_search_issues_pages_automatically_past_one_page(tmp_path, monkeypatch) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_search_issues("project = FC", limit=250)

    assert client.search_all_issues_calls[0]["limit"] == 250
    assert client.search_issues_calls == []
    assert result["result"]["returned"] == 150
    assert result["result"]["is_last"] is False
    assert "isLast" not in result["result"]
    assert result["result"]["nextPageToken"] == "tok-2"
    assert "total" not in result["result"]


def test_jira_search_issues_pages_automatically_tolerates_is_last_spelling_too(
    tmp_path, monkeypatch
) -> None:
    """F3: once `client.search_all_issues` is migrated to `is_last`, this must still work."""
    config = make_config(tmp_path)

    class FutureSpellingClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def search_all_issues(
            self, jql, *, limit=None, fields=None, page_size=100, project_key=None
        ):
            return {
                "issues": [{"key": f"FC-{n}"} for n in range(1, 151)],
                "returned": 150,
                "is_last": False,
                "nextPageToken": "tok-2",
            }

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: FutureSpellingClient())

    result = mcp_server.jira_search_issues("project = FC", limit=250)

    assert result["result"]["is_last"] is False
    assert "isLast" not in result["result"]
    assert result["result"]["nextPageToken"] == "tok-2"


def test_jira_search_issues_manual_next_page_token_does_not_switch_to_auto_paging(
    tmp_path, monkeypatch
) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_search_issues("project = FC", limit=250, next_page_token="tok-1")

    assert client.search_all_issues_calls == []
    assert client.search_issues_calls[0]["next_page_token"] == "tok-1"
    assert result["result"]["returned"] == 2


def test_jira_recent_issues_reports_returned_and_is_last_never_a_total(
    tmp_path, monkeypatch
) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_recent_issues(days=7, limit=20)

    assert result["ok"] is True
    assert "total" not in result["result"]
    assert result["result"]["returned"] == 2
    assert "updated >= -7d" in client.search_issues_calls[0]["jql"]


def test_jira_mine_reports_returned_and_is_last_never_a_total(tmp_path, monkeypatch) -> None:
    _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_mine(project="FC", limit=20)

    assert result["ok"] is True
    assert "total" not in result["result"]
    assert result["result"]["returned"] == 2


def test_jira_fc_open_reports_returned_and_is_last_never_a_total(tmp_path, monkeypatch) -> None:
    _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_fc_open(limit=20)

    assert result["ok"] is True
    assert result["result"]["returned"] == 2
    assert "total" not in result["result"]


def test_jira_by_status_reports_returned_and_is_last_never_a_total(tmp_path, monkeypatch) -> None:
    _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_by_status("In Progress", project="FC", limit=20)

    assert result["ok"] is True
    assert result["result"]["returned"] == 2
    assert "total" not in result["result"]


def test_jira_test_scope_issues_reports_returned_and_is_last_never_a_total(
    tmp_path, monkeypatch
) -> None:
    _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_test_scope_issues(limit=5)

    assert result["ok"] is True
    assert result["result"]["returned"] == 2
    assert "total" not in result["result"]


class CommentsFakeClient:
    def __init__(self, _config) -> None:
        self.list_comments_calls: list[dict[str, object]] = []
        self.list_all_comments_calls: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def list_comments(self, key, *, start_at=0, limit=20, order_by=None):
        self.list_comments_calls.append(
            {"key": key, "start_at": start_at, "limit": limit, "order_by": order_by}
        )
        return {"comments": [{"id": "3"}, {"id": "2"}], "total": 18}

    def list_all_comments(self, key, *, order_by=None, page_size=100):
        self.list_all_comments_calls.append({"key": key, "order_by": order_by})
        return {"comments": [{"id": str(n)} for n in range(1, 19)], "total": 18}


def _install_comments_client(monkeypatch, tmp_path) -> CommentsFakeClient:
    config = make_config(tmp_path)
    client = CommentsFakeClient(config)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    return client


def test_jira_list_comments_defaults_to_newest_first(tmp_path, monkeypatch) -> None:
    client = _install_comments_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_comments("FC-402")

    assert result["ok"] is True
    assert client.list_comments_calls[0]["order_by"] == "-created"
    assert result["returned"] == 2
    assert result["total"] == 18
    assert result["comments"] == [{"id": "3"}, {"id": "2"}]


def test_jira_list_comments_order_by_none_restores_native_oldest_first(
    tmp_path, monkeypatch
) -> None:
    client = _install_comments_client(monkeypatch, tmp_path)

    mcp_server.jira_list_comments("FC-402", order_by=None)

    assert client.list_comments_calls[0]["order_by"] is None


def test_jira_list_comments_all_pages_through_every_comment(tmp_path, monkeypatch) -> None:
    client = _install_comments_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_comments("FC-402", limit="all")

    assert client.list_all_comments_calls == [{"key": "FC-402", "order_by": "-created"}]
    assert client.list_comments_calls == []
    assert result["returned"] == 18
    assert result["total"] == 18


def test_jira_list_comments_refuses_a_negative_limit(tmp_path, monkeypatch) -> None:
    client = _install_comments_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_comments("FC-402", limit=-1)

    assert result == {"ok": False, "message": "limit must be at least 1 or 'all', got -1."}
    assert client.list_comments_calls == []
    assert client.list_all_comments_calls == []


def test_jira_list_comments_refuses_a_negative_start_at(tmp_path, monkeypatch) -> None:
    client = _install_comments_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_comments("FC-402", start_at=-5)

    assert result == {"ok": False, "message": "start_at must be non-negative, got -5."}
    assert client.list_comments_calls == []


_COMMENT_BODY_ADF = {
    "type": "doc",
    "version": 1,
    "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Original comment"}]}],
}


class DeleteCommentFakeClient:
    def __init__(self, _config, *, missing: bool = False) -> None:
        self.config = _config
        self.missing = missing
        self.deleted: list[tuple[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        assert fields == ["project"]
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def get_comment(self, issue_key, comment_id):
        if self.missing:
            raise JiraApiError("not found", status_code=404)
        return {
            "id": comment_id,
            "author": {"displayName": "Agent"},
            "body": _COMMENT_BODY_ADF,
        }

    def delete_comment(self, issue_key, comment_id):
        self.deleted.append((issue_key, comment_id))
        return {}


def _install_delete_comment_client(monkeypatch, tmp_path, **kwargs) -> DeleteCommentFakeClient:
    config = make_config(tmp_path)
    client = DeleteCommentFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_delete_comment_dry_run_shows_current_body_without_deleting(
    tmp_path, monkeypatch
) -> None:
    client = _install_delete_comment_client(monkeypatch, tmp_path)

    result = mcp_server.jira_delete_comment("FC-403", "555")

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["current_author"] == "Agent"
    assert client.deleted == []


def test_jira_delete_comment_dry_run_flattens_current_body_like_the_cli(
    tmp_path, monkeypatch
) -> None:
    """F8: current_body must be plain text, matching the CLI's delete-comment dry run."""
    _install_delete_comment_client(monkeypatch, tmp_path)

    result = mcp_server.jira_delete_comment("FC-403", "555")

    assert result["current_body"] == "Original comment"
    assert isinstance(result["current_body"], str)
    assert result["current_body_adf"] == _COMMENT_BODY_ADF


def test_jira_delete_comment_execute_deletes(tmp_path, monkeypatch) -> None:
    client = _install_delete_comment_client(monkeypatch, tmp_path)

    result = mcp_server.jira_delete_comment("FC-403", "555", execute=True)

    assert result == {"ok": True, "issue": "FC-403", "deleted_comment_id": "555"}
    assert client.deleted == [("FC-403", "555")]


def test_jira_delete_comment_refuses_a_missing_comment(tmp_path, monkeypatch) -> None:
    client = _install_delete_comment_client(monkeypatch, tmp_path, missing=True)

    result = mcp_server.jira_delete_comment("FC-403", "999", execute=True)

    assert result == {"ok": False, "message": "Comment 999 is not present on FC-403."}
    assert client.deleted == []


class UpdateCommentFakeClient:
    def __init__(self, _config, *, missing: bool = False) -> None:
        self.config = _config
        self.missing = missing
        self.updated: list[tuple[str, str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        assert fields == ["project"]
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def get_comment(self, issue_key, comment_id):
        if self.missing:
            raise JiraApiError("not found", status_code=404)
        return {
            "id": comment_id,
            "author": {"displayName": "Agent"},
            "body": _COMMENT_BODY_ADF,
        }

    def update_comment(self, issue_key, comment_id, body):
        self.updated.append((issue_key, comment_id, body))
        return {"id": comment_id, "updated": "2026-09-14T00:00:00.000+0000"}


def _install_update_comment_client(monkeypatch, tmp_path, **kwargs) -> UpdateCommentFakeClient:
    config = make_config(tmp_path)
    client = UpdateCommentFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_update_comment_dry_run_flattens_current_body_like_the_cli(
    tmp_path, monkeypatch
) -> None:
    """F8: current_body must be plain text, matching the CLI's edit-comment dry run, with the
    raw ADF still available under a clearly different key rather than lost.

    Sweep finding (b): the replacement text is `proposed_body`, matching the CLI's `edit-comment`
    dry run key for the identical fact; this surface used to call it `new_body` instead.
    """
    client = _install_update_comment_client(monkeypatch, tmp_path)

    result = mcp_server.jira_update_comment("FC-403", "555", "Replacement text")

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["current_body"] == "Original comment"
    assert isinstance(result["current_body"], str)
    assert result["current_body_adf"] == _COMMENT_BODY_ADF
    assert result["proposed_body"] == "Replacement text"
    assert "new_body" not in result
    assert client.updated == []


def test_jira_update_comment_execute_replaces_the_body(tmp_path, monkeypatch) -> None:
    client = _install_update_comment_client(monkeypatch, tmp_path)

    result = mcp_server.jira_update_comment("FC-403", "555", "Replacement text", execute=True)

    assert result["ok"] is True
    assert client.updated == [("FC-403", "555", "Replacement text")]


def test_jira_update_comment_refuses_a_missing_comment(tmp_path, monkeypatch) -> None:
    client = _install_update_comment_client(monkeypatch, tmp_path, missing=True)

    result = mcp_server.jira_update_comment("FC-403", "999", "Replacement text", execute=True)

    assert result == {"ok": False, "message": "Comment 999 is not present on FC-403."}
    assert client.updated == []


class IssueTypeFakeClient:
    def __init__(
        self,
        _config,
        *,
        allowed: bool = True,
        updated: str = "2026-09-01T00:00:00.000+0000",
        server_updated: str | None = None,
    ) -> None:
        self.config = _config
        self.allowed = allowed
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated
        self.set_calls: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        assert fields == ["project"]
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def get_edit_metadata(self, issue_key):
        allowed_values = (
            [{"name": "Bug"}, {"name": "Task"}] if self.allowed else [{"name": "Bug"}]
        )
        return {"fields": {"issuetype": {"allowedValues": allowed_values}}}

    def is_issue_type_allowed(self, issue_key, issue_type_name):
        allowed_values = self.get_edit_metadata(issue_key)["fields"]["issuetype"]["allowedValues"]
        needle = issue_type_name.casefold()
        return any(value["name"].casefold() == needle for value in allowed_values)

    def issue_version(self, issue_key):
        return self.updated

    def set_issue_type(self, issue_key, issue_type_name, *, expect_updated=None):
        if expect_updated is not None and expect_updated != self.server_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.server_updated!r}. Re-read the issue and "
                "retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.server_updated,
            )
        self.set_calls.append({"issue": issue_key, "issue_type_name": issue_type_name})
        return {}


def _install_issue_type_client(monkeypatch, tmp_path, **kwargs) -> IssueTypeFakeClient:
    config = make_config(tmp_path)
    client = IssueTypeFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_issue_type_options_lists_allowed_values(tmp_path, monkeypatch) -> None:
    """R3: the key is `allowed_values`, matching the CLI's `issue-type-options` output and the
    Jira editmeta field it comes from — not `allowed_issue_types`, which used to be this
    surface's own, differently spelled name for the identical fact.
    """
    _install_issue_type_client(monkeypatch, tmp_path)

    result = mcp_server.jira_issue_type_options("FC-404")

    assert result == {"ok": True, "issue": "FC-404", "allowed_values": ["Bug", "Task"]}
    assert "allowed_issue_types" not in result


def test_jira_set_issue_type_refuses_a_disallowed_type_without_writing(
    tmp_path, monkeypatch
) -> None:
    client = _install_issue_type_client(monkeypatch, tmp_path, allowed=False)

    result = mcp_server.jira_set_issue_type("FC-404", "Task", execute=True)

    assert result == {"ok": False, "message": "'Task' is not an allowed issue type for FC-404."}
    assert client.set_calls == []


def test_jira_set_issue_type_dry_run_reports_expect_updated(tmp_path, monkeypatch) -> None:
    client = _install_issue_type_client(monkeypatch, tmp_path)

    result = mcp_server.jira_set_issue_type("FC-404", "Bug")

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["expect_updated"] == client.updated
    assert client.set_calls == []


def test_jira_set_issue_type_execute_changes_the_type(tmp_path, monkeypatch) -> None:
    client = _install_issue_type_client(monkeypatch, tmp_path)

    result = mcp_server.jira_set_issue_type("FC-404", "Bug", execute=True)

    assert result == {"ok": True, "issue": "FC-404", "issue_type_name": "Bug"}
    assert client.set_calls == [{"issue": "FC-404", "issue_type_name": "Bug"}]


def test_jira_set_issue_type_refuses_a_concurrent_edit(tmp_path, monkeypatch) -> None:
    client = _install_issue_type_client(
        monkeypatch,
        tmp_path,
        updated="2026-09-01T00:00:00.000+0000",
        server_updated="2026-09-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_set_issue_type("FC-404", "Bug", execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.set_calls == []


class SprintFakeClient:
    """Models the same real Agile behaviour as the CLI's `_SprintStateFakeClient`: the write
    endpoints return 204 with no body (`{}`), and `apply_writes=False` simulates a write Jira
    accepted without actually changing membership -- the scenario the post-write verification
    (sprint hardening item 3) exists to catch.
    """

    def __init__(
        self,
        _config,
        *,
        sprint_field_present: bool = True,
        initial_sprints: dict[str, list[dict[str, object]]] | None = None,
        disallowed_project_for: str | None = None,
        apply_writes: bool = True,
    ) -> None:
        self.config = _config
        self.added: list[tuple[object, list[str]]] = []
        self.moved: list[list[str]] = []
        self.sprint_field_present = sprint_field_present
        self.sprint_field_value: object = [{"id": 42, "name": "Sprint 42"}]
        self._disallowed_project_for = disallowed_project_for
        self._apply_writes = apply_writes
        self._sprints: dict[str, list[dict[str, object]]] = {
            key: list(value) for key, value in (initial_sprints or {}).items()
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def list_boards(self, *, board_type=None, name=None, max_results=50):
        return {"values": [{"id": 7, "name": "FC board", "type": board_type or "scrum"}]}

    def list_board_sprints(self, board_id, *, state=None, max_results=50):
        assert board_id == 7
        return {"values": [{"id": 42, "name": "Sprint 42", "state": state or "active"}]}

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            project = "OTHER" if issue_key == self._disallowed_project_for else "FC"
            return {"key": issue_key, "fields": {"project": {"key": project}}}
        assert fields == ["customfield_10020"]
        fields_dict: dict[str, object] = {}
        if self.sprint_field_present:
            fields_dict["customfield_10020"] = self.sprint_field_value
        return {"key": issue_key, "fields": fields_dict}

    def get_issue_sprints(self, issue_key, *, sprint_field=None):
        assert sprint_field == "customfield_10020"
        return list(self._sprints.get(issue_key) or [])

    def add_issues_to_sprint(self, sprint_id, issue_keys):
        self.added.append((sprint_id, list(issue_keys)))
        if self._apply_writes:
            sprint = {"id": sprint_id, "name": f"Sprint {sprint_id}", "state": "active"}
            for issue_key in issue_keys:
                current = [
                    entry
                    for entry in self._sprints.get(issue_key, [])
                    if entry.get("id") != sprint_id
                ]
                current.append(sprint)
                self._sprints[issue_key] = current
        return {}

    def move_issues_to_backlog(self, issue_keys):
        self.moved.append(list(issue_keys))
        if self._apply_writes:
            for issue_key in issue_keys:
                self._sprints[issue_key] = []
        return {}


def _install_sprint_client(monkeypatch, tmp_path, **kwargs) -> SprintFakeClient:
    config = make_config(tmp_path)
    client = SprintFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_list_boards_returns_client_result(tmp_path, monkeypatch) -> None:
    """Sweep finding (a)/(b): `list_boards`/`list_board_sprints` used to return the bare raw
    Jira page with no `returned`/`is_last`, unlike the CLI's `boards`/`sprints` commands, which
    already curate exactly this shape (see `cli.boards`, `cli.sprints`). A missing `isLast` must
    surface as `None` (unknown), never default to `True`, matching `search_all_issues`.
    """
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_boards()

    assert result["ok"] is True
    assert result["boards"]["values"][0]["id"] == 7
    assert result["boards"]["returned"] == 1
    assert result["boards"]["is_last"] is None
    assert "total" not in result["boards"]


def test_jira_list_board_sprints_returns_client_result(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_board_sprints(7, state="active")

    assert result["ok"] is True
    assert result["sprints"]["values"][0]["id"] == 42
    assert result["sprints"]["returned"] == 1
    assert result["sprints"]["is_last"] is None


def test_jira_get_issue_sprints_returns_the_issues_sprints(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_get_issue_sprints("FC-405")

    assert result == {
        "ok": True,
        "issue": "FC-405",
        "sprint_field": "customfield_10020",
        "sprint_field_present": True,
        "sprints": [{"id": 42, "name": "Sprint 42"}],
    }


def test_jira_get_issue_sprints_marks_field_absent_rather_than_an_empty_list(
    tmp_path, monkeypatch
) -> None:
    """F14: JiraClient.get_issue_sprints documents that it collapses "field absent" and "field
    present but empty" into the same `[]`, and advises reading the field off get_issue directly
    when that distinction matters -- exactly what the CLI's issue-sprint command already does.
    This tool must make the same choice rather than silently reporting "no sprints" for what may
    really be a wrong sprint_field id or a permissions gap."""
    _install_sprint_client(monkeypatch, tmp_path, sprint_field_present=False)

    result = mcp_server.jira_get_issue_sprints("FC-405")

    assert result == {
        "ok": True,
        "issue": "FC-405",
        "sprint_field": "customfield_10020",
        "sprint_field_present": False,
        "sprints": [],
    }


def test_jira_add_issues_to_sprint_dry_run_then_execute(tmp_path, monkeypatch) -> None:
    """3: this fake actually moves FC-405 into sprint 42 on `add_issues_to_sprint`, so the
    `verified: True` below is proof `jira_add_issues_to_sprint` re-read membership rather than
    trusted the 204.
    """
    client = _install_sprint_client(monkeypatch, tmp_path)

    dry_run = mcp_server.jira_add_issues_to_sprint(42, ["FC-405"])

    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert client.added == []

    executed = mcp_server.jira_add_issues_to_sprint(42, ["FC-405"], execute=True)

    assert executed == {
        "ok": True,
        "sprint_id": 42,
        "issue_keys": ["FC-405"],
        "results": [
            {
                "key": "FC-405",
                "sprint_id": 42,
                "sprints": [{"id": 42, "name": "Sprint 42", "state": "active"}],
                "in_sprint": True,
                "expected_in_sprint": True,
                "verified": True,
            }
        ],
    }
    assert client.added == [(42, ["FC-405"])]


def test_jira_add_issues_to_sprint_execute_reports_failure_when_reread_does_not_confirm(
    tmp_path, monkeypatch
) -> None:
    """Regression: fails against the pre-hardening tool, which trusted the batched POST and
    never re-read membership afterwards, so it always reported success. `apply_writes=False`
    simulates Jira accepting the request (still 204/`{}`) without membership actually changing.
    """
    client = _install_sprint_client(monkeypatch, tmp_path, apply_writes=False)

    executed = mcp_server.jira_add_issues_to_sprint(42, ["FC-405"], execute=True)

    assert executed["ok"] is False
    assert executed["results"][0]["verified"] is False
    assert executed["results"][0]["in_sprint"] is False
    assert client.added == [(42, ["FC-405"])]


def test_jira_add_issues_to_sprint_refuses_an_empty_list(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_add_issues_to_sprint(42, [])

    assert result == {"ok": False, "message": "issue_keys must not be empty."}


def test_jira_move_issues_to_backlog_dry_run_then_execute(tmp_path, monkeypatch) -> None:
    """1/3: `sprint_id` is now required (breaking change); FC-405 starts in sprint 42 so the
    precondition passes, and this fake actually clears membership on `move_issues_to_backlog`,
    so `verified: True` below is proof of a re-read, not a trusted 204.
    """
    client = _install_sprint_client(
        monkeypatch,
        tmp_path,
        initial_sprints={"FC-405": [{"id": 42, "name": "Sprint 42", "state": "active"}]},
    )

    dry_run = mcp_server.jira_move_issues_to_backlog(42, ["FC-405"])

    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert dry_run["precondition"][0]["in_sprint"] is True
    assert client.moved == []

    executed = mcp_server.jira_move_issues_to_backlog(42, ["FC-405"], execute=True)

    assert executed == {
        "ok": True,
        "sprint_id": 42,
        "issue_keys": ["FC-405"],
        "results": [
            {
                "key": "FC-405",
                "sprint_id": 42,
                "sprints": [],
                "in_sprint": False,
                "expected_in_sprint": False,
                "verified": True,
            }
        ],
    }
    assert client.moved == [["FC-405"]]


def test_jira_move_issues_to_backlog_precondition_refuses_whole_batch_when_not_in_sprint(
    tmp_path, monkeypatch
) -> None:
    """1: sprint_id is a precondition this tool enforces, not something Jira checks -- refuse
    the whole batch before any write when a key is not currently in sprint_id.
    """
    client = _install_sprint_client(monkeypatch, tmp_path, initial_sprints={"FC-405": []})

    result = mcp_server.jira_move_issues_to_backlog(42, ["FC-405"], execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "sprint_precondition_failed"
    assert result["sprint_id"] == 42
    assert result["report"] == [
        {"key": "FC-405", "sprint_id": 42, "sprints": [], "in_sprint": False}
    ]
    assert client.moved == []


def test_jira_move_issues_to_backlog_execute_reports_failure_when_reread_still_shows_membership(
    tmp_path, monkeypatch
) -> None:
    """Regression: fails against the pre-hardening tool, which trusted the batched POST and
    never re-read membership afterwards, so it always reported success.
    """
    client = _install_sprint_client(
        monkeypatch,
        tmp_path,
        initial_sprints={"FC-405": [{"id": 42, "name": "Sprint 42", "state": "active"}]},
        apply_writes=False,
    )

    executed = mcp_server.jira_move_issues_to_backlog(42, ["FC-405"], execute=True)

    assert executed["ok"] is False
    assert executed["results"][0]["verified"] is False
    assert executed["results"][0]["in_sprint"] is True
    assert client.moved == [["FC-405"]]


def test_jira_move_issues_to_backlog_refuses_an_empty_list(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_move_issues_to_backlog(42, [])

    assert result == {"ok": False, "message": "issue_keys must not be empty."}


# --- G2: the concurrency guard threaded through five previously-unguarded MCP tools ----------
#
# jira_write_human_brief, jira_render_mermaid_diagrams, jira_embed_diagram,
# jira_embed_existing_png_attachment (covered above, next to its own fake), and
# jira_upsert_image_gallery each read a description (or attachment set), transform it, and
# write the whole field back with no guard at all before this fix: a concurrent edit landing
# between the read and the write was silently reverted and the tool still returned
# {"ok": True}. These tests pin the guard on the remaining four.


class HumanBriefFakeClient:
    def __init__(self, _config, *, updated="2026-01-01T00:00:00.000+0000", server_updated=None):
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated
        self.writes: list[dict[str, object]] = []
        self.expect_updated_calls: list[str | None] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        assert fields == ["updated"], fields
        return {"key": issue_key, "fields": {"updated": self.updated}}

    def edit_issue(self, issue_key, *, description=None, expect_updated=None, **_kwargs):
        self.expect_updated_calls.append(expect_updated)
        if expect_updated is not None and expect_updated != self.server_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.server_updated!r}. Re-read the issue and "
                "retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.server_updated,
            )
        self.writes.append(
            {"issue": issue_key, "description": description, "expect_updated": expect_updated}
        )
        return {"key": issue_key, "fields": {}}


def _install_human_brief_client(monkeypatch, tmp_path, **kwargs) -> HumanBriefFakeClient:
    config = make_config(tmp_path)
    client = HumanBriefFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_a, **_k: None)
    return client


def test_jira_write_human_brief_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    client = _install_human_brief_client(
        monkeypatch,
        tmp_path,
        updated="2026-01-01T00:00:00.000+0000",
        server_updated="2026-01-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_write_human_brief("FC-1", key="FC-1", execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.writes == []


def test_jira_write_human_brief_executes_when_updated_matches(tmp_path, monkeypatch) -> None:
    client = _install_human_brief_client(monkeypatch, tmp_path)

    result = mcp_server.jira_write_human_brief("Brief text.", key="FC-1", execute=True)

    assert result == {"ok": True, "issue": "FC-1"}
    assert len(client.writes) == 1


def test_jira_write_human_brief_refuses_when_updated_missing(tmp_path, monkeypatch) -> None:
    client = _install_human_brief_client(monkeypatch, tmp_path, updated=None)

    result = mcp_server.jira_write_human_brief("Brief text.", key="FC-1", execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.writes == []
    assert client.expect_updated_calls == []


def test_jira_write_human_brief_guard_false_ignores_missing_updated(tmp_path, monkeypatch) -> None:
    client = _install_human_brief_client(monkeypatch, tmp_path, updated=None)

    result = mcp_server.jira_write_human_brief(
        "Brief text.", key="FC-1", guard_concurrent_edit=False, execute=True
    )

    assert result == {"ok": True, "issue": "FC-1"}
    assert client.expect_updated_calls == [None]


class DriftLabelsFakeClient:
    def __init__(self, _config, *, updated="2026-01-01T00:00:00.000+0000", server_updated=None):
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated
        self.existing_labels = ["drift-open"]
        self.writes: list[dict[str, object]] = []
        self.expect_updated_calls: list[str | None] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        assert fields == ["labels", "updated"], fields
        return {
            "key": issue_key,
            "fields": {"labels": self.existing_labels, "updated": self.updated},
        }

    def edit_issue(self, issue_key, *, labels=None, expect_updated=None, **_kwargs):
        self.expect_updated_calls.append(expect_updated)
        if expect_updated is not None and expect_updated != self.server_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.server_updated!r}. Re-read the issue and "
                "retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.server_updated,
            )
        self.writes.append({"issue": issue_key, "labels": labels})
        return {"key": issue_key, "fields": {"labels": labels}}


def _install_drift_labels_client(monkeypatch, tmp_path, **kwargs) -> DriftLabelsFakeClient:
    config = make_config(tmp_path)
    client = DriftLabelsFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_a, **_k: None)
    return client


def test_jira_update_drift_labels_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    """G2 regression: a whole-array read-transform-write, exactly like the description
    rewrites, had no guard at all before this fix."""
    client = _install_drift_labels_client(
        monkeypatch,
        tmp_path,
        updated="2026-01-01T00:00:00.000+0000",
        server_updated="2026-01-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_update_drift_labels("FC-1", add=["drift-reviewed"], execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.writes == []


def test_jira_update_drift_labels_executes_when_updated_matches(tmp_path, monkeypatch) -> None:
    client = _install_drift_labels_client(monkeypatch, tmp_path)

    result = mcp_server.jira_update_drift_labels("FC-1", add=["drift-reviewed"], execute=True)

    assert result["ok"] is True
    assert set(result["labels"]) == {"drift-open", "drift-reviewed"}
    assert len(client.writes) == 1


def test_jira_update_drift_labels_refuses_when_updated_missing(tmp_path, monkeypatch) -> None:
    client = _install_drift_labels_client(monkeypatch, tmp_path, updated=None)

    result = mcp_server.jira_update_drift_labels("FC-1", add=["drift-reviewed"], execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.writes == []


def test_jira_update_drift_labels_guard_false_ignores_missing_updated(
    tmp_path, monkeypatch
) -> None:
    client = _install_drift_labels_client(monkeypatch, tmp_path, updated=None)

    result = mcp_server.jira_update_drift_labels(
        "FC-1", add=["drift-reviewed"], guard_concurrent_edit=False, execute=True
    )

    assert result["ok"] is True
    assert client.expect_updated_calls == [None]


def _gallery_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "title": "Screenshots",
                "sections": [
                    {
                        "heading": "Flow",
                        "rows": [
                            {
                                "file": "diagram.png",
                                "title": "Step one",
                                "description": "The first step.",
                                "alt": "Step one screenshot",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


class GalleryFakeClient:
    def __init__(self, _config, *, updated="2026-01-01T00:00:00.000+0000", server_updated=None):
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated
        self.attachment = {
            "id": "170501",
            "filename": "diagram.png",
            "content": "https://example.atlassian.net/attachments/170501",
        }
        self.writes: list[dict[str, object]] = []
        self.expect_updated_calls: list[str | None] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        assert fields == ["description", "attachment", "updated"], fields
        return {
            "key": issue_key,
            "fields": {
                "description": {"type": "doc", "version": 1, "content": []},
                "attachment": [self.attachment],
                "updated": self.updated,
            },
        }

    def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
        self.expect_updated_calls.append(expect_updated)
        if expect_updated is not None and expect_updated != self.server_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.server_updated!r}. Re-read the issue and "
                "retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.server_updated,
            )
        self.writes.append({"issue": issue_key, "description": description})
        return {"id": "1"}


def _install_gallery_client(monkeypatch, tmp_path, **kwargs) -> GalleryFakeClient:
    config = make_config(tmp_path)
    client = GalleryFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_a, **_k: None)
    return client


def test_jira_upsert_image_gallery_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    client = _install_gallery_client(
        monkeypatch,
        tmp_path,
        updated="2026-01-01T00:00:00.000+0000",
        server_updated="2026-01-01T00:05:00.000+0000",
    )
    manifest_path = _gallery_manifest(tmp_path)

    result = mcp_server.jira_upsert_image_gallery(
        str(manifest_path), key="FC-1", execute=True
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.writes == []


def test_jira_upsert_image_gallery_executes_when_updated_matches(tmp_path, monkeypatch) -> None:
    client = _install_gallery_client(monkeypatch, tmp_path)
    manifest_path = _gallery_manifest(tmp_path)

    result = mcp_server.jira_upsert_image_gallery(
        str(manifest_path), key="FC-1", execute=True
    )

    assert result["ok"] is True
    assert len(client.writes) == 1


def test_jira_upsert_image_gallery_refuses_when_updated_missing(tmp_path, monkeypatch) -> None:
    client = _install_gallery_client(monkeypatch, tmp_path, updated=None)
    manifest_path = _gallery_manifest(tmp_path)

    result = mcp_server.jira_upsert_image_gallery(
        str(manifest_path), key="FC-1", execute=True
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.writes == []


class EmbedDiagramMcpFakeClient:
    """Mirrors tests/test_cli_concurrency.py's `_make_embed_diagram_fake_client`: models Jira's
    real behaviour where uploading an attachment moves the issue's own `updated`, and a genuine
    concurrent editor can change the live `description` in the same window
    (`concurrent_description`, applied at the same point `add_attachment` bumps `updated`). The
    fixed tool's SECOND `get_issue` read (fields=["description", "updated"]), after the upload,
    must observe this live state, not whatever the FIRST read saw. `issue_version` mirrors the
    same live `updated` state for any code path that still reads it directly.
    """

    def __init__(
        self,
        _config,
        *,
        after_upload_updated,
        at_put_time_updated=None,
        initial_description=None,
        concurrent_description=None,
    ):
        self.state = {
            "updated": "T-before-upload",
            "description": initial_description or {"type": "doc", "version": 1, "content": []},
        }
        self.after_upload_updated = after_upload_updated
        self.at_put_time_updated = (
            at_put_time_updated if at_put_time_updated is not None else after_upload_updated
        )
        self.concurrent_description = concurrent_description
        self.writes: list[dict[str, object]] = []
        self.deleted: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        assert fields is not None, fields
        payload: dict[str, object] = {
            "key": issue_key,
            "fields": {"updated": self.state["updated"]},
        }
        if "description" in fields:
            payload["fields"]["description"] = self.state["description"]
        if "attachment" in fields:
            payload["fields"]["attachment"] = []
        return payload

    def add_attachment(self, issue_key, file_path):
        self.state["updated"] = self.after_upload_updated
        if self.concurrent_description is not None:
            self.state["description"] = self.concurrent_description
        return [{"id": "9001", "filename": Path(file_path).name}]

    def attachment_media_id(self, attachment_id, *, issue_key):
        return "media-diagram-1"

    def issue_version(self, issue_key):
        current = self.state["updated"]
        if not isinstance(current, str) or not current:
            raise JiraMissingConcurrencyTokenError(issue_key=issue_key)
        return current

    def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
        if expect_updated is not None and expect_updated != self.at_put_time_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.at_put_time_updated!r}. Re-read the issue "
                "and retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.at_put_time_updated,
            )
        self.writes.append({"issue": issue_key, "description": description})
        return {"id": "1"}

    def delete_attachment(self, attachment_id, *, issue_key):
        self.deleted.append(attachment_id)
        return {}


def _install_embed_diagram_mcp_client(
    monkeypatch, tmp_path, **kwargs
) -> EmbedDiagramMcpFakeClient:
    config = make_config(tmp_path)
    client = EmbedDiagramMcpFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_a, **_k: None)
    return client


def test_jira_embed_diagram_succeeds_despite_the_upload_moving_updated(
    tmp_path, monkeypatch
) -> None:
    """G1, carried to the MCP surface while adding G2's guard: the upload itself moves
    `updated`, so the token must be read after it, not from the initial combined read."""
    client = _install_embed_diagram_mcp_client(
        monkeypatch, tmp_path, after_upload_updated="2026-01-01T00:01:00.000+0000"
    )
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = mcp_server.jira_embed_diagram(str(png), key="FC-1", execute=True)

    assert result["ok"] is True, result
    assert len(client.writes) == 1


def test_jira_embed_diagram_refuses_on_genuine_concurrent_edit(tmp_path, monkeypatch) -> None:
    client = _install_embed_diagram_mcp_client(
        monkeypatch,
        tmp_path,
        after_upload_updated="2026-01-01T00:01:00.000+0000",
        at_put_time_updated="2026-01-01T00:05:00.000+0000",
    )
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = mcp_server.jira_embed_diagram(str(png), key="FC-1", execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.writes == []


def test_jira_embed_diagram_refuses_when_guard_on_and_updated_missing(
    tmp_path, monkeypatch
) -> None:
    client = _install_embed_diagram_mcp_client(monkeypatch, tmp_path, after_upload_updated=None)
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = mcp_server.jira_embed_diagram(str(png), key="FC-1", execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.writes == []


def test_jira_embed_diagram_guard_false_ignores_missing_updated(tmp_path, monkeypatch) -> None:
    client = _install_embed_diagram_mcp_client(monkeypatch, tmp_path, after_upload_updated=None)
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = mcp_server.jira_embed_diagram(
        str(png), key="FC-1", guard_concurrent_edit=False, execute=True
    )

    assert result["ok"] is True
    assert len(client.writes) == 1


def test_jira_embed_diagram_incorporates_concurrent_edit_landing_before_post_upload_read(
    tmp_path, monkeypatch
) -> None:
    """H1 regression on the MCP surface. Round 2's bug: build from the first (pre-upload) read,
    guard token from a later, separate `issue_version` call. The token would match (nothing
    changed the issue after the upload), so the write would go ahead and report a guarded
    success while silently dropping a concurrent editor's paragraph that arrived between the
    first read and the upload. The fix builds the write from the post-upload read instead, so
    this same edit must be INCORPORATED.
    """
    initial_description = {"type": "doc", "version": 1, "content": [_paragraph("Original content")]}
    concurrent_description = {
        "type": "doc",
        "version": 1,
        "content": [_paragraph("Original content"), _paragraph("Concurrent editor's note")],
    }
    client = _install_embed_diagram_mcp_client(
        monkeypatch,
        tmp_path,
        after_upload_updated="2026-01-01T00:01:00.000+0000",
        initial_description=initial_description,
        concurrent_description=concurrent_description,
    )
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = mcp_server.jira_embed_diagram(str(png), key="FC-1", execute=True)

    assert result["ok"] is True, result
    assert len(client.writes) == 1
    written_description = client.writes[0]["description"]
    written_texts = [
        item.get("text")
        for block in written_description["content"]
        if block.get("type") == "paragraph"
        for item in block.get("content", [])
    ]
    assert "Concurrent editor's note" in written_texts, (
        "the concurrent editor's paragraph must survive the write, not be silently dropped: "
        f"{written_description}"
    )


def test_jira_embed_diagram_refused_write_leaves_no_attachment_behind(
    tmp_path, monkeypatch
) -> None:
    """H2 regression: before this fix, jira_embed_diagram's except clauses never cleaned up the
    attachment it had already uploaded when the guarded write refused (unlike
    jira_render_mermaid_diagrams, which already did). A refused write must not orphan the
    upload.
    """
    client = _install_embed_diagram_mcp_client(
        monkeypatch,
        tmp_path,
        after_upload_updated="2026-01-01T00:01:00.000+0000",
        at_put_time_updated="2026-01-01T00:05:00.000+0000",
    )
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = mcp_server.jira_embed_diagram(str(png), key="FC-1", execute=True)

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.writes == []
    assert client.deleted == ["9001"], "the uploaded attachment must be cleaned up, not orphaned"


class MermaidFakeClient:
    """Fake for jira_render_mermaid_diagrams. Each `add_attachment` call (one per rendered
    diagram) bumps the live `updated` state, modelling Jira's real behaviour; `issue_version`
    reports it, exactly like `EmbedDiagramMcpFakeClient` above.
    """

    def __init__(
        self,
        _config,
        *,
        after_upload_updated,
        at_put_time_updated=None,
        concurrent_description=None,
    ):
        self.state = {"updated": "T-before-upload"}
        self.after_upload_updated = after_upload_updated
        self.at_put_time_updated = (
            at_put_time_updated if at_put_time_updated is not None else after_upload_updated
        )
        self.description = {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "codeBlock",
                    "attrs": {"language": "mermaid"},
                    "content": [{"type": "text", "text": "graph TD; A-->B;"}],
                }
            ],
        }
        self.concurrent_description = concurrent_description
        self.writes: list[dict[str, object]] = []
        self.uploaded: list[str] = []
        self.deleted: list[str] = []
        self._next_id = 9200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        payload: dict[str, object] = {
            "key": issue_key,
            "fields": {"updated": self.state["updated"]},
        }
        if fields is None or "description" in fields:
            payload["fields"]["description"] = self.description
        if fields is not None and "attachment" in fields:
            payload["fields"]["attachment"] = []
        return payload

    def add_attachment(self, issue_key, path):
        self._next_id += 1
        attachment_id = str(self._next_id)
        self.uploaded.append(attachment_id)
        self.state["updated"] = self.after_upload_updated
        if self.concurrent_description is not None:
            self.description = self.concurrent_description
        return [{"id": attachment_id, "filename": Path(path).name}]

    def attachment_media_id(self, attachment_id, *, issue_key):
        return f"media-{attachment_id}"

    def issue_version(self, issue_key):
        current = self.state["updated"]
        if not isinstance(current, str) or not current:
            raise JiraMissingConcurrencyTokenError(issue_key=issue_key)
        return current

    def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
        if expect_updated is not None and expect_updated != self.at_put_time_updated:
            raise JiraConcurrentEditError(
                f"Issue {issue_key} changed since it was read: expected updated="
                f"{expect_updated!r}, found {self.at_put_time_updated!r}. Re-read the issue "
                "and retry.",
                issue_key=issue_key,
                expected_updated=expect_updated,
                found_updated=self.at_put_time_updated,
            )
        self.writes.append({"issue": issue_key, "description": description})
        return {"id": "1"}

    def delete_attachment(self, attachment_id, *, issue_key):
        self.deleted.append(attachment_id)
        return {}


def _install_mermaid_client(monkeypatch, tmp_path, **kwargs) -> MermaidFakeClient:
    config = make_config(tmp_path)
    client = MermaidFakeClient(config, **kwargs)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_a, **_k: None)
    monkeypatch.setattr(
        mcp_server,
        "discover_mermaid_runtime",
        lambda *_a, **_k: SimpleNamespace(
            node=Path("node"), cli=Path("mmdc"), browser=Path("chromium")
        ),
    )

    def _fake_render(_source, path, *, runtime):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(_PNG_BYTES)

    monkeypatch.setattr(mcp_server, "render_mermaid_png", _fake_render)
    return client


def test_jira_render_mermaid_diagrams_succeeds_despite_the_upload_moving_updated(
    tmp_path, monkeypatch
) -> None:
    client = _install_mermaid_client(
        monkeypatch, tmp_path, after_upload_updated="2026-01-01T00:01:00.000+0000"
    )

    result = mcp_server.jira_render_mermaid_diagrams(
        key="FC-1", output_dir=str(tmp_path / "out"), execute=True
    )

    assert result["ok"] is True, result
    assert len(client.writes) == 1
    assert client.deleted == []


def test_jira_render_mermaid_diagrams_refuses_on_genuine_concurrent_edit_and_cleans_up_upload(
    tmp_path, monkeypatch
) -> None:
    """A real race must still refuse -- and, unlike embed-diagram, this tool already cleans up
    attachments it uploaded in the same call when the write that would reference them fails."""
    client = _install_mermaid_client(
        monkeypatch,
        tmp_path,
        after_upload_updated="2026-01-01T00:01:00.000+0000",
        at_put_time_updated="2026-01-01T00:05:00.000+0000",
    )

    result = mcp_server.jira_render_mermaid_diagrams(
        key="FC-1", output_dir=str(tmp_path / "out"), execute=True
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit"
    assert client.writes == []
    assert client.deleted == client.uploaded
    assert len(client.uploaded) == 1


def test_jira_render_mermaid_diagrams_refuses_when_guard_on_and_updated_missing(
    tmp_path, monkeypatch
) -> None:
    client = _install_mermaid_client(monkeypatch, tmp_path, after_upload_updated=None)

    result = mcp_server.jira_render_mermaid_diagrams(
        key="FC-1", output_dir=str(tmp_path / "out"), execute=True
    )

    assert result["ok"] is False
    assert result["error_type"] == "concurrent_edit_guard_unavailable"
    assert client.writes == []
    assert client.deleted == client.uploaded


def test_jira_render_mermaid_diagrams_incorporates_concurrent_edit_landing_before_post_upload_read(
    tmp_path, monkeypatch
) -> None:
    """H1 regression on the MCP surface's mermaid tool: a concurrent editor's change lands
    between the first read (which only finds the Mermaid block to render) and the post-upload
    second read that now supplies both content and guard token. The write must incorporate it.
    """
    concurrent_description = {
        "type": "doc",
        "version": 1,
        "content": [
            _paragraph("Concurrent editor's note"),
            {
                "type": "codeBlock",
                "attrs": {"language": "mermaid"},
                "content": [{"type": "text", "text": "graph TD; A-->B;"}],
            },
        ],
    }
    client = _install_mermaid_client(
        monkeypatch,
        tmp_path,
        after_upload_updated="2026-01-01T00:01:00.000+0000",
        concurrent_description=concurrent_description,
    )

    result = mcp_server.jira_render_mermaid_diagrams(
        key="FC-1", output_dir=str(tmp_path / "out"), execute=True
    )

    assert result["ok"] is True, result
    assert len(client.writes) == 1
    written_description = client.writes[0]["description"]
    written_texts = [
        item.get("text")
        for block in written_description["content"]
        if block.get("type") == "paragraph"
        for item in block.get("content", [])
    ]
    assert "Concurrent editor's note" in written_texts, (
        "the concurrent editor's paragraph must survive the write, not be silently dropped: "
        f"{written_description}"
    )


def test_jira_config_fields_reports_resolved_ids(tmp_path, monkeypatch) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)

    result = mcp_server.jira_config_fields()

    assert result == {
        "ok": True,
        "story_points_field": "customfield_10028",
        "sprint_field": "customfield_10020",
    }


def test_jira_config_fields_reflects_a_configured_override(tmp_path, monkeypatch) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_SPRINT_FIELD": "customfield_99999",
        }
    )
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)

    result = mcp_server.jira_config_fields()

    assert result["sprint_field"] == "customfield_99999"
