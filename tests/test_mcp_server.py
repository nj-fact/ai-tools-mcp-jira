from __future__ import annotations

from jira_agent import mcp_server
from jira_agent.client import JiraApiError, JiraConcurrentEditError, text_to_adf
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
    ) -> None:
        self.duplicate_heading = duplicate_heading
        self.updated = updated
        self.server_updated = server_updated if server_updated is not None else updated
        self.writes: list[dict[str, object]] = []
        self.expect_updated_calls: list[str | None] = []

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
                "updated": self.updated,
            },
        }

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
    def __init__(self, _config) -> None:
        self.description = {"type": "doc", "version": 1, "content": []}
        self.edits: list[dict[str, object]] = []
        self.attachment = {
            "id": "170501",
            "filename": "fc349-disbursement-flow.png",
            "mimeType": "image/png",
        }

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        assert fields == ["description", "attachment"]
        return {
            "key": issue_key,
            "fields": {"description": self.description, "attachment": [self.attachment]},
        }

    def attachment_media_id(self, attachment_id, *, issue_key):
        assert attachment_id == "170501"
        assert issue_key == "FC-349"
        return "media-fc349"

    def edit_issue_adf_description(self, issue_key, description):
        assert issue_key == "FC-349"
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


def test_jira_get_issue_default_fields_include_parent(tmp_path, monkeypatch) -> None:
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
    assert calls["fields"] == [*mcp_server.DEFAULT_FIELDS, "parent"]


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
    assert result["result"]["returned"] == 2
    assert result["result"]["isLast"] is True
    assert client.search_issues_calls[0]["limit"] == 20
    assert client.search_all_issues_calls == []


def test_jira_search_issues_pages_automatically_past_one_page(tmp_path, monkeypatch) -> None:
    client = _install_search_client(monkeypatch, tmp_path)

    result = mcp_server.jira_search_issues("project = FC", limit=250)

    assert client.search_all_issues_calls[0]["limit"] == 250
    assert client.search_issues_calls == []
    assert result["result"]["returned"] == 150
    assert result["result"]["isLast"] is False
    assert result["result"]["nextPageToken"] == "tok-2"
    assert "total" not in result["result"]


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
        return {"id": comment_id, "author": {"displayName": "Agent"}, "body": {"type": "doc"}}

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
    _install_issue_type_client(monkeypatch, tmp_path)

    result = mcp_server.jira_issue_type_options("FC-404")

    assert result == {"ok": True, "issue": "FC-404", "allowed_issue_types": ["Bug", "Task"]}


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
    def __init__(self, _config) -> None:
        self.config = _config
        self.added: list[tuple[object, list[str]]] = []
        self.moved: list[list[str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def list_boards(self, *, board_type=None, name=None, max_results=50):
        return {"values": [{"id": 7, "name": "FC board", "type": board_type or "scrum"}]}

    def list_board_sprints(self, board_id, *, state=None, max_results=50):
        assert board_id == 7
        return {"values": [{"id": 42, "name": "Sprint 42", "state": state or "active"}]}

    def get_issue_sprints(self, issue_key, *, sprint_field="customfield_10020"):
        assert issue_key == "FC-405"
        return [{"id": 42, "name": "Sprint 42"}]

    def get_issue(self, issue_key, *, fields=None, expand=None):
        assert fields == ["project"]
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def add_issues_to_sprint(self, sprint_id, issue_keys):
        self.added.append((sprint_id, list(issue_keys)))
        return {}

    def move_issues_to_backlog(self, issue_keys):
        self.moved.append(list(issue_keys))
        return {}


def _install_sprint_client(monkeypatch, tmp_path) -> SprintFakeClient:
    config = make_config(tmp_path)
    client = SprintFakeClient(config)
    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", lambda _config: client)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)
    return client


def test_jira_list_boards_returns_client_result(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_boards()

    assert result["ok"] is True
    assert result["boards"]["values"][0]["id"] == 7


def test_jira_list_board_sprints_returns_client_result(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_list_board_sprints(7, state="active")

    assert result["ok"] is True
    assert result["sprints"]["values"][0]["id"] == 42


def test_jira_get_issue_sprints_returns_the_issues_sprints(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_get_issue_sprints("FC-405")

    assert result == {"ok": True, "issue": "FC-405", "sprints": [{"id": 42, "name": "Sprint 42"}]}


def test_jira_add_issues_to_sprint_dry_run_then_execute(tmp_path, monkeypatch) -> None:
    client = _install_sprint_client(monkeypatch, tmp_path)

    dry_run = mcp_server.jira_add_issues_to_sprint(42, ["FC-405"])

    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert client.added == []

    executed = mcp_server.jira_add_issues_to_sprint(42, ["FC-405"], execute=True)

    assert executed == {"ok": True, "sprint_id": 42, "issue_keys": ["FC-405"]}
    assert client.added == [(42, ["FC-405"])]


def test_jira_add_issues_to_sprint_refuses_an_empty_list(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_add_issues_to_sprint(42, [])

    assert result == {"ok": False, "message": "issue_keys must not be empty."}


def test_jira_move_issues_to_backlog_dry_run_then_execute(tmp_path, monkeypatch) -> None:
    client = _install_sprint_client(monkeypatch, tmp_path)

    dry_run = mcp_server.jira_move_issues_to_backlog(["FC-405"])

    assert dry_run["ok"] is True
    assert dry_run["dry_run"] is True
    assert client.moved == []

    executed = mcp_server.jira_move_issues_to_backlog(["FC-405"], execute=True)

    assert executed == {"ok": True, "issue_keys": ["FC-405"]}
    assert client.moved == [["FC-405"]]


def test_jira_move_issues_to_backlog_refuses_an_empty_list(tmp_path, monkeypatch) -> None:
    _install_sprint_client(monkeypatch, tmp_path)

    result = mcp_server.jira_move_issues_to_backlog([])

    assert result == {"ok": False, "message": "issue_keys must not be empty."}
