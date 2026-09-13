from __future__ import annotations

import json

from typer.testing import CliRunner

from jira_agent import cli
from jira_agent.client import select_transition
from jira_agent.config import JiraConfig


def test_select_transition_matches_id_or_case_insensitive_name() -> None:
    data = {
        "transitions": [
            {"id": "11", "name": "Start Progress"},
            {"id": "21", "name": "Done"},
        ]
    }

    assert select_transition(data, "11") == {"id": "11", "name": "Start Progress"}
    assert select_transition(data, "done") == {"id": "21", "name": "Done"}
    assert select_transition(data, "missing") is None


def test_create_issue_cli_embeds_mermaid_after_creation(tmp_path, monkeypatch) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
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

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", FakeClient)
    monkeypatch.setattr(cli, "embed_mermaid_diagrams", fake_embed)
    monkeypatch.setattr(cli, "audit_write", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        cli.app,
        [
            "create-issue",
            "Diagram",
            "--issue-type",
            "Task",
            "--project",
            "FC",
            "--description",
            "```mermaid\nflowchart LR\nA --> B\n```",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Embedded 1 Mermaid diagram(s)." in result.output
    assert calls["embed"]["issue_key"] == "FC-123"


def _patch_description() -> dict[str, object]:
    """Indices: 0 h1, 1 h2 Observed, 2 paragraph, 3 h2 Acceptance criteria, 4 taskList."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 1},
                "content": [{"type": "text", "text": "Description wipe"}],
            },
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Observed"}],
            },
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "The payload replaced the whole field."}],
            },
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Acceptance criteria"}],
            },
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


def _install_patch_cli(monkeypatch, tmp_path) -> list[dict[str, object]]:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    writes: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, **_kwargs):
            return {
                "key": issue_key,
                "fields": {"project": {"key": "FC"}, "description": _patch_description()},
            }

        def edit_issue_adf_description(self, issue_key, description):
            writes.append({"issue": issue_key, "description": description})
            return {"updated": True}

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", FakeClient)
    monkeypatch.setattr(cli, "audit_write", lambda *_args, **_kwargs: None)
    return writes


def test_patch_description_section_cli_dry_run_shows_counts_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    writes = _install_patch_cli(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "patch-description-section",
            "FC-385",
            "--section",
            "Acceptance criteria",
            "--content",
            "## Acceptance criteria\n\n- [ ] Only the section changes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "would patch one description section" in result.output
    assert '"preserved_before": 3' in result.output
    assert '"original_node_count": 5' in result.output
    assert writes == []


def test_patch_description_section_cli_execute_keeps_the_untouched_nodes(
    tmp_path, monkeypatch
) -> None:
    writes = _install_patch_cli(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "patch-description-section",
            "FC-385",
            "--section",
            "Observed",
            "--content",
            "## Observed\n\nThe dry run showed a one-bullet payload.",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.output
    written = writes[0]["description"]
    assert written["content"][3:] == _patch_description()["content"][3:]
    assert written["content"][0] == _patch_description()["content"][0]


def test_patch_description_section_cli_refuses_a_wide_rewrite(tmp_path, monkeypatch) -> None:
    writes = _install_patch_cli(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "patch-description-section",
            "FC-385",
            "--section",
            "Description wipe",
            "--content",
            "# Description wipe\n\n- Only this bullet survives",
            "--execute",
        ],
    )

    assert result.exit_code == 2
    assert "Swallowed-section mismatch" in result.output
    assert "allow_wide_rewrite" in result.output
    assert writes == []


def test_patch_description_section_cli_allows_a_wide_rewrite_on_request(
    tmp_path, monkeypatch
) -> None:
    writes = _install_patch_cli(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "patch-description-section",
            "FC-385",
            "--section",
            "Description wipe",
            "--content",
            "# Description wipe\n\n- Only this bullet survives",
            "--expect-swallowed-heading",
            "Observed",
            "--expect-swallowed-heading",
            "Acceptance criteria",
            "--allow-wide-rewrite",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(writes[0]["description"]["content"]) == 2


class ClearParentCliFakeClient:
    def __init__(self, _config) -> None:
        self.parent = {"key": "FC-300"}
        self.writes = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        assert fields == ["parent"]
        return {"key": issue_key, "fields": {"parent": self.parent}}

    def edit_issue(self, issue_key, *, clear_parent=False, **_kwargs):
        assert issue_key == "FC-359"
        assert clear_parent is True
        self.writes += 1
        self.parent = None
        return {}


def test_clear_parent_cli_is_dry_run_by_default_and_verifies_execute(tmp_path, monkeypatch) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    client = ClearParentCliFakeClient(config)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", lambda _config: client)
    monkeypatch.setattr(cli, "audit_write", lambda *_args, **_kwargs: None)

    dry_run = CliRunner().invoke(cli.app, ["clear-parent", "FC-359"])
    executed = CliRunner().invoke(cli.app, ["clear-parent", "FC-359", "--execute"])

    assert dry_run.exit_code == 0, dry_run.output
    assert "would clear issue parent" in dry_run.output
    assert client.writes == 1
    assert executed.exit_code == 0, executed.output
    assert "Cleared parent of FC-359." in executed.output


class DeleteLinkCliFakeClient:
    def __init__(self, _config) -> None:
        self.deleted: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        assert fields == ["issuelinks"]
        if issue_key == "FC-354":
            links = [
                {
                    "id": "10001",
                    "type": {"name": "Relates"},
                    "outwardIssue": {"key": "FC-356"},
                }
            ]
        else:
            links = [
                {
                    "id": "10001",
                    "type": {"name": "Relates"},
                    "inwardIssue": {"key": "FC-354"},
                }
            ]
        return {"key": issue_key, "fields": {"issuelinks": links}}

    def delete_issue_link(self, link_id, *, issue_key):
        self.deleted.append(link_id)
        return {}


def test_delete_issue_link_cli_dry_run_resolves_without_delete(tmp_path, monkeypatch) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    client = DeleteLinkCliFakeClient(config)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", lambda _config: client)

    result = CliRunner().invoke(cli.app, ["delete-issue-link", "FC-354", "FC-356"])

    assert result.exit_code == 0, result.output
    assert "would delete issue link" in result.output
    assert '"id": "10001"' in result.output
    assert client.deleted == []


class LinkCreateCliFakeClient:
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


def test_link_issue_cli_dry_run_maps_fc_350_blocks_fc_352_to_jira_direction(
    tmp_path, monkeypatch
) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    client = LinkCreateCliFakeClient(config)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", lambda _config: client)

    result = CliRunner().invoke(cli.app, ["link-issue", "FC-350", "FC-352", "--type", "Blocks"])

    assert result.exit_code == 0, result.output
    assert '"inwardIssue": {' in result.output
    assert '"key": "FC-350"' in result.output
    assert client.created == []


class ExistingPngCliFakeClient:
    def __init__(self, _config) -> None:
        self.edits: list[dict[str, object]] = []

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
            "fields": {
                "description": {"type": "doc", "version": 1, "content": []},
                "attachment": [
                    {
                        "id": "170501",
                        "filename": "fc349-disbursement-flow.png",
                        "mimeType": "image/png",
                    }
                ],
            },
        }

    def attachment_media_id(self, attachment_id, *, issue_key):
        assert (attachment_id, issue_key) == ("170501", "FC-349")
        return "media-fc349"

    def edit_issue_adf_description(self, issue_key, description):
        self.edits.append(description)
        return {}


def test_embed_existing_png_cli_dry_run_resolves_exact_existing_attachment(
    tmp_path, monkeypatch
) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    client = ExistingPngCliFakeClient(config)
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", lambda _config: client)

    result = CliRunner().invoke(
        cli.app,
        [
            "embed-existing-png",
            "FC-349",
            "--filename",
            "fc349-disbursement-flow.png",
            "--width",
            "1600",
            "--height",
            "900",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "would embed existing PNG attachment" in result.output
    assert '"attachment_id": "170501"' in result.output
    assert client.edits == []


def test_patch_description_section_cli_prints_the_lists_as_json_on_a_refusal(
    tmp_path, monkeypatch
) -> None:
    writes = _install_patch_cli(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "patch-description-section",
            "FC-385",
            "--section",
            "Description wipe",
            "--content",
            "# Description wipe\n\n- Only this bullet",
            "--allow-wide-rewrite",
            "--execute",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload == {
        "error_type": "section_patch_refused",
        "swallowed_headings": ["Observed", "Acceptance criteria"],
        "expect_swallowed_headings": [],
        "whole_document": True,
    }
    assert writes == []
