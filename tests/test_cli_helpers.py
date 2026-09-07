from __future__ import annotations

import io
import json

from rich.console import Console
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


def test_edit_issue_cli_sets_fix_version_and_audits_it(tmp_path, monkeypatch) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    audited: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, **_kwargs):
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

        def edit_issue(self, issue_key, **kwargs):
            audited.append({"issue": issue_key, "kwargs": kwargs})
            return {"key": issue_key, "fields": {"fixVersions": [{"name": "FCL-MVP"}]}}

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", FakeClient)
    monkeypatch.setattr(cli, "audit_write", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        cli.app,
        ["edit-issue", "FC-380", "--fix-version", "FCL-MVP", "--execute"],
    )

    assert result.exit_code == 0, result.output
    assert audited == [
        {
            "issue": "FC-380",
            "kwargs": {
                "summary": None,
                "description": None,
                "labels": None,
                "fix_versions": ["FCL-MVP"],
                "parent_key": None,
                "priority": None,
                "story_points": None,
            },
        }
    ]


def test_edit_issue_cli_dry_run_shows_several_fix_versions_in_payload(
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

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, **_kwargs):
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(
        cli.app,
        [
            "edit-issue",
            "FC-380",
            "--fix-version",
            "FCL-MVP",
            "--fix-version",
            "FCL-1.1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    json_start = result.output.index("{\n")
    payload, _ = json.JSONDecoder().raw_decode(result.output[json_start:])
    assert payload["fields"]["fixVersions"] == [
        {"name": "FCL-MVP"},
        {"name": "FCL-1.1"},
    ]


MARKUP_HOSTILE_NAMES = [
    "Costs [in EUR/m2]",
    "[wip] Detail",
    "[open] Risks",
    "[#1234] Rework",
    "[/red] Detail",
    "L" * 120,
]


def _print_json_output(payload: object) -> str:
    with cli.data_console.capture() as captured:
        cli._print_json(payload)
    return captured.get()


def test_print_json_is_byte_exact_for_every_caller_of_the_shared_helper() -> None:
    """34 commands print JSON through this helper, so pin it here and not per command."""
    for name in MARKUP_HOSTILE_NAMES:
        payload = {"summary": name, "nested": [{"heading": name}]}
        output = _print_json_output(payload)

        assert json.loads(output) == payload
        assert output.rstrip("\n") == json.dumps(payload, indent=2, ensure_ascii=False)


def _patch_cli_document(name: str) -> dict[str, object]:
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Preamble."}]},
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Scope"}],
            },
            {
                "type": "heading",
                "attrs": {"level": 3},
                "content": [{"type": "text", "text": name}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "body"}]},
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Next"}],
            },
        ],
    }


def test_patch_description_section_cli_payload_survives_a_markup_hostile_heading(
    tmp_path, monkeypatch
) -> None:
    """The refusal payload is the correction, so the channel must not rewrite the names."""
    for name in MARKUP_HOSTILE_NAMES:
        config = JiraConfig.model_validate(
            {
                "JIRA_SITE_URL": "https://example.atlassian.net",
                "JIRA_EMAIL": "user@example.com",
                "JIRA_API_TOKEN": "token",
                "JIRA_WRITE_PROJECT_KEYS": "FC",
                "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
            }
        )
        document = _patch_cli_document(name)

        class FakeClient:
            def __init__(self, _config, _document=document):
                self.document = _document

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def get_issue(self, issue_key, **_kwargs):
                return {
                    "key": issue_key,
                    "fields": {"project": {"key": "FC"}, "description": self.document},
                }

            def edit_issue_adf_description(self, issue_key, description):
                raise AssertionError("must not write")

        monkeypatch.setattr(cli, "load_config", lambda _config=config: _config)
        monkeypatch.setattr(cli, "JiraClient", FakeClient)
        monkeypatch.setattr(cli, "audit_write", lambda *_args, **_kwargs: None)

        result = CliRunner().invoke(
            cli.app,
            [
                "patch-description-section",
                "FC-385",
                "--section",
                "Scope",
                "--content",
                "## Scope\n\nnew",
                "--execute",
            ],
        )

        assert result.exit_code == 2, f"{name!r}: {result.output}"
        assert result.exception is None or isinstance(result.exception, SystemExit)
        payload = json.loads(result.output[result.output.index("{") :])
        assert payload["swallowed_headings"] == [name], f"{name!r} came back altered"
        assert payload["error_type"] == "section_patch_refused"


def _on_a_terminal(console: Console) -> Console:
    """Rebuild a console with its own settings, forced to behave as a terminal.

    Highlighting and colour only appear on a terminal, so a capture in a test would never see
    them. Reading the settings off the real console keeps this honest: loosen any of the three
    on the module object and this probe loosens with it.
    """
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        width=80,
        markup=console._markup,
        highlight=console._highlight,
        soft_wrap=console.soft_wrap,
    )


def test_structured_output_adds_no_ansi_when_stdout_is_a_terminal() -> None:
    """A copy out of a real terminal has to be the same bytes as a copy out of a pipe."""
    text = json.dumps({"swallowed_headings": ["Costs [in EUR/m2]", "L" * 120]}, indent=2)
    probe = _on_a_terminal(cli.data_console)

    probe.print(text)
    written = probe.file.getvalue()

    assert "\x1b" not in written
    assert written.rstrip("\n") == text


def test_a_table_cell_keeps_jira_text_that_looks_like_markup() -> None:
    """Same defect class as the JSON channel, display only: rich reads markup in table cells."""
    issues = [
        {
            "key": "FC-385",
            "fields": {
                "summary": "Costs [in EUR/m2]",
                "status": {"name": "Open"},
                "assignee": {"displayName": "Ada"},
            },
        }
    ]

    with cli.table_console.capture() as captured:
        cli._print_issues_table(issues)

    assert "Costs [in EUR/m2]" in captured.get()
