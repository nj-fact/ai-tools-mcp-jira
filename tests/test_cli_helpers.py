from __future__ import annotations

import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from typer.testing import CliRunner

from jira_agent import cli
from jira_agent.client import JiraClient as RealJiraClient
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
                "fields": {
                    "project": {"key": "FC"},
                    "description": _patch_description(),
                    "updated": "2026-09-14T10:00:00.000+0200",
                },
            }

        def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
            writes.append(
                {
                    "issue": issue_key,
                    "description": description,
                    "expect_updated": expect_updated,
                }
            )
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

        def issue_version(self, issue_key):
            return "2026-09-14T10:00:00.000+0200"

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
                "expect_updated": "2026-09-14T10:00:00.000+0200",
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


# --- JSON purity: --json must print exactly one JSON document on stdout, nothing else -----------
#
# Click 8.4.1's CliRunner keeps stdout and stderr separate (checked directly below and confirmed
# by hand against this repo's pinned version): result.stdout and result.stderr are independent
# streams, so these tests assert on them separately instead of slicing result.output for a "{".


def _json_purity_config(tmp_path) -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


def test_click_runner_keeps_stdout_and_stderr_separate_in_this_click_version() -> None:
    """Pin the CliRunner behaviour this whole test group relies on.

    If a future click upgrade merges the streams again, this fails first and loudly, instead of
    every test below silently starting to check a merged stream.
    """
    import click
    import click.testing

    @click.command()
    def probe() -> None:
        click.echo("to stdout")
        click.echo("to stderr", err=True)

    # typer.testing.CliRunner.invoke expects a Typer app; this probes plain click, so use
    # click's own runner directly.
    result = click.testing.CliRunner().invoke(probe, [])

    assert result.stdout == "to stdout\n"
    assert result.stderr == "to stderr\n"


class _EditIssueFakeClient:
    def __init__(self, _config) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, **_kwargs):
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def issue_version(self, issue_key):
        return "2026-09-14T10:00:00.000+0200"

    def edit_issue(self, issue_key, **_kwargs):
        return {"key": issue_key, "fields": {"summary": "Updated summary"}}


def test_edit_issue_cli_json_dry_run_parses_directly_and_carries_dry_run_true(
    tmp_path, monkeypatch
) -> None:
    """The exact scenario from the defect report: edit-issue --json without --execute."""
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _EditIssueFakeClient)

    result = CliRunner().invoke(
        cli.app,
        ["edit-issue", "FC-380", "--summary", "Updated summary", "--json"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    # No find('{') slicing: the whole of stdout is the JSON document, nothing else.
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["fields"]["summary"] == "Updated summary"
    # Every prose line moved to stderr; none of it leaked onto stdout.
    assert "Dry run" not in result.stdout
    assert "Pass --execute" not in result.stdout
    assert "Dry run" in result.stderr


def test_edit_issue_cli_json_execute_parses_as_one_document_with_dry_run_false(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _EditIssueFakeClient)
    monkeypatch.setattr(cli, "audit_write", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        cli.app,
        ["edit-issue", "FC-380", "--summary", "Updated summary", "--execute", "--json"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is False
    assert payload["fields"]["summary"] == "Updated summary"
    assert result.stdout.strip() == json.dumps(payload, indent=2, ensure_ascii=False)


def test_issue_cli_json_round_trips_long_and_bracket_bearing_values_byte_identically(
    tmp_path, monkeypatch
) -> None:
    """A summary long enough to have triggered rich's word-wrap before the data_console fix,
    and a value containing a literal '[' that markup would have swallowed.
    """
    long_summary = "S" * 210 + " [scope note]"
    fake_issue = {
        "key": "FC-389",
        "fields": {
            "summary": long_summary,
            "parent": {"key": "FC-100", "fields": {"summary": "Parent [with brackets]"}},
        },
    }

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, **_kwargs):
            return fake_issue

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["issue", "FC-389", "--fields", "parent", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload == fake_issue


def test_search_cli_json_hides_next_page_token_from_stdout(tmp_path, monkeypatch) -> None:
    """search --json is curated: it goes through search_all_issues, not raw search_issues,

    and its JSON is `issues`/`returned`/`is_last`/`nextPageToken`, never a fabricated `total`.
    """
    fake_issues = [
        {
            "key": "FC-1",
            "fields": {
                "summary": "s",
                "status": {"name": "Open"},
                "assignee": {"displayName": "Ada"},
                "issuetype": {"name": "Task"},
                "project": {"key": "FC"},
                "updated": "2026-09-01T00:00:00.000+0000",
            },
        }
    ]
    fake_response = {
        "issues": fake_issues,
        "returned": 1,
        "isLast": False,
        "nextPageToken": "token-abc-123",
    }

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def search_all_issues(self, jql, **_kwargs):
            return fake_response

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["search", "project = FC", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["issues"] == fake_issues
    assert payload["returned"] == 1
    assert payload["is_last"] is False
    assert payload["nextPageToken"] == "token-abc-123"
    assert "total" not in payload
    # The token legitimately appears as a JSON field in the document above; what must not
    # appear is the separate human-readable "nextPageToken: ..." prose line the non-json path
    # prints after its table.
    assert "nextPageToken: token-abc-123" not in result.stdout


# --- Read-path fixes: agent-report comment paging, issue parent field, search/comments honesty --


def _agent_report_fake_client(total_comments: int) -> tuple[type, list[tuple[str, object]]]:
    """A fake JiraClient for `agent-report`, oldest-first source of truth like real Jira.

    Comment ids count up from 1000 in creation order (oldest first). `list_comments` and
    `list_all_comments` project a newest-first view only when asked for one, exactly like the
    real endpoint's `orderBy=-created`, so a test can tell a correctly-ordered fetch from an
    accidentally oldest-first one.
    """
    oldest_first = [
        {
            "id": str(1000 + i),
            "author": {"displayName": f"User {i}"},
            "created": f"2026-01-{i + 1:02d}T00:00:00.000+0000",
            "body": f"comment {i}",
        }
        for i in range(total_comments)
    ]
    newest_first = list(reversed(oldest_first))
    calls: list[tuple[str, object]] = []

    class FakeClient:
        def __init__(self, _config) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, **_kwargs):
            return {
                "key": issue_key,
                "fields": {
                    "summary": "S",
                    "status": {"name": "Open", "statusCategory": {"name": "To Do"}},
                    "assignee": None,
                    "reporter": None,
                    "issuetype": {"name": "Task"},
                    "project": {"key": "FC", "name": "First Cloud Loans"},
                    "labels": [],
                    "updated": "2026-01-01T00:00:00.000+0000",
                    "created": "2026-01-01T00:00:00.000+0000",
                    "description": None,
                },
            }

        def list_comments(self, issue_key, *, limit, order_by=None, start_at=0):
            calls.append(("list_comments", order_by))
            source = newest_first if order_by == "-created" else oldest_first
            return {"comments": source[:limit], "total": total_comments}

        def list_all_comments(self, issue_key, *, order_by=None, page_size=100):
            calls.append(("list_all_comments", order_by))
            source = newest_first if order_by == "-created" else oldest_first
            return {"comments": source, "total": total_comments}

        def get_transitions(self, issue_key):
            return {"transitions": []}

    return FakeClient, calls


def test_agent_report_cli_comments_all_returns_every_comment(tmp_path, monkeypatch) -> None:
    FakeClient, _calls = _agent_report_fake_client(total_comments=18)
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["agent-report", "FC-1", "--comments", "all", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["comment_count"] == 18
    assert payload["comments_returned"] == 18
    assert payload["comments_truncated"] is False


def test_agent_report_cli_comments_int_returns_newest_and_flags_truncation(
    tmp_path, monkeypatch
) -> None:
    FakeClient, calls = _agent_report_fake_client(total_comments=18)
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["agent-report", "FC-1", "--comments", "5", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["comment_count"] == 18
    assert payload["comments_returned"] == 5
    assert payload["comments_truncated"] is True
    returned_ids = [int(c["id"]) for c in payload["comments"]]
    assert returned_ids == sorted(returned_ids, reverse=True)
    assert ("list_comments", "-created") in calls


def test_agent_report_cli_rejects_a_comments_value_that_is_neither_int_nor_all(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))

    result = CliRunner().invoke(cli.app, ["agent-report", "FC-1", "--comments", "bogus"])

    assert result.exit_code != 0
    combined = result.stdout + result.stderr
    assert "bogus" in combined
    # Old typer-level int coercion also rejects "bogus" (exit 2), but its message never mentions
    # the "all" alternative; this pins the new, readable validation message instead of that
    # coincidental overlap.
    assert "all" in combined.lower()


def test_agent_report_cli_human_output_states_both_comment_numbers(tmp_path, monkeypatch) -> None:
    FakeClient, _calls = _agent_report_fake_client(total_comments=18)
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["agent-report", "FC-1", "--comments", "5"])

    assert result.exit_code == 0, result.stdout + result.stderr
    # The exact phrase, not bare "5"/"18": one of the fake's comment timestamps legitimately
    # contains "18" (2026-01-18), so a loose substring check would pass against the old code by
    # coincidence instead of because the new summary line exists.
    assert "comments: 5 of 18 returned (truncated)" in result.stdout


def test_issue_cli_json_shows_parent_field_with_no_fields_argument(tmp_path, monkeypatch) -> None:
    full_fields = {
        "summary": "S",
        "status": {"name": "Open"},
        "assignee": None,
        "issuetype": {"name": "Story"},
        "project": {"key": "FC"},
        "updated": "2026-09-01T00:00:00.000+0000",
        "parent": {"key": "FC-349", "fields": {"summary": "Parent epic"}},
    }

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, *, fields=None, **_kwargs):
            # A real Jira response only carries what was asked for; project the fixture the
            # same way so a DEFAULT_FIELDS regression actually starves this test of "parent".
            projected = (
                {name: full_fields[name] for name in fields if name in full_fields}
                if fields
                else dict(full_fields)
            )
            return {"key": issue_key, "fields": projected}

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["issue", "FC-389", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["fields"]["parent"]["key"] == "FC-349"


def test_search_cli_limit_pages_past_one_page(tmp_path, monkeypatch) -> None:
    def make_issues(start: int, count: int) -> list[dict[str, object]]:
        return [
            {"key": f"FC-{i}", "fields": {"summary": f"s{i}"}} for i in range(start, start + count)
        ]

    pages = {
        None: {"issues": make_issues(0, 100), "isLast": False, "nextPageToken": "p2"},
        "p2": {"issues": make_issues(100, 100), "isLast": False, "nextPageToken": "p3"},
        "p3": {"issues": make_issues(200, 50), "isLast": True},
    }
    calls: list[object] = []

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def search_issues(self, jql, *, limit, fields=None, next_page_token=None, project_key=None):
            calls.append(next_page_token)
            return pages[next_page_token]

        # Reuse the real paging algorithm against the fake single-page method above, instead of
        # re-implementing `search_all_issues`'s own already-tested pagination here.
        _MAX_SEARCH_PAGES = RealJiraClient._MAX_SEARCH_PAGES
        search_all_issues = RealJiraClient.search_all_issues

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["search", "project = FC", "--limit", "250", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["returned"] == 250
    assert payload["is_last"] is True
    assert "total" not in payload
    assert calls == [None, "p2", "p3"]


def test_search_cli_all_returns_everything_and_overrides_limit(tmp_path, monkeypatch) -> None:
    fake_result = {
        "issues": [{"key": f"FC-{i}", "fields": {"summary": f"s{i}"}} for i in range(7)],
        "returned": 7,
        "isLast": True,
    }
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def search_all_issues(self, jql, *, limit=None, fields=None, **_kwargs):
            captured["limit"] = limit
            return fake_result

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    # --limit is passed alongside --all on purpose: --all must win.
    result = CliRunner().invoke(
        cli.app, ["search", "project = FC", "--all", "--limit", "10", "--json"]
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["returned"] == 7
    assert payload["is_last"] is True
    assert captured["limit"] is None


def test_search_cli_count_reports_an_explicitly_approximate_total(tmp_path, monkeypatch) -> None:
    fake_result = {
        "issues": [{"key": "FC-1", "fields": {"summary": "s"}}],
        "returned": 1,
        "isLast": True,
    }

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def search_all_issues(self, jql, **_kwargs):
            return fake_result

        def approximate_issue_count(self, jql, **_kwargs):
            return 137

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    json_result = CliRunner().invoke(cli.app, ["search", "project = FC", "--count", "--json"])
    assert json_result.exit_code == 0, json_result.stdout + json_result.stderr
    payload = json.loads(json_result.stdout)
    assert payload["approximate_total"] == 137
    assert "total" not in {k for k in payload if k == "total"}

    human_result = CliRunner().invoke(cli.app, ["search", "project = FC", "--count"])
    assert human_result.exit_code == 0, human_result.stdout + human_result.stderr
    assert "approximate" in human_result.stdout.lower()


def test_comments_cli_defaults_newest_first_and_all_pages_everything(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    oldest_first = [
        {"id": "1", "author": {"displayName": "A"}, "created": "c1", "body": "first"},
        {"id": "2", "author": {"displayName": "A"}, "created": "c2", "body": "second"},
        {"id": "3", "author": {"displayName": "A"}, "created": "c3", "body": "third"},
    ]
    newest_first = list(reversed(oldest_first))

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def list_comments(self, issue_key, *, start_at=0, limit=20, order_by=None):
            calls.append(("list_comments", order_by))
            source = newest_first if order_by == "-created" else oldest_first
            return {"comments": source[:limit], "total": 3}

        def list_all_comments(self, issue_key, *, order_by=None, page_size=100):
            calls.append(("list_all_comments", order_by))
            source = newest_first if order_by == "-created" else oldest_first
            return {"comments": source, "total": 3}

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    default_result = CliRunner().invoke(cli.app, ["comments", "FC-1", "--json"])
    assert default_result.exit_code == 0, default_result.stdout + default_result.stderr
    assert ("list_comments", "-created") in calls

    all_result = CliRunner().invoke(cli.app, ["comments", "FC-1", "--all", "--json"])
    assert all_result.exit_code == 0, all_result.stdout + all_result.stderr
    all_payload = json.loads(all_result.stdout)
    assert all_payload["returned"] == all_payload["total"] == 3


# --- Every --json/--execute command must fold dry_run into its JSON (item d) --------------------


def _commands_with_json_and_execute() -> list[str]:
    """The typer app's own registered commands, not a hand-maintained guess.

    A future command wired with both --json and --execute has to appear here automatically, so
    it cannot silently skip the dry_run rule the way edit-issue and friends used to. A bare
    ``@app.command()`` leaves ``command.name`` as ``None``; typer derives the actual dashed CLI
    name from the function name at run time, via the same helper used here.
    """
    from typer.main import get_command_name

    names = []
    for command in cli.app.registered_commands:
        params = inspect.signature(command.callback).parameters
        if "json_output" in params and "execute" in params:
            names.append(command.name or get_command_name(command.callback.__name__))
    return sorted(names)


# Commands excluded from the sweep below, and why. Keep this empty unless a command genuinely
# cannot be driven by a fake client; every command discovered above is either exercised or named
# here, so a silently skipped command fails the coverage assertion in the test instead.
_JSON_EXECUTE_SWEEP_EXCLUSIONS: dict[str, str] = {}


class _JsonExecuteSweepFakeClient:
    """One generic fake client for the dry-run sweep below.

    Every command in the sweep only reads Jira state before its dry-run return, so this fake only
    has to answer the handful of read shapes those commands ask for, keyed on the exact `fields`
    list each call site uses. It also carries `config`, because a command may run the write-scope
    check before its dry-run return rather than after — `issue-type` does, deliberately, so a
    dry run reports up front whether the write would be permitted at all.
    """

    def __init__(self, config) -> None:
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def myself(self):
        return {"accountId": "acc-1", "displayName": "Fake User", "emailAddress": "fake@x.test"}

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": "FC"}}}
        if fields == ["parent"]:
            return {"key": issue_key, "fields": {"parent": None}}
        if fields == ["description"]:
            return {"key": issue_key, "fields": {"description": _patch_description()}}
        if fields == ["description", "updated"]:
            # patch-description-section reads its concurrency-guard token in the same request
            # that reads the description, so the token cannot straddle a change the way a
            # second, separate read could.
            return {
                "key": issue_key,
                "fields": {
                    "description": _patch_description(),
                    "updated": "2026-09-14T10:00:00.000+0200",
                },
            }
        if fields == ["description", "attachment"]:
            return {
                "key": issue_key,
                "fields": {
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "codeBlock",
                                "attrs": {"language": "mermaid"},
                                "content": [{"type": "text", "text": "flowchart LR\nA --> B"}],
                            }
                        ],
                    },
                    "attachment": [],
                },
            }
        return {"key": issue_key, "fields": {}}

    def get_transitions(self, issue_key):
        return {"transitions": [{"id": "31", "name": "Done", "to": {"name": "Done"}}]}

    def get_comment(self, issue_key, comment_id):
        return {
            "id": comment_id,
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Current body"}]}
                ],
            },
        }

    def is_issue_type_allowed(self, issue_key, issue_type_name):
        return True

    def get_edit_metadata(self, issue_key):
        return {"fields": {"issuetype": {"allowedValues": [{"name": "Bug"}, {"name": "Task"}]}}}


def _json_execute_sweep_args(tmp_path) -> dict[str, list[str]]:
    attach_file = tmp_path / "sweep-attachment.txt"
    attach_file.write_text("attachment body", encoding="utf-8")
    return {
        "add-comment": ["add-comment", "FC-1", "Test body", "--json"],
        "create-test-issue": ["create-test-issue", "Test summary", "--json"],
        "create-issue": [
            "create-issue",
            "Test summary",
            "--issue-type",
            "Task",
            "--project",
            "FC",
            "--json",
        ],
        "render-mermaid": ["render-mermaid", "FC-1", "--json"],
        "edit-issue": ["edit-issue", "FC-1", "--summary", "New summary", "--json"],
        "clear-parent": ["clear-parent", "FC-1", "--json"],
        "patch-description-section": [
            "patch-description-section",
            "FC-1",
            "--section",
            "Observed",
            "--content",
            "## Observed\n\nUpdated for the sweep.",
            "--json",
        ],
        "assign": ["assign", "FC-1", "--json"],
        "attach": ["attach", "FC-1", "--file", str(attach_file), "--json"],
        "transition": ["transition", "FC-1", "Done", "--json"],
        "edit-comment": [
            "edit-comment",
            "FC-1",
            "667647",
            "--body",
            "Corrected body.",
            "--json",
        ],
        "delete-comment": ["delete-comment", "FC-1", "667647", "--json"],
        "issue-type": ["issue-type", "FC-1", "Bug", "--json"],
        "sprint-add": ["sprint-add", "7340", "FC-1", "--json"],
        "sprint-remove": ["sprint-remove", "FC-1", "--json"],
    }


def test_every_json_and_execute_command_carries_dry_run_true_in_its_json(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _JsonExecuteSweepFakeClient)
    monkeypatch.setattr(
        cli,
        "discover_mermaid_runtime",
        lambda *_a, **_k: SimpleNamespace(
            node=Path("node"), cli=Path("mmdc"), browser=Path("chromium")
        ),
    )

    discovered = _commands_with_json_and_execute()
    args_by_command = _json_execute_sweep_args(tmp_path)
    uncovered = set(discovered) - set(args_by_command) - set(_JSON_EXECUTE_SWEEP_EXCLUSIONS)
    assert not uncovered, f"no dry-run args registered for: {sorted(uncovered)}"

    for name in discovered:
        if name in _JSON_EXECUTE_SWEEP_EXCLUSIONS:
            continue
        result = CliRunner().invoke(cli.app, args_by_command[name])
        assert result.exit_code == 0, f"{name}: {result.stdout}{result.stderr}"
        payload = json.loads(result.stdout)
        assert payload["dry_run"] is True, f"{name}: {payload}"
