from __future__ import annotations

import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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


# --- `transition` / `transitions` CLI commands (FC-492 backwards-move incident) --------------


class _TransitionFakeClient:
    """One configurable fake for the `transition`/`transitions` CLI command tests.

    `status_category=None` reproduces an issue read that could not resolve a `statusCategory`
    (the "direction could not be determined" case); every other case names the category directly.
    """

    def __init__(
        self,
        config,
        *,
        transitions: dict[str, object],
        status_category: str | None = "To Do",
        write_project: str = "FC",
        transition_result: dict[str, object] | None = None,
    ) -> None:
        self.config = config
        self._transitions = transitions
        self._status_category = status_category
        self._write_project = write_project
        self._transition_result = {} if transition_result is None else transition_result
        self.captured_transition_issue: dict[str, object] | None = None

    def __enter__(self) -> _TransitionFakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get_transitions(self, _issue_key: str) -> dict[str, object]:
        return self._transitions

    def get_issue(self, issue_key: str, *, fields=None, expand=None) -> dict[str, object]:
        if fields == ["status"]:
            status_fields = (
                {} if self._status_category is None
                else {"statusCategory": {"name": self._status_category}}
            )
            return {"key": issue_key, "fields": {"status": status_fields}}
        if fields == ["project"]:
            return {"key": issue_key, "fields": {"project": {"key": self._write_project}}}
        return {"key": issue_key, "fields": {}}

    def transition_issue(self, issue_key, transition_id, *, fields=None):
        self.captured_transition_issue = {
            "issue_key": issue_key,
            "transition_id": transition_id,
            "fields": fields,
        }
        return self._transition_result


def _patch_transition_client(monkeypatch, fake: _TransitionFakeClient) -> None:
    """Patch `cli.JiraClient` to return `fake`, binding the REAL config `_client()` loads onto
    it -- `ensure_issue_write_allowed` reads `client.config.write_issue_keys` /
    `write_project_keys`, so the fake must carry the actual `JiraConfig`, not the placeholder
    passed to its constructor.
    """

    def _factory(config: JiraConfig) -> _TransitionFakeClient:
        fake.config = config
        return fake

    monkeypatch.setattr(cli, "JiraClient", _factory)


def _transition_config(tmp_path) -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


def test_transition_cli_allows_a_forward_move_without_the_flag(tmp_path, monkeypatch) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "21",
                    "name": "Start Progress",
                    "to": {"name": "In Progress", "statusCategory": {"name": "In Progress"}},
                }
            ]
        },
        status_category="To Do",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transition", "FC-1", "Start Progress", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["current_category"] == "To Do"
    assert payload["transition"]["name"] == "Start Progress"


def test_transition_cli_allows_wont_do_straight_to_do_to_done(tmp_path, monkeypatch) -> None:
    """FC-458's real, probed transition: To Do straight to Done, skipping In Progress. Forward,
    must be allowed without --allow-backwards-transition.
    """
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "811",
                    "name": "Won't Do",
                    "to": {"name": "Closed", "statusCategory": {"name": "Done"}},
                    "fields": {
                        "resolution": {
                            "required": True,
                            "schema": {"type": "resolution", "system": "resolution"},
                            "allowedValues": [
                                {"id": "10000", "name": "Done"},
                                {"id": "10001", "name": "Won't Do"},
                            ],
                        }
                    },
                }
            ]
        },
        status_category="To Do",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(
        cli.app,
        ["transition", "FC-458", "Won't Do", "--field", "resolution=Won't Do", "--json"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["fields"] == {"resolution": {"name": "Won't Do"}}


def test_transition_cli_refuses_a_backwards_move_without_the_flag(tmp_path, monkeypatch) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "11",
                    "name": "Reopen",
                    "to": {"name": "Open", "statusCategory": {"name": "To Do"}},
                }
            ]
        },
        status_category="In Progress",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transition", "FC-492", "Reopen", "--json"])

    assert result.exit_code == cli.TRANSITION_DIRECTION_REFUSED_EXIT_CODE, (
        result.stdout + result.stderr
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error_type"] == "transition_direction_refused"
    assert payload["direction"] == "backward"
    assert payload["current_category"] == "In Progress"
    assert payload["target_category"] == "To Do"
    assert fake.captured_transition_issue is None  # refused before any write


def test_transition_cli_allows_a_backwards_move_with_the_flag(tmp_path, monkeypatch) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "11",
                    "name": "Reopen",
                    "to": {"name": "Open", "statusCategory": {"name": "To Do"}},
                }
            ]
        },
        status_category="In Progress",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(
        cli.app,
        ["transition", "FC-492", "Reopen", "--allow-backwards-transition", "--execute", "--json"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert fake.captured_transition_issue == {
        "issue_key": "FC-492",
        "transition_id": "11",
        "fields": None,
    }


def test_transition_cli_allows_a_same_category_move_without_the_flag(tmp_path, monkeypatch) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "5",
                    "name": "Reassign",
                    "to": {"name": "In Progress", "statusCategory": {"name": "In Progress"}},
                }
            ]
        },
        status_category="In Progress",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transition", "FC-1", "Reassign", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr


def test_transition_cli_refuses_when_current_category_cannot_be_read(tmp_path, monkeypatch) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "21",
                    "name": "Start Progress",
                    "to": {"name": "In Progress", "statusCategory": {"name": "In Progress"}},
                }
            ]
        },
        status_category=None,
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transition", "FC-1", "Start Progress", "--json"])

    assert result.exit_code == cli.TRANSITION_DIRECTION_REFUSED_EXIT_CODE, (
        result.stdout + result.stderr
    )
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "transition_direction_refused"
    assert payload["direction"] == "unknown"
    assert payload["current_category"] is None

    # The explicit flag is required to proceed, exactly as for a confirmed backward move.
    result2 = CliRunner().invoke(
        cli.app,
        ["transition", "FC-1", "Start Progress", "--allow-backwards-transition", "--json"],
    )
    assert result2.exit_code == 0, result2.stdout + result2.stderr


def test_transition_cli_unmatched_name_lists_available_transitions_with_categories(
    tmp_path, monkeypatch
) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "21",
                    "name": "Start Progress",
                    "to": {"name": "In Progress", "statusCategory": {"name": "In Progress"}},
                },
                {
                    "id": "31",
                    "name": "Done",
                    "to": {"name": "Done", "statusCategory": {"name": "Done"}},
                },
            ]
        },
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transition", "FC-1", "Bogus Name", "--json"])

    assert result.exit_code == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "transition_not_found"
    assert payload["available_transitions"] == [
        {"id": "21", "name": "Start Progress", "to": "In Progress", "to_category": "In Progress"},
        {"id": "31", "name": "Done", "to": "Done", "to_category": "Done"},
    ]


def test_transition_cli_empty_transitions_list_says_no_permission_or_no_applicable_transition(
    tmp_path, monkeypatch
) -> None:
    """Probed fact: FC-425 and FC-389 both returned an EMPTY transitions list from the live
    instance -- the common case, not the edge case. Must say so explicitly, not just fail.
    """
    fake = _TransitionFakeClient(None, transitions={"transitions": []})
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transition", "FC-425", "Anything", "--json"])

    assert result.exit_code == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "transition_not_found"
    assert payload["available_transitions"] == []
    assert "no permission or no applicable transition" in payload["message"]
    assert "not the end of the workflow" in payload["message"]


def test_transition_cli_refuses_before_writing_when_a_required_field_is_missing(
    tmp_path, monkeypatch
) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "811",
                    "name": "Won't Do",
                    "to": {"name": "Closed", "statusCategory": {"name": "Done"}},
                    "fields": {
                        "resolution": {
                            "required": True,
                            "schema": {"type": "resolution", "system": "resolution"},
                            "allowedValues": [
                                {"id": "10000", "name": "Done"},
                                {"id": "10001", "name": "Won't Do"},
                            ],
                        }
                    },
                }
            ]
        },
        status_category="To Do",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transition", "FC-458", "Won't Do", "--json"])

    assert result.exit_code == 1, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "transition_fields_missing"
    assert payload["missing_fields"] == [
        {"field": "resolution", "allowed_values": ["Done", "Won't Do"]}
    ]
    assert fake.captured_transition_issue is None  # refused before any write


def test_transition_cli_refuses_a_blank_required_field_before_writing(
    tmp_path, monkeypatch
) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "811",
                    "name": "Won't Do",
                    "to": {"name": "Closed", "statusCategory": {"name": "Done"}},
                    "fields": {"resolution": {"required": True, "schema": {"type": "resolution"}}},
                }
            ]
        },
        status_category="To Do",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(
        cli.app, ["transition", "FC-458", "Won't Do", "--field", "resolution=   ", "--json"]
    )

    assert result.exit_code == 1, result.stdout + result.stderr
    assert json.loads(result.stdout)["error_type"] == "transition_fields_missing"
    assert fake.captured_transition_issue is None


def test_transition_cli_execute_with_a_supplied_required_field_writes_it(
    tmp_path, monkeypatch
) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "811",
                    "name": "Won't Do",
                    "to": {"name": "Closed", "statusCategory": {"name": "Done"}},
                    "fields": {
                        "resolution": {
                            "required": True,
                            "schema": {"type": "resolution", "system": "resolution"},
                            "allowedValues": [
                                {"id": "10000", "name": "Done"},
                                {"id": "10001", "name": "Won't Do"},
                            ],
                        }
                    },
                }
            ]
        },
        status_category="To Do",
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(
        cli.app,
        [
            "transition",
            "FC-458",
            "Won't Do",
            "--field",
            "resolution=Won't Do",
            "--execute",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert fake.captured_transition_issue == {
        "issue_key": "FC-458",
        "transition_id": "811",
        "fields": {"resolution": {"name": "Won't Do"}},
    }


def test_transition_cli_rejects_a_malformed_field_option(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))

    result = CliRunner().invoke(cli.app, ["transition", "FC-1", "Done", "--field", "no-equals"])

    assert result.exit_code == 2, result.stdout + result.stderr


def test_transitions_cli_lists_category_column_and_labels_an_empty_list(
    tmp_path, monkeypatch
) -> None:
    fake = _TransitionFakeClient(
        None,
        transitions={
            "transitions": [
                {
                    "id": "21",
                    "name": "Start Progress",
                    "to": {"name": "In Progress", "statusCategory": {"name": "In Progress"}},
                }
            ]
        },
    )
    monkeypatch.setattr(cli, "load_config", lambda: _transition_config(tmp_path))
    _patch_transition_client(monkeypatch, fake)

    result = CliRunner().invoke(cli.app, ["transitions", "FC-1"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "In Progress" in result.stdout

    empty_fake = _TransitionFakeClient(None, transitions={"transitions": []})
    _patch_transition_client(monkeypatch, empty_fake)
    empty_result = CliRunner().invoke(cli.app, ["transitions", "FC-425"])
    assert empty_result.exit_code == 0, empty_result.stdout + empty_result.stderr
    # Normalize whitespace: Rich wraps this sentence across lines at terminal width, so a raw
    # substring check would pass or fail on wrap position rather than content.
    normalized = " ".join(empty_result.stdout.split())
    assert "no permission or no applicable transition" in normalized


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
            self.config = _config

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
            self.config = _config

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
        assert fields == ["description", "attachment", "updated"]
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
                "updated": "2026-09-14T10:00:00.000+0200",
            },
        }

    def attachment_media_id(self, attachment_id, *, issue_key):
        assert (attachment_id, issue_key) == ("170501", "FC-349")
        return "media-fc349"

    def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
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
            "--json",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload == {
        "error_type": "section_patch_refused",
        "swallowed_headings": ["Observed", "Acceptance criteria"],
        "expect_swallowed_headings": [],
        "whole_document": True,
        "dry_run": True,
    }
    assert writes == []


def test_patch_description_section_cli_refusal_prints_no_json_without_the_flag(
    tmp_path, monkeypatch
) -> None:
    """F12 regression: the refusal used to dump a JSON document to stdout unconditionally,
    even when the caller never passed --json. Without --json, stdout must stay empty and the
    detail must appear only as prose on stderr, matching every other refusal in this command.
    """
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
    assert result.stdout == ""
    assert "Patch refused:" in result.stderr
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
            self.config = _config

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
                "labels_add": None,
                "labels_remove": None,
                "fix_versions_add": None,
                "fix_versions_remove": None,
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
            self.config = _config

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
                    "fields": {
                        "project": {"key": "FC"},
                        "description": self.document,
                        "updated": "2026-09-14T10:00:00.000+0200",
                    },
                }

            def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
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
                "--json",
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


def test_issue_cli_json_api_error_emits_a_document_on_stdout_not_empty(
    tmp_path, monkeypatch
) -> None:
    """F5 regression: `_exit_api_error` used to write only to stderr, so `issue X --json`
    against a 404 exited non-zero with an EMPTY stdout. `json.load`/`jq` on that empty stream
    raises `Expecting value: line 1 column 1` — exactly the failure an agent hits when it
    trusts stdout on a --json invocation that failed.
    """
    from jira_agent.client import JiraApiError

    class FakeClient:
        def __init__(self, _config):
            self.config = _config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, **_kwargs):
            raise JiraApiError(f"Issue does not exist: {issue_key}", status_code=404)

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["issue", "FC-999", "--json"])

    assert result.exit_code == 1
    assert result.stdout != ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error_type"] == "jira_api_error"
    assert payload["status_code"] == 404


def test_issue_cli_no_json_api_error_leaves_stdout_empty(tmp_path, monkeypatch) -> None:
    """Without --json, the error stays prose-only on stderr; stdout is untouched, exactly the
    behaviour before `json_mode` existed on `_exit_api_error`.
    """
    from jira_agent.client import JiraApiError

    class FakeClient:
        def __init__(self, _config):
            self.config = _config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def get_issue(self, issue_key, **_kwargs):
            raise JiraApiError(f"Issue does not exist: {issue_key}", status_code=404)

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["issue", "FC-999"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Jira API error" in result.stderr


def test_assign_cli_json_safety_error_emits_a_document_on_stdout(tmp_path, monkeypatch) -> None:
    """F5 regression: `_exit_safety_error` had the identical empty-stdout defect for the
    write-safety refusal path, on every command that both supports --json and is gated by
    `ensure_issue_write_allowed`.
    """

    class FakeClient:
        def __init__(self, _config):
            self.config = _config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def myself(self):
            return {"accountId": "acc-1", "displayName": "Ada"}

        def get_issue(self, issue_key, **_kwargs):
            return {"key": issue_key, "fields": {"project": {"key": "OTHER"}}}

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["assign", "FC-1", "--execute", "--json"])

    assert result.exit_code == 3
    assert result.stdout != ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error_type"] == "write_safety_error"


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
            self.config = _config

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
        "is_last": False,
        "nextPageToken": "token-abc-123",
    }

    class FakeClient:
        def __init__(self, _config):
            self.config = _config

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


# --- R4: every search-backed command shares one curated shape, not just `search` ----------------
#
# Before this fix, `_run_search` took a `curated: bool = False` parameter and only `search`
# passed `curated=True`. `recent`, `mine`, `fc-open`, `by-status`, and `test-scope` emitted the
# raw Jira `/search/jql` response as-is under `--json` (never `returned`/`is_last`, sometimes a
# bare `nextPageToken` prose line, never a page-past-one-page walk) — the identical
# unverifiable-completeness defect `search` was fixed for, still live in five neighbouring
# commands whose MCP twins (`jira_recent_issues`, `jira_mine`, `jira_fc_open`, `jira_by_status`,
# `jira_test_scope_issues`) already used the curated shape. `curated` no longer exists: every
# search-backed command goes through the same `_run_search` path `search` always did.


class _CuratedSearchFakeClient:
    """A fake client whose `search_all_issues` return value proves the CLI command called
    through the paging path (not a raw single-page `search_issues` passthrough) and whose shape
    matches what `JiraClient.search_all_issues` itself now reports: `issues`, `returned`,
    `is_last`, never `total`.
    """

    def __init__(self, _config):
        self.config = _config
        self.search_all_issues_calls: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def search_all_issues(self, jql, **kwargs):
        self.search_all_issues_calls.append({"jql": jql, **kwargs})
        return {
            "issues": [{"key": "FC-9"}],
            "returned": 1,
            "is_last": True,
        }


@pytest.mark.parametrize(
    "args",
    [
        ["recent", "--json"],
        ["mine", "--json"],
        ["fc-open", "--json"],
        ["by-status", "In Progress", "--json"],
        ["test-scope", "--json"],
    ],
    ids=["recent", "mine", "fc-open", "by-status", "test-scope"],
)
def test_search_backed_commands_share_searchs_curated_json_shape(
    tmp_path, monkeypatch, args
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _CuratedSearchFakeClient)

    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["issues"] == [{"key": "FC-9"}]
    assert payload["returned"] == 1
    assert payload["is_last"] is True
    assert "total" not in payload
    # search_all_issues, not a raw single-page search_issues call: proves the paging path ran.
    assert "search_all_issues_calls" not in payload


@pytest.mark.parametrize(
    "args",
    [
        ["recent"],
        ["mine"],
        ["fc-open"],
        ["by-status", "In Progress"],
        ["test-scope"],
        ["search", "project = FC"],
    ],
    ids=["recent", "mine", "fc-open", "by-status", "test-scope", "search"],
)
def test_no_search_backed_command_emits_a_total(tmp_path, monkeypatch, args) -> None:
    """`/search/jql` never reports a match count; no curated command may invent one."""
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _CuratedSearchFakeClient)

    result = CliRunner().invoke(cli.app, [*args, "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert "total" not in payload
    assert "returned" in payload
    assert "is_last" in payload


def test_recent_and_mine_limit_options_have_no_upper_cap(tmp_path, monkeypatch) -> None:
    """R4: there is no external compatibility burden here, so the old `max=100` ceiling that
    `search --limit` never had (and that silently truncated a caller asking for more without any
    way to detect the truncation) is lifted from its five neighbouring commands too.
    """
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _CuratedSearchFakeClient)

    for args in (
        ["recent", "--limit", "500", "--json"],
        ["mine", "--limit", "500", "--json"],
        ["fc-open", "--limit", "500", "--json"],
        ["by-status", "In Progress", "--limit", "500", "--json"],
    ):
        result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 0, f"{args}: {result.stdout}{result.stderr}"


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


def test_agent_report_cli_human_output_states_both_comment_numbers_and_order(
    tmp_path, monkeypatch
) -> None:
    """G4 regression: human mode used to print the count and then a bare comment list, with no
    signal that `--comments all` changed from oldest-first to newest-first. Anyone who learned
    "the last line is the newest" would misread the oldest comment as the newest. The order must
    now sit right next to the count, not only reach `--json`'s `comments_order` key."""
    FakeClient, _calls = _agent_report_fake_client(total_comments=18)
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["agent-report", "FC-1", "--comments", "5"])

    assert result.exit_code == 0, result.stdout + result.stderr
    # The exact phrase, not bare "5"/"18": one of the fake's comment timestamps legitimately
    # contains "18" (2026-01-18), so a loose substring check would pass against the old code by
    # coincidence instead of because the new summary line exists.
    assert "comments: 5 of 18 returned (truncated, newest-first)" in result.stdout


def test_agent_report_cli_human_output_states_order_when_complete_too(
    tmp_path, monkeypatch
) -> None:
    FakeClient, _calls = _agent_report_fake_client(total_comments=18)
    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    result = CliRunner().invoke(cli.app, ["agent-report", "FC-1", "--comments", "all"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "comments: 18 of 18 returned (complete, newest-first)" in result.stdout


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
            self.config = _config

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
            self.config = _config

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
        "is_last": True,
    }
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _config):
            self.config = _config

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
        "is_last": True,
    }

    class FakeClient:
        def __init__(self, _config):
            self.config = _config

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


def test_search_cli_count_reports_a_missing_count_instead_of_dropping_it(
    tmp_path, monkeypatch
) -> None:
    """F7 regression: Jira's approximate-count response can omit `count` (`data.get("count")`
    is then `None`). The CLI used to test `approximate_total is not None` to decide whether to
    show the field at all, so a missing count silently disappeared from both --json and human
    output — `search --count --json` answered nothing about the question it was asked. The fix
    keys off whether --count was requested, not off whether Jira answered it.
    """
    fake_result = {
        "issues": [{"key": "FC-1", "fields": {"summary": "s"}}],
        "returned": 1,
        "is_last": True,
    }

    class FakeClient:
        def __init__(self, _config):
            self.config = _config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def search_all_issues(self, jql, **_kwargs):
            return fake_result

        def approximate_issue_count(self, jql, **_kwargs):
            return None

    monkeypatch.setattr(cli, "load_config", lambda: _json_purity_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", FakeClient)

    json_result = CliRunner().invoke(cli.app, ["search", "project = FC", "--count", "--json"])
    assert json_result.exit_code == 0, json_result.stdout + json_result.stderr
    payload = json.loads(json_result.stdout)
    assert "approximate_total" in payload
    assert payload["approximate_total"] is None

    human_result = CliRunner().invoke(cli.app, ["search", "project = FC", "--count"])
    assert human_result.exit_code == 0, human_result.stdout + human_result.stderr
    assert "approximate match count: unavailable" in human_result.stdout


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
            self.config = _config

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
        if fields == ["customfield_10020"]:
            # sprint-remove's precondition (and sprint-add/-remove's post-write reread) reads
            # this field through `get_issue_sprints`, reused below from the real client.
            return {
                "key": issue_key,
                "fields": {
                    "customfield_10020": [
                        {"id": 7340, "name": "Sweep Sprint", "state": "active"}
                    ]
                },
            }
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
        if fields == ["status"]:
            # `transition`'s direction check reads this. Same category as get_transitions' own
            # "Done" target below, so the sweep's dry run is a same-category move: allowed
            # without --allow-backwards-transition, exactly like every other command in this
            # sweep expects a clean dry run with no extra flag.
            return {"key": issue_key, "fields": {"status": {"statusCategory": {"name": "Done"}}}}
        return {"key": issue_key, "fields": {}}

    def get_transitions(self, issue_key):
        return {
            "transitions": [
                {
                    "id": "31",
                    "name": "Done",
                    "to": {"name": "Done", "statusCategory": {"name": "Done"}},
                }
            ]
        }

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

    # sprint-remove's precondition check calls this; reuse the real, already-tested parsing
    # logic on top of this fake's own `get_issue` rather than reimplementing it.
    get_issue_sprints = RealJiraClient.get_issue_sprints


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
        "sprint-remove": ["sprint-remove", "7340", "FC-1", "--json"],
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


def test_json_help_no_longer_claims_raw_response_for_curated_commands() -> None:
    """G5 regression: `recent`, `mine`, `assigned-to-me`, `fc-open`, `by-status`, `search`,
    `test-scope`, `issue-sprint`, and `issue-type-options` each synthesize a curated result
    document under `--json` (see `_run_search`, and each command's own body for the other two),
    but their `--json` help text used to say "Print the raw Jira JSON response" -- copied from
    the many commands where that literally is what happens. Checks the click-level help string
    directly, not rendered `--help` output, since Rich wraps that across lines and would make a
    substring check pass or fail on line-wrap width rather than content.
    """
    import typer as typer_module

    click_app = typer_module.main.get_command(cli.app)
    stale_claim = "print the raw jira json response"
    curated_commands = {
        "recent",
        "mine",
        "assigned-to-me",
        "fc-open",
        "by-status",
        "search",
        "test-scope",
        "issue-sprint",
        "issue-type-options",
    }
    for name in curated_commands:
        command = click_app.commands[name]
        json_param = next(p for p in command.params if p.name == "json_output")
        help_text = (json_param.help or "").lower()
        assert stale_claim not in help_text, f"{name}: {json_param.help!r}"
        assert "curated" in help_text, f"{name}: {json_param.help!r}"
