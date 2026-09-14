"""Tests for the sprint/board commands and the issue-type command (Task 1 and Task 2).

Every test drives the Typer app through ``CliRunner`` against a small hand-written fake client,
following the monkeypatch style already used in ``tests/test_cli_helpers.py``: ``cli.load_config``
and ``cli.JiraClient`` are replaced so nothing here touches the network.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from jira_agent import cli
from jira_agent.client import JiraConcurrentEditError
from jira_agent.config import JiraConfig


def _config(tmp_path, *, write_project_keys: str = "FC") -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": write_project_keys,
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


def _install(monkeypatch, config, client) -> None:
    # cli.JiraClient(config) is stubbed to hand back the already-built fake client below, so its
    # own constructor never sees `config`; write commands read `client.config`, so bind it here.
    client.config = config
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", lambda _config: client)


# --- boards / sprints ------------------------------------------------------------------------


class _BoardsFakeClient:
    def __init__(
        self,
        _config=None,
        *,
        is_last: bool = True,
        total: int | None = None,
        values: list[dict[str, object]] | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._is_last = is_last
        self._total = total
        self._values = (
            values
            if values is not None
            else [{"id": 814, "name": "First Mortgage Loan", "type": "scrum"}]
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def list_boards(self, *, board_type=None, name=None, max_results=50):
        self.calls.append({"board_type": board_type, "name": name, "max_results": max_results})
        response: dict[str, object] = {
            "maxResults": 50,
            "startAt": 0,
            "isLast": self._is_last,
            "values": self._values,
        }
        if self._total is not None:
            response["total"] = self._total
        return response


def test_boards_cli_renders_probed_shape(tmp_path, monkeypatch) -> None:
    _install(monkeypatch, _config(tmp_path), _BoardsFakeClient())

    result = CliRunner().invoke(cli.app, ["boards"])

    assert result.exit_code == 0, result.output
    assert "814" in result.stdout
    assert "First Mortgage Loan" in result.stdout
    assert "scrum" in result.stdout


def test_boards_cli_supports_json_and_forwards_filters(tmp_path, monkeypatch) -> None:
    client = _BoardsFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["boards", "--type", "scrum", "--name", "First", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["values"][0]["id"] == 814
    assert client.calls == [{"board_type": "scrum", "name": "First", "max_results": 50}]


def test_boards_cli_reports_more_remain_when_not_last_page(tmp_path, monkeypatch) -> None:
    """F4 regression: `boards` used to print only `values`, with no signal that more boards
    exist, even though the Agile response it already holds carries `isLast`. A board with more
    boards than `--max-results` (default 50) showed 50 and said nothing.
    """
    client = _BoardsFakeClient(is_last=False)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["boards"])

    assert result.exit_code == 0, result.output
    assert "returned 1 board(s); more remain" in result.stdout


def test_boards_cli_reports_complete_when_last_page(tmp_path, monkeypatch) -> None:
    client = _BoardsFakeClient(is_last=True)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["boards"])

    assert result.exit_code == 0, result.output
    assert "returned 1 board(s); complete" in result.stdout


def test_boards_cli_json_carries_returned_and_is_last(tmp_path, monkeypatch) -> None:
    client = _BoardsFakeClient(is_last=False)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["boards", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["returned"] == 1
    assert payload["is_last"] is False


def test_boards_cli_never_fabricates_a_total(tmp_path, monkeypatch) -> None:
    client = _BoardsFakeClient(is_last=False)  # no "total" in the fake Agile response
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["boards"])

    assert result.exit_code == 0, result.output
    assert "total" not in result.stdout.lower()


def test_boards_cli_reports_a_real_total_when_jira_supplies_one(tmp_path, monkeypatch) -> None:
    client = _BoardsFakeClient(is_last=False, total=120)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["boards"])

    assert result.exit_code == 0, result.output
    assert "total: 120" in result.stdout
    assert "Jira" in result.stdout


class _SprintsFakeClient:
    def __init__(
        self,
        _config=None,
        *,
        is_last: bool | None = None,
        total: int | None = None,
        values: list[dict[str, object]] | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._is_last = is_last
        self._total = total
        self._values = (
            values
            if values is not None
            else [
                {
                    "id": 7340,
                    "name": "10 SEPA-Mandate, Disbursement",
                    "state": "active",
                    "boardId": 814,
                    "startDate": "2026-08-01T00:00:00.000Z",
                    "endDate": "2026-08-15T00:00:00.000Z",
                }
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def list_board_sprints(self, board_id, *, state=None, max_results=50):
        self.calls.append({"board_id": board_id, "state": state, "max_results": max_results})
        response: dict[str, object] = {"values": self._values}
        if self._is_last is not None:
            response["isLast"] = self._is_last
        if self._total is not None:
            response["total"] = self._total
        return response


def test_sprints_cli_renders_probed_shape_and_supports_json(tmp_path, monkeypatch) -> None:
    client = _SprintsFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprints", "--board", "814", "--state", "active"])

    assert result.exit_code == 0, result.output
    assert "7340" in result.stdout
    assert "10 SEPA-Mandate, Disbursement" in result.stdout
    assert "active" in result.stdout
    assert client.calls == [{"board_id": 814, "state": "active", "max_results": 50}]

    result_json = CliRunner().invoke(cli.app, ["sprints", "--board", "814", "--json"])
    assert result_json.exit_code == 0, result_json.stdout + result_json.stderr
    payload = json.loads(result_json.stdout)
    assert payload["values"][0]["name"] == "10 SEPA-Mandate, Disbursement"


def test_sprints_cli_reports_more_remain_when_not_last_page(tmp_path, monkeypatch) -> None:
    """F4 regression: `sprints` used to print only `values`, with no signal that more sprints
    exist, even though the Agile response it already holds carries `isLast`. A board with more
    sprints than `--max-results` (default 50) showed 50 and said nothing.
    """
    client = _SprintsFakeClient(is_last=False)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprints", "--board", "814"])

    assert result.exit_code == 0, result.output
    assert "returned 1 sprint(s); more remain" in result.stdout


def test_sprints_cli_reports_complete_when_last_page(tmp_path, monkeypatch) -> None:
    client = _SprintsFakeClient(is_last=True)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprints", "--board", "814"])

    assert result.exit_code == 0, result.output
    assert "returned 1 sprint(s); complete" in result.stdout


def test_sprints_cli_json_carries_returned_and_is_last(tmp_path, monkeypatch) -> None:
    client = _SprintsFakeClient(is_last=False)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprints", "--board", "814", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["returned"] == 1
    assert payload["is_last"] is False


def test_sprints_cli_reports_a_real_total_when_jira_supplies_one(tmp_path, monkeypatch) -> None:
    client = _SprintsFakeClient(is_last=False, total=42)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprints", "--board", "814"])

    assert result.exit_code == 0, result.output
    assert "total: 42" in result.stdout
    assert "Jira" in result.stdout


# --- issue-sprint: three distinct states, never collapsed -------------------------------------


class _IssueSprintFakeClient:
    def __init__(self, fields: dict[str, object]) -> None:
        self._fields = fields

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        return {"key": issue_key, "fields": self._fields}


def test_issue_sprint_cli_issue_in_a_sprint(tmp_path, monkeypatch) -> None:
    sprint = {
        "id": 7340,
        "name": "10 SEPA-Mandate, Disbursement",
        "state": "active",
        "boardId": 814,
        "startDate": "2026-08-01T00:00:00.000Z",
        "endDate": "2026-08-15T00:00:00.000Z",
    }
    _install(
        monkeypatch,
        _config(tmp_path),
        _IssueSprintFakeClient({"customfield_10020": [sprint]}),
    )

    result = CliRunner().invoke(cli.app, ["issue-sprint", "FC-1", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "10 SEPA-Mandate, Disbursement" in result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "issue": "FC-1",
        "sprint_field": "customfield_10020",
        "sprint_field_present": True,
        "sprints": [sprint],
    }


def test_issue_sprint_cli_empty_sprint_list_says_no_sprint(tmp_path, monkeypatch) -> None:
    _install(
        monkeypatch,
        _config(tmp_path),
        _IssueSprintFakeClient({"customfield_10020": []}),
    )

    result = CliRunner().invoke(cli.app, ["issue-sprint", "FC-2", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    # Prose goes to stderr under --json so stdout carries only the JSON document.
    assert "is in no sprint" in result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "issue": "FC-2",
        "sprint_field": "customfield_10020",
        "sprint_field_present": True,
        "sprints": [],
    }


def test_issue_sprint_cli_field_absent_is_a_different_state_than_empty_list(
    tmp_path, monkeypatch
) -> None:
    _install(
        monkeypatch,
        _config(tmp_path),
        _IssueSprintFakeClient({}),  # no customfield_10020 key at all
    )

    result = CliRunner().invoke(cli.app, ["issue-sprint", "FC-3", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    # Prose goes to stderr under --json so stdout carries only the JSON document.
    assert "did not include the sprint field" in result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "issue": "FC-3",
        "sprint_field": "customfield_10020",
        "sprint_field_present": False,
        "sprints": [],
    }


# --- sprint-add ---------------------------------------------------------------------------------


class _SprintAddFakeClient:
    def __init__(self, _config=None, *, disallowed_project_for: str | None = None) -> None:
        self.config = _config
        self._disallowed_project_for = disallowed_project_for
        self.add_calls: list[tuple[object, list[str]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        assert fields == ["project"]
        project = "OTHER" if issue_key == self._disallowed_project_for else "FC"
        return {"key": issue_key, "fields": {"project": {"key": project}}}

    def add_issues_to_sprint(self, sprint_id, issue_keys):
        self.add_calls.append((sprint_id, list(issue_keys)))
        return {"ok": True}


def test_sprint_add_cli_dry_run_writes_nothing_and_names_every_issue(tmp_path, monkeypatch) -> None:
    client = _SprintAddFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprint-add", "7340", "FC-1", "FC-2", "FC-3", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "FC-1" in result.stdout
    assert "FC-2" in result.stdout
    assert "FC-3" in result.stdout
    payload = json.loads(result.stdout)
    assert payload == {
        "sprint_id": 7340,
        "issue_keys": ["FC-1", "FC-2", "FC-3"],
        "safety": [
            {"key": "FC-1", "project": "FC", "allowed_by": "project allowlist"},
            {"key": "FC-2", "project": "FC", "allowed_by": "project allowlist"},
            {"key": "FC-3", "project": "FC", "allowed_by": "project allowlist"},
        ],
        "dry_run": True,
    }
    assert client.add_calls == []


def test_sprint_add_cli_execute_calls_client_once_with_all_keys_and_audits(
    tmp_path, monkeypatch
) -> None:
    client = _SprintAddFakeClient()
    audited: list[dict[str, object]] = []
    _install(monkeypatch, _config(tmp_path), client)
    monkeypatch.setattr(
        cli,
        "audit_write",
        lambda config, **kwargs: audited.append(kwargs),
    )

    result = CliRunner().invoke(
        cli.app, ["sprint-add", "7340", "FC-1", "FC-2", "FC-3", "--execute", "--json"]
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.add_calls == [(7340, ["FC-1", "FC-2", "FC-3"])]
    assert len(audited) == 1
    assert audited[0]["action"] == "sprint_add"
    assert audited[0]["issue_key"] is None
    assert audited[0]["payload"] == {"sprint_id": 7340, "issue_keys": ["FC-1", "FC-2", "FC-3"]}
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is False


def test_sprint_add_cli_safety_gate_rejects_one_key_no_write_for_any(tmp_path, monkeypatch) -> None:
    # Five keys, the fourth belongs to a project that is not on the write allow-list.
    client = _SprintAddFakeClient(disallowed_project_for="FC-4")
    audited: list[dict[str, object]] = []
    _install(monkeypatch, _config(tmp_path), client)
    monkeypatch.setattr(cli, "audit_write", lambda config, **kwargs: audited.append(kwargs))

    result = CliRunner().invoke(
        cli.app,
        ["sprint-add", "7340", "FC-1", "FC-2", "FC-3", "FC-4", "FC-5", "--execute"],
    )

    assert result.exit_code == 3, result.stdout + result.stderr
    assert client.add_calls == []
    assert audited == []


def test_sprint_add_cli_dry_run_also_catches_the_safety_gate(tmp_path, monkeypatch) -> None:
    """F9 regression: the dry run used to return before ever opening a client, so it never ran
    the write-safety gate `--execute` runs. A dry run that reports success for a batch
    `--execute` would go on to refuse does not predict what `--execute` does, which is the whole
    point of a dry run.
    """
    client = _SprintAddFakeClient(disallowed_project_for="FC-4")
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(
        cli.app,
        ["sprint-add", "7340", "FC-1", "FC-2", "FC-3", "FC-4", "FC-5"],
    )

    assert result.exit_code == 3, result.stdout + result.stderr
    assert client.add_calls == []


# --- sprint-remove -------------------------------------------------------------------------------


class _SprintRemoveFakeClient:
    def __init__(self, _config=None) -> None:
        self.remove_calls: list[list[str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def move_issues_to_backlog(self, issue_keys):
        self.remove_calls.append(list(issue_keys))
        return {"ok": True}


def test_sprint_remove_cli_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    client = _SprintRemoveFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprint-remove", "FC-1", "FC-2", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "issue_keys": ["FC-1", "FC-2"],
        "safety": [
            {"key": "FC-1", "project": "FC", "allowed_by": "project allowlist"},
            {"key": "FC-2", "project": "FC", "allowed_by": "project allowlist"},
        ],
        "dry_run": True,
    }
    assert client.remove_calls == []


def test_sprint_remove_cli_execute_moves_to_backlog_and_audits(tmp_path, monkeypatch) -> None:
    client = _SprintRemoveFakeClient()
    audited: list[dict[str, object]] = []
    _install(monkeypatch, _config(tmp_path), client)
    monkeypatch.setattr(cli, "audit_write", lambda config, **kwargs: audited.append(kwargs))

    result = CliRunner().invoke(cli.app, ["sprint-remove", "FC-1", "FC-2", "--execute", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.remove_calls == [["FC-1", "FC-2"]]
    assert audited == [
        {
            "action": "sprint_remove",
            "issue_key": None,
            "payload": {"issue_keys": ["FC-1", "FC-2"]},
            "result": {"ok": True},
        }
    ]


class _SprintRemoveDisallowedFakeClient:
    """Like `_SprintRemoveFakeClient`, but one key belongs to a project off the write allow-list,
    so the safety gate must reject it — this is the fixture the F9 dry-run regression test needs
    and the plain fake above has no way to express."""

    def __init__(self, _config=None, *, disallowed_project_for: str | None = None) -> None:
        self.config = _config
        self._disallowed_project_for = disallowed_project_for
        self.remove_calls: list[list[str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        project = "OTHER" if issue_key == self._disallowed_project_for else "FC"
        return {"key": issue_key, "fields": {"project": {"key": project}}}

    def move_issues_to_backlog(self, issue_keys):
        self.remove_calls.append(list(issue_keys))
        return {"ok": True}


def test_sprint_remove_cli_dry_run_also_catches_the_safety_gate(tmp_path, monkeypatch) -> None:
    """F9 regression: the dry run used to return before ever opening a client, so it never ran
    the write-safety gate `--execute` runs. A dry run that reports success for a batch
    `--execute` would go on to refuse does not predict what `--execute` does.
    """
    client = _SprintRemoveDisallowedFakeClient(disallowed_project_for="FC-2")
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprint-remove", "FC-1", "FC-2"])

    assert result.exit_code == 3, result.stdout + result.stderr
    assert client.remove_calls == []


# --- issue-type -----------------------------------------------------------------------------


class _IssueTypeFakeClient:
    def __init__(
        self,
        _config=None,
        *,
        allowed: bool = True,
        allowed_values: list[str] | None = None,
        version: str = "2026-09-14T10:00:00.000+0000",
        concurrent_error: JiraConcurrentEditError | None = None,
    ) -> None:
        self._allowed = allowed
        self._allowed_values = allowed_values or ["Story", "Bug", "Task"]
        self._version = version
        self._concurrent_error = concurrent_error
        self.set_calls: list[dict[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def is_issue_type_allowed(self, issue_key, issue_type_name):
        return self._allowed

    def get_edit_metadata(self, issue_key):
        return {
            "fields": {
                "issuetype": {
                    "allowedValues": [{"name": name} for name in self._allowed_values],
                }
            }
        }

    def get_issue(self, issue_key, *, fields=None, expand=None):
        return {"key": issue_key, "fields": {"project": {"key": "FC"}}}

    def issue_version(self, issue_key):
        return self._version

    def set_issue_type(self, issue_key, issue_type_name, *, expect_updated=None):
        if self._concurrent_error is not None:
            raise self._concurrent_error
        self.set_calls.append(
            {
                "issue_key": issue_key,
                "issue_type_name": issue_type_name,
                "expect_updated": expect_updated,
            }
        )
        return {"key": issue_key, "fields": {"issuetype": {"name": issue_type_name}}}


def test_issue_type_cli_refuses_disallowed_type_lists_allowed_values_no_write(
    tmp_path, monkeypatch
) -> None:
    client = _IssueTypeFakeClient(allowed=False, allowed_values=["Story", "Bug", "Task"])
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["issue-type", "FC-1", "Epic", "--execute"])

    assert result.exit_code == 2, result.stdout + result.stderr
    assert "Story" in result.output
    assert "Bug" in result.output
    assert "Task" in result.output
    assert client.set_calls == []


def test_issue_type_cli_execute_allowed_type_calls_set_issue_type_with_expect_updated(
    tmp_path, monkeypatch
) -> None:
    client = _IssueTypeFakeClient(allowed=True, version="2026-09-14T10:00:00.000+0000")
    audited: list[dict[str, object]] = []
    _install(monkeypatch, _config(tmp_path), client)
    monkeypatch.setattr(cli, "audit_write", lambda config, **kwargs: audited.append(kwargs))

    result = CliRunner().invoke(cli.app, ["issue-type", "FC-1", "Bug", "--execute", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.set_calls == [
        {
            "issue_key": "FC-1",
            "issue_type_name": "Bug",
            "expect_updated": "2026-09-14T10:00:00.000+0000",
        }
    ]
    assert len(audited) == 1
    assert audited[0]["payload"]["expect_updated"] == "2026-09-14T10:00:00.000+0000"
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is False


def test_issue_type_cli_concurrent_edit_exits_nonzero_no_write(tmp_path, monkeypatch) -> None:
    error = JiraConcurrentEditError(
        "Issue FC-1 changed since it was read",
        issue_key="FC-1",
        expected_updated="2026-09-14T10:00:00.000+0000",
        found_updated="2026-09-14T10:05:00.000+0000",
    )
    client = _IssueTypeFakeClient(allowed=True, concurrent_error=error)
    audited: list[dict[str, object]] = []
    _install(monkeypatch, _config(tmp_path), client)
    monkeypatch.setattr(cli, "audit_write", lambda config, **kwargs: audited.append(kwargs))

    result = CliRunner().invoke(cli.app, ["issue-type", "FC-1", "Bug", "--execute"])

    assert result.exit_code == 4, result.stdout + result.stderr
    assert "2026-09-14T10:00:00.000+0000" in result.output
    assert "2026-09-14T10:05:00.000+0000" in result.output
    assert client.set_calls == []
    assert audited == []
