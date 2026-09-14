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


def test_issue_sprint_cli_null_field_value_is_the_no_sprint_state_not_field_absent(
    tmp_path, monkeypatch
) -> None:
    """Verified fact: an issue in no sprint on this Jira site returns the field PRESENT with the
    value `null` -- not `[]`, and not absent -- confirmed via `jql=project = FC AND Sprint IS
    EMPTY`. This must land in the same "no sprint" bucket as an empty list
    (`sprint_field_present=True`, `sprints=[]`), not the "field absent" bucket.

    This already passes against the current code: `fields.get(sprint_field,
    _SPRINT_FIELD_MISSING)` returns `None` here, and `None is not _SPRINT_FIELD_MISSING`, so
    `field_present` is correctly `True`, and `sprint_list = raw if isinstance(raw, list) else []`
    correctly folds `None` into `[]`. Recorded as a regression test rather than left unverified,
    per the hardening brief's own instruction to check what the current code does with `null`
    before assuming it needs a fix -- stashing the src change and rerunning this test confirms it
    was already correct; there is no equivalent failing-before state to demonstrate.
    """
    _install(
        monkeypatch,
        _config(tmp_path),
        _IssueSprintFakeClient({"customfield_10020": None}),
    )

    result = CliRunner().invoke(cli.app, ["issue-sprint", "FC-4", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "is in no sprint" in result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "issue": "FC-4",
        "sprint_field": "customfield_10020",
        "sprint_field_present": True,
        "sprints": [],
    }


# --- sprint-add / sprint-remove -----------------------------------------------------------------
#
# One stateful fake models the real Agile write behaviour both commands depend on: the write
# endpoints return 204 with no body (modelled here as returning `{}`), so the only way a caller
# can tell whether a write actually took effect is to re-read the issue's sprint field afterwards
# -- exactly what `sprint_membership_report` does. `apply_writes=False` simulates a write Jira
# accepted (still returns `{}`) that did not actually change membership, the scenario the
# post-write verification exists to catch; without it, a fake that always "just works" could never
# fail the verification path the way a stale membership record for real can.


class _SprintStateFakeClient:
    def __init__(
        self,
        _config=None,
        *,
        initial_sprints: dict[str, list[dict[str, object]]] | None = None,
        disallowed_project_for: str | None = None,
        apply_writes: bool = True,
        sprint_field: str = "customfield_10020",
    ) -> None:
        self.config = _config
        self._disallowed_project_for = disallowed_project_for
        self._apply_writes = apply_writes
        self._sprint_field = sprint_field
        # key -> the sprint objects it currently sits in; absent/empty means "in no sprint",
        # rendered as Jira's own `null` shape by get_issue below, not `[]`.
        self._sprints: dict[str, list[dict[str, object]]] = {
            key: list(value) for key, value in (initial_sprints or {}).items()
        }
        self.add_calls: list[tuple[object, list[str]]] = []
        self.remove_calls: list[list[str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        if fields == ["project"]:
            project = "OTHER" if issue_key == self._disallowed_project_for else "FC"
            return {"key": issue_key, "fields": {"project": {"key": project}}}
        assert fields == [self._sprint_field], fields
        return {
            "key": issue_key,
            "fields": {self._sprint_field: self._sprints.get(issue_key) or None},
        }

    def get_issue_sprints(self, issue_key, *, sprint_field=None):
        assert sprint_field == self._sprint_field
        return list(self._sprints.get(issue_key) or [])

    def add_issues_to_sprint(self, sprint_id, issue_keys):
        self.add_calls.append((sprint_id, list(issue_keys)))
        if self._apply_writes:
            sprint = {"id": sprint_id, "name": f"Sprint {sprint_id}", "state": "active"}
            for issue_key in issue_keys:
                current = [s for s in self._sprints.get(issue_key, []) if s.get("id") != sprint_id]
                current.append(sprint)
                self._sprints[issue_key] = current
        return {}

    def move_issues_to_backlog(self, issue_keys):
        self.remove_calls.append(list(issue_keys))
        if self._apply_writes:
            for issue_key in issue_keys:
                self._sprints[issue_key] = []
        return {}


# --- sprint-add ---------------------------------------------------------------------------------


def test_sprint_add_cli_dry_run_writes_nothing_and_names_every_issue(tmp_path, monkeypatch) -> None:
    client = _SprintStateFakeClient()
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


def test_sprint_add_cli_execute_verifies_membership_by_reread_and_reports_per_key(
    tmp_path, monkeypatch
) -> None:
    """3: a 204 proves only that Jira accepted the request. This fake actually moves the keys
    into the sprint on `add_issues_to_sprint` (see `_SprintStateFakeClient`), so a passing
    `verified: True` here is proof the code re-read rather than trusted the response.
    """
    client = _SprintStateFakeClient()
    audited: list[dict[str, object]] = []
    _install(monkeypatch, _config(tmp_path), client)
    monkeypatch.setattr(cli, "audit_write", lambda config, **kwargs: audited.append(kwargs))

    result = CliRunner().invoke(
        cli.app, ["sprint-add", "7340", "FC-1", "FC-2", "FC-3", "--execute", "--json"]
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.add_calls == [(7340, ["FC-1", "FC-2", "FC-3"])]
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is False
    assert payload["ok"] is True
    assert payload["results"] == [
        {
            "key": key,
            "sprint_id": 7340,
            "sprints": [{"id": 7340, "name": "Sprint 7340", "state": "active"}],
            "in_sprint": True,
            "expected_in_sprint": True,
            "verified": True,
        }
        for key in ("FC-1", "FC-2", "FC-3")
    ]
    assert len(audited) == 1
    assert audited[0]["action"] == "sprint_add"
    assert audited[0]["issue_key"] is None
    assert audited[0]["payload"] == {"sprint_id": 7340, "issue_keys": ["FC-1", "FC-2", "FC-3"]}
    assert audited[0]["result"]["verification"] == payload["results"]


def test_sprint_add_cli_execute_reports_failure_and_exits_nonzero_when_reread_does_not_confirm(
    tmp_path, monkeypatch
) -> None:
    """Regression: fails against the pre-hardening code, which trusted the batched POST's
    success and never re-read membership at all, so it always reported success and exit 0 here.
    `apply_writes=False` simulates Jira accepting the request (still 204/`{}`) without the
    membership actually changing -- exactly the gap between "Jira accepted it" and "the issue is
    in the sprint" this hardening closes.
    """
    client = _SprintStateFakeClient(apply_writes=False)
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(
        cli.app, ["sprint-add", "7340", "FC-1", "FC-2", "--execute", "--json"]
    )

    assert result.exit_code == cli.SPRINT_VERIFICATION_EXIT_CODE, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert all(entry["verified"] is False for entry in payload["results"])
    assert all(entry["in_sprint"] is False for entry in payload["results"])


def test_sprint_add_cli_safety_gate_rejects_one_key_no_write_for_any(tmp_path, monkeypatch) -> None:
    # Five keys, the fourth belongs to a project that is not on the write allow-list.
    client = _SprintStateFakeClient(disallowed_project_for="FC-4")
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
    client = _SprintStateFakeClient(disallowed_project_for="FC-4")
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(
        cli.app,
        ["sprint-add", "7340", "FC-1", "FC-2", "FC-3", "FC-4", "FC-5"],
    )

    assert result.exit_code == 3, result.stdout + result.stderr
    assert client.add_calls == []


# --- sprint-remove --------------------------------------------------------------------------


def test_sprint_remove_cli_old_signature_no_longer_parses(tmp_path, monkeypatch) -> None:
    """1: breaking change, deliberate. `sprint-remove KEY...` (no sprint id) used to be valid;
    SPRINT_ID is now a required first argument, so the old invocation must fail argument parsing
    rather than silently treating the first key as a sprint id would (it can't here, because it
    is not an int, but the point is this is refused, loudly, before any client is even built).
    """
    client = _SprintStateFakeClient(initial_sprints={"FC-1": [], "FC-2": []})
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprint-remove", "FC-1", "FC-2"])

    assert result.exit_code == 2, result.stdout + result.stderr
    assert "SPRINT_ID" in result.stderr
    assert client.remove_calls == []


def test_sprint_remove_cli_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    sprint = {"id": 7340, "name": "Sprint 7340", "state": "active"}
    client = _SprintStateFakeClient(initial_sprints={"FC-1": [sprint], "FC-2": [sprint]})
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprint-remove", "7340", "FC-1", "FC-2", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "sprint_id": 7340,
        "issue_keys": ["FC-1", "FC-2"],
        "safety": [
            {"key": "FC-1", "project": "FC", "allowed_by": "project allowlist"},
            {"key": "FC-2", "project": "FC", "allowed_by": "project allowlist"},
        ],
        "precondition": [
            {"key": "FC-1", "sprint_id": 7340, "sprints": [sprint], "in_sprint": True},
            {"key": "FC-2", "sprint_id": 7340, "sprints": [sprint], "in_sprint": True},
        ],
        "dry_run": True,
    }
    assert client.remove_calls == []


def test_sprint_remove_cli_precondition_refuses_whole_batch_when_a_key_is_elsewhere(
    tmp_path, monkeypatch
) -> None:
    """1: the precondition is checked and enforced by this command, not forwarded to Jira (the
    backlog endpoint takes no sprint id at all). FC-2 is in a different sprint and FC-3 is in no
    sprint at all; both must be listed, and NEITHER key -- including FC-1, which is correctly in
    the target sprint -- may be written, on the dry run or on --execute.
    """
    target = {"id": 7340, "name": "Sprint 7340", "state": "active"}
    elsewhere = {"id": 9001, "name": "Other Sprint", "state": "active"}
    client = _SprintStateFakeClient(
        initial_sprints={"FC-1": [target], "FC-2": [elsewhere], "FC-3": []}
    )
    _install(monkeypatch, _config(tmp_path), client)

    for execute_flag in ([], ["--execute"]):
        result = CliRunner().invoke(
            cli.app, ["sprint-remove", "7340", "FC-1", "FC-2", "FC-3", "--json", *execute_flag]
        )

        assert result.exit_code == cli.SPRINT_PRECONDITION_EXIT_CODE, (
            result.stdout + result.stderr
        )
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["error_type"] == "sprint_precondition_failed"
        assert payload["sprint_id"] == 7340
        report_by_key = {entry["key"]: entry for entry in payload["report"]}
        assert report_by_key["FC-1"]["in_sprint"] is True
        assert report_by_key["FC-2"]["in_sprint"] is False
        assert report_by_key["FC-2"]["sprints"] == [elsewhere]
        assert report_by_key["FC-3"]["in_sprint"] is False
        assert report_by_key["FC-3"]["sprints"] == []
        assert client.remove_calls == []


def test_sprint_remove_cli_execute_verifies_removal_by_reread_and_reports_per_key(
    tmp_path, monkeypatch
) -> None:
    """3: this fake actually clears sprint membership on `move_issues_to_backlog` (see
    `_SprintStateFakeClient`), so a passing `verified: True` here is proof the code re-read
    membership rather than trusted the 204.
    """
    sprint = {"id": 7340, "name": "Sprint 7340", "state": "active"}
    client = _SprintStateFakeClient(initial_sprints={"FC-1": [sprint], "FC-2": [sprint]})
    audited: list[dict[str, object]] = []
    _install(monkeypatch, _config(tmp_path), client)
    monkeypatch.setattr(cli, "audit_write", lambda config, **kwargs: audited.append(kwargs))

    result = CliRunner().invoke(
        cli.app, ["sprint-remove", "7340", "FC-1", "FC-2", "--execute", "--json"]
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.remove_calls == [["FC-1", "FC-2"]]
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["results"] == [
        {
            "key": key,
            "sprint_id": 7340,
            "sprints": [],
            "in_sprint": False,
            "expected_in_sprint": False,
            "verified": True,
        }
        for key in ("FC-1", "FC-2")
    ]
    assert audited == [
        {
            "action": "sprint_remove",
            "issue_key": None,
            "payload": {"sprint_id": 7340, "issue_keys": ["FC-1", "FC-2"]},
            "result": {"verification": payload["results"]},
        }
    ]


def test_sprint_remove_cli_execute_failure_when_reread_still_shows_membership(
    tmp_path, monkeypatch
) -> None:
    """Regression: fails against the pre-hardening code, which trusted the batched POST and
    never re-read membership afterwards, so it always reported success here. `apply_writes=False`
    simulates Jira accepting the request without the issue actually leaving the sprint.
    """
    sprint = {"id": 7340, "name": "Sprint 7340", "state": "active"}
    client = _SprintStateFakeClient(
        initial_sprints={"FC-1": [sprint], "FC-2": [sprint]}, apply_writes=False
    )
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(
        cli.app, ["sprint-remove", "7340", "FC-1", "FC-2", "--execute", "--json"]
    )

    assert result.exit_code == cli.SPRINT_VERIFICATION_EXIT_CODE, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert all(entry["verified"] is False for entry in payload["results"])
    assert all(entry["in_sprint"] is True for entry in payload["results"])


def test_sprint_remove_cli_dry_run_also_catches_the_safety_gate(tmp_path, monkeypatch) -> None:
    """F9 regression: the dry run used to return before ever opening a client, so it never ran
    the write-safety gate `--execute` runs. A dry run that reports success for a batch
    `--execute` would go on to refuse does not predict what `--execute` does.
    """
    sprint = {"id": 7340, "name": "Sprint 7340", "state": "active"}
    client = _SprintStateFakeClient(
        initial_sprints={"FC-1": [sprint], "FC-2": [sprint]},
        disallowed_project_for="FC-2",
    )
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["sprint-remove", "7340", "FC-1", "FC-2"])

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


# --- sprint field on `issue` and search-backed commands (item 4) ----------------------------
#
# A caller should not need `issue-sprint` just to see an issue's sprint: `issue` and every
# search-backed command request the config-resolved sprint field alongside the shared
# `DEFAULT_FIELDS` when the caller left `--fields` at its default, and render it as its name and
# state, never the raw Jira object. An explicit --fields is used exactly as given (matching the
# existing "explicit fields are not augmented" rule for `parent`), so it must be neither
# requested nor rendered.


_SPRINT_ON_ISSUE = {"id": 7340, "name": "10 SEPA-Mandate, Disbursement", "state": "active"}


class _IssueWithSprintFakeClient:
    def __init__(self, _config=None) -> None:
        self.config = _config
        self.get_issue_calls: list[list[str] | None] = []
        self.search_calls: list[list[str] | None] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get_issue(self, issue_key, *, fields=None, expand=None):
        self.get_issue_calls.append(fields)
        fields_dict = {"summary": "S"}
        if fields is None or "customfield_10020" in fields:
            fields_dict["customfield_10020"] = [_SPRINT_ON_ISSUE]
        return {"key": issue_key, "fields": fields_dict}

    def search_all_issues(self, jql, *, limit=None, fields=None, **_kwargs):
        self.search_calls.append(fields)
        fields_dict = {"summary": "S"}
        if fields is None or "customfield_10020" in fields:
            fields_dict["customfield_10020"] = [_SPRINT_ON_ISSUE]
        return {
            "issues": [{"key": "FC-1", "fields": fields_dict}],
            "returned": 1,
            "is_last": True,
        }


def test_issue_cli_default_fields_request_and_render_the_configured_sprint_field(
    tmp_path, monkeypatch
) -> None:
    client = _IssueWithSprintFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["issue", "FC-1"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.get_issue_calls == [[*cli.DEFAULT_FIELDS, "customfield_10020"]]
    assert "10 SEPA-Mandate, Disbursement" in result.stdout
    assert "active" in result.stdout


def test_issue_cli_json_passes_the_raw_sprint_field_through(tmp_path, monkeypatch) -> None:
    client = _IssueWithSprintFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["issue", "FC-1", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["fields"]["customfield_10020"] == [_SPRINT_ON_ISSUE]


def test_issue_cli_explicit_fields_does_not_request_or_render_sprint(
    tmp_path, monkeypatch
) -> None:
    client = _IssueWithSprintFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["issue", "FC-1", "--fields", "summary"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.get_issue_calls == [["summary"]]
    assert "10 SEPA-Mandate, Disbursement" not in result.stdout


def test_search_cli_default_fields_request_and_render_the_configured_sprint_field(
    tmp_path, monkeypatch
) -> None:
    client = _IssueWithSprintFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["search", "project = FC"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.search_calls == [[*cli.DEFAULT_FIELDS, "customfield_10020"]]
    assert "10 SEPA-Mandate, Disbursement" in result.stdout


def test_search_cli_explicit_fields_does_not_request_or_render_sprint(
    tmp_path, monkeypatch
) -> None:
    client = _IssueWithSprintFakeClient()
    _install(monkeypatch, _config(tmp_path), client)

    result = CliRunner().invoke(cli.app, ["search", "project = FC", "--fields", "summary"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert client.search_calls == [["summary"]]
    assert "10 SEPA-Mandate, Disbursement" not in result.stdout


# --- config-fields ----------------------------------------------------------------------------


def test_config_fields_cli_shows_resolved_field_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))

    result = CliRunner().invoke(cli.app, ["config-fields"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "story_points_field: customfield_10028" in result.stdout
    assert "sprint_field: customfield_10020" in result.stdout


def test_config_fields_cli_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))

    result = CliRunner().invoke(cli.app, ["config-fields", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "story_points_field": "customfield_10028",
        "sprint_field": "customfield_10020",
    }


def test_config_fields_cli_reflects_a_configured_override(tmp_path, monkeypatch) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_SPRINT_FIELD": "customfield_99999",
        }
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)

    result = CliRunner().invoke(cli.app, ["config-fields", "--json"])

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["sprint_field"] == "customfield_99999"
