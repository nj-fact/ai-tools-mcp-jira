"""Tests for the optimistic-concurrency guard on whole-field rewrites, and for the new
``edit-comment`` / ``delete-comment`` commands.

Every fake client here plays the same role the real ``JiraClient`` guard machinery plays in
``client.py``: ``read_updated`` is the ``updated`` timestamp a command captures when it reads the
issue, and ``server_updated`` is what the server actually holds by the time the write is
attempted. A mismatch between the two stands in for a concurrent edit landing inside the race
window the guard narrows. This mirrors ``JiraClient._guarded_issue_put``, which re-checks
``updated`` immediately before its own PUT and raises ``JiraConcurrentEditError`` on a mismatch;
none of that machinery is reimplemented here, only its observable contract.

No network: every test monkeypatches ``cli.load_config`` and ``cli.JiraClient`` with fakes, the
same style ``tests/test_cli_helpers.py`` already uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from jira_agent import cli
from jira_agent.client import JiraConcurrentEditError
from jira_agent.config import JiraConfig

ISSUE_KEY = "FC-1"
READ_UPDATED = "2026-01-01T00:00:00.000+0000"
CHANGED_UPDATED = "2026-01-01T00:05:00.000+0000"


def _config(tmp_path: Path) -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            # The issue-key allowlist lets ensure_issue_write_allowed skip its own get_issue
            # call, so every fake below only has to answer the calls this test is about.
            "JIRA_WRITE_ISSUE_KEYS": ISSUE_KEY,
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


def _sample_description() -> dict[str, Any]:
    """A two-section ADF description: patching one section is never the whole document."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 1},
                "content": [{"type": "text", "text": "Title"}],
            },
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Observed"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "Original body."}]},
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Next"}],
            },
            {"type": "paragraph", "content": [{"type": "text", "text": "Other body."}]},
        ],
    }


def _make_patch_fake_client(*, read_updated: str, server_updated: str) -> tuple[type, list[dict]]:
    """Fake client for patch-description-section. ``writes`` records only completed writes,
    exactly like the real client only calls its PUT after the guard check passes."""
    writes: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_issue(self, issue_key: str, *, fields: list[str] | None = None, **_kwargs: Any):
            assert fields == ["description", "updated"], fields
            return {
                "key": issue_key,
                "fields": {"description": _sample_description(), "updated": read_updated},
            }

        def edit_issue_adf_description(
            self, issue_key: str, description: dict[str, Any], *, expect_updated: str | None = None
        ) -> dict[str, Any]:
            if expect_updated is not None and expect_updated != server_updated:
                raise JiraConcurrentEditError(
                    f"Issue {issue_key} changed since it was read: expected updated="
                    f"{expect_updated!r}, found {server_updated!r}. Re-read the issue and retry.",
                    issue_key=issue_key,
                    expected_updated=expect_updated,
                    found_updated=server_updated,
                )
            writes.append(
                {
                    "issue": issue_key,
                    "description": description,
                    "expect_updated": expect_updated,
                }
            )
            return {"id": "99", "updated": True}

    return FakeClient, writes


def _make_edit_issue_fake_client(
    *, read_updated: str, server_updated: str
) -> tuple[type, list[dict]]:
    """Fake client for edit-issue. ``writes`` records only completed writes."""
    writes: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def issue_version(self, issue_key: str) -> str:
            return read_updated

        def edit_issue(
            self, issue_key: str, *, expect_updated: str | None = None, **kwargs: Any
        ) -> dict[str, Any]:
            if expect_updated is not None and expect_updated != server_updated:
                raise JiraConcurrentEditError(
                    f"Issue {issue_key} changed since it was read: expected updated="
                    f"{expect_updated!r}, found {server_updated!r}. Re-read the issue and retry.",
                    issue_key=issue_key,
                    expected_updated=expect_updated,
                    found_updated=server_updated,
                )
            writes.append({"issue": issue_key, "expect_updated": expect_updated, "kwargs": kwargs})
            return {"key": issue_key, "fields": {}}

    return FakeClient, writes


def _run(monkeypatch, tmp_path: Path, fake_client: type, args: list[str], *, audits=None):
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", fake_client)
    if audits is None:
        monkeypatch.setattr(cli, "audit_write", lambda *_a, **_k: None)
    else:
        monkeypatch.setattr(cli, "audit_write", lambda *a, **k: audits.append((a, k)))
    return CliRunner().invoke(cli.app, args)


# --- patch-description-section --------------------------------------------------------------


def test_patch_description_section_execute_refuses_on_concurrent_edit(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_patch_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "patch-description-section",
            ISSUE_KEY,
            "--section",
            "Observed",
            "--content",
            "## Observed\n\nNew body.",
            "--execute",
        ],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert ISSUE_KEY in result.stderr
    assert READ_UPDATED in result.stderr
    assert CHANGED_UPDATED in result.stderr
    assert writes == []


def test_patch_description_section_execute_writes_when_updated_matches(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_patch_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "patch-description-section",
            ISSUE_KEY,
            "--section",
            "Observed",
            "--content",
            "## Observed\n\nNew body.",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] == READ_UPDATED


def test_patch_description_section_no_concurrency_guard_writes_despite_change(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_patch_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "patch-description-section",
            ISSUE_KEY,
            "--section",
            "Observed",
            "--content",
            "## Observed\n\nNew body.",
            "--no-concurrency-guard",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] is None


def test_patch_description_section_json_refusal_is_one_machine_readable_document(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_patch_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "patch-description-section",
            ISSUE_KEY,
            "--section",
            "Observed",
            "--content",
            "## Observed\n\nNew body.",
            "--execute",
            "--json",
        ],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "concurrent_edit"
    assert payload["expected_updated"] == READ_UPDATED
    assert payload["found_updated"] == CHANGED_UPDATED
    assert payload["dry_run"] is False
    assert writes == []


def test_patch_description_section_docstring_no_longer_claims_no_optimistic_concurrency() -> None:
    doc = (cli.patch_description_section.__doc__ or "").lower()
    assert "there is no optimistic concurrency" not in doc
    assert "concurrency" in doc
    assert "no-concurrency-guard" in doc


# --- edit-issue ------------------------------------------------------------------------------


def test_edit_issue_execute_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--description", "New description text.", "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert ISSUE_KEY in result.stderr
    assert READ_UPDATED in result.stderr
    assert CHANGED_UPDATED in result.stderr
    assert writes == []


def test_edit_issue_execute_writes_when_updated_matches(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--description", "New description text.", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] == READ_UPDATED


def test_edit_issue_no_concurrency_guard_writes_despite_change(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "edit-issue",
            ISSUE_KEY,
            "--description",
            "New description text.",
            "--no-concurrency-guard",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] is None


def test_edit_issue_json_refusal_is_one_machine_readable_document(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "edit-issue",
            ISSUE_KEY,
            "--description",
            "New description text.",
            "--execute",
            "--json",
        ],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "concurrent_edit"
    assert payload["expected_updated"] == READ_UPDATED
    assert payload["found_updated"] == CHANGED_UPDATED
    assert payload["dry_run"] is False
    assert writes == []


# --- edit-comment / delete-comment ----------------------------------------------------------


def _make_comment_fake_client(*, current_text: str) -> tuple[type, dict[str, Any]]:
    state: dict[str, Any] = {"updates": [], "deletes": [], "get_comment_calls": 0}
    current_comment = {
        "id": "10099",
        "body": {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": current_text}]}],
        },
    }

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_comment(self, issue_key: str, comment_id: str) -> dict[str, Any]:
            state["get_comment_calls"] += 1
            return current_comment

        def update_comment(self, issue_key: str, comment_id: str, body: str) -> dict[str, Any]:
            state["updates"].append({"issue": issue_key, "comment_id": comment_id, "body": body})
            return {
                "id": comment_id,
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": body}]}],
                },
            }

        def delete_comment(self, issue_key: str, comment_id: str) -> dict[str, Any]:
            state["deletes"].append({"issue": issue_key, "comment_id": comment_id})
            return {}

    return FakeClient, state


def test_edit_comment_dry_run_shows_current_and_proposed_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    FakeClient, state = _make_comment_fake_client(current_text="Fehlerhafter deutscher Text.")

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-comment", ISSUE_KEY, "10099", "--body", "Korrigierter deutscher Text."],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Fehlerhafter deutscher Text." in result.stdout
    assert "Korrigierter deutscher Text." in result.stdout
    assert state["updates"] == []


def test_edit_comment_json_dry_run_carries_current_and_proposed_body(tmp_path, monkeypatch) -> None:
    FakeClient, state = _make_comment_fake_client(current_text="Fehlerhafter deutscher Text.")

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-comment", ISSUE_KEY, "10099", "--body", "Korrigierter deutscher Text.", "--json"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["current_body"] == "Fehlerhafter deutscher Text."
    assert payload["proposed_body"] == "Korrigierter deutscher Text."
    assert payload["dry_run"] is True
    assert state["updates"] == []


def test_edit_comment_execute_calls_update_comment_once_and_audits(tmp_path, monkeypatch) -> None:
    FakeClient, state = _make_comment_fake_client(current_text="Fehlerhafter deutscher Text.")
    audits: list[tuple[tuple, dict]] = []

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "edit-comment",
            ISSUE_KEY,
            "10099",
            "--body",
            "Korrigierter deutscher Text.",
            "--execute",
        ],
        audits=audits,
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert state["updates"] == [
        {"issue": ISSUE_KEY, "comment_id": "10099", "body": "Korrigierter deutscher Text."}
    ]
    assert len(audits) == 1
    _, audit_kwargs = audits[0]
    assert audit_kwargs["action"] == "edit_comment"
    assert audit_kwargs["issue_key"] == ISSUE_KEY


def test_delete_comment_dry_run_prints_current_body_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    FakeClient, state = _make_comment_fake_client(current_text="Zu löschender Kommentar.")

    result = _run(monkeypatch, tmp_path, FakeClient, ["delete-comment", ISSUE_KEY, "10099"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Zu löschender Kommentar." in result.stdout
    assert state["deletes"] == []


def test_delete_comment_execute_calls_delete_comment_once_and_audits(tmp_path, monkeypatch) -> None:
    FakeClient, state = _make_comment_fake_client(current_text="Zu löschender Kommentar.")
    audits: list[tuple[tuple, dict]] = []

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["delete-comment", ISSUE_KEY, "10099", "--execute"],
        audits=audits,
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert state["deletes"] == [{"issue": ISSUE_KEY, "comment_id": "10099"}]
    assert len(audits) == 1
    _, audit_kwargs = audits[0]
    assert audit_kwargs["action"] == "delete_comment"
    assert audit_kwargs["issue_key"] == ISSUE_KEY


# --- Integration seam: issue-type was built alongside these commands, in a separate worktree,
# and arrived with its own inline copy of the refusal instead of the shared helper. That copy
# printed prose only, so under --json a lost-update refusal produced an empty stdout and a
# non-zero exit — the caller could parse nothing and had to read prose to learn why. These two
# tests pin the unified behaviour so the duplicate cannot come back.


class _IssueTypeFakeClient:
    """Answers the reads `issue-type` makes, and fails its write with a lost-update race."""

    def __init__(self, config) -> None:
        self.config = config
        self.writes: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def is_issue_type_allowed(self, issue_key, issue_type_name):
        return True

    def get_edit_metadata(self, issue_key):
        return {"fields": {"issuetype": {"allowedValues": [{"name": "Bug"}, {"name": "Task"}]}}}

    def issue_version(self, issue_key):
        return READ_UPDATED

    def set_issue_type(self, issue_key, issue_type_name, *, expect_updated=None):
        raise JiraConcurrentEditError(
            f"{issue_key} changed since it was read",
            issue_key=issue_key,
            expected_updated=expect_updated,
            found_updated=CHANGED_UPDATED,
        )


def test_issue_type_concurrent_refusal_is_machine_readable_under_json(
    tmp_path, monkeypatch
) -> None:
    clients: list[_IssueTypeFakeClient] = []

    def _factory(config):
        client = _IssueTypeFakeClient(config)
        clients.append(client)
        return client

    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _factory)
    monkeypatch.setattr(cli, "audit_write", lambda *_a, **_k: None)

    result = CliRunner().invoke(cli.app, ["issue-type", ISSUE_KEY, "Bug", "--execute", "--json"])

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    # The whole of stdout is the refusal document: no find('{') slicing, no prose mixed in.
    payload = json.loads(result.stdout)
    assert payload["error_type"] == "concurrent_edit"
    assert payload["issue"] == ISSUE_KEY
    assert payload["expected_updated"] == READ_UPDATED
    assert payload["found_updated"] == CHANGED_UPDATED
    assert clients and clients[0].writes == []


def test_issue_type_refusal_uses_the_same_exit_code_as_the_other_guarded_commands(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", _IssueTypeFakeClient)
    monkeypatch.setattr(cli, "audit_write", lambda *_a, **_k: None)

    result = CliRunner().invoke(cli.app, ["issue-type", ISSUE_KEY, "Bug", "--execute"])

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE
    assert "Concurrent edit refused:" in result.stderr
    assert CHANGED_UPDATED in result.stderr
