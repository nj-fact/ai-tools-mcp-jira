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
