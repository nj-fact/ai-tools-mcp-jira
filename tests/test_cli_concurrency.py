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
from jira_agent import mermaid as mermaid_module
from jira_agent.client import (
    JiraApiError,
    JiraConcurrentEditError,
    JiraMissingConcurrencyTokenError,
    adf_contains_media_id,
)
from jira_agent.config import JiraConfig
from jira_agent.mermaid import MermaidRuntime

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


def _make_patch_fake_client(
    *, read_updated: str, server_updated: str, put_time_missing_token: bool = False
) -> tuple[type, list[dict]]:
    """Fake client for patch-description-section. ``writes`` records only completed writes,
    exactly like the real client only calls its PUT after the guard check passes.

    ``put_time_missing_token`` models H4: the FIRST read (``get_issue`` above) returned a
    perfectly usable ``read_updated``, so ``require_concurrency_token`` never objects, but the
    real client's own ``_guarded_issue_put`` re-reads ``updated`` immediately before its PUT and
    can *itself* find no usable token there -- a later, independent failure from the one
    ``read_updated=None`` models. That must still produce the shared guard-unavailable refusal,
    not the generic API-error path.
    """
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
            if expect_updated is not None and put_time_missing_token:
                raise JiraMissingConcurrencyTokenError(issue_key=issue_key)
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


def test_patch_description_section_refuses_when_guard_on_and_updated_missing(
    tmp_path, monkeypatch
) -> None:
    """F2 regression: Jira's response can omit `fields.updated` (a partial or unusual response).

    Before the fix, `read_updated = issue_fields.get("updated")` silently became `None`,
    `expect_updated` became `None` too, and the guarded write path treated that exactly like
    `--no-concurrency-guard` was passed: an unguarded whole-description overwrite went through,
    reported as `dry_run: false` with exit 0 and no warning, even though the caller asked for the
    default guard. The fix must refuse instead of silently writing unguarded.
    """
    FakeClient, writes = _make_patch_fake_client(read_updated=None, server_updated=READ_UPDATED)

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
    assert "Concurrency guard unavailable" in result.stderr
    assert ISSUE_KEY in result.stderr
    assert writes == []


def test_patch_description_section_missing_token_refusal_is_one_machine_readable_document(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_patch_fake_client(read_updated=None, server_updated=READ_UPDATED)

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
    assert payload["error_type"] == "concurrent_edit_guard_unavailable"
    assert payload["issue"] == ISSUE_KEY
    assert payload["dry_run"] is False
    assert writes == []


def test_patch_description_section_put_time_missing_token_is_guard_unavailable_not_api_error(
    tmp_path, monkeypatch
) -> None:
    """H4 regression. `patch-description-section` splits its read/build and its write into two
    separate try blocks; the write-containing one used to catch only `JiraConcurrentEditError`
    and the generic `JiraApiError`, so when the pre-PUT re-check inside `_guarded_issue_put`
    itself found no usable token (distinct from the FIRST read having none, which the other try
    block already handled), it fell through to the generic API-error handler: exit 1,
    error_type "jira_api_error", instead of the same guard-unavailable refusal (exit 4,
    "concurrent_edit_guard_unavailable") every other guarded command gives for this condition.
    """
    FakeClient, writes = _make_patch_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED, put_time_missing_token=True
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
    assert payload["error_type"] == "concurrent_edit_guard_unavailable"
    assert writes == []


def test_patch_description_section_no_concurrency_guard_writes_despite_missing_token(
    tmp_path, monkeypatch
) -> None:
    """Passing --no-concurrency-guard explicitly opts out, so a missing token must not refuse."""
    FakeClient, writes = _make_patch_fake_client(read_updated=None, server_updated=READ_UPDATED)

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


def test_patch_description_section_docstring_no_longer_claims_no_optimistic_concurrency() -> None:
    doc = (cli.patch_description_section.__doc__ or "").lower()
    assert "there is no optimistic concurrency" not in doc
    assert "concurrency" in doc
    assert "no-concurrency-guard" in doc


# --- G6: all four --json error documents share the same envelope -----------------------------


def _no_allowlist_config(tmp_path: Path) -> JiraConfig:
    """A config with a project allowlist but no issue-key allowlist, so `assign FC-1` actually
    reaches the project check instead of short-circuiting through `_config`'s own
    `JIRA_WRITE_ISSUE_KEYS` allowlist for FC-1.
    """
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


def test_all_four_json_error_documents_share_the_same_envelope_keys(tmp_path, monkeypatch) -> None:
    """G6 regression: `_exit_concurrent_edit_error`'s JSON document was the only one of the four
    without `"ok": false` -- the others (`_exit_api_error`, `_exit_safety_error`,
    `_exit_missing_concurrency_token_error`) already carried it. A caller doing `data["ok"]`,
    which the README invites, got a KeyError exactly when a write was refused for a lost-update
    race. Pins that all four now share at least `{"ok": False, "error_type": <str>}`.
    """
    from jira_agent.client import JiraApiError

    concurrent_client, _ = _make_patch_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )
    concurrent_result = _run(
        monkeypatch,
        tmp_path,
        concurrent_client,
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

    missing_client, _ = _make_patch_fake_client(read_updated=None, server_updated=READ_UPDATED)
    missing_result = _run(
        monkeypatch,
        tmp_path,
        missing_client,
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

    class ApiErrorClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> ApiErrorClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_issue(self, issue_key: str, **_kwargs: Any):
            raise JiraApiError(f"Issue does not exist: {issue_key}", status_code=404)

    api_result = _run(monkeypatch, tmp_path, ApiErrorClient, ["issue", "FC-999", "--json"])

    class SafetyErrorClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> SafetyErrorClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def myself(self):
            return {"accountId": "acc-1", "displayName": "Ada"}

        def get_issue(self, issue_key: str, **_kwargs: Any):
            return {"key": issue_key, "fields": {"project": {"key": "OTHER"}}}

    monkeypatch.setattr(cli, "load_config", lambda: _no_allowlist_config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", SafetyErrorClient)
    safety_result = CliRunner().invoke(cli.app, ["assign", "FC-1", "--execute", "--json"])

    documents = {
        "concurrent_edit": json.loads(concurrent_result.stdout),
        "concurrent_edit_guard_unavailable": json.loads(missing_result.stdout),
        "jira_api_error": json.loads(api_result.stdout),
        "write_safety_error": json.loads(safety_result.stdout),
    }
    for name, payload in documents.items():
        assert payload["ok"] is False, f"{name}: {payload}"
        assert isinstance(payload.get("error_type"), str) and payload["error_type"], (
            f"{name}: {payload}"
        )


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


def test_edit_issue_atomic_label_ops_alone_skip_the_guard_read_entirely(
    tmp_path, monkeypatch
) -> None:
    """A call carrying only --add-label/--remove-label (no whole-field option at all) has
    nothing for a concurrency token to protect, so it must not read `issue_version` at all, even
    though --concurrency-guard defaults on."""
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )
    read_calls: list[str] = []
    original_issue_version = FakeClient.issue_version

    def issue_version(self, issue_key: str) -> str:
        read_calls.append(issue_key)
        return original_issue_version(self, issue_key)

    FakeClient.issue_version = issue_version

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--add-label", "reviewed", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert read_calls == [], "atomic-only edit-issue must not read issue_version at all"
    assert len(writes) == 1
    assert writes[0]["expect_updated"] is None
    assert writes[0]["kwargs"]["labels_add"] == ["reviewed"]


def test_edit_issue_mixed_whole_field_and_atomic_still_reads_the_guard(
    tmp_path, monkeypatch
) -> None:
    """Combining an unrelated whole-field option with an atomic op still guards the whole-field
    part: --summary together with --add-label is one PUT carrying both `fields` and `update`."""
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "edit-issue",
            ISSUE_KEY,
            "--summary",
            "New title",
            "--add-label",
            "reviewed",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] == READ_UPDATED
    assert writes[0]["kwargs"]["summary"] == "New title"
    assert writes[0]["kwargs"]["labels_add"] == ["reviewed"]


def test_edit_issue_refuses_to_combine_whole_array_labels_with_atomic_label_ops(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--labels", "a,b", "--add-label", "c"],
    )

    assert result.exit_code == 2, result.stdout + result.stderr
    assert "--labels" in result.stderr
    assert "--add-label" in result.stderr
    assert writes == []


def test_edit_issue_refuses_to_combine_whole_array_fix_version_with_atomic_ops(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--fix-version", "FCL-MVP", "--remove-fix-version", "FCL-1.0"],
    )

    assert result.exit_code == 2, result.stdout + result.stderr
    assert "--fix-version" in result.stderr
    assert "--remove-fix-version" in result.stderr
    assert writes == []


def test_edit_issue_labels_empty_list_clear_still_takes_the_guard(tmp_path, monkeypatch) -> None:
    """K2 regression: `edit_issue_payload` now omits an entirely EMPTY `fields` dict, but
    `--labels ""` (an explicit whole-array clear) is not that -- it keys `fields` with an empty
    list, which is a non-empty dict even though the list inside it is empty. The guard-read
    decision (`payload.get("fields")`) must still fire for this case, or it collapses into the
    atomic-only skip path that needs no guard."""
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )
    read_calls: list[str] = []
    original_issue_version = FakeClient.issue_version

    def issue_version(self, issue_key: str) -> str:
        read_calls.append(issue_key)
        return original_issue_version(self, issue_key)

    FakeClient.issue_version = issue_version

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--labels", "", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert read_calls == [ISSUE_KEY], "labels=[] is a whole-field clear and must take the guard"
    assert len(writes) == 1
    assert writes[0]["expect_updated"] == READ_UPDATED
    assert writes[0]["kwargs"]["labels"] == []


def test_edit_issue_refuses_a_blank_add_label_value(tmp_path, monkeypatch) -> None:
    """K4: an empty or whitespace-only atomic value is refused locally, naming the option, rather
    than sent to Jira for it to reject."""
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--add-label", "  ", "--execute"],
    )

    assert result.exit_code == 2, result.stdout + result.stderr
    assert "--add-label" in result.stderr
    assert writes == []


def test_edit_issue_combines_whole_array_labels_with_atomic_op_on_a_different_field(
    tmp_path, monkeypatch
) -> None:
    """K7: the README-advertised combination -- a whole-array write on one field together with
    an atomic operation on a DIFFERENT field -- must not be refused; only set-and-update of the
    SAME field is."""
    FakeClient, writes = _make_edit_issue_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["edit-issue", ISSUE_KEY, "--labels", "a,b", "--add-fix-version", "V1", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["kwargs"]["labels"] == ["a", "b"]
    assert writes[0]["kwargs"]["fix_versions_add"] == ["V1"]


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


# --- F11: the concurrency guard carried to the sibling whole-description-rewrite commands ----
#
# human-brief, embed-diagram, embed-existing-png, and upsert-image-gallery each read a
# description (or attachment set), transform it, and write the whole field back, exactly like
# edit-issue and patch-description-section. Before this fix none of the four passed
# expect_updated at all, so a concurrent edit landing between their read and their write was
# always silently lost. These tests pin the guard on all four.


def _make_human_brief_fake_client(*, read_updated: str, server_updated: str) -> tuple[type, list]:
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
            self, issue_key: str, *, description: str | None = None, expect_updated=None, **_kw
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
                {"issue": issue_key, "description": description, "expect_updated": expect_updated}
            )
            return {"key": issue_key, "fields": {}}

    return FakeClient, writes


def test_human_brief_execute_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_human_brief_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["human-brief", ISSUE_KEY, "--text", "Brief text.", "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert writes == []


def test_human_brief_execute_writes_when_updated_matches(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_human_brief_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["human-brief", ISSUE_KEY, "--text", "Brief text.", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] == READ_UPDATED


def test_human_brief_no_concurrency_guard_writes_despite_change(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_human_brief_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["human-brief", ISSUE_KEY, "--text", "Brief text.", "--no-concurrency-guard", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] is None


# --- drift-labels: converted from a guarded whole-array read-transform-write (G8) to atomic ---
# add/remove operations. Add/remove alone need no read and no guard at all; --clear is the one
# path that still reads `labels` once, to name the drift-* labels currently present as explicit
# remove operations, and that read still carries no guard either (see cli.drift_labels).


def _make_stateful_drift_labels_client(initial_labels: list[str]) -> tuple[type, dict[str, Any]]:
    """A fake that models Jira's own `add`/`remove` semantics against ONE stored array, the way
    the real Jira API does server-side -- not a client-side whole-array replacement. This is what
    lets a test prove a concurrently added label survives an atomic write: the fake applies each
    operation against whatever `state["labels"]` holds at write time, exactly like Jira does.
    """
    state: dict[str, Any] = {"labels": list(initial_labels), "get_issue_calls": 0}

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_issue(self, issue_key: str, *, fields: list[str] | None = None, **_kwargs: Any):
            assert fields == ["labels"], fields
            state["get_issue_calls"] += 1
            return {"key": issue_key, "fields": {"labels": list(state["labels"])}}

        def edit_issue(
            self,
            issue_key: str,
            *,
            labels_add: list[str] | None = None,
            labels_remove: list[str] | None = None,
            expect_updated: str | None = None,
            **_kw: Any,
        ) -> dict[str, Any]:
            assert expect_updated is None, "atomic label ops must never carry a guard token"
            for label in labels_add or []:
                if label not in state["labels"]:
                    state["labels"].append(label)
            for label in labels_remove or []:
                if label in state["labels"]:
                    state["labels"].remove(label)
            return {"key": issue_key, "fields": {}}

    return FakeClient, state


def test_drift_labels_add_writes_atomic_ops_with_no_read_and_no_guard(
    tmp_path, monkeypatch
) -> None:
    FakeClient, state = _make_stateful_drift_labels_client(["drift-open"])

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["drift-labels", ISSUE_KEY, "--add", "reviewed", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert state["get_issue_calls"] == 0, "add/remove alone must not read the issue at all"
    assert set(state["labels"]) == {"drift-open", "drift-reviewed"}


def test_drift_labels_two_concurrent_adds_both_survive(tmp_path, monkeypatch) -> None:
    """The point of this task: two agents adding different drift labels to the same issue, one
    after the other with no coordination, must both land. A whole-array read-transform-write
    computed by the first call would have raced the second; an atomic add cannot, because each
    is applied against whatever the server currently holds, not against a value either call read.
    """
    FakeClient, state = _make_stateful_drift_labels_client(["drift-open"])

    first = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["drift-labels", ISSUE_KEY, "--add", "reviewed", "--execute"],
    )
    second = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["drift-labels", ISSUE_KEY, "--add", "urgent", "--execute"],
    )

    assert first.exit_code == 0, first.stdout + first.stderr
    assert second.exit_code == 0, second.stdout + second.stderr
    assert set(state["labels"]) == {"drift-open", "drift-reviewed", "drift-urgent"}


def test_drift_labels_clear_reads_once_and_removes_only_drift_prefixed_labels(
    tmp_path, monkeypatch
) -> None:
    FakeClient, state = _make_stateful_drift_labels_client(
        ["jira-agent", "drift-old", "drift-keep"]
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["drift-labels", ISSUE_KEY, "--add", "new", "--clear", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert state["get_issue_calls"] == 1, "--clear reads the issue exactly once"
    assert set(state["labels"]) == {"jira-agent", "drift-new"}


def test_drift_labels_clear_does_not_remove_a_drift_label_added_after_its_read(
    tmp_path, monkeypatch
) -> None:
    """--clear's read is a snapshot: a drift label that lands after it is never named in an
    explicit remove operation, so it survives, exactly like add/remove alone already do for
    anything they were not told about."""
    FakeClient, state = _make_stateful_drift_labels_client(["drift-old"])
    original_get_issue = FakeClient.get_issue

    def get_issue_then_concurrent_add(self, issue_key, **kwargs):
        result = original_get_issue(self, issue_key, **kwargs)
        state["labels"].append("drift-concurrent")
        return result

    FakeClient.get_issue = get_issue_then_concurrent_add

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["drift-labels", ISSUE_KEY, "--clear", "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert state["labels"] == ["drift-concurrent"]


def test_drift_labels_dry_run_shows_operations_not_a_guessed_resulting_array(
    tmp_path, monkeypatch
) -> None:
    FakeClient, state = _make_stateful_drift_labels_client(["drift-open"])

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["drift-labels", ISSUE_KEY, "--add", "reviewed", "--remove", "open"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert state["get_issue_calls"] == 0
    json_start = result.stdout.index("{\n")
    payload, _ = json.JSONDecoder().raw_decode(result.stdout[json_start:])
    # No "fields" key at all -- not even an empty one -- when the request is purely atomic.
    assert payload == {
        "payload": {
            "update": {"labels": [{"add": "drift-reviewed"}, {"remove": "drift-open"}]},
        }
    }
    # Nothing was written: the dry run must not guess what the array would end up holding.
    assert state["labels"] == ["drift-open"]


def test_drift_labels_no_longer_offers_a_concurrency_guard_flag(tmp_path, monkeypatch) -> None:
    """The old `--concurrency-guard/--no-concurrency-guard` option protected a whole-array
    rewrite that no longer exists on any drift-labels path (add/remove need no read at all;
    --clear's read only selects which remove operations to send). A flag that no longer does
    anything is worse than one that was never documented, so it is removed, not just unused."""
    FakeClient, _state = _make_stateful_drift_labels_client(["drift-open"])

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["drift-labels", ISSUE_KEY, "--add", "reviewed", "--no-concurrency-guard", "--execute"],
    )

    assert result.exit_code != 0
    assert "no such option" in result.stdout.lower() + result.stderr.lower()


# A minimal valid PNG header (signature + IHDR), reused from tests/test_client.py's own
# png_dimensions fixture: decodes to (774, 70).
_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x03\x06\x00\x00\x00F\x08\x02\x00\x00\x00"


def _make_embed_diagram_fake_client(
    *,
    after_upload_updated: str | None,
    at_put_time_updated: str | None = None,
    initial_description: dict[str, Any] | None = None,
    concurrent_description: dict[str, Any] | None = None,
    delete_attachment_raises: bool = False,
) -> tuple[type, list, list]:
    """Fake client for embed-diagram.

    Models Jira's real behaviour: uploading an attachment moves the issue's own `updated` on
    the server, AND a genuine concurrent editor can change the live `description` in the same
    window (`concurrent_description`, applied at the same point `add_attachment` bumps
    `updated` -- this fake has no finer time resolution than "before the upload" vs. "after
    it"). The fixed code's SECOND `get_issue` read (fields=["description", "updated"]), issued
    after the upload, must observe this live state, not the state the FIRST read saw. A fake
    whose `get_issue` always returned the same frozen document regardless of when it was called
    could never reproduce the round-2 bug (content built from the first read, token from a
    later one) because it would verify the test rather than the thing.

    `add_attachment` bumps this fake's live `updated` state to `after_upload_updated` (`None`
    models Jira still returning no usable timestamp after the upload) and, when given, switches
    the live description to `concurrent_description`. `issue_version` mirrors the same live
    `updated` state for any code path that still reads it directly. `at_put_time_updated` is the
    ground truth `_guarded_issue_put`'s own re-check finds immediately before its PUT: it
    defaults to `after_upload_updated` (nothing but this process's own upload happened) and only
    needs overriding to model a genuine third party editing the issue in the narrow window that
    remains after the post-upload read.
    """
    writes: list[dict[str, Any]] = []
    deleted: list[str] = []
    state: dict[str, Any] = {
        "updated": "T-before-upload",
        "description": initial_description or {"type": "doc", "version": 1, "content": []},
    }
    resolved_at_put_time = (
        at_put_time_updated if at_put_time_updated is not None else after_upload_updated
    )

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_issue(self, issue_key: str, *, fields=None, expand=None):
            assert fields is not None, fields
            payload: dict[str, Any] = {"key": issue_key, "fields": {"updated": state["updated"]}}
            if "description" in fields:
                payload["fields"]["description"] = state["description"]
            if "attachment" in fields:
                payload["fields"]["attachment"] = []
            return payload

        def add_attachment(self, issue_key: str, file_path):
            state["updated"] = after_upload_updated
            if concurrent_description is not None:
                state["description"] = concurrent_description
            return [{"id": "9001", "filename": Path(file_path).name}]

        def attachment_media_id(self, attachment_id: str, *, issue_key: str) -> str:
            return "media-diagram-1"

        def issue_version(self, issue_key: str) -> str:
            current = state["updated"]
            if not isinstance(current, str) or not current:
                raise JiraMissingConcurrencyTokenError(issue_key=issue_key)
            return current

        def edit_issue_adf_description(self, issue_key: str, description, *, expect_updated=None):
            if expect_updated is not None and expect_updated != resolved_at_put_time:
                raise JiraConcurrentEditError(
                    f"Issue {issue_key} changed since it was read: expected updated="
                    f"{expect_updated!r}, found {resolved_at_put_time!r}. Re-read the issue and "
                    "retry.",
                    issue_key=issue_key,
                    expected_updated=expect_updated,
                    found_updated=resolved_at_put_time,
                )
            writes.append(
                {"issue": issue_key, "description": description, "expect_updated": expect_updated}
            )
            return {"id": "1"}

        def delete_attachment(self, attachment_id: str, *, issue_key: str) -> dict[str, Any]:
            deleted.append(attachment_id)
            if delete_attachment_raises:
                raise JiraApiError("cleanup delete failed")
            return {}

    return FakeClient, writes, deleted


def test_embed_diagram_execute_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    """A genuine third-party edit landing after the late read must still refuse."""
    FakeClient, writes, deleted = _make_embed_diagram_fake_client(
        after_upload_updated=READ_UPDATED, at_put_time_updated=CHANGED_UPDATED
    )
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["embed-diagram", ISSUE_KEY, "--file", str(png), "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert writes == []
    # H2: the refused write must not orphan the attachment this command already uploaded.
    assert deleted == ["9001"]


def test_embed_diagram_execute_writes_when_updated_matches(tmp_path, monkeypatch) -> None:
    FakeClient, writes, deleted = _make_embed_diagram_fake_client(after_upload_updated=READ_UPDATED)
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["embed-diagram", ISSUE_KEY, "--file", str(png), "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1


def test_embed_diagram_refuses_when_guard_on_and_updated_missing(tmp_path, monkeypatch) -> None:
    """F2's lesson carried to embed-diagram: a missing `updated` with the guard on must refuse,
    not silently write unguarded."""
    FakeClient, writes, deleted = _make_embed_diagram_fake_client(after_upload_updated=None)
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["embed-diagram", ISSUE_KEY, "--file", str(png), "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert "Concurrency guard unavailable" in result.stderr
    assert writes == []
    # H2: a missing-token refusal must not orphan the attachment this command already uploaded.
    assert deleted == ["9001"]


def test_embed_diagram_cleanup_failure_does_not_mask_the_original_refusal(
    tmp_path, monkeypatch
) -> None:
    """H2: the cleanup delete is itself best-effort and must not replace the ORIGINAL error
    (the concurrent-edit refusal) with whatever the cleanup attempt raised.
    """
    FakeClient, writes, deleted = _make_embed_diagram_fake_client(
        after_upload_updated=READ_UPDATED,
        at_put_time_updated=CHANGED_UPDATED,
        delete_attachment_raises=True,
    )
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["embed-diagram", ISSUE_KEY, "--file", str(png), "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert "Concurrent edit refused:" in result.stderr
    assert writes == []
    assert deleted == ["9001"], "cleanup must still be attempted even though it will fail"


def test_embed_diagram_execute_succeeds_despite_upload_moving_updated_no_orphan(
    tmp_path, monkeypatch
) -> None:
    """G1 regression. embed-diagram's own attachment upload moves the issue's `updated` on the
    Jira side. Before the fix, the guard token was captured BEFORE that upload from the initial
    `get_issue` read, so it was stale by construction: this exact scenario -- no external editor
    at all -- refused every guarded execution, after the attachment had already been uploaded,
    orphaning it with no description referencing it. This must succeed, and the written
    description must actually embed the uploaded attachment's media.
    """
    FakeClient, writes, deleted = _make_embed_diagram_fake_client(after_upload_updated=READ_UPDATED)
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["embed-diagram", ISSUE_KEY, "--file", str(png), "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] == READ_UPDATED
    assert adf_contains_media_id(writes[0]["description"], "media-diagram-1")


def _paragraph(text: str) -> dict[str, Any]:
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


def test_embed_diagram_execute_incorporates_concurrent_edit_landing_before_post_upload_read(
    tmp_path, monkeypatch
) -> None:
    """H1 regression. A concurrent editor's change lands in the window between the FIRST read
    (the one that only resolves --delete-filename and feeds the dry-run display) and the
    post-upload SECOND read that now supplies both the content and the guard token.

    Round 2's bug built the write from the FIRST read's description while reading the token from
    a separate, later `issue_version` call: the token would match (nothing further changed after
    the upload), so the write would go ahead and REPORT a guarded success, but the written
    document would silently drop the concurrent editor's paragraph, because it was never in the
    document round 2 built from. The fix must build from the post-upload read instead, so this
    same concurrent edit is INCORPORATED into the write, not lost by it.
    """
    initial_description = {"type": "doc", "version": 1, "content": [_paragraph("Original content")]}
    concurrent_description = {
        "type": "doc",
        "version": 1,
        "content": [_paragraph("Original content"), _paragraph("Concurrent editor's note")],
    }
    FakeClient, writes, deleted = _make_embed_diagram_fake_client(
        after_upload_updated=READ_UPDATED,
        initial_description=initial_description,
        concurrent_description=concurrent_description,
    )
    png = tmp_path / "diagram.png"
    png.write_bytes(_PNG_BYTES)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["embed-diagram", ISSUE_KEY, "--file", str(png), "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    written_description = writes[0]["description"]
    assert adf_contains_media_id(written_description, "media-diagram-1")
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


# --- render-mermaid: H1/H5's other read-mutate-write-whole-field shape ----------------------

_MERMAID_CODE_BLOCK = {
    "type": "codeBlock",
    "attrs": {"language": "mermaid"},
    "content": [{"type": "text", "text": "flowchart LR\nA --> B"}],
}


def _make_render_mermaid_fake_client(
    *,
    after_upload_updated: str | None,
    at_put_time_updated: str | None = None,
    initial_description: dict[str, Any] | None = None,
    concurrent_description: dict[str, Any] | None = None,
) -> tuple[type, list]:
    """Fake client for render-mermaid. Same shape as `_make_embed_diagram_fake_client`: uploading
    a rendered PNG moves the issue's own `updated`, and `concurrent_description` models a genuine
    concurrent editor's change becoming visible by the time of the post-upload SECOND read.
    """
    writes: list[dict[str, Any]] = []
    state: dict[str, Any] = {
        "updated": "T-before-upload",
        "description": initial_description
        or {"type": "doc", "version": 1, "content": [_MERMAID_CODE_BLOCK]},
    }
    resolved_at_put_time = (
        at_put_time_updated if at_put_time_updated is not None else after_upload_updated
    )

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_issue(self, issue_key: str, *, fields=None, expand=None):
            assert fields is not None, fields
            payload: dict[str, Any] = {"key": issue_key, "fields": {"updated": state["updated"]}}
            if "description" in fields:
                payload["fields"]["description"] = state["description"]
            if "attachment" in fields:
                payload["fields"]["attachment"] = []
            return payload

        def add_attachment(self, issue_key: str, path):
            state["updated"] = after_upload_updated
            if concurrent_description is not None:
                state["description"] = concurrent_description
            return [{"id": "9101", "filename": Path(path).name}]

        def attachment_media_id(self, attachment_id: str, *, issue_key: str) -> str:
            return "media-mermaid-1"

        def issue_version(self, issue_key: str) -> str:
            current = state["updated"]
            if not isinstance(current, str) or not current:
                raise JiraMissingConcurrencyTokenError(issue_key=issue_key)
            return current

        def edit_issue_adf_description(self, issue_key: str, description, *, expect_updated=None):
            if expect_updated is not None and expect_updated != resolved_at_put_time:
                raise JiraConcurrentEditError(
                    f"Issue {issue_key} changed since it was read: expected updated="
                    f"{expect_updated!r}, found {resolved_at_put_time!r}. Re-read the issue and "
                    "retry.",
                    issue_key=issue_key,
                    expected_updated=expect_updated,
                    found_updated=resolved_at_put_time,
                )
            writes.append(
                {"issue": issue_key, "description": description, "expect_updated": expect_updated}
            )
            return {"id": "1"}

        def delete_attachment(self, attachment_id: str, *, issue_key: str) -> dict[str, Any]:
            return {}

    return FakeClient, writes


def _patch_mermaid_rendering(monkeypatch, tmp_path: Path) -> None:
    def fake_render(_source, output_path, *, runtime):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x03\x06\x00\x00\x00F\x08\x02\x00\x00\x00"
        )
        return Path(output_path)

    monkeypatch.setattr(mermaid_module, "render_mermaid_png", fake_render)
    monkeypatch.setattr(
        cli,
        "discover_mermaid_runtime",
        lambda *_a, **_k: MermaidRuntime(
            node=tmp_path / "node", cli=tmp_path / "cli.js", browser=tmp_path / "chrome"
        ),
    )


def test_render_mermaid_execute_succeeds_despite_upload_moving_updated(
    tmp_path, monkeypatch
) -> None:
    """G1-style regression carried to render-mermaid: its own upload moves `updated`, so a token
    read before it (or from the first, pre-upload read) would be stale by construction."""
    _patch_mermaid_rendering(monkeypatch, tmp_path)
    FakeClient, writes = _make_render_mermaid_fake_client(after_upload_updated=READ_UPDATED)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["render-mermaid", ISSUE_KEY, "--output-dir", str(tmp_path / "out"), "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] == READ_UPDATED
    assert adf_contains_media_id(writes[0]["description"], "media-mermaid-1")


def test_render_mermaid_refuses_on_genuine_concurrent_edit(tmp_path, monkeypatch) -> None:
    _patch_mermaid_rendering(monkeypatch, tmp_path)
    FakeClient, writes = _make_render_mermaid_fake_client(
        after_upload_updated=READ_UPDATED, at_put_time_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["render-mermaid", ISSUE_KEY, "--output-dir", str(tmp_path / "out"), "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert writes == []


def test_render_mermaid_refuses_when_guard_on_and_updated_missing(tmp_path, monkeypatch) -> None:
    _patch_mermaid_rendering(monkeypatch, tmp_path)
    FakeClient, writes = _make_render_mermaid_fake_client(after_upload_updated=None)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["render-mermaid", ISSUE_KEY, "--output-dir", str(tmp_path / "out"), "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert "Concurrency guard unavailable" in result.stderr
    assert writes == []


def test_render_mermaid_no_concurrency_guard_writes_despite_missing_token(
    tmp_path, monkeypatch
) -> None:
    _patch_mermaid_rendering(monkeypatch, tmp_path)
    FakeClient, writes = _make_render_mermaid_fake_client(after_upload_updated=None)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "render-mermaid",
            ISSUE_KEY,
            "--output-dir",
            str(tmp_path / "out"),
            "--no-concurrency-guard",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    assert writes[0]["expect_updated"] is None


def test_render_mermaid_execute_incorporates_concurrent_edit_landing_before_post_upload_read(
    tmp_path, monkeypatch
) -> None:
    """H1 regression for render-mermaid, the second command sharing embed-diagram's shape: a
    concurrent editor's change lands between the FIRST read (which only finds the Mermaid blocks
    to render) and the post-upload SECOND read that now supplies both content and guard token.
    The write must incorporate it, not silently drop it the way round 2's token-after/content-
    from-before split would have.
    """
    initial_description = {
        "type": "doc",
        "version": 1,
        "content": [_paragraph("Original content"), _MERMAID_CODE_BLOCK],
    }
    concurrent_description = {
        "type": "doc",
        "version": 1,
        "content": [
            _paragraph("Original content"),
            _paragraph("Concurrent editor's note"),
            _MERMAID_CODE_BLOCK,
        ],
    }
    _patch_mermaid_rendering(monkeypatch, tmp_path)
    FakeClient, writes = _make_render_mermaid_fake_client(
        after_upload_updated=READ_UPDATED,
        initial_description=initial_description,
        concurrent_description=concurrent_description,
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["render-mermaid", ISSUE_KEY, "--output-dir", str(tmp_path / "out"), "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1
    written_description = writes[0]["description"]
    assert adf_contains_media_id(written_description, "media-mermaid-1")
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


def _make_embed_existing_png_fake_client(
    *, read_updated: str, server_updated: str
) -> tuple[type, list]:
    writes: list[dict[str, Any]] = []
    attachment = {"id": "170501", "filename": "diagram.png", "mimeType": "image/png"}

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_issue(self, issue_key: str, *, fields=None, expand=None):
            if fields == ["description", "attachment", "updated"]:
                return {
                    "key": issue_key,
                    "fields": {
                        "description": {"type": "doc", "version": 1, "content": []},
                        "attachment": [attachment],
                        "updated": read_updated,
                    },
                }
            # The post-write verification re-read, made after a successful write.
            return {
                "key": issue_key,
                "fields": {
                    "description": {
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
                                            "id": "media-existing-1",
                                            "type": "file",
                                            "collection": "",
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                    "attachment": [attachment],
                },
            }

        def attachment_media_id(self, attachment_id: str, *, issue_key: str) -> str:
            return "media-existing-1"

        def edit_issue_adf_description(self, issue_key: str, description, *, expect_updated=None):
            if expect_updated is not None and expect_updated != server_updated:
                raise JiraConcurrentEditError(
                    f"Issue {issue_key} changed since it was read: expected updated="
                    f"{expect_updated!r}, found {server_updated!r}. Re-read the issue and retry.",
                    issue_key=issue_key,
                    expected_updated=expect_updated,
                    found_updated=server_updated,
                )
            writes.append(
                {"issue": issue_key, "description": description, "expect_updated": expect_updated}
            )
            return {"id": "1"}

    return FakeClient, writes


def test_embed_existing_png_execute_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_embed_existing_png_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "embed-existing-png",
            ISSUE_KEY,
            "--filename",
            "diagram.png",
            "--width",
            "800",
            "--height",
            "600",
            "--execute",
        ],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert writes == []


def test_embed_existing_png_execute_writes_when_updated_matches(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_embed_existing_png_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        [
            "embed-existing-png",
            ISSUE_KEY,
            "--filename",
            "diagram.png",
            "--width",
            "800",
            "--height",
            "600",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1


def _write_gallery_manifest(tmp_path: Path) -> Path:
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


def _make_upsert_image_gallery_fake_client(
    *, read_updated: str, server_updated: str
) -> tuple[type, list]:
    writes: list[dict[str, Any]] = []
    attachment = {
        "id": "170501",
        "filename": "diagram.png",
        "content": "https://example.atlassian.net/attachments/170501",
    }

    class FakeClient:
        def __init__(self, config: JiraConfig) -> None:
            self.config = config

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def get_issue(self, issue_key: str, *, fields=None, expand=None):
            assert fields == ["description", "attachment", "updated"], fields
            return {
                "key": issue_key,
                "fields": {
                    "description": {"type": "doc", "version": 1, "content": []},
                    "attachment": [attachment],
                    "updated": read_updated,
                },
            }

        def edit_issue_adf_description(self, issue_key: str, description, *, expect_updated=None):
            if expect_updated is not None and expect_updated != server_updated:
                raise JiraConcurrentEditError(
                    f"Issue {issue_key} changed since it was read: expected updated="
                    f"{expect_updated!r}, found {server_updated!r}. Re-read the issue and retry.",
                    issue_key=issue_key,
                    expected_updated=expect_updated,
                    found_updated=server_updated,
                )
            writes.append(
                {"issue": issue_key, "description": description, "expect_updated": expect_updated}
            )
            return {"id": "1"}

    return FakeClient, writes


def test_upsert_image_gallery_execute_refuses_on_concurrent_edit(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_upsert_image_gallery_fake_client(
        read_updated=READ_UPDATED, server_updated=CHANGED_UPDATED
    )
    manifest_path = _write_gallery_manifest(tmp_path)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["upsert-image-gallery", ISSUE_KEY, "--manifest", str(manifest_path), "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert writes == []


def test_upsert_image_gallery_execute_writes_when_updated_matches(tmp_path, monkeypatch) -> None:
    FakeClient, writes = _make_upsert_image_gallery_fake_client(
        read_updated=READ_UPDATED, server_updated=READ_UPDATED
    )
    manifest_path = _write_gallery_manifest(tmp_path)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["upsert-image-gallery", ISSUE_KEY, "--manifest", str(manifest_path), "--execute"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(writes) == 1


def test_upsert_image_gallery_refuses_when_guard_on_and_updated_missing(
    tmp_path, monkeypatch
) -> None:
    FakeClient, writes = _make_upsert_image_gallery_fake_client(
        read_updated=None, server_updated=READ_UPDATED
    )
    manifest_path = _write_gallery_manifest(tmp_path)

    result = _run(
        monkeypatch,
        tmp_path,
        FakeClient,
        ["upsert-image-gallery", ISSUE_KEY, "--manifest", str(manifest_path), "--execute"],
    )

    assert result.exit_code == cli.CONCURRENT_EDIT_EXIT_CODE, result.stdout + result.stderr
    assert "Concurrency guard unavailable" in result.stderr
    assert writes == []
