from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jira_agent.config import JiraConfig


def audit_write(
    config: JiraConfig,
    *,
    action: str,
    issue_key: str | None,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    path = Path(config.audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": action,
        "issue_key": issue_key,
        "credential_source": config.credential_source_for(issue_key),
        "payload": payload or {},
        "result": result or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def audit_committed_write(
    config: JiraConfig,
    *,
    action: str,
    issue_key: str | None,
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    writer: Callable[..., None] | None = None,
) -> dict[str, Any] | None:
    """Audit a Jira mutation that is already committed without converting it into failure.

    Once Jira accepted a non-idempotent mutation, an audit-file or JSON-serialization failure
    cannot safely be raised as an ordinary command failure: a caller may retry and duplicate the
    remote write. Return an explicit committed-warning document instead. Callers must surface it
    while retaining a success/partial-success state and continue any best-effort cleanup.
    """
    selected_writer = writer or audit_write
    try:
        selected_writer(
            config,
            action=action,
            issue_key=issue_key,
            payload=payload,
            result=result,
        )
    except Exception as exc:
        warning = committed_audit_warning(action=action, exc=exc)
        if result is not None:
            # Preserve the remote payload/IDs while making the partial-success state available
            # to callers that return this result object directly.
            result["audit_warning"] = warning
            result["partial_success"] = True
        return warning
    return None


def committed_audit_warning(*, action: str, exc: Exception) -> dict[str, Any]:
    """Describe an audit failure that happened after Jira accepted the mutation."""
    return {
        "error_type": "audit_write_failed_after_commit",
        "committed": True,
        "action": action,
        "message": (
            "Jira accepted the mutation, but writing the local audit record failed: "
            f"{type(exc).__name__}: {exc}"
        ),
    }


def committed_verification_warning(*, action: str, exc: Exception) -> dict[str, Any]:
    """Describe a verification read that failed after Jira accepted the mutation."""
    warning: dict[str, Any] = {
        "error_type": "verification_unavailable_after_commit",
        "committed": True,
        "action": action,
        "message": (
            "Jira accepted the mutation, but its post-write verification read failed: "
            f"{type(exc).__name__}: {exc}"
        ),
    }
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        warning["status_code"] = status_code
    return warning


def committed_sequence_warning(*, failed_step: str, exc: Exception) -> dict[str, Any]:
    """Describe a later failed step after at least one earlier remote step committed."""
    warning: dict[str, Any] = {
        "error_type": "sequence_step_failed_after_commit",
        "committed": True,
        "failed_step": failed_step,
        "message": (
            "Earlier Jira mutation step(s) are committed, but the later "
            f"{failed_step} step failed: {type(exc).__name__}: {exc}"
        ),
    }
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        warning["status_code"] = status_code
    return warning
