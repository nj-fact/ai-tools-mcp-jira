from __future__ import annotations

import json
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
        "payload": payload or {},
        "result": result or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
