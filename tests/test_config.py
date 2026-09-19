from __future__ import annotations

from pathlib import Path

import pytest

from jira_agent.config import load_config


def _write_env(path: Path, token_file: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "JIRA_SITE_URL=https://example.atlassian.net",
                "JIRA_EMAIL=user@example.com",
                "JIRA_API_TOKEN=stale-token",
                f"JIRA_API_TOKEN_FILE={token_file}",
            ]
        ),
        encoding="utf-8",
    )


def test_load_config_prefers_api_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("JIRA_SITE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_API_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)

    token_file = tmp_path / "jira.token"
    token_file.write_text("file-token\n", encoding="utf-8")
    env_file = tmp_path / ".env"
    _write_env(env_file, token_file)

    config = load_config(env_file)

    assert config.api_token.get_secret_value() == "file-token"


def test_load_config_rejects_unreadable_api_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("JIRA_SITE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_API_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)

    token_file = tmp_path / "missing.token"
    env_file = tmp_path / ".env"
    _write_env(env_file, token_file)

    with pytest.raises(ValueError, match="Cannot read JIRA_API_TOKEN_FILE"):
        load_config(env_file)
