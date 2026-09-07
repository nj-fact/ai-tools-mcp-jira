from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from jira_agent import config as config_module
from jira_agent.config import (
    JiraConfig,
    load_config,
    normalize_project_key,
    reset_environment_cache,
    resolve_environment,
)


def make_env(**extra: str) -> dict[str, str]:
    env = {
        "JIRA_SITE_URL": "https://example.atlassian.net",
        "JIRA_EMAIL": "user@example.com",
        "JIRA_API_TOKEN": "token",
    }
    env.update(extra)
    return env


def test_config_without_overrides_keeps_default_credentials() -> None:
    config = JiraConfig.model_validate(make_env())

    assert config.project_credentials == {}
    assert config.credential_source_for("FC-1") == "default"
    credentials = config.credentials_for("FC-1")
    assert credentials.site_url == "https://example.atlassian.net"
    assert credentials.email == "user@example.com"
    assert credentials.api_token.get_secret_value() == "token"


def test_project_override_is_parsed_and_site_url_falls_back() -> None:
    config = JiraConfig.model_validate(
        make_env(
            JIRA_PROJECT__FINRP__EMAIL="finrp@example.com",
            JIRA_PROJECT__FINRP__API_TOKEN="finrp-token",
        )
    )

    credentials = config.project_credentials["FINRP"]
    assert credentials.email == "finrp@example.com"
    assert credentials.api_token.get_secret_value() == "finrp-token"
    assert credentials.site_url == "https://example.atlassian.net"


def test_project_override_site_url_wins_and_is_normalized() -> None:
    config = JiraConfig.model_validate(
        make_env(
            JIRA_PROJECT__FINRP__EMAIL="finrp@example.com",
            JIRA_PROJECT__FINRP__API_TOKEN="finrp-token",
            JIRA_PROJECT__FINRP__SITE_URL="https://other.atlassian.net/",
        )
    )

    assert config.project_credentials["FINRP"].site_url == "https://other.atlassian.net"


def test_project_key_in_variable_name_is_case_insensitive() -> None:
    config = JiraConfig.model_validate(
        make_env(
            JIRA_PROJECT__finrp__EMAIL="finrp@example.com",
            JIRA_PROJECT__FinRp__API_TOKEN="finrp-token",
        )
    )

    assert list(config.project_credentials) == ["FINRP"]
    assert config.credentials_for("FINRP-7").email == "finrp@example.com"


def test_partial_override_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="JIRA_PROJECT__FINRP__EMAIL"):
        JiraConfig.model_validate(make_env(JIRA_PROJECT__FINRP__API_TOKEN="finrp-token"))


def test_empty_override_value_counts_as_missing() -> None:
    with pytest.raises(ValidationError, match="JIRA_PROJECT__FINRP__API_TOKEN"):
        JiraConfig.model_validate(
            make_env(
                JIRA_PROJECT__FINRP__EMAIL="finrp@example.com",
                JIRA_PROJECT__FINRP__API_TOKEN="  ",
            )
        )


def test_unrecognized_project_credential_variable_is_a_validation_error() -> None:
    with pytest.raises(ValidationError, match="Unrecognized per-project credential variable"):
        JiraConfig.model_validate(make_env(JIRA_PROJECT__FINRP__TOKEN="finrp-token"))


def test_credentials_for_resolves_project_and_issue_keys() -> None:
    config = JiraConfig.model_validate(
        make_env(
            JIRA_PROJECT__FINRP__EMAIL="finrp@example.com",
            JIRA_PROJECT__FINRP__API_TOKEN="finrp-token",
        )
    )

    assert config.credentials_for("FINRP").email == "finrp@example.com"
    assert config.credentials_for("finrp-123").email == "finrp@example.com"
    assert config.credentials_for("FC-1").email == "user@example.com"
    assert config.credentials_for(None).email == "user@example.com"
    assert config.credential_source_for("FINRP-9") == "FINRP"
    assert config.credential_source_for("FC-1") == "default"
    assert config.credential_source_for(None) == "default"


def test_normalize_project_key_handles_issue_keys_and_blanks() -> None:
    assert normalize_project_key(None) is None
    assert normalize_project_key("") is None
    assert normalize_project_key("  ") is None
    assert normalize_project_key("fc") == "FC"
    assert normalize_project_key(" finrp-123 ") == "FINRP"
    assert normalize_project_key("FINRP") == "FINRP"


# --- Environment layering -------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_environment_cache():
    yield
    reset_environment_cache()


@pytest.fixture(autouse=True)
def _isolated_file_lookup(monkeypatch):
    """Keep the developer's own .env out of the suite.

    Without this the checkout fallback in config._default_file would resolve to
    this repository whenever a test leaves its tmp_path without a .env, and the
    real values would leak into the assertions. Tests that mean to exercise the
    fallback set PROJECT_ROOT to a tmp_path of their own.
    """
    monkeypatch.delenv(config_module.ENV_FILE_VARIABLE, raising=False)
    monkeypatch.delenv(config_module.ENVRC_FILE_VARIABLE, raising=False)
    monkeypatch.setattr(config_module, "PROJECT_ROOT", None)


def write_env(path: Path, **values: str) -> Path:
    path.write_text("\n".join(f"{name}={value}" for name, value in values.items()) + "\n")
    return path


def resolve(tmp_path: Path, monkeypatch) -> dict[str, str]:
    monkeypatch.chdir(tmp_path)
    return resolve_environment()


def test_env_file_values_are_layered_over_the_process_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JIRA_SITE_URL", "https://from-process.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "process@example.com")
    write_env(tmp_path / ".env", JIRA_SITE_URL="https://from-env-file.atlassian.net")

    merged = resolve(tmp_path, monkeypatch)

    assert merged["JIRA_SITE_URL"] == "https://from-env-file.atlassian.net"
    # Untouched process variables survive the merge.
    assert merged["JIRA_EMAIL"] == "process@example.com"


def test_envrc_overrides_env_file(tmp_path, monkeypatch) -> None:
    write_env(tmp_path / ".env", JIRA_WRITE_PROJECT_KEYS="FC", JIRA_EMAIL="env@example.com")
    (tmp_path / ".envrc").write_text("export JIRA_WRITE_PROJECT_KEYS=FC,ITHD\n")

    merged = resolve(tmp_path, monkeypatch)

    assert merged["JIRA_WRITE_PROJECT_KEYS"] == "FC,ITHD"
    assert merged["JIRA_EMAIL"] == "env@example.com"


def test_envrc_edits_are_picked_up_without_a_restart(tmp_path, monkeypatch) -> None:
    write_env(tmp_path / ".env", JIRA_WRITE_PROJECT_KEYS="FC")
    envrc = tmp_path / ".envrc"
    envrc.write_text("export JIRA_WRITE_PROJECT_KEYS=FC,FINRP\n")
    monkeypatch.chdir(tmp_path)

    assert resolve_environment()["JIRA_WRITE_PROJECT_KEYS"] == "FC,FINRP"

    envrc.write_text("export JIRA_WRITE_PROJECT_KEYS=FC,FINRP,ITHD\n")

    assert resolve_environment()["JIRA_WRITE_PROJECT_KEYS"] == "FC,FINRP,ITHD"


def test_env_file_edits_are_picked_up_without_a_restart(tmp_path, monkeypatch) -> None:
    env_file = write_env(tmp_path / ".env", JIRA_WRITE_PROJECT_KEYS="FC")
    monkeypatch.chdir(tmp_path)

    assert resolve_environment()["JIRA_WRITE_PROJECT_KEYS"] == "FC"

    write_env(env_file, JIRA_WRITE_PROJECT_KEYS="FC,ITHD")

    assert resolve_environment()["JIRA_WRITE_PROJECT_KEYS"] == "FC,ITHD"


def test_value_dropped_from_the_files_restores_the_process_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JIRA_EMAIL", "process@example.com")
    env_file = write_env(tmp_path / ".env", JIRA_EMAIL="env@example.com")
    monkeypatch.chdir(tmp_path)

    assert resolve_environment()["JIRA_EMAIL"] == "env@example.com"

    env_file.write_text("\n")

    assert resolve_environment()["JIRA_EMAIL"] == "process@example.com"


def test_value_dropped_from_the_files_disappears_when_it_was_file_only(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("JIRA_WRITE_ISSUE_KEYS", raising=False)
    env_file = write_env(tmp_path / ".env", JIRA_WRITE_ISSUE_KEYS="SAFE-1")
    monkeypatch.chdir(tmp_path)

    assert resolve_environment()["JIRA_WRITE_ISSUE_KEYS"] == "SAFE-1"

    env_file.write_text("\n")

    assert "JIRA_WRITE_ISSUE_KEYS" not in resolve_environment()
    assert "JIRA_WRITE_ISSUE_KEYS" not in os.environ


def test_missing_files_are_not_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JIRA_EMAIL", "process@example.com")

    assert resolve(tmp_path, monkeypatch)["JIRA_EMAIL"] == "process@example.com"


def test_envrc_directives_are_ignored_and_includes_are_followed(tmp_path, monkeypatch) -> None:
    write_env(tmp_path / ".env", JIRA_EMAIL="env@example.com")
    # The shape this project actually uses: the .envrc carries no credentials of
    # its own, only a flake reference and an include of the local override file.
    (tmp_path / ".envrc").write_text(
        "use flake ~/.nix-config#mcp-jira\n"
        "\n"
        "# Per-developer overrides\n"
        "source_env_if_exists .envrc.local\n"
    )
    (tmp_path / ".envrc.local").write_text('export JIRA_EMAIL="local@example.com"\n')

    assert resolve(tmp_path, monkeypatch)["JIRA_EMAIL"] == "local@example.com"


def test_include_created_later_invalidates_the_cache(tmp_path, monkeypatch) -> None:
    write_env(tmp_path / ".env", JIRA_EMAIL="env@example.com")
    (tmp_path / ".envrc").write_text("source_env_if_exists .envrc.local\n")
    monkeypatch.chdir(tmp_path)

    assert resolve_environment()["JIRA_EMAIL"] == "env@example.com"

    (tmp_path / ".envrc.local").write_text("export JIRA_EMAIL=local@example.com\n")

    assert resolve_environment()["JIRA_EMAIL"] == "local@example.com"


def test_missing_optional_include_is_skipped(tmp_path, monkeypatch) -> None:
    (tmp_path / ".envrc").write_text(
        "source_env_if_exists .envrc.local\nexport JIRA_EMAIL=envrc@example.com\n"
    )

    assert resolve(tmp_path, monkeypatch)["JIRA_EMAIL"] == "envrc@example.com"


def test_envrc_assignment_after_an_include_wins(tmp_path, monkeypatch) -> None:
    (tmp_path / ".envrc").write_text(
        "source_env_if_exists .envrc.local\nexport JIRA_EMAIL=after@example.com\n"
    )
    (tmp_path / ".envrc.local").write_text("export JIRA_EMAIL=local@example.com\n")

    assert resolve(tmp_path, monkeypatch)["JIRA_EMAIL"] == "after@example.com"


def test_envrc_dotenv_directive_reads_a_dotenv_file(tmp_path, monkeypatch) -> None:
    (tmp_path / ".envrc").write_text("dotenv_if_exists .env.shared\n")
    write_env(tmp_path / ".env.shared", JIRA_EMAIL="shared@example.com")

    assert resolve(tmp_path, monkeypatch)["JIRA_EMAIL"] == "shared@example.com"


def test_envrc_quotes_and_inline_comments_are_stripped(tmp_path, monkeypatch) -> None:
    (tmp_path / ".envrc").write_text(
        "export JIRA_EMAIL='quoted@example.com'\n"
        "JIRA_WRITE_PROJECT_KEYS=FC,ITHD  # only these two\n"
    )

    merged = resolve(tmp_path, monkeypatch)

    assert merged["JIRA_EMAIL"] == "quoted@example.com"
    assert merged["JIRA_WRITE_PROJECT_KEYS"] == "FC,ITHD"


def test_envrc_unset_removes_a_value_from_the_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("JIRA_WRITE_ISSUE_KEYS", raising=False)
    write_env(tmp_path / ".env", JIRA_WRITE_ISSUE_KEYS="SAFE-1")
    (tmp_path / ".envrc").write_text("unset JIRA_WRITE_ISSUE_KEYS\n")

    assert "JIRA_WRITE_ISSUE_KEYS" not in resolve(tmp_path, monkeypatch)


def test_envrc_unset_also_hides_a_process_value(tmp_path, monkeypatch) -> None:
    # Shell semantics: unset removes the name outright, it does not fall back
    # to whatever the process environment carried.
    monkeypatch.setenv("JIRA_WRITE_ISSUE_KEYS", "SAFE-1")
    (tmp_path / ".envrc").write_text("unset JIRA_WRITE_ISSUE_KEYS\n")

    assert "JIRA_WRITE_ISSUE_KEYS" not in resolve(tmp_path, monkeypatch)
    assert "JIRA_WRITE_ISSUE_KEYS" not in os.environ


def test_dropped_unset_restores_the_process_value(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JIRA_WRITE_ISSUE_KEYS", "SAFE-1")
    envrc = tmp_path / ".envrc"
    envrc.write_text("unset JIRA_WRITE_ISSUE_KEYS\n")
    monkeypatch.chdir(tmp_path)

    assert "JIRA_WRITE_ISSUE_KEYS" not in resolve_environment()

    envrc.write_text("\n")

    assert resolve_environment()["JIRA_WRITE_ISSUE_KEYS"] == "SAFE-1"


def test_include_cycle_terminates(tmp_path, monkeypatch) -> None:
    (tmp_path / ".envrc").write_text(
        "source_env .envrc.local\nexport JIRA_EMAIL=envrc@example.com\n"
    )
    (tmp_path / ".envrc.local").write_text("source_env .envrc\n")

    assert resolve(tmp_path, monkeypatch)["JIRA_EMAIL"] == "envrc@example.com"


def test_load_config_layers_envrc_over_env_file(tmp_path, monkeypatch) -> None:
    write_env(
        tmp_path / ".env",
        JIRA_SITE_URL="https://example.atlassian.net",
        JIRA_EMAIL="user@example.com",
        JIRA_API_TOKEN="token",
        JIRA_WRITE_PROJECT_KEYS="FC",
    )
    (tmp_path / ".envrc").write_text("export JIRA_WRITE_PROJECT_KEYS=FC,ITHD\n")
    monkeypatch.chdir(tmp_path)

    assert load_config().write_project_keys == ["FC", "ITHD"]


def test_mermaid_variables_reach_os_environ(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("JIRA_MERMAID_NODE_PATH", raising=False)
    write_env(tmp_path / ".env", JIRA_MERMAID_NODE_PATH="/usr/bin/node")

    resolve(tmp_path, monkeypatch)

    assert os.environ["JIRA_MERMAID_NODE_PATH"] == "/usr/bin/node"


# --- Where the files are looked up ----------------------------------------


def test_project_root_is_found_for_a_src_layout(tmp_path) -> None:
    root = tmp_path / "checkout"
    package = root / "src" / "jira_agent"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "jira-agent"\n')

    assert config_module._find_project_root(package / "config.py") == root


def test_project_root_is_found_for_a_flat_layout(tmp_path) -> None:
    root = tmp_path / "checkout"
    package = root / "jira_agent"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "jira-agent"\n')

    assert config_module._find_project_root(package / "config.py") == root


def test_project_root_is_none_for_an_installed_distribution(tmp_path) -> None:
    package = tmp_path / "lib" / "python3.12" / "site-packages" / "jira_agent"
    package.mkdir(parents=True)

    assert config_module._find_project_root(package / "config.py") is None


def test_checkout_root_is_used_when_the_working_directory_has_no_env_file(
    tmp_path, monkeypatch
) -> None:
    # The MCP server case: launched as a module with a foreign working directory.
    root = tmp_path / "checkout"
    root.mkdir()
    write_env(root / ".env", JIRA_EMAIL="checkout@example.com")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)

    assert resolve_environment()["JIRA_EMAIL"] == "checkout@example.com"


def test_working_directory_env_file_wins_over_the_checkout_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    write_env(root / ".env", JIRA_EMAIL="checkout@example.com")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    write_env(elsewhere / ".env", JIRA_EMAIL="working-directory@example.com")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)

    assert resolve_environment()["JIRA_EMAIL"] == "working-directory@example.com"


def test_env_file_variable_wins_over_both(tmp_path, monkeypatch) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    write_env(root / ".env", JIRA_EMAIL="checkout@example.com")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    write_env(elsewhere / ".env", JIRA_EMAIL="working-directory@example.com")
    pinned = write_env(tmp_path / "pinned.env", JIRA_EMAIL="pinned@example.com")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv(config_module.ENV_FILE_VARIABLE, str(pinned))

    assert resolve_environment()["JIRA_EMAIL"] == "pinned@example.com"


def test_env_file_variable_accepts_a_relative_path(tmp_path, monkeypatch) -> None:
    write_env(tmp_path / ".env", JIRA_EMAIL="working-directory@example.com")
    (tmp_path / "config").mkdir()
    write_env(tmp_path / "config" / "shared.env", JIRA_EMAIL="relative@example.com")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config_module.ENV_FILE_VARIABLE, "config/shared.env")

    assert resolve_environment()["JIRA_EMAIL"] == "relative@example.com"


def test_envrc_file_variable_is_honoured_too(tmp_path, monkeypatch) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    write_env(root / ".env", JIRA_EMAIL="checkout@example.com", JIRA_WRITE_PROJECT_KEYS="FC")
    pinned = tmp_path / "pinned.envrc"
    pinned.write_text("export JIRA_WRITE_PROJECT_KEYS=FC,ITHD\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv(config_module.ENVRC_FILE_VARIABLE, str(pinned))

    merged = resolve_environment()

    assert merged["JIRA_EMAIL"] == "checkout@example.com"
    assert merged["JIRA_WRITE_PROJECT_KEYS"] == "FC,ITHD"


def test_explicit_arguments_win_over_the_variable(tmp_path, monkeypatch) -> None:
    explicit = write_env(tmp_path / "explicit.env", JIRA_EMAIL="explicit@example.com")
    pinned = write_env(tmp_path / "pinned.env", JIRA_EMAIL="pinned@example.com")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config_module.ENV_FILE_VARIABLE, str(pinned))

    merged = resolve_environment(explicit, tmp_path / "missing.envrc")

    assert merged["JIRA_EMAIL"] == "explicit@example.com"


def test_env_file_in_the_files_cannot_redirect_the_lookup(tmp_path, monkeypatch) -> None:
    # The override is a bootstrap value: it decides which file is read, so it
    # must not be readable from that file.
    write_env(
        tmp_path / ".env",
        JIRA_ENV_FILE=str(tmp_path / "other.env"),
        JIRA_EMAIL="working-directory@example.com",
    )
    write_env(tmp_path / "other.env", JIRA_EMAIL="other@example.com")
    monkeypatch.chdir(tmp_path)

    assert resolve_environment()["JIRA_EMAIL"] == "working-directory@example.com"
    assert resolve_environment()["JIRA_EMAIL"] == "working-directory@example.com"


def test_missing_checkout_root_falls_back_to_the_working_directory(tmp_path, monkeypatch) -> None:
    write_env(tmp_path / ".env", JIRA_EMAIL="working-directory@example.com")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", None)
    monkeypatch.chdir(tmp_path)

    assert resolve_environment()["JIRA_EMAIL"] == "working-directory@example.com"


def test_load_config_reads_the_checkout_root_from_a_foreign_working_directory(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    write_env(
        root / ".env",
        JIRA_SITE_URL="https://example.atlassian.net",
        JIRA_EMAIL="user@example.com",
        JIRA_API_TOKEN="token",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    monkeypatch.chdir(elsewhere)

    assert load_config().site_url == "https://example.atlassian.net"
