from __future__ import annotations

import pytest
from pydantic import ValidationError

from jira_agent.config import JiraConfig, normalize_project_key


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
