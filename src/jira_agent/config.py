from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

DEFAULT_CREDENTIAL_SOURCE = "default"

_PROJECT_CREDENTIAL_PATTERN = re.compile(
    r"JIRA_PROJECT__(?P<project>.+)__(?P<field>EMAIL|API_TOKEN|SITE_URL)"
)


def normalize_project_key(key: str | None) -> str | None:
    """Normalize a project or issue key for credential lookup ("finrp-123" -> "FINRP")."""
    if key is None:
        return None
    normalized = key.strip().upper()
    if not normalized:
        return None
    return normalized.split("-", 1)[0]


class ProjectCredentials(BaseModel):
    site_url: str
    email: str
    api_token: SecretStr

    @field_validator("site_url")
    @classmethod
    def normalize_site_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("A Jira site URL is required")
        if not normalized.startswith("https://"):
            raise ValueError("A Jira site URL must start with https://")
        return normalized

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("A Jira email is required")
        return normalized

    @field_validator("api_token")
    @classmethod
    def validate_api_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("A Jira API token is required")
        return value


class JiraConfig(BaseModel):
    site_url: str = Field(alias="JIRA_SITE_URL")
    email: str = Field(alias="JIRA_EMAIL")
    api_token: SecretStr = Field(alias="JIRA_API_TOKEN")
    timeout_seconds: float = Field(default=20.0, alias="JIRA_TIMEOUT_SECONDS")
    test_jql: str = Field(
        default="project = FC AND updated >= -365d ORDER BY updated DESC",
        alias="JIRA_TEST_JQL",
    )
    test_project_key: str = Field(default="FC", alias="JIRA_TEST_PROJECT_KEY")
    test_issue_type: str = Field(default="Task", alias="JIRA_TEST_ISSUE_TYPE")
    test_issue_key: str | None = Field(default=None, alias="JIRA_TEST_ISSUE_KEY")
    write_project_keys: list[str] = Field(
        default_factory=lambda: ["FC"], alias="JIRA_WRITE_PROJECT_KEYS"
    )
    write_issue_keys: list[str] = Field(default_factory=list, alias="JIRA_WRITE_ISSUE_KEYS")
    audit_log_path: str = Field(default=".jira-agent/audit.jsonl", alias="JIRA_AUDIT_LOG_PATH")
    story_points_field: str = Field(default="customfield_10028", alias="JIRA_STORY_POINTS_FIELD")
    project_credentials: dict[str, ProjectCredentials] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def collect_project_credentials(cls, data: Any) -> Any:
        """Discover JIRA_PROJECT__<PROJECTKEY>__EMAIL/API_TOKEN/SITE_URL overrides.

        Project keys are case-insensitive in the variable name and normalized to uppercase.
        SITE_URL is optional per project and falls back to JIRA_SITE_URL.
        """
        if not isinstance(data, dict):
            return data
        overrides: dict[str, dict[str, str]] = {}
        for name, value in data.items():
            if not isinstance(name, str) or not name.startswith("JIRA_PROJECT__"):
                continue
            match = _PROJECT_CREDENTIAL_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(
                    f"Unrecognized per-project credential variable {name}; expected "
                    "JIRA_PROJECT__<PROJECTKEY>__EMAIL, "
                    "JIRA_PROJECT__<PROJECTKEY>__API_TOKEN, or "
                    "JIRA_PROJECT__<PROJECTKEY>__SITE_URL"
                )
            project_key = match.group("project").upper()
            overrides.setdefault(project_key, {})[match.group("field").lower()] = str(value)
        if not overrides:
            return data

        resolved: dict[str, dict[str, str]] = {}
        for project_key, fields in sorted(overrides.items()):
            missing = [name for name in ("email", "api_token") if not fields.get(name, "").strip()]
            if missing:
                raise ValueError(
                    f"Incomplete credential override for Jira project {project_key}: "
                    + " and ".join(
                        f"JIRA_PROJECT__{project_key}__{name.upper()}" for name in missing
                    )
                    + " must be set as well; a project override needs both EMAIL and API_TOKEN"
                )
            resolved[project_key] = {
                "site_url": fields.get("site_url") or str(data.get("JIRA_SITE_URL") or ""),
                "email": fields["email"],
                "api_token": fields["api_token"],
            }
        data = dict(data)
        data["project_credentials"] = resolved
        return data

    def credential_source_for(self, key: str | None) -> str:
        """Name the credential set serving a project or issue key: the project key or "default"."""
        project_key = normalize_project_key(key)
        if project_key is not None and project_key in self.project_credentials:
            return project_key
        return DEFAULT_CREDENTIAL_SOURCE

    def credentials_for(self, key: str | None) -> ProjectCredentials:
        """Return the credentials for a project or issue key, falling back to the defaults."""
        source = self.credential_source_for(key)
        if source != DEFAULT_CREDENTIAL_SOURCE:
            return self.project_credentials[source]
        return ProjectCredentials(
            site_url=self.site_url,
            email=self.email,
            api_token=self.api_token,
        )

    @field_validator("site_url")
    @classmethod
    def normalize_site_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("JIRA_SITE_URL is required")
        if not normalized.startswith("https://"):
            raise ValueError("JIRA_SITE_URL must start with https://")
        return normalized

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("JIRA_EMAIL is required")
        return normalized

    @field_validator("api_token")
    @classmethod
    def validate_api_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("JIRA_API_TOKEN is required")
        return value

    @field_validator("test_jql")
    @classmethod
    def validate_test_jql(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("JIRA_TEST_JQL must not be empty")
        return normalized

    @field_validator("test_project_key", "test_issue_type")
    @classmethod
    def validate_test_scope_value(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Jira test scope values must not be empty")
        return normalized

    @field_validator("test_issue_key")
    @classmethod
    def normalize_test_issue_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("story_points_field")
    @classmethod
    def validate_story_points_field(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.startswith("customfield_") or not normalized[12:].isdigit():
            raise ValueError(
                "JIRA_STORY_POINTS_FIELD must look like customfield_10028; "
                "the numeric id differs per Jira site"
            )
        return normalized

    @field_validator("write_project_keys", "write_issue_keys", mode="before")
    @classmethod
    def parse_csv_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("Expected comma-separated string or list")


def load_config(env_file: str | Path = ".env") -> JiraConfig:
    load_dotenv(env_file, override=False)
    return JiraConfig.model_validate(dict(os.environ))
