from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator


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
