from __future__ import annotations

import os
from pathlib import Path

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


def load_config(env_file: str | Path = ".env") -> JiraConfig:
    load_dotenv(env_file, override=False)
    return JiraConfig.model_validate(dict(os.environ))
