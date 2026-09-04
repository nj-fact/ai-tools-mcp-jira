from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

DEFAULT_CREDENTIAL_SOURCE = "default"

DEFAULT_ENV_FILE = ".env"
DEFAULT_ENVRC_FILE = ".envrc"

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


# --- Environment layering -------------------------------------------------
#
# Values are merged lowest to highest precedence:
#   1. the real process environment
#   2. .env      (dotenv syntax, python-dotenv)
#   3. .envrc    (direnv file; assignments and source/dotenv includes only)
#
# Both files are re-checked on every resolve, so an edit takes effect on the
# next tool call instead of at the next process start.

_MAX_INCLUDE_DEPTH = 8

_INCLUDE_PATTERN = re.compile(
    r"^(?P<directive>source_env_if_exists|source_env|dotenv_if_exists|dotenv)"
    r"\s+(?P<argument>\S.*)$"
)
_ASSIGNMENT_PATTERN = re.compile(r"^(?:export\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$")
_UNSET_PATTERN = re.compile(r"^(?:unset|unset_var)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$")

_Stamp = tuple[int, int]
# A layer maps a name to its value, or to None when the file unsets it.
_Layer = dict[str, str | None]

_resolve_lock = threading.Lock()
# Cache per root file: the stamps of every file that contributed (missing ones
# included, so creating .envrc.local later invalidates the entry) plus the
# values that parse produced.
_layer_cache: dict[tuple[Path, bool], tuple[dict[Path, _Stamp | None], _Layer]] = {}
# Keys this module wrote into os.environ, mapped to the value that was there
# before, so a key dropped from the files can be restored instead of lingering.
_process_overrides: dict[str, str | None] = {}


def _stamp(path: Path) -> _Stamp | None:
    """Change marker for a file, or None when it does not exist."""
    try:
        status = path.stat()
    except OSError:
        return None
    return (status.st_mtime_ns, status.st_size)


def _strip_inline_comment(value: str) -> str:
    """Drop a whitespace-preceded trailing # comment from an unquoted value."""
    for separator in (" #", "\t#"):
        value = value.split(separator, 1)[0]
    return value.strip()


def _unquote(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1]
    return _strip_inline_comment(stripped)


def _absorb(
    path: Path,
    *,
    as_envrc: bool,
    values: _Layer,
    stamps: dict[Path, _Stamp | None],
    seen: set[Path],
    depth: int,
) -> None:
    """Merge one env file into values, following .envrc include directives.

    Every file consulted is recorded in stamps, whether or not it exists.
    """
    key = path.resolve()
    stamp = _stamp(path)
    stamps.setdefault(key, stamp)
    if stamp is None or key in seen or depth > _MAX_INCLUDE_DEPTH:
        return
    seen.add(key)

    if not as_envrc:
        for name, value in dotenv_values(path).items():
            if value is not None:
                values[name] = value
        return

    # A .envrc is a shell script. Only plain assignments and include directives
    # are honoured; `use flake`, `layout`, conditionals and command
    # substitution are ignored because evaluating them would mean running a
    # shell on every tool call.
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        include = _INCLUDE_PATTERN.match(line)
        if include is not None:
            target = _unquote(include.group("argument"))
            if target:
                nested = Path(target)
                if not nested.is_absolute():
                    nested = path.parent / nested
                _absorb(
                    nested,
                    as_envrc=include.group("directive").startswith("source_env"),
                    values=values,
                    stamps=stamps,
                    seen=seen,
                    depth=depth + 1,
                )
            continue

        unset = _UNSET_PATTERN.match(line)
        if unset is not None:
            values[unset.group("name")] = None
            continue

        assignment = _ASSIGNMENT_PATTERN.match(line)
        if assignment is not None:
            values[assignment.group("name")] = _unquote(assignment.group("value"))


def _file_layer(path: Path, *, as_envrc: bool) -> _Layer:
    """Values from one env file, reparsed only when it or an include changed."""
    cache_key = (path.resolve(), as_envrc)
    cached = _layer_cache.get(cache_key)
    if cached is not None:
        stamps, values = cached
        if all(_stamp(known) == stamp for known, stamp in stamps.items()):
            return dict(values)

    values: _Layer = {}
    stamps: dict[Path, _Stamp | None] = {}
    _absorb(path, as_envrc=as_envrc, values=values, stamps=stamps, seen=set(), depth=0)
    _layer_cache[cache_key] = (stamps, dict(values))
    return values


def _base_environment() -> dict[str, str]:
    """The process environment as it was before this module wrote to it."""
    base = dict(os.environ)
    for name, original in _process_overrides.items():
        if original is None:
            base.pop(name, None)
        else:
            base[name] = original
    return base


def _publish(file_values: _Layer) -> None:
    """Mirror file values into os.environ for consumers that read it directly.

    jira_agent.mermaid picks up JIRA_MERMAID_* and PUPPETEER_EXECUTABLE_PATH
    from os.environ, so the files have to reach it there too.
    """
    for name in set(_process_overrides) - set(file_values):
        original = _process_overrides.pop(name)
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original
    for name, value in file_values.items():
        if name not in _process_overrides:
            _process_overrides[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def resolve_environment(
    env_file: str | Path = DEFAULT_ENV_FILE,
    envrc_file: str | Path = DEFAULT_ENVRC_FILE,
) -> dict[str, str]:
    """Merge process environment, .env and .envrc, re-reading both files."""
    with _resolve_lock:
        file_values = _file_layer(Path(env_file), as_envrc=False)
        file_values.update(_file_layer(Path(envrc_file), as_envrc=True))
        merged = _base_environment()
        for name, value in file_values.items():
            if value is None:
                merged.pop(name, None)
            else:
                merged[name] = value
        _publish(file_values)
        return merged


def reset_environment_cache() -> None:
    """Forget parsed files and restore os.environ. Intended for tests."""
    with _resolve_lock:
        _layer_cache.clear()
        _publish({})


def load_config(
    env_file: str | Path = DEFAULT_ENV_FILE,
    envrc_file: str | Path = DEFAULT_ENVRC_FILE,
) -> JiraConfig:
    return JiraConfig.model_validate(resolve_environment(env_file, envrc_file))
