from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from jira_agent.config import JiraConfig


class JiraApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class JiraClient:
    def __init__(self, config: JiraConfig, *, http_client: httpx.Client | None = None) -> None:
        self.config = config
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=f"{config.site_url}/rest/api/3",
            auth=(config.email, config.api_token.get_secret_value()),
            timeout=config.timeout_seconds,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "jira-agent/0.1",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> JiraClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def myself(self) -> dict[str, Any]:
        return self._request("GET", "/myself")

    def get_issue(
        self,
        issue_key: str,
        *,
        fields: Iterable[str] | None = None,
        expand: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {}
        if fields:
            params["fields"] = ",".join(fields)
        if expand:
            params["expand"] = ",".join(expand)
        return self._request("GET", f"/issue/{issue_key}", params=params)

    def search_issues(
        self,
        jql: str,
        *,
        limit: int = 20,
        fields: Iterable[str] | None = None,
        next_page_token: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "jql": jql,
            "maxResults": limit,
        }
        if fields:
            params["fields"] = list(fields)
        if next_page_token:
            params["nextPageToken"] = next_page_token
        return self._request("GET", "/search/jql", params=params)

    def list_comments(
        self,
        issue_key: str,
        *,
        start_at: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/issue/{issue_key}/comment",
            params={"startAt": start_at, "maxResults": limit},
        )

    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/issue/{issue_key}/comment",
            json={"body": text_to_adf(body)},
        )

    def get_transitions(self, issue_key: str) -> dict[str, Any]:
        return self._request("GET", f"/issue/{issue_key}/transitions")

    def transition_issue(self, issue_key: str, transition_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/issue/{issue_key}/transitions",
            json={"transition": {"id": transition_id}},
        )

    def create_metadata(
        self,
        *,
        project_key: str,
        issue_type_name: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "projectKeys": project_key,
            "expand": "projects.issuetypes.fields",
        }
        if issue_type_name:
            params["issuetypeNames"] = issue_type_name
        return self._request("GET", "/issue/createmeta", params=params)

    def create_issue(
        self,
        *,
        project_key: str,
        issue_type_name: str,
        summary: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/issue",
            json=create_issue_payload(
                project_key=project_key,
                issue_type_name=issue_type_name,
                summary=summary,
                description=description,
            ),
        )

    def assign_issue(self, issue_key: str, account_id: str | None) -> dict[str, Any]:
        return self._request(
            "PUT",
            f"/issue/{issue_key}/assignee",
            json={"accountId": account_id},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:
            raise JiraApiError(f"Jira request failed before receiving a response: {exc}") from exc

        if response.status_code >= 400:
            raise self._api_error(response)

        if not response.content:
            return {}

        try:
            data = response.json()
        except ValueError as exc:
            raise JiraApiError(
                "Jira returned a non-JSON response",
                status_code=response.status_code,
                details=response.text[:500],
            ) from exc

        if not isinstance(data, dict):
            raise JiraApiError(
                "Jira returned an unexpected JSON payload",
                status_code=response.status_code,
                details=data,
            )
        return data

    def _api_error(self, response: httpx.Response) -> JiraApiError:
        try:
            details: Any = response.json()
        except ValueError:
            details = response.text[:500]

        message = f"Jira API returned HTTP {response.status_code}"
        if isinstance(details, dict):
            errors = details.get("errorMessages") or details.get("errors")
            if errors:
                message = f"{message}: {errors}"

        return JiraApiError(message, status_code=response.status_code, details=details)


def text_to_adf(text: str) -> dict[str, Any]:
    paragraphs = []
    for paragraph in text.splitlines() or [""]:
        if paragraph:
            paragraphs.append(
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": paragraph}],
                }
            )
        else:
            paragraphs.append({"type": "paragraph"})

    return {
        "type": "doc",
        "version": 1,
        "content": paragraphs,
    }


def create_issue_payload(
    *,
    project_key: str,
    issue_type_name: str,
    summary: str,
    description: str | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "issuetype": {"name": issue_type_name},
        "summary": summary,
    }
    if description:
        fields["description"] = text_to_adf(description)
    return {"fields": fields}


def select_transition(
    data: dict[str, Any],
    transition_name_or_id: str,
) -> dict[str, Any] | None:
    transitions_data = data.get("transitions") or []
    if not isinstance(transitions_data, list):
        return None

    needle = transition_name_or_id.casefold()
    for transition in transitions_data:
        if not isinstance(transition, dict):
            continue
        transition_id = str(transition.get("id") or "")
        name = str(transition.get("name") or "")
        if transition_id == transition_name_or_id or name.casefold() == needle:
            return transition
    return None
