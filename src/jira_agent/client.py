from __future__ import annotations

import re
import struct
from collections.abc import Iterable
from pathlib import Path
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

    def search_assignable_users(
        self,
        query: str,
        *,
        issue_key: str | None = None,
        project_key: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query": query,
            "maxResults": limit,
        }
        if issue_key:
            params["issueKey"] = issue_key
        if project_key:
            params["project"] = project_key
        return self._request("GET", "/user/assignable/search", params=params)

    def edit_issue(
        self,
        issue_key: str,
        *,
        summary: str | None = None,
        description: str | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "PUT",
            f"/issue/{issue_key}",
            json=edit_issue_payload(
                summary=summary,
                description=description,
                labels=labels,
            ),
        )

    def edit_issue_adf_description(
        self,
        issue_key: str,
        description: dict[str, Any],
    ) -> dict[str, Any]:
        return self._request(
            "PUT",
            f"/issue/{issue_key}",
            json={"fields": {"description": description}},
        )

    def link_issues(
        self,
        *,
        outward_issue_key: str,
        inward_issue_key: str,
        link_type: str = "Relates",
        comment: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/issueLink",
            json=issue_link_payload(
                outward_issue_key=outward_issue_key,
                inward_issue_key=inward_issue_key,
                link_type=link_type,
                comment=comment,
            ),
        )

    def add_attachment(self, issue_key: str, file_path: str | Path) -> list[dict[str, Any]]:
        path = Path(file_path)
        with path.open("rb") as handle:
            try:
                response = self._client.request(
                    "POST",
                    f"/issue/{issue_key}/attachments",
                    headers={"X-Atlassian-Token": "no-check"},
                    files={"file": (path.name, handle)},
                )
            except httpx.HTTPError as exc:
                raise JiraApiError(
                    f"Jira request failed before receiving a response: {exc}"
                ) from exc

        if response.status_code >= 400:
            raise self._api_error(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise JiraApiError(
                "Jira returned a non-JSON response",
                status_code=response.status_code,
                details=response.text[:500],
            ) from exc
        if not isinstance(data, list):
            raise JiraApiError(
                "Jira returned an unexpected attachment payload",
                status_code=response.status_code,
                details=data,
            )
        return data

    def delete_attachment(self, attachment_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/attachment/{attachment_id}")

    def attachment_media_id(self, attachment_id: str) -> str:
        try:
            response = self._client.get(
                f"/attachment/content/{attachment_id}",
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise JiraApiError(f"Jira request failed before receiving a response: {exc}") from exc

        if response.status_code >= 400:
            raise self._api_error(response)
        location = response.headers.get("location") or ""
        match = re.search(r"/file/([^/]+)/binary", location)
        if not match:
            raise JiraApiError(
                "Jira attachment redirect did not expose a media id",
                status_code=response.status_code,
                details={"location": location},
            )
        return match.group(1)

    def get_changelog(
        self,
        issue_key: str,
        *,
        start_at: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/issue/{issue_key}/changelog",
            params={"startAt": start_at, "maxResults": limit},
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

        if not isinstance(data, dict | list):
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
    return {
        "type": "doc",
        "version": 1,
        "content": _markdown_blocks_to_adf(text),
    }


def png_dimensions(path: str | Path) -> tuple[int, int]:
    with Path(path).open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    return struct.unpack(">II", header[16:24])


def media_single_adf(
    *,
    media_id: str,
    alt: str,
    width: int,
    height: int,
    collection: str = "",
) -> dict[str, Any]:
    return {
        "type": "mediaSingle",
        "attrs": {"layout": "center"},
        "content": [
            {
                "type": "media",
                "attrs": {
                    "id": media_id,
                    "type": "file",
                    "collection": collection,
                    "alt": alt,
                    "width": width,
                    "height": height,
                },
            }
        ],
    }


def replace_diagram_with_media(
    adf: dict[str, Any],
    *,
    media_id: str,
    alt: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    replacement = media_single_adf(media_id=media_id, alt=alt, width=width, height=height)
    content = list(adf.get("content") or [])
    new_content: list[dict[str, Any]] = []
    replaced = False

    for block in content:
        if not replaced and block.get("type") == "codeBlock":
            language = (block.get("attrs") or {}).get("language")
            if language == "mermaid":
                new_content.append(
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Rendered architecture diagram:"}],
                    }
                )
                new_content.append(replacement)
                replaced = True
                continue
        if not replaced and block.get("type") == "mediaSingle":
            new_content.append(replacement)
            replaced = True
            continue
        new_content.append(block)

    if not replaced:
        new_content.extend(
            [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Rendered architecture diagram:"}],
                },
                replacement,
            ]
        )

    return {**adf, "content": new_content}


def _markdown_blocks_to_adf(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines() or [""]
    blocks: list[dict[str, Any]] = []
    index = 0
    task_index = 1

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            blocks.append({"type": "paragraph"})
            index += 1
            continue

        code_match = re.match(r"^```([A-Za-z0-9_-]+)?\s*$", stripped)
        if code_match:
            language = code_match.group(1)
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "```":
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            block: dict[str, Any] = {
                "type": "codeBlock",
                "content": [{"type": "text", "text": "\n".join(code_lines)}],
            }
            if language:
                block["attrs"] = {"language": language}
            blocks.append(block)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            blocks.append(
                {
                    "type": "heading",
                    "attrs": {"level": min(len(heading_match.group(1)), 6)},
                    "content": _inline_text_to_adf(heading_match.group(2)),
                }
            )
            index += 1
            continue

        if _is_table_start(lines, index):
            table, index = _table_to_adf(lines, index)
            blocks.append(table)
            continue

        task_match = re.match(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$", line)
        if task_match:
            items: list[dict[str, Any]] = []
            while index < len(lines):
                item_match = re.match(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$", lines[index])
                if not item_match:
                    break
                state = "DONE" if item_match.group(1).lower() == "x" else "TODO"
                items.append(
                    {
                        "type": "taskItem",
                        "attrs": {"localId": f"task-{task_index}", "state": state},
                        "content": _inline_text_to_adf(item_match.group(2)),
                    }
                )
                task_index += 1
                index += 1
            blocks.append(
                {
                    "type": "taskList",
                    "attrs": {"localId": f"task-list-{task_index - len(items)}"},
                    "content": items,
                }
            )
            continue

        if re.match(r"^\s*[-*]\s+", line):
            items: list[dict[str, Any]] = []
            while index < len(lines):
                item_match = re.match(r"^\s*[-*]\s+(.+?)\s*$", lines[index])
                if not item_match:
                    break
                items.append(
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": _inline_text_to_adf(item_match.group(1)),
                            }
                        ],
                    }
                )
                index += 1
            blocks.append({"type": "bulletList", "content": items})
            continue

        blocks.append({"type": "paragraph", "content": _inline_text_to_adf(line)})
        index += 1

    return blocks


def _inline_text_to_adf(text: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    cursor = 0
    for match in re.finditer(r"(`([^`]+)`|\*\*(.+?)\*\*)", text):
        if match.start() > cursor:
            parts.append({"type": "text", "text": text[cursor : match.start()]})
        if match.group(2) is not None:
            parts.append(
                {
                    "type": "text",
                    "text": match.group(2),
                    "marks": [{"type": "code"}],
                }
            )
        else:
            parts.append(
                {
                    "type": "text",
                    "text": match.group(3),
                    "marks": [{"type": "strong"}],
                }
            )
        cursor = match.end()
    if cursor < len(text):
        parts.append({"type": "text", "text": text[cursor:]})
    return parts or [{"type": "text", "text": ""}]


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return (
        _is_table_row(lines[index])
        and _is_table_row(lines[index + 1])
        and _is_table_separator(lines[index + 1])
    )


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_to_adf(lines: list[str], index: int) -> tuple[dict[str, Any], int]:
    header = _table_cells(lines[index])
    rows: list[dict[str, Any]] = [_table_row_to_adf(header, header=True)]
    index += 2

    while index < len(lines) and _is_table_row(lines[index]):
        rows.append(_table_row_to_adf(_table_cells(lines[index]), header=False))
        index += 1

    table = {
        "type": "table",
        "attrs": {"isNumberColumnEnabled": False, "layout": "default"},
        "content": rows,
    }
    return table, index


def _table_row_to_adf(cells: list[str], *, header: bool) -> dict[str, Any]:
    cell_type = "tableHeader" if header else "tableCell"
    return {
        "type": "tableRow",
        "content": [
            {
                "type": cell_type,
                "attrs": {},
                "content": [
                    {
                        "type": "paragraph",
                        "content": _inline_text_to_adf(cell),
                    }
                ],
            }
            for cell in cells
        ],
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


def edit_issue_payload(
    *,
    summary: str | None = None,
    description: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if summary is not None:
        fields["summary"] = summary
    if description is not None:
        fields["description"] = text_to_adf(description)
    if labels is not None:
        fields["labels"] = labels
    if not fields:
        raise ValueError("At least one editable field must be provided")
    return {"fields": fields}


def issue_link_payload(
    *,
    outward_issue_key: str,
    inward_issue_key: str,
    link_type: str = "Relates",
    comment: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": {"name": link_type},
        "outwardIssue": {"key": outward_issue_key},
        "inwardIssue": {"key": inward_issue_key},
    }
    if comment:
        payload["comment"] = {"body": text_to_adf(comment)}
    return payload


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
