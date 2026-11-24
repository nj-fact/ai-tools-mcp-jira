from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from jira_agent import cli, mcp_server
from jira_agent.client import (
    MAX_ATTACHMENT_BYTES_CEILING,
    JiraApiError,
    JiraAttachmentAmbiguousError,
    JiraAttachmentNotFoundError,
    JiraAttachmentNotTextError,
    JiraAttachmentTooLargeError,
    JiraClient,
    decode_attachment_text,
    resolve_issue_attachment,
)
from jira_agent.config import JiraConfig

runner = CliRunner()


def _config(tmp_path: Path) -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "secret-token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


# --- Client download_attachment tests --------------------------------------------------------


def test_download_attachment_strips_auth_on_cross_origin_redirect(tmp_path: Path) -> None:
    seen: list[tuple[str, str | None]] = []
    payload = b"revision: 1\nrisk: high\n"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization")))
        if request.url.host == "example.atlassian.net":
            assert request.url.path == "/rest/api/3/attachment/content/42"
            assert request.headers.get("authorization") is not None
            return httpx.Response(
                303,
                headers={"Location": "https://api.media.atlassian.com/file/abc/binary?sig=xyz"},
                request=request,
            )
        assert request.url.host == "api.media.atlassian.com"
        assert request.headers.get("authorization") is None, (
            "must never leak Jira auth cross-origin"
        )
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": "text/yaml; charset=utf-8"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with JiraClient(_config(tmp_path), transport=transport) as client:
        result = client.download_attachment("42", issue_key="FC-1280")

    assert len(seen) == 2
    assert seen[0][1] is not None
    assert seen[1][1] is None
    assert result["bytes"] == payload
    assert result["size_bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["content_type"] == "text/yaml; charset=utf-8"


def test_download_attachment_keeps_auth_on_same_origin_redirect(tmp_path: Path) -> None:
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("authorization"))
        if request.url.path == "/rest/api/3/attachment/content/42":
            return httpx.Response(
                302,
                headers={"Location": "/secure/attachment/42/agent-spec.yaml"},
                request=request,
            )
        return httpx.Response(200, content=b"ok: true\n", request=request)

    with JiraClient(_config(tmp_path), transport=httpx.MockTransport(handler)) as client:
        result = client.download_attachment("42", issue_key="FC-1280")

    assert result["bytes"] == b"ok: true\n"
    assert len(seen_auth) == 2
    assert all(header is not None for header in seen_auth)


def test_download_attachment_refuses_non_http_redirect_scheme(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "file:///etc/passwd"}, request=request)

    with JiraClient(_config(tmp_path), transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JiraApiError, match="unsupported scheme"):
            client.download_attachment("42", issue_key="FC-1280")


def test_download_attachment_enforces_streaming_and_header_size_cap(tmp_path: Path) -> None:
    def header_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 20,
            headers={"Content-Length": "20"},
            request=request,
        )

    with JiraClient(_config(tmp_path), transport=httpx.MockTransport(header_handler)) as client:
        with pytest.raises(JiraAttachmentTooLargeError) as exc_info:
            client.download_attachment("42", issue_key="FC-1280", max_bytes=10)
        assert exc_info.value.observed_bytes == 20
        assert exc_info.value.max_bytes == 10

    class ChunkedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"123456"
            yield b"789012"

    def stream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedStream(), request=request)

    with JiraClient(_config(tmp_path), transport=httpx.MockTransport(stream_handler)) as client:
        with pytest.raises(JiraAttachmentTooLargeError) as exc_info:
            client.download_attachment("42", issue_key="FC-1280", max_bytes=10)
        assert exc_info.value.max_bytes == 10


@pytest.mark.parametrize("bad_limit", [0, -5, MAX_ATTACHMENT_BYTES_CEILING + 1, True])
def test_download_attachment_rejects_invalid_max_bytes(tmp_path: Path, bad_limit: Any) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"ok", request=req))
    with JiraClient(_config(tmp_path), transport=transport) as client:
        with pytest.raises(JiraApiError, match="max_bytes"):
            client.download_attachment("42", issue_key="FC-1280", max_bytes=bad_limit)


# --- Shared resolver & decoder tests ---------------------------------------------------------


def test_resolve_issue_attachment_handles_duplicate_filenames_and_pinning() -> None:
    attachments = [
        {"id": "100", "filename": "agent-spec.yaml", "size": 50, "created": "2026-11-23T10:00:00Z"},
        {"id": "205", "filename": "agent-spec.yaml", "size": 60, "created": "2026-11-24T12:00:00Z"},
    ]

    with pytest.raises(JiraAttachmentAmbiguousError) as exc_info:
        resolve_issue_attachment("FC-1280", attachments, filename="agent-spec.yaml")
    assert [m["id"] for m in exc_info.value.matches] == ["100", "205"]

    newest = resolve_issue_attachment(
        "FC-1280", attachments, filename="agent-spec.yaml", select_newest=True
    )
    assert newest["id"] == "205"

    pinned = resolve_issue_attachment(
        "FC-1280", attachments, filename="agent-spec.yaml", attachment_id="100"
    )
    assert pinned["id"] == "100"


def test_resolve_issue_attachment_rejects_missing_or_path_traversal() -> None:
    with pytest.raises(JiraApiError, match="Either attachment_id or filename"):
        resolve_issue_attachment("FC-1280", [])

    with pytest.raises(JiraApiError, match="plain attachment filename"):
        resolve_issue_attachment("FC-1280", [], filename="../agent-spec.yaml")

    with pytest.raises(JiraAttachmentNotFoundError):
        resolve_issue_attachment(
            "FC-1280", [{"id": "1", "filename": "other.md"}], filename="a.yaml"
        )


def test_decode_attachment_text_refuses_binary_and_invalid_utf8() -> None:
    with pytest.raises(JiraAttachmentNotTextError, match="NUL bytes"):
        decode_attachment_text(b"\x89PNG\r\n\x1a\n\x00\x00", mime_type="image/png")

    with pytest.raises(JiraAttachmentNotTextError, match="invalid UTF-8"):
        decode_attachment_text(b"\xff\xfe\xfd", mime_type="application/octet-stream")


# --- MCP tool tests --------------------------------------------------------------------------


class _FakeAttachmentClient:
    def __init__(self, attachments: list[dict[str, Any]], content_by_id: dict[str, bytes]) -> None:
        self.attachments = attachments
        self.content_by_id = content_by_id
        self.download_calls: list[str] = []

    def __enter__(self) -> _FakeAttachmentClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def get_issue(self, issue_key: str, *, fields: list[str] | None = None) -> dict[str, Any]:
        assert fields == ["attachment"]
        return {"key": issue_key, "fields": {"attachment": self.attachments}}

    def download_attachment(
        self, attachment_id: str, *, issue_key: str | None = None, max_bytes: int = 1_048_576
    ) -> dict[str, Any]:
        self.download_calls.append(attachment_id)
        raw = self.content_by_id[attachment_id]
        if len(raw) > max_bytes:
            raise JiraAttachmentTooLargeError(
                max_bytes=max_bytes, observed_bytes=len(raw), attachment_id=attachment_id
            )
        return {
            "bytes": raw,
            "size_bytes": len(raw),
            "content_type": "text/yaml",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }


def test_mcp_jira_read_attachment_reads_yaml_and_verifies_ownership(monkeypatch) -> None:
    yaml_bytes = b"revision: 2\nrisk: high\nnotes: '[red] literal'\n"
    fake = _FakeAttachmentClient(
        attachments=[
            {
                "id": "9001",
                "filename": "agent-spec.yaml",
                "mimeType": "text/yaml",
                "size": len(yaml_bytes),
                "created": "2026-11-24T18:00:00.000+0100",
            }
        ],
        content_by_id={"9001": yaml_bytes},
    )
    monkeypatch.setattr(mcp_server, "_client", lambda: fake)

    ok = mcp_server.jira_read_attachment(key="FC-1280", filename="agent-spec.yaml")
    assert ok["ok"] is True
    assert ok["issue"] == "FC-1280"
    assert ok["attachment"]["id"] == "9001"
    assert ok["content"] == yaml_bytes.decode("utf-8")
    assert ok["size_bytes"] == len(yaml_bytes)
    assert ok["sha256"] == hashlib.sha256(yaml_bytes).hexdigest()
    assert fake.download_calls == ["9001"]

    # Ownership check: an attachment_id not present on FC-1280 must never be downloaded
    denied = mcp_server.jira_read_attachment(key="FC-1280", attachment_id="7777")
    assert denied["ok"] is False
    assert denied["error_type"] == "attachment_not_found"
    assert fake.download_calls == ["9001"]


def test_mcp_jira_read_attachment_refuses_metadata_oversize_before_download(monkeypatch) -> None:
    fake = _FakeAttachmentClient(
        attachments=[{"id": "9002", "filename": "huge.yaml", "size": 5000}],
        content_by_id={"9002": b"x" * 5000},
    )
    monkeypatch.setattr(mcp_server, "_client", lambda: fake)

    result = mcp_server.jira_read_attachment(key="FC-1280", filename="huge.yaml", max_bytes=100)
    assert result["ok"] is False
    assert result["error_type"] == "attachment_too_large"
    assert fake.download_calls == []


def test_mcp_jira_read_attachment_refuses_binary(monkeypatch) -> None:
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    fake = _FakeAttachmentClient(
        attachments=[
            {"id": "9003", "filename": "diag.png", "mimeType": "image/png", "size": len(png)}
        ],
        content_by_id={"9003": png},
    )
    monkeypatch.setattr(mcp_server, "_client", lambda: fake)

    result = mcp_server.jira_read_attachment(key="FC-1280", filename="diag.png")
    assert result["ok"] is False
    assert result["error_type"] == "attachment_not_text"


# --- CLI read-attachment tests ---------------------------------------------------------------


def test_cli_read_attachment_plain_json_and_binary_output(tmp_path: Path, monkeypatch) -> None:
    yaml_bytes = b"revision: 1\nlabel: '[red] keep literal'\n"
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    fake = _FakeAttachmentClient(
        attachments=[
            {
                "id": "501",
                "filename": "agent-spec.yaml",
                "mimeType": "text/yaml",
                "size": len(yaml_bytes),
            },
            {
                "id": "502",
                "filename": "diagram.png",
                "mimeType": "image/png",
                "size": len(png_bytes),
            },
        ],
        content_by_id={"501": yaml_bytes, "502": png_bytes},
    )
    monkeypatch.setattr(cli, "_config", lambda: _config(tmp_path))
    monkeypatch.setattr(cli, "JiraClient", lambda _cfg: fake)

    # 1. Plain stdout: exact literal UTF-8 bytes without Rich markup interpretation
    r_plain = runner.invoke(
        cli.app, ["read-attachment", "FC-1280", "--filename", "agent-spec.yaml"]
    )
    assert r_plain.exit_code == 0
    assert r_plain.stdout == yaml_bytes.decode("utf-8")

    # 2. JSON stdout: matches MCP shape
    r_json = runner.invoke(
        cli.app, ["read-attachment", "FC-1280", "--filename", "agent-spec.yaml", "--json"]
    )
    assert r_json.exit_code == 0
    doc = json.loads(r_json.stdout)
    assert doc["ok"] is True
    assert doc["issue"] == "FC-1280"
    assert doc["content"] == yaml_bytes.decode("utf-8")
    assert doc["sha256"] == hashlib.sha256(yaml_bytes).hexdigest()

    # 3. Binary download via --output
    out_file = tmp_path / "downloaded" / "diagram.png"
    r_bin = runner.invoke(
        cli.app,
        [
            "read-attachment",
            "FC-1280",
            "--filename",
            "diagram.png",
            "--output",
            str(out_file),
            "--json",
        ],
    )
    assert r_bin.exit_code == 0
    assert out_file.read_bytes() == png_bytes
    bin_doc = json.loads(r_bin.stdout)
    assert bin_doc["ok"] is True
    assert bin_doc["output_path"] == str(out_file)

    # 4. Missing attachment emits structured JSON error on --json and exits 2
    r_missing = runner.invoke(
        cli.app, ["read-attachment", "FC-1280", "--filename", "nope.yaml", "--json"]
    )
    assert r_missing.exit_code == 2
    err_doc = json.loads(r_missing.stdout)
    assert err_doc["ok"] is False
    assert err_doc["error_type"] == "attachment_not_found"
