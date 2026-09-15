from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from jira_agent.client import JiraApiError, text_to_adf
from jira_agent.mermaid import (
    MermaidRuntime,
    discover_mermaid_runtime,
    embed_mermaid_diagrams,
    mermaid_sources_from_adf,
    render_mermaid_png,
)


def test_mermaid_sources_from_adf_returns_code_blocks_in_order() -> None:
    adf = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "codeBlock",
                "attrs": {"language": "mermaid"},
                "content": [{"type": "text", "text": "flowchart LR\nA --> B"}],
            },
            {
                "type": "codeBlock",
                "attrs": {"language": "java"},
                "content": [{"type": "text", "text": "class A {}"}],
            },
            {
                "type": "codeBlock",
                "attrs": {"language": "mermaid"},
                "content": [{"type": "text", "text": "sequenceDiagram\nA->>B: Next"}],
            },
        ],
    }

    assert mermaid_sources_from_adf(adf) == [
        "flowchart LR\nA --> B",
        "sequenceDiagram\nA->>B: Next",
    ]


def test_discover_mermaid_runtime_uses_explicit_paths(tmp_path, monkeypatch) -> None:
    node = tmp_path / "node.exe"
    cli = tmp_path / "cli.js"
    browser = tmp_path / "chrome.exe"
    for path in (node, cli, browser):
        path.write_text("test", encoding="utf-8")
    monkeypatch.setenv("JIRA_MERMAID_NODE_PATH", str(node))
    monkeypatch.setenv("JIRA_MERMAID_CLI_PATH", str(cli))
    monkeypatch.setenv("JIRA_MERMAID_BROWSER_PATH", str(browser))

    assert discover_mermaid_runtime(tmp_path) == MermaidRuntime(
        node=node.resolve(),
        cli=cli.resolve(),
        browser=browser.resolve(),
    )


def test_render_mermaid_png_invokes_local_runtime(tmp_path, monkeypatch) -> None:
    runtime = MermaidRuntime(
        node=tmp_path / "node.exe",
        cli=tmp_path / "cli.js",
        browser=tmp_path / "chrome.exe",
    )
    output = tmp_path / "diagram.png"
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        Path(command[command.index("-o") + 1]).write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert render_mermaid_png("flowchart LR\nA --> B", output, runtime=runtime) == output
    assert seen["command"][:2] == [str(runtime.node), str(runtime.cli)]
    assert seen["kwargs"]["input"] == "flowchart LR\nA --> B"
    assert seen["kwargs"]["env"]["PUPPETEER_EXECUTABLE_PATH"] == str(runtime.browser)


def test_embed_mermaid_diagrams_renders_before_upload_and_replaces_blocks(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []

    def fake_render(_source, output_path, *, runtime):
        events.append(f"render:{Path(output_path).name}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 320, 180)
        )
        return Path(output_path)

    monkeypatch.setattr("jira_agent.mermaid.render_mermaid_png", fake_render)

    description = text_to_adf(
        "```mermaid\nflowchart LR\nA --> B\n```\n\n```mermaid\nsequenceDiagram\nA->>B: Next\n```"
    )

    class FakeClient:
        def get_issue(self, _issue_key, *, fields=None, expand=None):
            events.append("get_issue")
            return {"fields": {"description": description, "updated": "T-fresh"}}

        def add_attachment(self, _issue_key, path):
            events.append(f"upload:{Path(path).name}")
            return [{"id": str(len(events)), "filename": Path(path).name}]

        def attachment_media_id(self, attachment_id, *, issue_key):
            return f"media-{issue_key}-{attachment_id}"

        def edit_issue_adf_description(self, _issue_key, description, *, expect_updated=None):
            events.append("edit")
            self.description = description
            self.expect_updated = expect_updated
            return {"updated": True}

        def delete_attachment(self, attachment_id, *, issue_key):
            events.append(f"delete:{issue_key}:{attachment_id}")

    client = FakeClient()
    result = embed_mermaid_diagrams(
        client,
        issue_key="FC-123",
        description=description,
        attachments=[
            {"id": "old-1", "filename": "fc-123-mermaid-1.png"},
            {"id": "old-2", "filename": "fc-123-mermaid-2.png"},
        ],
        output_dir=tmp_path,
        runtime=MermaidRuntime(tmp_path / "node", tmp_path / "cli", tmp_path / "browser"),
    )

    assert events[:2] == [
        "render:fc-123-mermaid-1.png",
        "render:fc-123-mermaid-2.png",
    ]
    assert result["diagram_count"] == 2
    assert result["deleted_attachment_ids"] == ["old-1", "old-2"]
    assert [block["type"] for block in client.description["content"]] == [
        "mediaSingle",
        "mediaSingle",
    ]
    # The guard token comes from the post-upload get_issue read, not from a token captured
    # before the uploads (which would be stale by construction; see H1/G1).
    assert client.expect_updated == "T-fresh"


def test_embed_mermaid_audits_commit_before_best_effort_cleanup_failure(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []

    def fake_render(_source, output_path, *, runtime):
        Path(output_path).write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 10, 10)
        )
        return Path(output_path)

    monkeypatch.setattr("jira_agent.mermaid.render_mermaid_png", fake_render)
    description = text_to_adf("```mermaid\nflowchart LR\nA --> B\n```")

    class FakeClient:
        def get_issue(self, _issue_key, *, fields=None, expand=None):
            return {"fields": {"description": description, "updated": "T-fresh"}}

        def add_attachment(self, _issue_key, path):
            return [{"id": "9001", "filename": Path(path).name}]

        def attachment_media_id(self, attachment_id, *, issue_key):
            return "media-9001"

        def edit_issue_adf_description(self, issue_key, description, *, expect_updated=None):
            events.append("description_write")
            return {"updated": True}

        def delete_attachment(self, attachment_id, *, issue_key):
            events.append("cleanup_delete")
            raise JiraApiError("cleanup failed")

    result = embed_mermaid_diagrams(
        FakeClient(),
        issue_key="FC-123",
        description=description,
        attachments=[{"id": "8000", "filename": "fc-123-mermaid-1.png"}],
        output_dir=tmp_path,
        runtime=MermaidRuntime(tmp_path / "node", tmp_path / "cli", tmp_path / "browser"),
        on_description_committed=lambda _commit: events.append("audit"),
    )

    assert events == ["description_write", "audit", "cleanup_delete"]
    assert result["deleted_attachment_ids"] == []
    assert result["cleanup_failures"] == [
        {"attachment_id": "8000", "message": "cleanup failed"}
    ]


def test_embed_mermaid_diagrams_rolls_back_uploads_when_description_edit_fails(
    tmp_path, monkeypatch
) -> None:
    deleted: list[str] = []

    def fake_render(_source, output_path, *, runtime):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 10, 10))
        return Path(output_path)

    monkeypatch.setattr("jira_agent.mermaid.render_mermaid_png", fake_render)

    class FakeClient:
        def get_issue(self, _issue_key, *, fields=None, expand=None):
            return {
                "fields": {
                    "description": text_to_adf("```mermaid\nflowchart LR\nA --> B\n```"),
                    "updated": "T-fresh",
                }
            }

        def add_attachment(self, _issue_key, path):
            return [{"id": "new-1", "filename": Path(path).name}]

        def attachment_media_id(self, _attachment_id, *, issue_key):
            return f"media-{issue_key}"

        def edit_issue_adf_description(self, _issue_key, _description, *, expect_updated=None):
            raise JiraApiError("edit failed")

        def delete_attachment(self, attachment_id, *, issue_key):
            deleted.append(f"{issue_key}:{attachment_id}")

    with pytest.raises(JiraApiError, match="edit failed"):
        embed_mermaid_diagrams(
            FakeClient(),
            issue_key="FC-123",
            description=text_to_adf("```mermaid\nflowchart LR\nA --> B\n```"),
            output_dir=tmp_path,
            runtime=MermaidRuntime(tmp_path / "node", tmp_path / "cli", tmp_path / "browser"),
        )

    assert deleted == ["FC-123:new-1"]
