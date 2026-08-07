from __future__ import annotations

import subprocess
from pathlib import Path

from jira_agent.mermaid import (
    MermaidRuntime,
    discover_mermaid_runtime,
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
