from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MermaidRuntime:
    node: Path
    cli: Path
    browser: Path


def discover_mermaid_runtime(repo_root: str | Path | None = None) -> MermaidRuntime:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    user_profile = Path(os.environ.get("USERPROFILE", ""))
    program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))

    node = _first_executable(
        os.environ.get("JIRA_MERMAID_NODE_PATH"),
        shutil.which("node"),
        program_files / "nodejs" / "node.exe",
        user_profile
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
        / "bin"
        / "node.exe",
    )
    cli = _first_file(
        os.environ.get("JIRA_MERMAID_CLI_PATH"),
        root / "node_modules" / "@mermaid-js" / "mermaid-cli" / "src" / "cli.js",
    )
    browser = _first_executable(
        os.environ.get("JIRA_MERMAID_BROWSER_PATH"),
        os.environ.get("PUPPETEER_EXECUTABLE_PATH"),
        shutil.which("chrome"),
        shutil.which("msedge"),
        program_files / "Google" / "Chrome" / "Application" / "chrome.exe",
        program_files_x86 / "Google" / "Chrome" / "Application" / "chrome.exe",
        program_files / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        program_files_x86 / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    )
    if not node:
        raise RuntimeError(
            "Node.js was not found. Run scripts/setup_windows.ps1 or set JIRA_MERMAID_NODE_PATH."
        )
    if not cli:
        raise RuntimeError(
            "Mermaid CLI was not found. Run pnpm install in the jira-agent repository or set "
            "JIRA_MERMAID_CLI_PATH."
        )
    if not browser:
        raise RuntimeError(
            "Chrome or Edge was not found. Install one or set JIRA_MERMAID_BROWSER_PATH."
        )
    return MermaidRuntime(node=node, cli=cli, browser=browser)


def mermaid_sources_from_adf(adf: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for block in adf.get("content") or []:
        if block.get("type") != "codeBlock":
            continue
        if (block.get("attrs") or {}).get("language") != "mermaid":
            continue
        source = "".join(
            str(item.get("text") or "")
            for item in block.get("content") or []
            if item.get("type") == "text"
        ).strip()
        if source:
            sources.append(source)
    return sources


def render_mermaid_png(
    source: str,
    output_path: str | Path,
    *,
    runtime: MermaidRuntime | None = None,
    scale: int = 2,
) -> Path:
    selected = runtime or discover_mermaid_runtime()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PUPPETEER_EXECUTABLE_PATH"] = str(selected.browser)
    process = subprocess.run(
        [
            str(selected.node),
            str(selected.cli),
            "-i",
            "-",
            "-o",
            str(output),
            "-b",
            "white",
            "-s",
            str(scale),
            "-q",
        ],
        input=source,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    if process.returncode != 0:
        details = (process.stderr or process.stdout or "Unknown Mermaid error").strip()
        raise RuntimeError(f"Mermaid rendering failed: {details}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Mermaid rendering produced no PNG: {output}")
    return output


def _first_file(*candidates: object) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file():
            return path.resolve()
    return None


def _first_executable(*candidates: object) -> Path | None:
    return _first_file(*candidates)
