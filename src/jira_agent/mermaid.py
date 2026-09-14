from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jira_agent.client import (
    JiraApiError,
    png_dimensions,
    replace_diagram_with_media,
    require_concurrency_token,
)


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


def embed_mermaid_diagrams(
    client: Any,
    *,
    issue_key: str,
    description: dict[str, Any],
    attachments: list[dict[str, Any]] | None = None,
    output_dir: str | Path,
    filenames: list[str] | None = None,
    alt_texts: list[str] | None = None,
    runtime: MermaidRuntime | None = None,
    guard_concurrent_edit: bool = True,
) -> dict[str, Any]:
    """Render, upload, and embed every top-level Mermaid block. Guarded by default.

    All diagrams are rendered before the first Jira write. Uploads are rolled back if a later
    upload, the guard-token read, or the description write fails.

    `description` is read only to find the Mermaid source blocks to render and to size the
    attachment list; it is NOT what gets written. Uploading each rendered PNG is itself a write
    that moves this issue's own `updated` on the Jira side, so a guard token captured before or
    during the uploads (and the `description` argument itself, however it was obtained) would be
    stale by construction and would refuse the write after every attachment had already been
    uploaded, orphaning them.

    Instead, after every upload this function re-reads the issue's current `description` and
    `updated` together in one request, rebuilds the final ADF from THAT read, and guards the
    write with THAT same read's token. The guard therefore covers only the span from this
    post-upload read to the write; a concurrent edit landing before that read is incorporated
    into the write rather than lost, because the final ADF is built from the fresh document, not
    from the possibly-stale `description` argument. This is still check-then-act, not atomic:
    Jira Cloud's issue PUT has no conditional-write mechanism, and a change landing between the
    post-upload read and the write is still lost. Pass `guard_concurrent_edit=False` for the old
    unguarded behaviour.
    """
    sources = mermaid_sources_from_adf(description)
    if not sources:
        return {"ok": True, "issue": issue_key, "diagram_count": 0, "attachments": []}

    names = filenames or [
        f"{issue_key.lower()}-mermaid-{index}.png" for index in range(1, len(sources) + 1)
    ]
    alts = alt_texts or [f"Mermaid diagram {index}" for index in range(1, len(sources) + 1)]
    if len(names) != len(sources) or len(alts) != len(sources):
        raise ValueError("filenames and alt_texts must match the Mermaid block count.")
    if len(set(names)) != len(names):
        raise ValueError("Diagram filenames must be unique.")
    for name in names:
        path = Path(name)
        if path.name != name or path.suffix.lower() != ".png":
            raise ValueError(f"Diagram filename must be a plain PNG filename: {name}")

    selected_runtime = runtime or discover_mermaid_runtime()
    destination = Path(output_dir)
    paths = [destination / name for name in names]
    for source, path in zip(sources, paths, strict=True):
        render_mermaid_png(source, path, runtime=selected_runtime)

    previous_ids = {
        str(attachment.get("id"))
        for attachment in (attachments or [])
        if attachment.get("filename") in names
    }
    uploaded_ids: list[str] = []
    uploaded_attachments: list[dict[str, Any]] = []
    media: list[tuple[str, str, int, int]] = []
    try:
        for path, alt in zip(paths, alts, strict=True):
            uploaded = client.add_attachment(issue_key, path)
            if not uploaded:
                raise JiraApiError(f"Jira returned no attachment metadata for {path.name}.")
            attachment = uploaded[0]
            attachment_id = str(attachment.get("id"))
            uploaded_ids.append(attachment_id)
            uploaded_attachments.append(attachment)
            media_id = client.attachment_media_id(attachment_id, issue_key=issue_key)
            width, height = png_dimensions(path)
            media.append((media_id, alt, width, height))

        # Every upload above already moved this issue's `updated` on the Jira side, so content
        # and guard token are read together here, after the last mutation this function makes,
        # rather than built from the (now possibly stale) `description` argument.
        fresh_issue = client.get_issue(issue_key, fields=["description", "updated"])
        fresh_fields = fresh_issue.get("fields") or {}
        updated_description = fresh_fields.get("description") or {
            "type": "doc",
            "version": 1,
            "content": [],
        }
        for media_id, alt, width, height in media:
            updated_description = replace_diagram_with_media(
                updated_description,
                media_id=media_id,
                alt=alt,
                width=width,
                height=height,
            )
        expect_updated = require_concurrency_token(
            issue_key, fresh_fields.get("updated"), guard_enabled=guard_concurrent_edit
        )

        result = client.edit_issue_adf_description(
            issue_key, updated_description, expect_updated=expect_updated
        )
    except (JiraApiError, OSError, ValueError):
        for attachment_id in uploaded_ids:
            try:
                client.delete_attachment(attachment_id, issue_key=issue_key)
            except JiraApiError:
                pass
        raise

    deleted_ids: list[str] = []
    for attachment_id in sorted(previous_ids - set(uploaded_ids)):
        client.delete_attachment(attachment_id, issue_key=issue_key)
        deleted_ids.append(attachment_id)

    return {
        "ok": True,
        "issue": issue_key,
        "diagram_count": len(sources),
        "attachments": uploaded_attachments,
        "attachment_ids": uploaded_ids,
        "deleted_attachment_ids": deleted_ids,
        "files": [str(path) for path in paths],
        "result": result,
    }


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
