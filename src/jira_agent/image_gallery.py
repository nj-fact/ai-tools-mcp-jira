from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def newest_attachments_by_filename(
    attachments: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the newest Jira attachment for each filename."""
    selected: dict[str, dict[str, Any]] = {}
    for attachment in attachments:
        filename = str(attachment.get("filename", ""))
        if not filename:
            continue
        current = selected.get(filename)
        if current is None or _attachment_sort_key(attachment) > _attachment_sort_key(current):
            selected[filename] = attachment
    return selected


def load_image_gallery_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not str(data.get("title", "")).strip():
        raise ValueError("Image gallery manifest requires a title.")
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("Image gallery manifest requires at least one section.")

    filenames: list[str] = []
    for section in sections:
        if not isinstance(section, dict) or not str(section.get("heading", "")).strip():
            raise ValueError("Every image gallery section requires a heading.")
        rows = section.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Every image gallery section requires at least one row.")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Every image gallery row must be an object.")
            for field in ("file", "title", "description", "alt"):
                if not str(row.get(field, "")).strip():
                    raise ValueError(f"Every image gallery row requires {field}.")
            filename = Path(str(row["file"])).name
            if filename != row["file"] or Path(filename).suffix.lower() != ".png":
                raise ValueError(f"Gallery file must be a plain PNG filename: {row['file']}")
            filenames.append(filename)

    if len(filenames) != len(set(filenames)):
        raise ValueError("Image gallery filenames must be unique.")
    return data


def image_gallery_blocks(
    manifest: dict[str, Any],
    *,
    attachment_urls: dict[str, str],
    documentation_url: str | None = None,
) -> list[dict[str, Any]]:
    title = str(manifest["title"])
    blocks: list[dict[str, Any]] = [_heading(title, level=3)]
    intro = str(manifest.get("intro", "")).strip()
    if intro:
        blocks.append(_paragraph(intro))
    if documentation_url:
        blocks.append(
            _paragraph(
                "Open the complete mockup documentation",
                href=documentation_url,
            )
        )

    for section in manifest["sections"]:
        blocks.append(_heading(str(section["heading"]), level=4))
        rows = [_table_row(["Step or situation", "Intended experience", "Screenshot"], True)]
        for row in section["rows"]:
            filename = str(row["file"])
            href = attachment_urls.get(filename)
            if not href:
                raise ValueError(f"No Jira attachment URL found for {filename}.")
            rows.append(
                _table_row(
                    [str(row["title"]), str(row["description"]), ("Open screenshot", href)],
                    False,
                )
            )
        blocks.append(
            {
                "type": "table",
                "attrs": {"isNumberColumnEnabled": False, "layout": "wide"},
                "content": rows,
            }
        )
    return blocks


def upsert_top_level_section(
    adf: dict[str, Any],
    *,
    title: str,
    blocks: list[dict[str, Any]],
    before_heading: str | None = None,
) -> dict[str, Any]:
    content = list(adf.get("content") or [])
    start = _heading_index(content, title)
    if start is not None:
        level = int((content[start].get("attrs") or {}).get("level", 3))
        end = start + 1
        while end < len(content):
            node = content[end]
            if (
                node.get("type") == "heading"
                and int((node.get("attrs") or {}).get("level", 6)) <= level
            ):
                break
            end += 1
        content[start:end] = blocks
    else:
        insert_at = _heading_index(content, before_heading) if before_heading else None
        if insert_at is None:
            insert_at = len(content)
        content[insert_at:insert_at] = blocks
    return {**adf, "type": "doc", "version": 1, "content": content}


def _heading_index(content: list[dict[str, Any]], title: str | None) -> int | None:
    if not title:
        return None
    for index, node in enumerate(content):
        if node.get("type") == "heading" and _node_text(node).strip() == title:
            return index
    return None


def _attachment_sort_key(attachment: dict[str, Any]) -> tuple[int, str]:
    identifier = str(attachment.get("id", ""))
    return (int(identifier) if identifier.isdigit() else -1, identifier)


def _node_text(node: dict[str, Any]) -> str:
    return "".join(
        str(child.get("text", ""))
        for child in node.get("content") or []
        if child.get("type") == "text"
    )


def _heading(text: str, *, level: int) -> dict[str, Any]:
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": text}],
    }


def _paragraph(text: str, *, href: str | None = None) -> dict[str, Any]:
    node: dict[str, Any] = {"type": "text", "text": text}
    if href:
        node["marks"] = [{"type": "link", "attrs": {"href": href}}]
    return {"type": "paragraph", "content": [node]}


def _table_row(cells: list[str | tuple[str, str]], header: bool) -> dict[str, Any]:
    cell_type = "tableHeader" if header else "tableCell"
    content: list[dict[str, Any]] = []
    for cell in cells:
        if isinstance(cell, tuple):
            text, href = cell
            paragraph = _paragraph(text, href=href)
        else:
            paragraph = _paragraph(cell)
        content.append(
            {
                "type": cell_type,
                "attrs": {},
                "content": [paragraph],
            }
        )
    return {"type": "tableRow", "content": content}
