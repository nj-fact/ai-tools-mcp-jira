from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jira_agent.client import (
    adf_heading_text,
    assert_adf_slice_replacement,
    find_adf_section_bounds,
)


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
    """Replace the named section, or insert it before before_heading, or append it.

    The section extent comes from the shared ADF helper, so an ambiguous title and a heading
    without a usable level are refused here for the same reason a section patch refuses them,
    and the result passes the same slice-replacement check. Insertion is the degenerate slice
    start == end, so both paths are covered by it.
    """
    source = adf.get("content")
    before = list(source) if isinstance(source, list) else []
    content = list(before)
    bounds = find_adf_section_bounds(content, title)
    if bounds is not None:
        start, end = bounds
    else:
        insert_at = _heading_index(content, before_heading) if before_heading else None
        start = end = len(content) if insert_at is None else insert_at
    content[start:end] = blocks
    assert_adf_slice_replacement(
        before,
        content,
        start=start,
        end=end,
        new_nodes=blocks,
        document=adf if isinstance(source, list) else None,
    )
    return {**adf, "type": "doc", "version": 1, "content": content}


def _heading_index(content: list[dict[str, Any]], title: str | None) -> int | None:
    """Return the first heading with this text. An insertion anchor, not a section."""
    if not title:
        return None
    wanted = title.strip().casefold()
    for index, node in enumerate(content):
        if node.get("type") == "heading" and adf_heading_text(node).strip().casefold() == wanted:
            return index
    return None


def _attachment_sort_key(attachment: dict[str, Any]) -> tuple[int, str]:
    identifier = str(attachment.get("id", ""))
    return (int(identifier) if identifier.isdigit() else -1, identifier)


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
