from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class IssueLinkResolutionError(ValueError):
    """The fresh Jira reads did not identify exactly the directed link requested."""


class AttachmentResolutionError(ValueError):
    """The fresh Jira attachment list did not identify exactly one requested PNG."""


def resolve_png_attachment(
    issue: dict[str, Any],
    *,
    filename: str,
    attachment_id: str | None = None,
) -> dict[str, Any]:
    """Resolve one exact, attached PNG from a fresh issue response.

    A duplicated filename is refused unless the caller pins the attachment id, and a pinned id
    must still belong to that filename. The Jira attachment MIME type is checked rather than
    trusting the filename suffix alone.
    """
    path = Path(filename)
    if path.name != filename or path.suffix.lower() != ".png":
        raise AttachmentResolutionError(
            f"Attachment filename must be a plain PNG filename: {filename}"
        )

    attachments = (issue.get("fields") or {}).get("attachment") or []
    if not isinstance(attachments, list):
        attachments = []
    matches = [
        attachment
        for attachment in attachments
        if isinstance(attachment, dict) and attachment.get("filename") == filename
    ]
    if attachment_id is not None:
        matches = [item for item in matches if str(item.get("id")) == str(attachment_id)]
    if len(matches) != 1:
        pin = f" with id {attachment_id}" if attachment_id is not None else ""
        raise AttachmentResolutionError(
            f"Expected exactly one attachment named {filename!r}{pin}; found {len(matches)}."
        )

    attachment = matches[0]
    mime_type = str(attachment.get("mimeType") or "")
    if mime_type.casefold() != "image/png":
        raise AttachmentResolutionError(
            f"Attachment {attachment.get('id')} named {filename!r} is not a PNG "
            f"(mimeType={mime_type!r})."
        )
    if attachment.get("id") is None:
        raise AttachmentResolutionError(f"Attachment {filename!r} has no Jira attachment id.")
    return attachment


def resolve_directed_issue_link(
    *,
    outward_issue: dict[str, Any],
    inward_issue: dict[str, Any],
    outward_key: str,
    inward_key: str,
    link_type: str,
    link_id: str | None = None,
) -> dict[str, str]:
    """Resolve one directed Jira issue link from fresh reads of both endpoints.

    Jira returns a link relative to the issue being read: the outward endpoint lists the other
    issue as ``outwardIssue`` and the inward endpoint lists it as ``inwardIssue``. Requiring
    both views prevents a stale or mismatched id from deleting a different link.
    """
    outward_candidates = _directed_link_candidates(
        outward_issue,
        other_key=inward_key,
        other_side="outwardIssue",
        link_type=link_type,
    )
    inward_candidates = _directed_link_candidates(
        inward_issue,
        other_key=outward_key,
        other_side="inwardIssue",
        link_type=link_type,
    )

    if link_id is not None:
        expected_id = str(link_id)
        outward_candidates = [item for item in outward_candidates if item["id"] == expected_id]
        inward_candidates = [item for item in inward_candidates if item["id"] == expected_id]

    if len(outward_candidates) != 1 or len(inward_candidates) != 1:
        requested = f" with id {link_id}" if link_id is not None else ""
        raise IssueLinkResolutionError(
            "Expected exactly one directed "
            f"{link_type!r} link from {outward_key} to {inward_key}{requested}; found "
            f"{len(outward_candidates)} from {outward_key} and {len(inward_candidates)} from "
            f"{inward_key}."
        )

    outward = outward_candidates[0]
    inward = inward_candidates[0]
    if outward["id"] != inward["id"]:
        raise IssueLinkResolutionError(
            f"Fresh reads disagree on the link id for {outward_key} -> {inward_key}: "
            f"{outward['id']} versus {inward['id']}."
        )
    return {
        "id": outward["id"],
        "type": link_type,
        "outward_key": outward_key,
        "inward_key": inward_key,
    }


def _directed_link_candidates(
    issue: dict[str, Any],
    *,
    other_key: str,
    other_side: str,
    link_type: str,
) -> list[dict[str, str]]:
    links = (issue.get("fields") or {}).get("issuelinks") or []
    if not isinstance(links, list):
        return []
    candidates = []
    for link in links:
        if not isinstance(link, dict):
            continue
        endpoint = link.get(other_side) or {}
        link_name = (link.get("type") or {}).get("name")
        link_id = link.get("id")
        if (
            str((endpoint or {}).get("key") or "") == other_key
            and link_name == link_type
            and link_id is not None
        ):
            candidates.append({"id": str(link_id)})
    return candidates


def issue_contains_link_id(issue: dict[str, Any], link_id: str) -> bool:
    """Return whether a freshly read issue still exposes this Jira link id."""
    links = (issue.get("fields") or {}).get("issuelinks") or []
    return any(isinstance(link, dict) and str(link.get("id")) == str(link_id) for link in links)


def read_text_source(
    text: str | None = None,
    file_path: str | Path | None = None,
    section: str | None = None,
) -> str:
    if text and file_path:
        raise ValueError("Use either text or file_path, not both")
    if file_path:
        source = Path(file_path).read_text(encoding="utf-8")
    elif text:
        source = text
    else:
        raise ValueError("Either text or file_path is required")
    return markdown_section(source, section) if section else source


def markdown_section(markdown: str, heading: str) -> str:
    """Return the body below one Markdown heading, up to the next peer heading."""
    wanted = re.sub(r"^#{1,6}\s+", "", heading.strip()).casefold()
    lines = markdown.splitlines()
    start: int | None = None
    level: int | None = None

    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match or match.group(2).strip().casefold() != wanted:
            continue
        if start is not None:
            raise ValueError(f"Markdown section is ambiguous: {heading}")
        start = index + 1
        level = len(match.group(1))

    if start is None or level is None:
        raise ValueError(f"Markdown section not found: {heading}")

    end = len(lines)
    for index in range(start, len(lines)):
        match = re.match(r"^(#{1,6})\s+", lines[index])
        if match and len(match.group(1)) <= level:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def factory_contract_comment_body(contract: str, *, source_name: str | None = None) -> str:
    lines = ["Factory Contract"]
    if source_name:
        lines.append(f"Source: {source_name}")
    lines.extend(["", contract])
    return "\n".join(lines)


def drift_reconcile_comment_body(
    *,
    summary: str,
    decision: str | None = None,
    details: str | None = None,
) -> str:
    lines = ["Drift Reconcile", "", f"Summary: {summary}"]
    if decision:
        lines.append(f"Decision: {decision}")
    if details:
        lines.extend(["", "Details:", details])
    return "\n".join(lines)


def normalize_drift_label(value: str) -> str:
    label = value.strip().lower().replace(" ", "-").replace("_", "-")
    if not label:
        raise ValueError("Drift label must not be empty")
    if not label.startswith("drift-"):
        label = f"drift-{label}"
    return label


def drift_label_operations(
    *,
    add: list[str] | None = None,
    remove: list[str] | None = None,
    existing_labels: list[str] | None = None,
    clear: bool = False,
) -> tuple[list[str], list[str]]:
    """Compute the (add, remove) label VALUES for an atomic Jira `labels` `update`, never a
    whole replacement array.

    `add` and `remove` alone need no read at all: each named label becomes its own `add` or
    `remove` operation, applied by Jira against whatever the array currently holds server-side,
    regardless of what else changed there since. That is `drift-labels`' ordinary path, and it
    carries no concurrency guard because there is no read-modify-write left to protect.

    `clear=True` is the one path that still needs a read: to remove "every label starting with
    drift-", the caller first has to know which labels currently do. Pass that read's `labels` as
    `existing_labels`; only the ones already present and drift-prefixed become explicit `remove`
    operations, so a drift label some other, concurrent write adds after that read was taken is
    never named here and survives, the same way `add`/`remove` alone already leaves everything
    it was not told about untouched. This still needs no concurrency guard: the read only decides
    WHICH remove operations to send, and each one, once sent, is itself safe against a concurrent
    edit the same way any other `remove` operation is.

    A label named in both `add` and `remove` ends up added, never removed, the same precedence
    the previous whole-array implementation had (it applied `remove` before `add` against one
    Python set) -- preserved here so the net *labels present* outcome for identical inputs against
    an unchanged issue does not change, only how it is sent to Jira.
    """
    add_labels = sorted({normalize_drift_label(label) for label in add or []})
    remove_labels = {normalize_drift_label(label) for label in remove or []}
    if clear:
        remove_labels |= {label for label in existing_labels or [] if label.startswith("drift-")}
    remove_labels -= set(add_labels)
    return add_labels, sorted(remove_labels)
