from __future__ import annotations

from pathlib import Path


def read_text_source(text: str | None = None, file_path: str | Path | None = None) -> str:
    if text and file_path:
        raise ValueError("Use either text or file_path, not both")
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    if text:
        return text
    raise ValueError("Either text or file_path is required")


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


def update_drift_labels(
    existing_labels: list[str],
    *,
    add: list[str] | None = None,
    remove: list[str] | None = None,
    clear: bool = False,
) -> list[str]:
    labels = set(existing_labels)
    if clear:
        labels = {label for label in labels if not label.startswith("drift-")}
    for label in remove or []:
        labels.discard(normalize_drift_label(label))
    for label in add or []:
        labels.add(normalize_drift_label(label))
    return sorted(labels)
