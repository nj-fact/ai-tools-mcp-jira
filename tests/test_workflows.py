from __future__ import annotations

from jira_agent.workflows import (
    drift_reconcile_comment_body,
    factory_contract_comment_body,
    normalize_drift_label,
    update_drift_labels,
)


def test_factory_contract_comment_body_includes_source() -> None:
    assert factory_contract_comment_body("contract", source_name="factory.md") == (
        "Factory Contract\nSource: factory.md\n\ncontract"
    )


def test_drift_reconcile_comment_body_formats_optional_sections() -> None:
    assert drift_reconcile_comment_body(
        summary="Description differs from contract.",
        decision="Update Jira.",
        details="Field A changed.",
    ) == (
        "Drift Reconcile\n\n"
        "Summary: Description differs from contract.\n"
        "Decision: Update Jira.\n\n"
        "Details:\n"
        "Field A changed."
    )


def test_normalize_drift_label_adds_prefix_and_normalizes_spacing() -> None:
    assert normalize_drift_label("Needs Review") == "drift-needs-review"
    assert normalize_drift_label("drift-open") == "drift-open"


def test_update_drift_labels_can_add_remove_and_clear() -> None:
    assert update_drift_labels(
        ["jira-agent", "drift-old", "drift-keep"],
        add=["new"],
        remove=["keep"],
        clear=True,
    ) == ["drift-new", "jira-agent"]
