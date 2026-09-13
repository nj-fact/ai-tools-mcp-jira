from __future__ import annotations

from jira_agent.workflows import (
    drift_reconcile_comment_body,
    factory_contract_comment_body,
    markdown_section,
    normalize_drift_label,
    read_text_source,
    resolve_directed_issue_link,
    update_drift_labels,
)


def _link_issue(*, key: str, link: dict[str, object]) -> dict[str, object]:
    return {"key": key, "fields": {"issuelinks": [link]}}


def test_resolve_directed_issue_link_requires_one_fresh_matching_view_per_endpoint() -> None:
    outward = _link_issue(
        key="FC-354",
        link={
            "id": "10001",
            "type": {"name": "Relates"},
            "inwardIssue": {"key": "FC-356"},
        },
    )
    inward = _link_issue(
        key="FC-356",
        link={
            "id": "10001",
            "type": {"name": "Relates"},
            "outwardIssue": {"key": "FC-354"},
        },
    )

    assert resolve_directed_issue_link(
        outward_issue=outward,
        inward_issue=inward,
        outward_key="FC-354",
        inward_key="FC-356",
        link_type="Relates",
    ) == {
        "id": "10001",
        "type": "Relates",
        "outward_key": "FC-354",
        "inward_key": "FC-356",
    }


def test_resolve_directed_issue_link_refuses_ambiguity_and_mismatched_id() -> None:
    outward = _link_issue(
        key="FC-354",
        link={
            "id": "10001",
            "type": {"name": "Relates"},
            "inwardIssue": {"key": "FC-356"},
        },
    )
    outward["fields"]["issuelinks"].append(  # type: ignore[index]
        {
            "id": "10002",
            "type": {"name": "Relates"},
            "inwardIssue": {"key": "FC-356"},
        }
    )
    inward = _link_issue(
        key="FC-356",
        link={
            "id": "10001",
            "type": {"name": "Relates"},
            "outwardIssue": {"key": "FC-354"},
        },
    )

    for link_id, message in [
        (None, "found 2 from FC-354 and 1 from FC-356."),
        ("10002", "found 1 from FC-354 and 0 from FC-356."),
    ]:
        try:
            resolve_directed_issue_link(
                outward_issue=outward,
                inward_issue=inward,
                outward_key="FC-354",
                inward_key="FC-356",
                link_type="Relates",
                link_id=link_id,
            )
        except ValueError as exc:
            assert str(exc).endswith(message)
        else:
            raise AssertionError("Expected ValueError")


def test_read_text_source_can_select_one_markdown_section(tmp_path) -> None:
    source = tmp_path / "issue.md"
    source.write_text(
        "# Wrapper\n\n## Exact description\n\n### Outcome\nReady.\n\n## Notes\nLocal only.\n",
        encoding="utf-8",
    )

    assert read_text_source(file_path=source, section="Exact description") == (
        "### Outcome\nReady."
    )


def test_markdown_section_rejects_missing_or_ambiguous_headings() -> None:
    for source, message in [
        ("## Other\nText", "Markdown section not found: Exact"),
        ("## Exact\nOne\n## Exact\nTwo", "Markdown section is ambiguous: Exact"),
    ]:
        try:
            markdown_section(source, "Exact")
        except ValueError as exc:
            assert str(exc) == message
        else:
            raise AssertionError("Expected ValueError")


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
