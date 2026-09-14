from __future__ import annotations

from jira_agent.workflows import (
    drift_label_operations,
    drift_reconcile_comment_body,
    factory_contract_comment_body,
    markdown_section,
    normalize_drift_label,
    read_text_source,
    resolve_directed_issue_link,
    resolve_png_attachment,
)


def test_resolve_png_attachment_requires_one_exact_current_png() -> None:
    issue = {
        "fields": {
            "attachment": [
                {
                    "id": "170501",
                    "filename": "fc349-disbursement-flow.png",
                    "mimeType": "image/png",
                },
                {
                    "id": "170502",
                    "filename": "fc349-disbursement-flow.png",
                    "mimeType": "image/png",
                },
            ]
        }
    }

    try:
        resolve_png_attachment(issue, filename="fc349-disbursement-flow.png")
    except ValueError as exc:
        assert str(exc).endswith("found 2.")
    else:
        raise AssertionError("Expected ValueError")

    assert resolve_png_attachment(
        issue,
        filename="fc349-disbursement-flow.png",
        attachment_id="170502",
    )["id"] == "170502"


def test_resolve_png_attachment_rejects_non_png_mime_and_non_plain_filename() -> None:
    issue = {
        "fields": {
            "attachment": [
                {"id": "170501", "filename": "flow.png", "mimeType": "image/jpeg"},
            ]
        }
    }

    for filename, message in [
        ("flow.png", "is not a PNG"),
        ("nested/flow.png", "must be a plain PNG filename"),
    ]:
        try:
            resolve_png_attachment(issue, filename=filename)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def _link_issue(*, key: str, link: dict[str, object]) -> dict[str, object]:
    return {"key": key, "fields": {"issuelinks": [link]}}


def test_resolve_directed_issue_link_matches_jira_endpoint_representation() -> None:
    outward = _link_issue(
        key="FC-354",
        link={
            "id": "10001",
            "type": {"name": "Relates"},
            "outwardIssue": {"key": "FC-356"},
        },
    )
    inward = _link_issue(
        key="FC-356",
        link={
            "id": "10001",
            "type": {"name": "Relates"},
            "inwardIssue": {"key": "FC-354"},
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
            "outwardIssue": {"key": "FC-356"},
        },
    )
    outward["fields"]["issuelinks"].append(  # type: ignore[index]
        {
            "id": "10002",
            "type": {"name": "Relates"},
            "outwardIssue": {"key": "FC-356"},
        }
    )
    inward = _link_issue(
        key="FC-356",
        link={
            "id": "10001",
            "type": {"name": "Relates"},
            "inwardIssue": {"key": "FC-354"},
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


def test_drift_label_operations_add_and_remove_need_no_existing_labels() -> None:
    """The ordinary path: add/remove alone never look at `existing_labels` (it is not even
    passed here), because each named label becomes its own atomic operation regardless of what
    else the array currently holds."""
    assert drift_label_operations(add=["New"], remove=["Old"]) == (["drift-new"], ["drift-old"])


def test_drift_label_operations_clear_removes_only_existing_drift_labels() -> None:
    """NEW contract for the old test_update_drift_labels_can_add_remove_and_clear case: `clear`
    turns every drift-* label found in `existing_labels` into an explicit remove operation
    instead of folding survivors into a whole replacement array. `jira-agent` (not drift-*) never
    appears in either list -- it was never a candidate for removal, and this function does not
    report array survivors at all, only the operations to send."""
    assert drift_label_operations(
        add=["new"],
        remove=["keep"],
        existing_labels=["jira-agent", "drift-old", "drift-keep"],
        clear=True,
    ) == (["drift-new"], ["drift-keep", "drift-old"])


def test_drift_label_operations_add_wins_over_remove_for_the_same_label() -> None:
    """Preserves the previous whole-array implementation's precedence (remove processed before
    add against one Python set): a label named in both lists is never sent as a remove
    operation, only as an add."""
    assert drift_label_operations(add=["open"], remove=["open"]) == (["drift-open"], [])


def test_drift_label_operations_clear_alone_needs_no_add_or_remove() -> None:
    assert drift_label_operations(existing_labels=["drift-a", "drift-b", "keep"], clear=True) == (
        [],
        ["drift-a", "drift-b"],
    )
