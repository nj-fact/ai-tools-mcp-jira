"""Mechanical cross-surface parity checks.

The CLI (`cli.py`) and the MCP server (`mcp_server.py`) are two faces of one client
(`client.py`), read by the same kind of caller (an AI agent) that has no way to know which face
answered it. This review (R1-R4, September 2026) found the same underlying fact spelled or
shaped two different ways across the two surfaces four separate times: a default field list that
silently dropped `parent` on one side, a paging-completeness flag read as an unconditional
default on the client, an `allowed_issue_types`/`allowed_values` naming split, and a `curated`
flag that only one of six search-backed CLI commands ever set. The sweep for the same class
found two more: `jira_list_boards`/`jira_list_board_sprints` omitting the `returned`/`is_last`
curation their CLI equivalents (`boards`/`sprints`) already had, and `jira_update_comment`
spelling the CLI's `proposed_body` as `new_body`.

A reading-only review will not catch the next one of these; a test that fails the moment a
literal reappears, or the moment the two surfaces' default field lists diverge again, will. That
is what this module is for — it is not a review artifact, it is regression coverage for the
finding class itself.
"""

from __future__ import annotations

import inspect

from jira_agent import cli, mcp_server
from jira_agent import client as client_module


def test_default_fields_are_the_single_shared_list() -> None:
    """R2: `cli.DEFAULT_FIELDS` and `mcp_server.DEFAULT_FIELDS` must be the exact same object,
    both imported from `client.DEFAULT_FIELDS`, rather than two separately maintained copies.
    This is the actual R2 deliverable: something must fail the instant either surface goes back
    to defining its own list, with or without `parent`.
    """
    assert cli.DEFAULT_FIELDS is client_module.DEFAULT_FIELDS
    assert mcp_server.DEFAULT_FIELDS is client_module.DEFAULT_FIELDS
    assert cli.DEFAULT_FIELDS == mcp_server.DEFAULT_FIELDS
    assert "parent" in cli.DEFAULT_FIELDS, "the whole point of R2 was that both surfaces see it"


def test_issue_type_allowed_values_helper_is_shared_not_duplicated() -> None:
    """R3's fix reads cleanly only if both surfaces call the *same* extraction function; two
    independently written functions that happen to agree today can still drift tomorrow the way
    the key name itself did.
    """
    assert cli.issue_type_allowed_values is client_module.issue_type_allowed_values
    assert mcp_server.issue_type_allowed_values is client_module.issue_type_allowed_values


def test_no_stale_key_spelling_survives_in_either_surface_source() -> None:
    """Mechanical regression for sweep finding (b): the two facts this review found spelled two
    ways across `cli.py` and `mcp_server.py` (R3's `allowed_issue_types` vs `allowed_values`, and
    `jira_update_comment`'s `new_body` vs the CLI's `proposed_body`). Grepping the *source*, not
    just today's call sites, catches a reintroduction anywhere in either file — including a copy
    left behind in a docstring or a new call site added later — not only the one place each was
    found and fixed.
    """
    cli_source = inspect.getsource(cli)
    mcp_source = inspect.getsource(mcp_server)

    # Checked as a quoted dict-key literal, not a bare word: both files legitimately name the old
    # spelling in backticks, in prose, explaining what changed and why — that history is useful
    # and is not what this guards against. A literal `"allowed_issue_types"` or `"new_body"`
    # reappearing as an actual key is the regression.
    for banned in ('"allowed_issue_types"', '"new_body"'):
        assert banned not in cli_source, f"{banned} reappeared in cli.py"
        assert banned not in mcp_source, f"{banned} reappeared in mcp_server.py"


def test_search_backed_paths_share_the_same_completeness_keys_never_a_total() -> None:
    """Both surfaces must spell "how many issues came back" and "did this response return
    everything" the same way — `returned` and `is_last` — never Jira's raw `isLast`, and never a
    fabricated `total` (`/search/jql` does not have one). `cli._run_search` and
    `mcp_server._search_result` are what every curated command and tool funnels through (R4: there
    is only one shape now, not a `curated` flag choosing between two), so reading their own
    source is reading the actual contract every caller gets, not a duplicated network fixture.
    """
    cli_source = inspect.getsource(cli._run_search)
    mcp_source = inspect.getsource(mcp_server._search_result)

    for key in ('"returned"', '"is_last"'):
        assert key in cli_source, f"{key} missing from cli._run_search"
        assert key in mcp_source, f"{key} missing from mcp_server._search_result"

    assert '"total"' not in cli_source
    assert '"total"' not in mcp_source
    # R4: no `curated` *parameter* left to pick between an honest shape and the old raw one.
    # ("curated" the adjective legitimately survives in prose describing the now-only shape.)
    assert "curated:" not in inspect.getsource(cli)
    assert "curated=" not in inspect.getsource(cli)


def test_search_all_issues_docstring_matches_its_own_contract() -> None:
    """R1: the docstring is the only place a caller reads the termination rule before trusting
    the aggregate `is_last`; if the code and the docstring disagree, the docstring wins the
    caller's trust and loses the argument. Pin the load-bearing phrases directly rather than
    relying on a human re-reading them after the next edit.
    """
    doc = client_module.JiraClient.search_all_issues.__doc__ or ""
    assert "never carries a" in doc and "`total`" in doc
    assert "`None`" in doc, "must document that a missing signal reports unknown, not complete"
    assert "nextPageToken" in doc
