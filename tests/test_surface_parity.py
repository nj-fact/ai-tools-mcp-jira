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

import ast
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


# --- G8: a mechanical guard against the next unguarded whole-field rewrite -------------------
#
# G2 found five MCP tools that read a description (or labels/attachment set), transformed it,
# and wrote the whole field back with no `expect_updated` at all — the guard this round exists
# for, simply absent from the surface AI agents actually call. This file's own parity test
# passed clean while that gap existed: its assertions pinned three literals that happened to
# drift once, not the shape of the defect itself. A guard that reads as coverage but cannot fail
# is worse than none. This enumerates every call, mechanically, so a new one is caught by the
# suite instead of by the next adversarial review.

_GUARDED_EDIT_METHODS = frozenset({"edit_issue", "edit_issue_adf_description"})

# (module label, enclosing function name, client method name) -> why this specific call is
# deliberately unguarded. A call landing here is a decision, recorded once, not an oversight;
# see the matching docstring/comment at the call site itself for the same reasoning.
_NO_GUARD_ALLOWLIST: dict[tuple[str, str, str], str] = {
    ("cli", "clear_parent", "edit_issue"): (
        "Sets one relationship field (parent) to a single, specific, verified target state "
        "(null) rather than reading and rewriting a whole computed field; the post-write reread "
        "independently confirms that target state regardless of what a concurrent editor did to "
        "parent in between."
    ),
    ("mcp_server", "jira_clear_issue_parent", "edit_issue"): (
        "Same reasoning as cli.clear_parent: a single-field clear verified by reread, not a "
        "read-transform-whole-field rewrite."
    ),
}


class _EditCallVisitor(ast.NodeVisitor):
    """Collect every `<expr>.edit_issue(...)` / `<expr>.edit_issue_adf_description(...)` call in
    a module, tagged with the name of its innermost enclosing function and whether it passes
    `expect_updated` as a keyword argument. Deliberately does not look at the receiver expression
    (it is always some spelling of the Jira client), only the attribute name and the call's own
    keywords, since that is exactly what `_guarded_issue_put` gates on at runtime.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []
        self._stack: list[str] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in _GUARDED_EDIT_METHODS:
            guarded = any(kw.arg == "expect_updated" for kw in node.keywords)
            function_name = self._stack[-1] if self._stack else "<module>"
            self.calls.append((function_name, node.func.attr, guarded))
        self.generic_visit(node)


def _edit_calls(source: str) -> list[tuple[str, str, bool]]:
    visitor = _EditCallVisitor()
    visitor.visit(ast.parse(source))
    return visitor.calls


def test_every_edit_issue_call_is_guarded_or_on_the_named_allowlist() -> None:
    """A new call to `edit_issue` / `edit_issue_adf_description` that omits `expect_updated`
    must fail this test, not pass silently the way G2's five missing guards did. Verified by
    deliberately introducing an unguarded call during remediation and confirming this test (and
    only this test in this class) catches it before reverting.
    """
    modules = {"cli": inspect.getsource(cli), "mcp_server": inspect.getsource(mcp_server)}
    unresolved: list[str] = []
    seen_allowlist_keys: set[tuple[str, str, str]] = set()

    for module_name, source in modules.items():
        for function_name, method_name, guarded in _edit_calls(source):
            if guarded:
                continue
            key = (module_name, function_name, method_name)
            if key in _NO_GUARD_ALLOWLIST:
                seen_allowlist_keys.add(key)
                continue
            unresolved.append(f"{module_name}.{function_name} calls .{method_name}(...) unguarded")

    assert not unresolved, (
        "unguarded whole-field rewrite call(s) found; pass expect_updated or add a named, "
        "justified entry to _NO_GUARD_ALLOWLIST:\n" + "\n".join(unresolved)
    )

    stale = set(_NO_GUARD_ALLOWLIST) - seen_allowlist_keys
    assert not stale, f"allowlist entries no longer match any unguarded call: {sorted(stale)}"
