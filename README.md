# Jira Agent

Small local Jira API tool for Codex, Claude Code, and a future MCP wrapper.

## Setup

Fill `.env`:

```env
JIRA_SITE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=your-api-token
JIRA_TEST_JQL=project = FC AND updated >= -365d ORDER BY updated DESC
JIRA_TEST_PROJECT_KEY=FC
JIRA_TEST_ISSUE_TYPE=Task
JIRA_TEST_ISSUE_KEY=FC-123
JIRA_WRITE_PROJECT_KEYS=FC
JIRA_WRITE_ISSUE_KEYS=
JIRA_AUDIT_LOG_PATH=.jira-agent/audit.jsonl
```

### Where configuration is read from

Values are merged from three layers, lowest to highest precedence:

1. the real process environment
2. `.env` — dotenv syntax, read by `python-dotenv`
3. `.envrc` — a direnv file, so a direnv setup can override `.env`

Both files are re-checked on **every** CLI command and MCP tool call and reparsed
when they changed, so an edit takes effect on the next call instead of at the next
process start. Files are only reparsed when their mtime or size moved, and a file
that does not exist yet is watched too — creating `.envrc.local` later is picked up.

#### Which files are read

Neither file is looked for in the working directory alone — the MCP server is started
as `python3 -m jira_agent.mcp_server` and inherits whatever working directory its host
process happens to have. `.env` is resolved highest to lowest precedence:

1. `$JIRA_ENV_FILE`, absolute or relative to the working directory
2. `.env` in the working directory, when it is actually there
3. `.env` next to the `pyproject.toml` of the checkout the package was imported from

`.envrc` follows the same order via `$JIRA_ENVRC_FILE`. Step 3 is skipped for an
installed (non-editable) distribution, which has no checkout to fall back to; point
`$JIRA_ENV_FILE` at the file in that case. `$JIRA_ENV_FILE` is read from the process
environment only: setting it inside `.env` or `.envrc` has no effect, because it is the
value that decides which file is read in the first place.

`.envrc` is a shell script, and it is **not** executed. Only these lines are honoured:

- `KEY=value` and `export KEY=value` (surrounding quotes and trailing `# comments`
  are stripped; no variable expansion or command substitution)
- `source_env <file>` / `source_env_if_exists <file>` — included as a `.envrc`
- `dotenv <file>` / `dotenv_if_exists <file>` — included as a `.env`
- `unset KEY` / `unset_var KEY` — removes the name entirely, cancelling a value
  from `.env` or from the process environment

Everything else — `use flake`, `layout`, `PATH_add`, conditionals — is ignored.
So a `.envrc` that only carries `use flake` plus `source_env_if_exists .envrc.local`
contributes exactly what `.envrc.local` assigns. If you need the full direnv
semantics, export the values from your shell instead; the process environment is
the base layer.

### Per-project credentials

Some Jira projects require a different user/API token on the same site. Add per-project
overrides with `JIRA_PROJECT__<PROJECTKEY>__*` variables; every other project keeps using
the default `JIRA_EMAIL`/`JIRA_API_TOKEN`:

```env
JIRA_PROJECT__FINRP__EMAIL=finrp-user@example.com
JIRA_PROJECT__FINRP__API_TOKEN=finrp-api-token
# Optional; falls back to JIRA_SITE_URL:
# JIRA_PROJECT__FINRP__SITE_URL=https://your-domain.atlassian.net
```

The project key is case-insensitive in the variable name and is normalized to uppercase.
`EMAIL` and `API_TOKEN` must both be set for a project; a partial override fails config
validation. Requests are matched by the target project or issue key (`FINRP-123` uses the
`FINRP` credentials); requests without a project target, such as `jira_myself` or plain
JQL searches, use the default credentials. `jira_myself`, `jira_search_issues`, `jira_mine`,
and `jira_by_status` accept an optional project argument to run under that project's
credential set. Write audit records name the credential set used as `credential_source`
(the project key or `default`); tokens are never logged. Restart the MCP server (or the
Claude/Codex session hosting it) after changing `.env`.

Install and run:

```bash
uv sync
uv run jira-agent myself
uv run jira-agent mcp-config
uv run jira-agent test-scope --limit 5
uv run jira-agent mine --project FC --limit 5
uv run jira-agent fc-open --limit 5
uv run jira-agent by-status Open --project FC --limit 5
uv run jira-agent user-search Norman --issue FC-107
uv run jira-agent create-meta --project FC
uv run jira-agent create-test-issue "jira-agent smoke test" --description "Dry-run by default."
uv run jira-agent create-issue "Diagram example" --issue-type Task --description-file issue.md
uv run jira-agent render-mermaid FC-107
uv run jira-agent test-issue
uv run jira-agent agent-report
uv run jira-agent agent-report FC-107 --changelog 5 --json
uv run jira-agent agent-report FC-107 --comments all --json
uv run jira-agent agent-plan-comment --next-step "Run implementation checks."
uv run jira-agent human-brief --file ./human-brief.md
uv run jira-agent factory-contract --file ./factory-contract.md --mode comment
uv run jira-agent factory-contract --file ./factory-contract.md --mode attachment
uv run jira-agent drift-reconcile-comment --summary "Jira description drift detected."
uv run jira-agent drift-labels --add needs-review --remove resolved
uv run jira-agent edit-issue --labels jira-agent,test
uv run jira-agent assign-user Norman
uv run jira-agent link-issue FC-107 FC-101 --type Relates
uv run jira-agent delete-issue-link FC-354 FC-356 --type Relates
uv run jira-agent clear-parent FC-359
uv run jira-agent attach --file ./some-report.txt
uv run jira-agent delete-attachment FC-107 --attachment-id 12345
uv run jira-agent embed-diagram FC-107 --file ./architecture.png --delete-filename old.svg
uv run jira-agent embed-existing-png FC-349 --filename fc349-disbursement-flow.png --width <WIDTH> --height <HEIGHT>
uv run jira-agent changelog --limit 5
uv run jira-agent audit-log --limit 10
uv run jira-agent smoke-write
uv run jira-agent assign
uv run jira-agent assign --execute
uv run jira-agent recent --limit 5
uv run jira-agent search "updated >= -30d ORDER BY updated DESC" --limit 5
uv run jira-agent search "project = FC" --all --json
uv run jira-agent search "project = FC" --count
uv run jira-agent issue ABC-123
uv run jira-agent comments ABC-123 --limit 5
uv run jira-agent comments ABC-123 --all --order oldest
uv run jira-agent transitions ABC-123
uv run jira-agent add-comment ABC-123 "Investigated by local agent." --execute
uv run jira-agent transition ABC-123 "In Progress" --execute
uv run python scripts/smoke_mcp.py
```

Without `uv`, use a local virtual environment. The generated MCP configuration uses that
environment's Python executable and therefore does not depend on `uv` being on the client PATH:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\jira-agent.exe myself
.venv\Scripts\jira-agent.exe mcp-config
.venv\Scripts\python.exe scripts\smoke_mcp.py
```

On a fresh Windows system, the repository setup script installs the Python development extras and
the pinned Mermaid CLI, verifies that an installed Chrome or Edge browser can be reused, and runs
the local verification suite. It runs the Jira MCP smoke test as well when `.env` is configured:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

Create commands are dry-run by default. They accept Markdown text or a UTF-8 Markdown file; use
`--description-section` to file one heading body from a larger preparation document:

```powershell
.venv\Scripts\jira-agent.exe create-issue `
  "Establish the immutable external-evidence foundation" `
  --project FC `
  --issue-type Task `
  --description-file C:\path\to\story-draft.md `
  --description-section "Exact proposed Jira description" `
  --labels FA101,MVP,risk-high `
  --parent FC-123 `
  --priority High `
  --story-points 8
```

`--description-section` selects a heading body out of the Markdown you supply, and that extract
becomes the whole description field. On `edit-issue` that means every other section of the existing
issue is dropped. To change one section of a live description and keep the rest, use
`patch-description-section`, which reads the current ADF and rewrites only the matched section:

```bash
uv run jira-agent patch-description-section FC-385 \
  --section "Acceptance criteria" \
  --content-file ./acceptance-criteria.md \
  --mode replace
```

A section is the matched heading plus every following top-level node up to the next heading of the
same or a lower level, so `## A` also covers its `### A.1` and stops before `## B`. An unknown
heading and an ambiguous heading are refused. An ADF level that is missing or out of range is never
guessed: on the matched heading it is refused, and below it the section ends there.

`--mode replace` and `--mode delete` pass two independent controls. Neither inspects a node to
decide whether it counts as content, and neither substitutes for the other.

The first is arithmetic. If the slice is the whole content list, nothing outside it survives, and
that needs `--allow-wide-rewrite`.

The second is a naming duty. Every heading inside the slice below the one you named is a section
the patch removes on top of it, and each has to be named with a repeated
`--expect-swallowed-heading`, in document order. Absent, short, long, differing, or supplied when
the slice removes nothing is a mismatch and is refused; the refusal prints the exact list in full,
so the correction is a copy. A heading with no text is printed and named as `<untitled>`.
`--allow-wide-rewrite` covers the first control only and does not satisfy the naming duty.

```bash
uv run jira-agent patch-description-section FC-385 \
  --section "Scope" \
  --content-file ./scope.md \
  --expect-swallowed-heading "In scope" \
  --expect-swallowed-heading "Out of scope"
```

A section is removed only if you named it. What the tool does not do is judge whether what remains
is meaningful — there is no such detection. Naming the only section of a description carries no
naming duty, because there is no unnamed section to lose, but the first control still applies: if
that section is also the whole content list — it starts at the first node and nothing follows it —
`--allow-wide-rewrite` is required. A section that starts at the first node but is bounded by a
later heading is not the whole content list and needs no flag. A separate rule still refuses any patch that would leave the
description with no content at all.

A refusal prints each name as a quoted JSON array, so a heading containing a comma stays one name
and an empty list is not the word "nothing". The MCP tool returns `swallowed_headings` and
`expect_swallowed_headings` as lists alongside the message, and the CLI prints the same JSON, so
the correction is a copy of data rather than a parse of a sentence.

`--mode append` and `--mode prepend` remove nothing, so neither control applies to them, and
`--no-include-heading` keeps the existing heading node and changes only the body below it. The dry
run prints the matched heading, the replaced index range, `swallowed_headings`, `whole_document`,
and the top-level node counts before and after.

One limitation worth knowing: new checklist items are numbered past the highest task localId
already present, which keeps them unique within the description but does not renumber anything
that was already there. The command reads the description and writes the field back whole, but
by default this is guarded against exactly the race that would otherwise cause: see
[Concurrency guard on whole-field rewrites](#concurrency-guard-on-whole-field-rewrites).

`--story-points` writes to the Jira custom field named by `JIRA_STORY_POINTS_FIELD`, which defaults
to `customfield_10028`. That numeric id differs per Jira site: look yours up under
`/rest/api/3/field` and set the variable in `.env` if it is not the default. Whole values are sent as
integers, halves such as `0.5` are preserved, and `edit-issue` takes the same option.

Every sprint-reading and sprint-writing command or tool reads and renders the field named by
`JIRA_SPRINT_FIELD`, which defaults to `customfield_10020` (the id probed on this Jira site,
schema `com.pyxis.greenhopper.jira:gh-sprint`). That id is also per-site: if a wrong id is
configured, Jira returns HTTP 200 with the field silently absent rather than an error, so check
`config-fields` (or `jira_config_fields`) to see which id is actually in effect before trusting a
row that renders "in no sprint."

## Safe parent and issue-link reconciliation

`edit-issue` leaves an issue parent unchanged when `--parent` is omitted and sets a parent when
`--parent FC-123` is supplied. Clearing is deliberately a separate operation so an omitted value
can never be interpreted as removal:

```bash
uv run jira-agent clear-parent FC-359
uv run jira-agent clear-parent FC-359 --execute
```

The command first reads the current parent, defaults to a dry run, and after `--execute` rereads
the issue and refuses a success result unless the parent is `null`.

`delete-issue-link` also defaults to dry run. It reads both endpoints freshly and resolves exactly
one directed link with the requested type before deletion. The endpoint records must agree on the
same Jira link id; missing, mismatched, or duplicate matches are refused. Pass `--link-id` when a
specific id is already known to pin the resolution further:

```bash
uv run jira-agent delete-issue-link FC-354 FC-356 --type Relates
uv run jira-agent delete-issue-link FC-354 FC-356 --type Relates --link-id 10001 --execute
```

After an executed deletion the tool rereads both issues and refuses success if either still lists
that link id. Both operations retain the configured project/issue write allowlists and write an
audit record after an execution attempt.

For directional Jira link types, `link-issue OUTWARD_KEY INWARD_KEY --type TYPE` uses business
direction: `OUTWARD_KEY` displays the type's outward label and `INWARD_KEY` displays its inward
label. For example, the work item which blocks another is supplied first:

```bash
uv run jira-agent link-issue FC-350 FC-352 --type Blocks
```

Jira's create REST payload uses the inverse field names (`inwardIssue: FC-350`,
`outwardIssue: FC-352`) to produce that display. This corrects earlier releases, which sent the
two fields in the intuitive but wrong order and therefore created directional links reversed.
Existing directional links are not migrated; inspect and reconcile them deliberately with the
dry-run `delete-issue-link` command before recreating them.

To restore an inline diagram without uploading a second copy, `embed-existing-png` resolves an
existing attachment by its exact PNG filename. A duplicate filename is refused unless
`--attachment-id` pins one current attachment, and the MIME type must be `image/png`. Display
dimensions are explicit because they are never guessed:

```bash
uv run jira-agent embed-existing-png FC-349 \
  --filename fc349-disbursement-flow.png --width <WIDTH> --height <HEIGHT>
uv run jira-agent embed-existing-png FC-349 \
  --filename fc349-disbursement-flow.png --width <WIDTH> --height <HEIGHT> --execute
```

The command resolves and validates Jira's media id before its dry run, uses the same normal
diagram placement behavior as `embed-diagram`, and rereads the description after execution to
prove that exact media id is inline. `--replace-existing` is stricter: it replaces only one media
item that belongs to an attachment of the same filename; zero or multiple candidates are refused.

## Current Scope

- Read current Jira user with `myself`
- Fetch one issue with `issue`, including its `parent` by default; a `--fields` value Jira does
  not return for the issue is called out instead of being silently dropped
- Search issues with JQL using Jira Cloud REST API v3 enhanced search. This endpoint never reports
  a total match count, so every search-backed command below states how many issues it returned
  and whether more remain (`returned`/`is_last`, never a fabricated `total`) instead of just one of
  them doing so; `search`'s own `--limit` pages automatically past one page, `--all` pages through
  every match (and wins over `--limit` when both are given), and `--count` opts into one extra,
  explicitly-approximate count from `/search/approximate-count`
- Fetch recently updated issues with `recent`, and friendly searches with `mine`,
  `assigned-to-me`, `fc-open`, and `by-status` — all share `search`'s curated `returned`/`is_last`
  shape and its uncapped `--limit` paging, not a raw single-page passthrough
- Run read-only smoke checks inside the configured FC test scope with `test-scope`, curated the
  same way (its own `--limit` stays capped at 20, deliberately, as a smoke query)
- Inspect create metadata and create dry-run/real test issues in the configured FC project
- Create arbitrary allowed issue types with labels, parent, priority, story points, and
  Markdown-to-ADF descriptions; optionally select one Markdown section from a preparation file
- Fetch and assign the configured test issue
- Build compact agent-friendly issue reports with `agent-report`. `--comments` takes an integer
  (the newest that many) or the literal `all` to page through every comment; both fetch via
  `orderBy=-created` and so both return newest first, and the report carries an explicit
  `comments_order: "newest-first"` key so a reader never has to infer the order from which
  option was passed. The report and its human rendering always state both the true comment
  count and how many were actually returned, so a caller can tell a short list from a complete
  one
- Prepare standardized agent comments with `agent-plan-comment`
- Write Human Briefs into Jira descriptions with `human-brief`, guarded against a concurrent
  edit by default the same way `edit-issue` is: an `--no-concurrency-guard` opt-out is available.
- Store Factory Contracts as comments or attachments with `factory-contract`
- Post Drift Reconcile comments with `drift-reconcile-comment`
- Add or remove `drift-*` labels atomically with `drift-labels` (`--clear` removes every
  existing `drift-*` label); see
  [Atomic label and fix-version operations](#atomic-label-and-fix-version-operations)
- Edit summary, description, labels, parent, priority, and story points with `edit-issue`; add or
  remove individual labels and fix versions atomically on the same command with `--add-label`/
  `--remove-label` and `--add-fix-version`/`--remove-fix-version`, alongside the whole-array
  `--labels`/`--fix-version` (see
  [Atomic label and fix-version operations](#atomic-label-and-fix-version-operations))
- Clear an issue parent explicitly with `clear-parent`, with a post-write reread
- Rewrite one section of an existing description with `patch-description-section`, which
  replaces, appends to, prepends to, or deletes the matched section and carries every top-level
  node outside it over unchanged; any further section the slice removes has to be named with
  `--expect-swallowed-heading`, and a slice that is the whole field needs `--allow-wide-rewrite`
- Search assignable users with `user-search` and assign by query with `assign-user`
- Link issues with `link-issue` and delete one freshly resolved, direction-validated link with
  `delete-issue-link`
- Attach local files with `attach`
- Delete issue attachments with `delete-attachment`
- Upload PNG diagrams and embed them in descriptions with `embed-diagram`
- Embed an existing, exact PNG attachment without reuploading it using `embed-existing-png`
- Automatically render and embed Mermaid blocks after `create-issue` (disable with
  `--no-render-mermaid`) and retrofit an existing issue with `render-mermaid`
- Insert or replace a linked screenshot gallery from existing attachments with
  `upsert-image-gallery`
- `embed-diagram`, `embed-existing-png`, `upsert-image-gallery`, and `render-mermaid` each
  rewrite the whole description field, exactly like `patch-description-section`, and are guarded
  the same way by default. `embed-existing-png` and `upsert-image-gallery` perform no
  Jira-changing call between their read and their write, so the same read that fetches the
  description and attachments also fetches `updated`. `embed-diagram` and `render-mermaid`
  cannot do that: uploading a file is itself a write that moves the issue's own `updated`, so a
  description and token both taken from their first read (before the upload) would already be
  stale by the time of the write. Each instead re-reads the description and `updated`
  together, in one request, immediately after its own upload(s), and builds the write from that
  fresh read; a concurrent edit landing before that post-upload read is incorporated into the
  write rather than lost. All four refuse the write (exit code 4) if `updated` no longer matches
  immediately before it, and `--no-concurrency-guard` opts out of any of them.
- Render every Mermaid code block locally with the pinned Mermaid CLI, upload the generated PNGs,
  and embed them with `jira_render_mermaid_diagrams`
- Render Markdown headings, wrapped paragraphs, bullet and numbered lists, task checkboxes, tables,
  inline code, emphasis, and HTTP links as native Jira ADF
- Treat Markdown blank lines as source readability only; Jira headings, paragraphs, lists, and
  tables provide their own layout spacing, so no empty ADF paragraphs are written
- Inspect issue history with `changelog` or `agent-report --changelog`
- Inspect local write audit records with `audit-log`
- Run bundled write smoke checks against the configured test issue
- List and add comments. `comments` defaults to newest first (`--order oldest` to opt back into
  Jira's natural order), and `--all` pages through every comment instead of the `--limit` page
- List and execute issue transitions

Write actions default to dry-run. Pass `--execute` to actually modify Jira.
Write actions are blocked unless the issue belongs to `JIRA_WRITE_PROJECT_KEYS`
or the issue key is listed in `JIRA_WRITE_ISSUE_KEYS`.
Executed write actions are appended to `JIRA_AUDIT_LOG_PATH`.

With `--json`, a command's stdout is exactly one JSON document and nothing else — safe to pipe
into `jq` or `json.load` with no slicing. Every other line (dry-run notices, safety scope, the
`Pass --execute ...` reminder) prints to stderr instead. For a command that supports `--execute`,
that JSON document also carries a top-level `dry_run` boolean, `true` when the write was not
performed and `false` when it was, so a `--json` caller can tell a dry-run preview apart from a
completed write without having read the (now stderr-only) prose.

This holds on the error paths too: a Jira API failure or a write-safety refusal under `--json`
still prints its prose to stderr, but stdout is not left empty. It carries one document instead,
with `ok: false`, an `error_type` (`jira_api_error` or `write_safety_error`), and the failure's
own detail (a `status_code` where there is one, and the message). Piping a failed `--json`
invocation into `json.load` used to raise `Expecting value: line 1 column 1` on the empty stream;
it now parses, so a caller can tell a transport failure apart from a refusal without falling back
to reading stderr prose.

## MCP Server

The same core is exposed as an MCP server over stdio:

```bash
uv run jira-agent-mcp
```

Example MCP server command for local clients:

```json
{
  "command": "C:\\path\\to\\mcp-jira\\.venv\\Scripts\\python.exe",
  "args": ["-m", "jira_agent.mcp_server"],
  "cwd": "C:\\path\\to\\mcp-jira"
}
```

Available MCP tools:

- `jira_myself`
- `jira_get_issue`
- `jira_search_issues`
- `jira_recent_issues`
- `jira_mine`
- `jira_fc_open`
- `jira_by_status`
- `jira_test_scope_issues`
- `jira_create_issue`
- `jira_create_test_issue`
- `jira_get_test_issue`
- `jira_assign_issue`
- `jira_agent_report`
- `jira_edit_issue`
- `jira_clear_issue_parent`
- `jira_patch_description_section`
- `jira_agent_plan_comment`
- `jira_write_human_brief`
- `jira_store_factory_contract`
- `jira_post_drift_reconcile_comment`
- `jira_update_drift_labels`
- `jira_user_search`
- `jira_assign_user`
- `jira_list_comments`
- `jira_add_comment`
- `jira_update_comment`
- `jira_list_transitions`
- `jira_link_issue`
- `jira_delete_issue_link`
- `jira_attach_file`
- `jira_delete_attachment`
- `jira_embed_diagram`
- `jira_embed_existing_png_attachment`
- `jira_upsert_image_gallery`
- `jira_render_mermaid_diagrams`
- `jira_changelog`
- `jira_transition_issue`
- `jira_get_issue_sprints`
- `jira_add_issues_to_sprint`
- `jira_move_issues_to_backlog`
- `jira_config_fields`

Run an end-to-end MCP smoke test:

```bash
uv run python scripts/smoke_mcp.py
uv run python scripts/smoke_mcp.py --issue FC-107
```

The smoke test starts `jira-agent-mcp` over stdio, verifies the expected tool list, and calls `jira_agent_report`.

## Concurrency guard on whole-field rewrites

`edit-issue`, `patch-description-section`, `human-brief`, `embed-diagram`, `embed-existing-png`,
`upsert-image-gallery`, and `render-mermaid` each read an issue, then write a whole field back.
By default all seven are guarded: the command captures the issue's `updated` timestamp — as part
of its read where nothing changes the issue in between, or with a dedicated read immediately
before the write when something does (`embed-diagram` and `render-mermaid`'s own uploads) — then
refuses the write with exit code 4 (distinct from the generic API error's 1 and the write-safety
error's 3) if `updated` no longer matches immediately before the write. `issue-type` carries the
same guard unconditionally, with no opt-out flag. The refusal names the issue and both timestamps
in prose, and, under `--json`, in a JSON document carrying `error_type: "concurrent_edit"`,
`expected_updated`, `found_updated`, and `dry_run`.

This is a check-then-act narrowing of the race window, not an atomic compare-and-swap: Jira
Cloud exposes no conditional write (no ETag / If-Match) on the issue resource, so a change that
lands inside that narrowed window is still silently lost. Pass `--no-concurrency-guard` on any of
the seven opt-out-capable commands for the previous unguarded behaviour.

On `edit-issue` specifically, the guard only fires for a call that includes at least one
whole-field rewrite (`--summary`, `--description`, the whole-array `--labels`/`--fix-version`,
`--parent`, `--priority`, or `--story-points`): a call built entirely from the atomic
`--add-label`/`--remove-label`/`--add-fix-version`/`--remove-fix-version` options below reads
nothing and carries no guard, because there is no whole-array read for a stale token to protect.

`drift-labels` no longer appears in the list above or carries a `--concurrency-guard` option at
all. It used to read the whole `labels` array, compute a new array in Python, and write it back —
exactly the shape the guard exists to protect — but every one of its paths (`--add`, `--remove`,
and `--clear`) now sends atomic `add`/`remove` operations instead of a whole-array replacement.
See [Atomic label and fix-version operations](#atomic-label-and-fix-version-operations) for what
changed and why none of its three paths need the guard any more.

## Atomic label and fix-version operations

Jira's `editmeta` advertises which update operations a field accepts. Probed read-only against
this site: `labels` reports `operations: ["add", "set", "remove"]` with plain string items, and
`fixVersions` reports `operations: ["set", "add", "remove"]` with version-object items
(`{"name": ...}`). This tool assumes `add`/`remove` are available for both fields on that basis,
sent through Jira's issue PUT `update` section (`{"update": {"labels": [{"add": "x"}, ...]}}`)
rather than the `fields` section every other write here uses. Jira applies `add`/`remove`
against whatever the field holds server-side at write time, not against a value this tool read
earlier, so there is no read-modify-write left for a concurrent editor's own change to lose, and
no `expect_updated` guard is needed or accepted for a call built only from these operations.

`edit-issue` exposes this as `--add-label`/`--remove-label` and `--add-fix-version`/
`--remove-fix-version`, each repeatable, alongside the pre-existing whole-array `--labels` and
`--fix-version`:

```bash
uv run jira-agent edit-issue FC-107 --add-label needs-review --remove-label stale --execute
uv run jira-agent edit-issue FC-107 --add-fix-version FCL-MVP --execute
```

`--labels` and `--fix-version` still replace the whole array outright — genuinely useful when a
caller wants to set the exact contents — and still carry the concurrency guard described above,
because that whole-array replacement CAN silently drop a label or fix version a concurrent editor
added since this process last read the issue. `--add-label`/`--remove-label` and
`--add-fix-version`/`--remove-fix-version` cannot. Combining the whole-array form with the atomic
form for the SAME field is refused locally, naming both options, rather than sent to Jira: it
rejects a PUT that both sets and updates one field in the same request.

```bash
uv run jira-agent edit-issue FC-107 --labels a,b --add-label c
# Edit error: Cannot combine --labels with --add-label/--remove-label: Jira rejects a PUT that
# both sets and updates the same field. --labels replaces the whole array and carries the
# concurrency guard; --add-label/--remove-label are atomic and carry none. Use one form or the
# other.
```

The MCP `jira_edit_issue` tool takes the same four operations as `labels_add`/`labels_remove` and
`fix_versions_add`/`fix_versions_remove`, with the identical refusal (naming `labels`/
`labels_add`/`labels_remove` or `fix_versions`/`fix_versions_add`/`fix_versions_remove`) when a
caller mixes the two forms for one field.

`drift-labels` (and the MCP `jira_update_drift_labels`) is the atomic form's main customer: `--add`
and `--remove` need no read of the issue at all — each named label becomes its own `add`/`remove`
operation, so two agents adding different drift labels to the same issue at the same time cannot
lose each other's work. `--clear` is the one path that still reads the issue, once, to find out
which labels currently start with `drift-`; only those become explicit `remove` operations, so a
drift label another writer adds after that read is never named and survives — the same guarantee
`--add`/`--remove` already give everything they were not told about. None of the three paths
carries a concurrency guard: `--add`/`--remove` have no read to guard, and `--clear`'s read only
selects which `remove` operations to send, each of which is itself safe against a concurrent edit
once sent.

Every dry run under this section prints the exact operations it would send, never a guessed
resulting array: the real result depends on server-side state this tool has not read, so showing
anything else would be a guess dressed up as a preview.

Two comment commands fill a gap where only `add-comment` existed:

- `edit-comment <KEY> <COMMENT_ID> --body TEXT | --body-file PATH` replaces one comment's body.
  Dry run shows the comment's current body next to the proposed replacement, in prose and, under
  `--json`, as `current_body` / `proposed_body`.
- `delete-comment <KEY> <COMMENT_ID>` deletes one comment. Since deletion is irreversible, dry
  run prints the full current body — the only chance to notice a wrong comment id before it is
  gone for good.

Both default to dry-run behind `--execute`, go through the same write-scope check as every other
write command, and append to the audit log on execution.

## Sprints, boards, and issue-type changes

These commands automate two things a human previously did by hand in the Jira board UI: moving
several issues between sprints, and turning a Task into a Bug (previously impossible in place, so
the workaround was creating a duplicate issue and retiring the original).

- `boards [--type scrum|kanban] [--name TEXT] [--max-results N]` — list Jira Agile boards, so a
  caller can find a board id without opening the UI. `--max-results` defaults to 50 and this
  command never fetches a second page, so both output modes report whether more boards remain:
  a `returned N; more remain` / `returned N; complete` line in human mode from the response's own
  `isLast`, and `returned`/`is_last` alongside the raw response under `--json`. No total is
  fabricated; Jira's own `total`, when the response carries one, is reported and labelled as
  coming from Jira.
- `sprints --board ID [--state active|future|closed] [--max-results N]` — list one board's
  sprints. Same truncation reporting as `boards`: `--max-results` defaults to 50, and both output
  modes say whether more sprints remain instead of showing 50 and staying silent.
- `issue-sprint <KEY>` — read-only, no `--execute`. Reports which sprint(s) an issue is in, and
  renders three distinct states rather than collapsing any two of them: the sprint field absent
  from the Jira response entirely (`sprint_field_present: false`), the field present with `null`
  or an empty list (`sprint_field_present: true`, `sprints: []`; this Jira site returns `null`,
  not `[]`, for "in no sprint", verified via `Sprint IS EMPTY` JQL, and both render identically
  here), and the field present with one or more sprints. The `issue`, `search`, `mine`,
  `fc-open`, `by-status`, `recent`, and `test-scope` commands keep the same first distinction in
  their own Sprint column: a field genuinely absent from a row (this Jira site's sprint field id
  may differ from the configured one, or `JIRA_SPRINT_FIELD` may be misconfigured — Jira returns
  HTTP 200 and silently drops an unrecognized field id, with no error of its own) renders as
  `? (field not returned)`, never as the same `-` a genuine empty sprint list gets, and triggers
  the same "Jira did not return requested field(s)" warning `issue` already prints for any other
  missing field.
- `sprint-add <SPRINT_ID> <KEY>...` — add one or more issues to a sprint. This is a single
  batched `POST /sprint/{id}/issue` call carrying every key at once, not a per-issue loop, so
  there is no "key 7 of 13 failed, the rest already moved" state to report on the Jira side. The
  write-safety gate is checked for every key before the dry run reports success, not only before
  `--execute` writes, so a clean dry run predicts what `--execute` does; a key that fails the
  gate stops the command before any issue, including earlier ones in the same invocation, is
  written, and the dry run reports the same per-key `safety` result `--execute` would have used.
  The write is audited as soon as Jira accepts it, before the post-write reread that verifies it
  runs; if that reread itself fails, the write is reported as accepted but unverified
  (`verified: null`, exit code 6) rather than as a failed write with no audit record.
- `sprint-remove <SPRINT_ID> <KEY>...` — move one or more issues to the backlog. `SPRINT_ID` is a
  **precondition this command enforces, not a filter Jira applies**: the underlying endpoint
  (`POST /rest/agile/1.0/backlog/issue`) takes no sprint id at all. It is documented as
  equivalent to removing the FUTURE and ACTIVE sprints from the given issues — ALL of them, on
  every key, not only the one named — and it leaves CLOSED sprints untouched. An issue can be in
  more than one non-closed sprint at once (a long-lived issue on a board that has moved through
  several sprints), so before any write every key's sprint field is re-read and the whole batch
  is refused, nothing written for any key, if either: a key is not currently in `SPRINT_ID`
  (exit code 5), or a key sits in another non-closed sprint besides `SPRINT_ID` (exit code 7) —
  that write would silently clear a sprint nobody named. Pass `--allow-multi-sprint-removal` to
  proceed anyway once you have seen what it would additionally clear. The dry run lists, per key,
  every sprint it will actually leave, not only `SPRINT_ID`. Like `sprint-add`, the write is
  audited as soon as Jira accepts it, before the post-write reread; `results` reports, per key,
  whether it left every sprint the dry run said it would, naming any it is still in, and an
  unverifiable reread is reported as `verified: null` (exit code 6) rather than a failed write.
  Reading `SPRINT_ID` membership is itself a check-then-act race, not an atomic guarantee: a
  concurrent change between that read and the write is not caught, the same hedge that applies
  to the concurrency guard elsewhere in this README.
- `issue-type <KEY> <NEW_TYPE>` — change one issue's issue type. Defaults to dry-run; pass
  `--execute` to write. Before writing, it calls `is_issue_type_allowed` and refuses locally, with
  no Jira round trip, listing the allowed values from `editmeta` when `NEW_TYPE` is not among
  them. When executing, it captures `issue_version` immediately before the write and passes it as
  the concurrency guard `expect_updated`; if the issue changed since it was read, the write is
  refused and the command exits with code 4, naming both the expected and the found `updated`
  timestamp. A Jira-side rejection not predicted by `editmeta` (for example a required field on
  the target type) surfaces Jira's own error text rather than being swallowed.
- `issue-type-options <KEY>` — read-only. Lists the issue types `<KEY>` may become, from the same
  `editmeta` allowed-values list `issue-type` checks against.
- `config-fields` — read-only, no Jira call. Shows the custom field ids this tool has resolved
  from config: `story_points_field` and `sprint_field`, both instance-specific Jira custom field
  ids resolved the same way (a `JIRA_STORY_POINTS_FIELD` / `JIRA_SPRINT_FIELD` env var, or this
  site's probed default), so a caller can check which ids are actually in effect before running
  anything that writes rather than guessing.

`sprint-add` and `sprint-remove` add three exit codes beyond the general ones (1 generic API
error, 2 bad arguments, 3 write-safety refusal, 4 concurrent-edit refusal): 5 fires before any
write when a key is not currently in `SPRINT_ID` (`sprint-remove` only); 6 fires after a write
Jira accepted when the post-write reread did not confirm it, OR when that reread itself failed
and the write's outcome is therefore unknown rather than failed (`verified: null`); 7 fires
before any write, like 5, but for a different reason specific to `sprint-remove` — a key IS in
`SPRINT_ID` but ALSO in another non-closed sprint the write would silently clear too. Under
`--json`, the three failure states carry distinct `error_type` values —
`sprint_precondition_failed`, `sprint_multi_removal_refused`, and
`sprint_verification_unavailable` — so a script never has to parse prose to tell them apart.

Sprint and issue-type writes follow the same dry-run-by-default, `--execute`-to-write,
`JIRA_WRITE_PROJECT_KEYS`/`JIRA_WRITE_ISSUE_KEYS`-gated, and audit-logged rules as every other
write command above, including the `--json` contract: stdout carries exactly one JSON document
with a top-level `dry_run` boolean, and prose moves to stderr.
