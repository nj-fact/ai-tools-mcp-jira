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
uv run jira-agent issue ABC-123
uv run jira-agent comments ABC-123 --limit 5
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
naming duty, because there is no unnamed section to lose, but the first control still applies: an
ordinary single-section description starts at the first node, so its slice is the whole content
list and `--allow-wide-rewrite` is required. Only a description with something in front of the
section goes through unflagged. A separate rule still refuses any patch that would leave the
description with no content at all.

A refusal prints each name as a quoted JSON array, so a heading containing a comma stays one name
and an empty list is not the word "nothing". The MCP tool returns `swallowed_headings` and
`expect_swallowed_headings` as lists alongside the message, and the CLI prints the same JSON, so
the correction is a copy of data rather than a parse of a sentence.

`--mode append` and `--mode prepend` remove nothing, so neither control applies to them, and
`--no-include-heading` keeps the existing heading node and changes only the body below it. The dry
run prints the matched heading, the replaced index range, `swallowed_headings`, `whole_document`,
and the top-level node counts before and after.

Two limitations worth knowing. The command reads the description and writes the field back whole
with no optimistic concurrency, so an edit that lands between the read and the write is silently
reverted. And new checklist items are numbered past the highest task localId already present, which
keeps them unique within the description but does not renumber anything that was already there.

`--story-points` writes to the Jira custom field named by `JIRA_STORY_POINTS_FIELD`, which defaults
to `customfield_10028`. That numeric id differs per Jira site: look yours up under
`/rest/api/3/field` and set the variable in `.env` if it is not the default. Whole values are sent as
integers, halves such as `0.5` are preserved, and `edit-issue` takes the same option.

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
- Fetch one issue with `issue`
- Search issues with JQL using Jira Cloud REST API v3 enhanced search
- Fetch recently updated issues with `recent`
- Use friendly searches with `mine`, `assigned-to-me`, `fc-open`, and `by-status`
- Run read-only smoke checks inside the configured FC test scope with `test-scope`
- Inspect create metadata and create dry-run/real test issues in the configured FC project
- Create arbitrary allowed issue types with labels, parent, priority, story points, and
  Markdown-to-ADF descriptions; optionally select one Markdown section from a preparation file
- Fetch and assign the configured test issue
- Build compact agent-friendly issue reports with `agent-report`
- Prepare standardized agent comments with `agent-plan-comment`
- Write Human Briefs into Jira descriptions with `human-brief`
- Store Factory Contracts as comments or attachments with `factory-contract`
- Post Drift Reconcile comments with `drift-reconcile-comment`
- Set or remove `drift-*` labels with `drift-labels`
- Edit summary, description, labels, parent, priority, and story points with `edit-issue`
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
- Render every Mermaid code block locally with the pinned Mermaid CLI, upload the generated PNGs,
  and embed them with `jira_render_mermaid_diagrams`
- Render Markdown headings, wrapped paragraphs, bullet and numbered lists, task checkboxes, tables,
  inline code, emphasis, and HTTP links as native Jira ADF
- Treat Markdown blank lines as source readability only; Jira headings, paragraphs, lists, and
  tables provide their own layout spacing, so no empty ADF paragraphs are written
- Inspect issue history with `changelog` or `agent-report --changelog`
- Inspect local write audit records with `audit-log`
- Run bundled write smoke checks against the configured test issue
- List and add comments
- List and execute issue transitions

Write actions default to dry-run. Pass `--execute` to actually modify Jira.
Write actions are blocked unless the issue belongs to `JIRA_WRITE_PROJECT_KEYS`
or the issue key is listed in `JIRA_WRITE_ISSUE_KEYS`.
Executed write actions are appended to `JIRA_AUDIT_LOG_PATH`.

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

Run an end-to-end MCP smoke test:

```bash
uv run python scripts/smoke_mcp.py
uv run python scripts/smoke_mcp.py --issue FC-107
```

The smoke test starts `jira-agent-mcp` over stdio, verifies the expected tool list, and calls `jira_agent_report`.
