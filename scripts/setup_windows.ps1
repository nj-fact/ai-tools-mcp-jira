param(
    [string]$PythonCommand = "python",
    [string]$PnpmCommand = "pnpm"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-Tool {
    param(
        [string]$Command,
        [string[]]$Candidates
    )

    $resolved = Get-Command $Command -ErrorAction SilentlyContinue
    if ($resolved) {
        return $resolved.Source
    }
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    throw "Required tool '$Command' was not found."
}

$python = Resolve-Tool $PythonCommand @(
    "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)
$node = Resolve-Tool "node" @(
    "$env:ProgramFiles\nodejs\node.exe",
    "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
)
$pnpm = Resolve-Tool $PnpmCommand @(
    "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"
)

$env:PATH = "$(Split-Path -Parent $node);$env:PATH"
$env:PUPPETEER_SKIP_DOWNLOAD = "true"

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
        & $python -m venv .venv
    }
    & .venv\Scripts\python.exe -m pip install -e ".[dev]"
    & $pnpm install --frozen-lockfile
    & .venv\Scripts\python.exe -c "from jira_agent.mermaid import discover_mermaid_runtime; runtime = discover_mermaid_runtime(); print(f'Mermaid runtime: node={runtime.node}; cli={runtime.cli}; browser={runtime.browser}')"
    & .venv\Scripts\python.exe -m pytest
    & .venv\Scripts\python.exe -m ruff check src tests scripts
    & .venv\Scripts\python.exe -m ruff format --check src tests scripts
    if (Test-Path -LiteralPath ".env") {
        & .venv\Scripts\python.exe scripts\smoke_mcp.py
    }
    else {
        Write-Host "Skipping Jira MCP smoke test because .env is not configured yet."
    }
}
finally {
    Pop-Location
}
