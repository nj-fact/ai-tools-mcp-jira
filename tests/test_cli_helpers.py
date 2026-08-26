from __future__ import annotations

from typer.testing import CliRunner

from jira_agent import cli
from jira_agent.client import select_transition
from jira_agent.config import JiraConfig


def test_select_transition_matches_id_or_case_insensitive_name() -> None:
    data = {
        "transitions": [
            {"id": "11", "name": "Start Progress"},
            {"id": "21", "name": "Done"},
        ]
    }

    assert select_transition(data, "11") == {"id": "11", "name": "Start Progress"}
    assert select_transition(data, "done") == {"id": "21", "name": "Done"}
    assert select_transition(data, "missing") is None


def test_create_issue_cli_embeds_mermaid_after_creation(tmp_path, monkeypatch) -> None:
    config = JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    calls: dict[str, object] = {}

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def create_issue(self, **kwargs):
            calls["create"] = kwargs
            return {"id": "10001", "key": "FC-123"}

    def fake_embed(client, **kwargs):
        calls["embed_client"] = client
        calls["embed"] = kwargs
        return {
            "ok": True,
            "diagram_count": 1,
            "files": [str(tmp_path / "fc-123-mermaid-1.png")],
            "attachment_ids": ["42"],
            "deleted_attachment_ids": [],
            "result": {"updated": True},
        }

    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(cli, "JiraClient", FakeClient)
    monkeypatch.setattr(cli, "embed_mermaid_diagrams", fake_embed)
    monkeypatch.setattr(cli, "audit_write", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        cli.app,
        [
            "create-issue",
            "Diagram",
            "--issue-type",
            "Task",
            "--project",
            "FC",
            "--description",
            "```mermaid\nflowchart LR\nA --> B\n```",
            "--execute",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Embedded 1 Mermaid diagram(s)." in result.output
    assert calls["embed"]["issue_key"] == "FC-123"
