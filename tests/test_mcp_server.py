from __future__ import annotations

from jira_agent import mcp_server
from jira_agent.client import text_to_adf
from jira_agent.config import JiraConfig


def make_config(tmp_path) -> JiraConfig:
    return JiraConfig.model_validate(
        {
            "JIRA_SITE_URL": "https://example.atlassian.net",
            "JIRA_EMAIL": "user@example.com",
            "JIRA_API_TOKEN": "token",
            "JIRA_WRITE_PROJECT_KEYS": "FC",
            "JIRA_AUDIT_LOG_PATH": str(tmp_path / "audit.jsonl"),
        }
    )


def test_jira_create_issue_dry_run_reports_mermaid_diagrams() -> None:
    result = mcp_server.jira_create_issue(
        project_key="FC",
        issue_type_name="Task",
        summary="Diagram",
        description="```mermaid\nflowchart LR\nA --> B\n```",
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["mermaid_diagrams"] == 1


def test_jira_create_issue_embeds_mermaid_after_creation(tmp_path, monkeypatch) -> None:
    config = make_config(tmp_path)
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

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)
    monkeypatch.setattr(mcp_server, "embed_mermaid_diagrams", fake_embed)
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    result = mcp_server.jira_create_issue(
        project_key="FC",
        issue_type_name="Task",
        summary="Diagram",
        description="```mermaid\nflowchart LR\nA --> B\n```",
        execute=True,
    )

    assert result["ok"] is True
    assert result["issue"]["key"] == "FC-123"
    assert result["mermaid"]["diagram_count"] == 1
    assert calls["embed"]["issue_key"] == "FC-123"
    assert calls["embed"]["description"] == text_to_adf("```mermaid\nflowchart LR\nA --> B\n```")


def test_jira_create_issue_can_disable_mermaid_embedding(tmp_path, monkeypatch) -> None:
    config = make_config(tmp_path)

    class FakeClient:
        def __init__(self, _config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def create_issue(self, **_kwargs):
            return {"id": "10001", "key": "FC-123"}

    monkeypatch.setattr(mcp_server, "load_config", lambda: config)
    monkeypatch.setattr(mcp_server, "JiraClient", FakeClient)
    monkeypatch.setattr(
        mcp_server,
        "embed_mermaid_diagrams",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not embed")),
    )
    monkeypatch.setattr(mcp_server, "audit_write", lambda *_args, **_kwargs: None)

    result = mcp_server.jira_create_issue(
        project_key="FC",
        issue_type_name="Task",
        summary="Diagram",
        description="```mermaid\nflowchart LR\nA --> B\n```",
        execute=True,
        render_mermaid=False,
    )

    assert result == {"ok": True, "issue": {"id": "10001", "key": "FC-123"}}
