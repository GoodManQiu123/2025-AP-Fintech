import json
import os

import pytest

openai = pytest.importorskip("openai")  # Skip this file entirely if openai not installed.

from core.llm.chat_agent import ChatAgent  # type: ignore


def test_chat_agent_construct_and_export(tmp_path, monkeypatch):
    """Smoke test: construct ChatAgent and export empty histories (no API call)."""

    # Avoid accidental network/API calls by pointing to a dummy key and not calling send().
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "sk-dummy"))

    agent = ChatAgent(system_prompt="You are a JSON-only test agent.", json_mode=True, verbose=False)

    dialog_path = tmp_path / "dialog.json"
    full_path = tmp_path / "full.json"

    agent.export_dialog_json(dialog_path)
    agent.export_full_json(full_path)

    # Files should exist and be valid JSON.
    for p in (dialog_path, full_path):
        assert p.exists()
        json.loads(p.read_text(encoding="utf-8"))
