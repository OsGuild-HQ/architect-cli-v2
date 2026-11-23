from __future__ import annotations

import json
from pathlib import Path

import pytest

from architect.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    SessionStore,
    describe_call,
)


def _call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": '{"path": "a.py"}'},
    }


def test_session_file_created_lazily(store: SessionStore, tmp_path: Path) -> None:
    session = store.create(root_path=tmp_path, model="m", base_url="u")
    assert not session.exists
    session.message({"role": "user", "content": "hi"}, display_content="hi")
    lines = [json.loads(line) for line in session.path.read_text().splitlines()]
    assert lines[0]["type"] == "meta"
    assert lines[0]["session_id"] == session.id
    assert lines[1]["role"] == "user"


def test_roundtrip_with_tools(store: SessionStore, tmp_path: Path) -> None:
    s = store.create(root_path=tmp_path, model="m", base_url="u")
    s.message({"role": "user", "content": "PROMPT"}, display_content="shown")
    s.message({"role": "assistant", "content": None, "tool_calls": [_call("a")]}, thinking="hmm")
    s.message({"role": "tool", "tool_call_id": "a", "content": "file text"}, name="read")
    s.message({"role": "assistant", "content": "final"}, thinking="")

    loaded = store.load(s.path)
    roles = [m["role"] for m in loaded.conversation.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert loaded.conversation.messages[0]["content"] == "PROMPT"
    assert [d.kind for d in loaded.display] == ["user", "assistant", "tool", "assistant"]
    assert loaded.display[0].text == "shown"
    assert loaded.display[2].text == "read a.py"
    # Resumed history with None content must not break token estimation (v1 crash).
    assert loaded.conversation.estimated_tokens() > 0


def test_compaction_replay_drops_covered_messages(store: SessionStore, tmp_path: Path) -> None:
    """v1 kept every message after a compaction, so resumed sessions re-sent everything."""
    s = store.create(root_path=tmp_path, model="m", base_url="u")
    for i in range(3):
        s.message({"role": "user", "content": f"q{i}"})
        s.message({"role": "assistant", "content": f"a{i}"})
    s.write({"type": "compaction", "summary": "S1", "covered_message_count": 4})
    s.message({"role": "user", "content": "q3"})
    s.write({"type": "compaction", "summary": "S2", "covered_message_count": 1})
    loaded = store.load(s.path)
    assert loaded.conversation.summary == "S2"
    assert [m["content"] for m in loaded.conversation.messages] == ["a2", "q3"]


def test_rollback_and_clear_replay(store: SessionStore, tmp_path: Path) -> None:
    s = store.create(root_path=tmp_path, model="m", base_url="u")
    s.message({"role": "user", "content": "q"})
    s.message({"role": "assistant", "content": None, "tool_calls": [_call("a")]})
    s.write({"type": "rollback", "message_count": 1})
    assert [m["role"] for m in store.load(s.path).conversation.messages] == ["user"]
    s.write({"type": "clear"})
    s.message({"role": "user", "content": "fresh"})
    loaded = store.load(s.path)
    assert [m["content"] for m in loaded.conversation.messages] == ["fresh"]
    assert [d.text for d in loaded.display] == ["fresh"]


def test_dangling_tool_call_is_repaired_on_send(store: SessionStore, tmp_path: Path) -> None:
    """A crash mid-tool-group (no rollback record) must not poison the session."""
    s = store.create(root_path=tmp_path, model="m", base_url="u")
    s.message({"role": "user", "content": "q"})
    s.message({"role": "assistant", "content": None, "tool_calls": [_call("a"), _call("b")]})
    s.message({"role": "tool", "tool_call_id": "a", "content": "A"})
    loaded = store.load(s.path)
    loaded.conversation.append({"role": "user", "content": "again"})
    api = loaded.conversation.api_messages("SYS")
    assert [m["role"] for m in api] == ["system", "user", "user"]


def test_bad_lines_are_skipped(store: SessionStore, tmp_path: Path) -> None:
    s = store.create(root_path=tmp_path, model="m", base_url="u")
    s.message({"role": "user", "content": "q"})
    with s.path.open("a") as handle:
        handle.write("{not json\n")
    s.message({"role": "assistant", "content": "a"})
    assert len(store.load(s.path).conversation) == 2


def test_find_and_list(store: SessionStore, tmp_path: Path) -> None:
    a = store.create(root_path=tmp_path, model="m", base_url="u")
    a.message({"role": "user", "content": "first question"})
    b = store.create(root_path=tmp_path / "other", model="m", base_url="u")
    b.message({"role": "user", "content": "second"})
    empty = store.create(root_path=tmp_path, model="m", base_url="u")
    empty.write({"type": "event", "name": "noop"})

    assert store.find(a.id) == a.path
    assert store.find(a.id[:6]) == a.path
    with pytest.raises(SessionNotFoundError):
        store.find("nope")
    listed = store.list(tmp_path)
    assert [i.id for i in listed] == [a.id]  # other root and empty session excluded
    assert listed[0].title == "first question"
    assert store.latest(tmp_path) == a.path


def test_find_ambiguous(store: SessionStore) -> None:
    store.directory.mkdir(parents=True)
    for name in ("2026-01-01T00-00-00Z-abc11111.jsonl", "2026-01-01T00-00-01Z-abc22222.jsonl"):
        (store.directory / name).write_text("")
    with pytest.raises(AmbiguousSessionError) as info:
        store.find("abc")
    assert set(info.value.matches) == {"abc11111", "abc22222"}


def test_describe_call() -> None:
    assert describe_call("grep", '{"pattern": "TODO", "include": "*.py"}') == "grep 'TODO' in *.py"
    assert describe_call("glob", "not json") == "glob"
