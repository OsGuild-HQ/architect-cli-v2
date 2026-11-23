from __future__ import annotations

from architect.conversation import Conversation, estimate_tokens, repair, transcript


def call(call_id: str, name: str = "read") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def test_estimate_tokens_handles_none() -> None:
    """v1 crashed computing len(None) after resuming a session with tool calls."""
    conv = Conversation([{"role": "assistant", "content": None, "tool_calls": [call("a")]}])
    assert estimate_tokens(None) == 0
    assert conv.estimated_tokens() > 0


def test_repair_keeps_complete_groups() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [call("a"), call("b")]},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "assistant", "content": "done"},
    ]
    repaired = repair(messages)
    assert [m["role"] for m in repaired] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [m["tool_call_id"] for m in repaired[2:4]] == ["a", "b"]  # reordered to call order


def test_repair_drops_incomplete_group_and_orphans() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "zzz", "content": "orphan"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [call("a"), call("b")]},
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "user", "content": "next"},
    ]
    assert [m["role"] for m in repair(messages)] == ["user", "user"]


def test_repair_keeps_text_of_incomplete_group() -> None:
    messages = [{"role": "assistant", "content": "Let me look.", "tool_calls": [call("a")]}]
    assert repair(messages) == [{"role": "assistant", "content": "Let me look."}]


def test_repair_rejects_missing_ids() -> None:
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [call("")]},
        {"role": "tool", "tool_call_id": "", "content": "x"},
    ]
    assert repair(messages) == []


def test_api_messages_includes_summary() -> None:
    conv = Conversation([{"role": "user", "content": "hi"}], summary="earlier stuff")
    api = conv.api_messages("SYS")
    assert api[0] == {"role": "system", "content": "SYS"}
    assert "earlier stuff" in api[1]["content"]
    assert api[2]["content"] == "hi"


def _turns() -> list[dict]:
    big = "x" * 4000  # ~1000 tokens
    return [
        {"role": "user", "content": big},
        {"role": "assistant", "content": None, "tool_calls": [call("a")]},
        {"role": "tool", "tool_call_id": "a", "content": big},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": big},
        {"role": "assistant", "content": "answer 2"},
        {"role": "user", "content": "latest"},
    ]


def test_compaction_split_only_at_turn_boundaries() -> None:
    conv = Conversation(_turns())
    covered = conv.compaction_split(keep_tokens=1500)
    assert covered == 4  # first turn (with its tool group) goes; turns 2 and 3 stay
    kept = conv.messages[covered:]
    assert kept[0]["role"] == "user"


def test_compaction_split_keeps_latest_turn_even_if_huge() -> None:
    conv = Conversation(_turns())
    assert conv.compaction_split(keep_tokens=1) == 6
    assert Conversation([{"role": "user", "content": "x"}]).compaction_split(1) is None


def test_apply_compaction() -> None:
    conv = Conversation(_turns())
    conv.apply_compaction("S", 4)
    assert conv.summary == "S"
    assert len(conv) == 3


def test_transcript_clips_tool_output() -> None:
    text = transcript(_turns(), tool_result_chars=10)
    assert "assistant called read({})" in text
    assert "…(clipped)" in text
