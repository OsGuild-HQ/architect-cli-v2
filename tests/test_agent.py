from __future__ import annotations

import json
import threading
from dataclasses import replace

from architect.agent import AgentCallbacks
from architect.config import Settings
from architect.sessions import SessionStore
from conftest import (
    AgentHarness,
    FakeStream,
    bad_request,
    completion,
    connection_error,
    text_reply,
    tool_reply,
)


def roles(h: AgentHarness) -> list[str]:
    return [m["role"] for m in h.agent.conversation.messages]


def test_plain_answer(make_agent) -> None:
    h = make_agent([text_reply("Hello there", thinking="pondering")])
    out = h.agent.run("hi")
    assert out.status == "done"
    assert out.text == "Hello there"
    assert out.usage is not None and out.usage.total == 15
    assert roles(h) == ["user", "assistant"]
    call = h.client.calls[0]
    assert call["stream_options"] == {"include_usage": True}
    assert call["tools"]


def test_tool_messages_stay_in_history(make_agent) -> None:
    """v1 bug: tool calls/results were only in a local list, lost after the turn."""
    h = make_agent(
        [
            tool_reply(("c1", "read", '{"path": "src/app.py"}')),
            text_reply("It returns 42."),
            text_reply("Still 42."),
        ]
    )
    assert h.agent.run("what does run return?").text == "It returns 42."
    assert roles(h) == ["user", "assistant", "tool", "assistant"]
    assert "return 42" in h.agent.conversation.messages[2]["content"]

    h.agent.run("and again?")
    sent = h.client.calls[2]["messages"]
    assert any(m.get("role") == "tool" and "return 42" in m["content"] for m in sent)


def test_tool_calls_detected_without_tool_calls_finish_reason(make_agent) -> None:
    """Some backends finish with "stop" even when returning tool calls."""
    h = make_agent(
        [tool_reply(("c1", "glob", '{"pattern": "**/*.py"}'), finish="stop"), text_reply("ok")]
    )
    out = h.agent.run("list python files")
    assert out.tool_calls == 1
    assert "main.py" in h.agent.conversation.messages[2]["content"]


def test_missing_tool_call_id_is_generated(make_agent) -> None:
    h = make_agent([tool_reply(("", "glob", '{"pattern": "*.md"}')), text_reply("ok")])
    h.agent.run("x")
    assistant = h.agent.conversation.messages[1]
    generated = assistant["tool_calls"][0]["id"]
    assert generated.startswith("call_")
    assert h.agent.conversation.messages[2]["tool_call_id"] == generated


def test_bad_tool_arguments_reported_to_model(make_agent) -> None:
    h = make_agent([tool_reply(("c1", "read", "{not json")), text_reply("sorry")])
    h.agent.run("x")
    assert "not valid JSON" in h.agent.conversation.messages[2]["content"]


def test_tool_limit_forces_final_answer(make_agent, settings: Settings) -> None:
    h = make_agent(
        [
            tool_reply(("c1", "glob", '{"pattern": "*"}')),
            tool_reply(("c2", "glob", '{"pattern": "*"}')),
            text_reply("final"),
        ],
        settings=replace(settings, max_tool_iterations=2),
    )
    out = h.agent.run("x")
    assert out.text == "final"
    assert out.hit_tool_limit
    assert "tools" not in h.client.calls[-1]  # last round sent without tools


def test_tools_unsupported_fallback(make_agent) -> None:
    notices: list[str] = []
    h = make_agent(
        [
            bad_request("this model does not support tools"),
            text_reply("plain"),
            text_reply("again"),
        ],
        callbacks=AgentCallbacks(on_notice=notices.append),
    )
    assert h.agent.run("x").text == "plain"
    assert notices and "support tools" in notices[0]
    h.agent.run("y")
    assert "tools" not in h.client.calls[-1]  # remembered


def test_stream_options_fallback(make_agent) -> None:
    h = make_agent(
        [bad_request("Unrecognized parameter stream_options"), text_reply("ok"), text_reply("ok")]
    )
    assert h.agent.run("x").status == "done"
    assert "stream_options" not in h.client.calls[1]
    h.agent.run("y")
    assert "stream_options" not in h.client.calls[2]


def test_cancel_mid_tool_group_rolls_back(make_agent, store: SessionStore) -> None:
    """v1 left an assistant tool-call message without results → every later request 400'd."""
    started = threading.Event()
    release = threading.Event()

    def ask(question: str, options: list[str]) -> str | None:
        started.set()
        release.wait(5)
        return "A"

    h = make_agent(
        [tool_reply(("c1", "glob", '{"pattern": "*"}'), ("c2", "question", '{"question": "?"}'))],
        ask=ask,
    )
    worker = threading.Thread(target=lambda: setattr(h, "out", h.agent.run("x")))
    worker.start()
    assert started.wait(5)
    h.agent.cancel()
    release.set()
    worker.join(5)

    assert h.out.status == "cancelled"  # type: ignore[attr-defined]
    assert roles(h) == ["user"]
    resumed = store.load(h.session.path)
    assert [m["role"] for m in resumed.conversation.messages] == ["user"]


def test_cancel_during_stream(make_agent) -> None:
    h = make_agent([])

    def cancel_at_fourth_chunk(index: int) -> None:
        if index == 3:
            h.agent.cancel()

    stream = FakeStream(text_reply("a long answer " * 20), before_chunk=cancel_at_fourth_chunk)
    h.client.completions.script.append(stream)
    out = h.agent.run("x")
    assert out.status == "cancelled"
    assert stream.closed
    assert roles(h) == ["user"]


def test_api_error_is_reported(make_agent) -> None:
    h = make_agent([connection_error()])
    out = h.agent.run("x")
    assert out.status == "error"
    assert "Connection error" in (out.error or "")
    assert roles(h) == ["user"]


def test_compaction_runs_and_persists(make_agent, settings: Settings, store: SessionStore) -> None:
    tight = replace(settings, max_context_tokens=1000)
    big = "z" * 3000
    statuses: list[str] = []
    h = make_agent(
        [text_reply("a1"), completion("SUMMARY"), text_reply("a2")],
        settings=tight,
        callbacks=AgentCallbacks(on_status=statuses.append),
    )
    h.agent.run(big)
    out = h.agent.run("short follow-up")
    assert out.compacted
    assert "compacting" in statuses
    assert h.agent.conversation.summary == "SUMMARY"
    assert roles(h) == ["user", "assistant"]
    sent = h.client.calls[2]["messages"]
    assert "SUMMARY" in sent[1]["content"]

    resumed = store.load(h.session.path)
    assert resumed.conversation.summary == "SUMMARY"
    assert resumed.conversation.messages == h.agent.conversation.messages


def test_compaction_failure_continues(make_agent, settings: Settings) -> None:
    tight = replace(settings, max_context_tokens=1000)
    err = connection_error()
    notices: list[str] = []
    h = make_agent(
        [text_reply("a1"), err, text_reply("a2")],
        settings=tight,
        callbacks=AgentCallbacks(on_notice=notices.append),
    )
    h.agent.run("z" * 3000)
    assert h.agent.run("next").status == "done"
    assert any("compact" in n for n in notices)


def test_session_mirrors_history(make_agent, store: SessionStore) -> None:
    h = make_agent(
        [tool_reply(("c1", "read", json.dumps({"path": "main.py"}))), text_reply("done")]
    )
    h.agent.run("go", display="go (shown)")
    resumed = store.load(h.session.path)
    assert resumed.conversation.messages == h.agent.conversation.messages
    assert resumed.display[0].text == "go (shown)"
