"""End-to-end UI tests with Textual's pilot and a scripted fake API."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from textual.pilot import Pilot
from textual.widgets import Static

from architect.config import ConfigFile, Settings
from architect.sessions import SessionStore
from architect.ui.app import ArchitectApp
from architect.ui.question_modal import QuestionModal
from conftest import FakeClient, text_reply, tool_reply


async def wait_for(pilot: Pilot, condition: Callable[[], bool], timeout: float = 5.0) -> None:
    waited = 0.0
    while not condition():
        await pilot.pause(0.05)
        waited += 0.05
        assert waited < timeout, "timed out waiting for condition"


def chat_text(app: ArchitectApp) -> str:
    parts = []
    for widget in app.chat.query("*"):
        raw = getattr(widget, "raw", None)
        if isinstance(raw, str):
            parts.append(raw)
        elif isinstance(widget, Static) and not hasattr(widget, "raw"):
            parts.append(str(widget.content))
    return "\n".join(parts)


def make_app(tmp_path: Path, project: Path, settings: Settings, script: list, **kw) -> ArchitectApp:
    return ArchitectApp(
        settings=settings,
        root=project,
        store=kw.pop("store", SessionStore(tmp_path / "sessions")),
        client=FakeClient(script),
        **kw,
    )


async def test_send_with_tool_call(tmp_path: Path, project: Path, settings: Settings) -> None:
    script = [
        tool_reply(("c1", "read", '{"path": "src/app.py"}'), content="Let me look."),
        text_reply("It returns **42**.", thinking="checking the file"),
    ]
    app = make_app(tmp_path, project, settings, script)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press(*"hi", "ctrl+j", *"see @main.py", "enter")
        await wait_for(pilot, lambda: len(app.agent.conversation) == 4 and not app.is_streaming)
        text = chat_text(app)
        assert "Let me look." in text  # pre-tool text stays visible
        assert "→ read src/app.py  ✓" in text
        assert "It returns **42**." in text
        assert "checking the file" in text
        sent = app.agent.llm.client.calls[0]["messages"]  # type: ignore[attr-defined]
        assert "print('hello')" in sent[-1]["content"]  # @main.py attached
        assert sent[-1]["content"].endswith("hi\nsee @main.py")
        assert "session" in app.sub_title


async def test_secret_reference_not_attached(
    tmp_path: Path, project: Path, settings: Settings
) -> None:
    app = make_app(tmp_path, project, settings, [text_reply("ok")])
    async with app.run_test() as pilot:
        app.submit("check @.env please")
        await wait_for(pilot, lambda: not app.is_streaming and len(app.agent.conversation) == 2)
        sent = json.dumps(app.agent.llm.client.calls[0]["messages"])  # type: ignore[attr-defined]
        assert "sk-secret" not in sent
        assert "@.env not attached" in chat_text(app)


async def test_question_modal_answer(tmp_path: Path, project: Path, settings: Settings) -> None:
    script = [
        tool_reply(("q1", "question", json.dumps({"question": "Which?", "options": ["A", "B"]}))),
        text_reply("Going with B."),
    ]
    app = make_app(tmp_path, project, settings, script)
    async with app.run_test() as pilot:
        app.submit("do it")
        await wait_for(pilot, lambda: isinstance(app.screen, QuestionModal))
        await pilot.press("2")
        await wait_for(pilot, lambda: not app.is_streaming)
        assert app.agent.conversation.messages[2]["content"] == "User answer: B"


async def test_cancel_while_question_open(
    tmp_path: Path, project: Path, settings: Settings
) -> None:
    script = [tool_reply(("q1", "question", '{"question": "Which?"}'))]
    app = make_app(tmp_path, project, settings, script)
    async with app.run_test() as pilot:
        app.submit("do it")
        await wait_for(pilot, lambda: isinstance(app.screen, QuestionModal))
        await pilot.press("ctrl+c")
        await wait_for(pilot, lambda: not app.is_streaming)
        assert not isinstance(app.screen, QuestionModal)
        assert [m["role"] for m in app.agent.conversation.messages] == ["user"]
        assert "cancelled" in chat_text(app)


async def test_resume_renders_and_continues(
    tmp_path: Path, project: Path, settings: Settings
) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = store.create(root_path=project, model="m", base_url="u")
    call = {
        "id": "a",
        "type": "function",
        "function": {"name": "glob", "arguments": '{"pattern": "*"}'},
    }
    session.message({"role": "user", "content": "old question"}, display_content="old question")
    session.message({"role": "assistant", "content": None, "tool_calls": [call]})
    session.message({"role": "tool", "tool_call_id": "a", "content": "main.py"})
    session.message({"role": "assistant", "content": "old answer"})
    resumed = store.load(session.path)

    # No usage chunk: v1 crashed estimating tokens over a None-content message here.
    app = make_app(
        tmp_path,
        project,
        settings,
        [text_reply("new answer", usage=False)],
        store=store,
        resume=resumed,
    )
    async with app.run_test() as pilot:
        text = chat_text(app)
        assert "old question" in text and "old answer" in text and "→ glob *" in text
        app.submit("follow up")
        await wait_for(pilot, lambda: not app.is_streaming and len(app.agent.conversation) == 6)
        sent = app.agent.llm.client.calls[0]["messages"]  # type: ignore[attr-defined]
        assert [m["role"] for m in sent] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
        ]
        assert app.agent.session.id in app.sub_title


async def test_new_chat_and_help(tmp_path: Path, project: Path, settings: Settings) -> None:
    app = make_app(tmp_path, project, settings, [text_reply("first")])
    async with app.run_test() as pilot:
        app.submit("hello")
        await wait_for(pilot, lambda: not app.is_streaming and len(app.agent.conversation) == 2)
        app.submit("/new")
        await pilot.pause()
        assert len(app.agent.conversation) == 0
        app.submit("/help")
        await pilot.pause()
        assert "Ctrl+B" in chat_text(app)


async def test_tab_completes_file_reference(
    tmp_path: Path, project: Path, settings: Settings
) -> None:
    app = make_app(tmp_path, project, settings, [])
    async with app.run_test() as pilot:
        await wait_for(pilot, lambda: bool(app.files.snapshot()))
        await pilot.press(*"look at @ap", "tab")
        assert app.chat_input.text == "look at @src/app.py "


async def test_preferences_persist(tmp_path: Path, project: Path, settings: Settings) -> None:
    config = ConfigFile(tmp_path / "config.json")
    app = make_app(tmp_path, project, settings, [], config_file=config)
    async with app.run_test() as pilot:
        assert not (tmp_path / "config.json").exists()  # startup writes nothing
        await pilot.press("ctrl+b")
        app.theme = "textual-dark"
        await pilot.pause()
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved == {"show_sidebar": False, "theme": "textual-dark"}


async def test_ctrl_c_twice_quits(tmp_path: Path, project: Path, settings: Settings) -> None:
    app = make_app(tmp_path, project, settings, [])
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        assert app.is_running
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.return_code == 0


async def test_quit_with_question_open_does_not_hang(
    tmp_path: Path, project: Path, settings: Settings
) -> None:
    script = [tool_reply(("q1", "question", '{"question": "Which?", "options": ["A"]}'))]
    app = make_app(tmp_path, project, settings, script)
    async with app.run_test() as pilot:
        app.submit("do it")
        await wait_for(pilot, lambda: isinstance(app.screen, QuestionModal))
        await pilot.press("ctrl+q")
        await pilot.pause()
    workers = [w for w in app.workers if w.group == "agent"]
    assert all(w.is_finished for w in workers)


async def test_digits_typed_in_answer_box_are_text(
    tmp_path: Path, project: Path, settings: Settings
) -> None:
    script = [
        tool_reply(("q1", "question", '{"question": "How many?", "options": ["one", "two"]}')),
        text_reply("ok"),
    ]
    app = make_app(tmp_path, project, settings, script)
    async with app.run_test() as pilot:
        app.submit("count")
        await wait_for(pilot, lambda: isinstance(app.screen, QuestionModal))
        await pilot.press("tab", "tab", *"12", "enter")  # option-1 → option-2 → input
        await wait_for(pilot, lambda: not app.is_streaming)
        assert app.agent.conversation.messages[2]["content"] == "User answer: 12"
