"""The Textual application: transcript, input, file tree, and agent wiring.

Threading model: the agent runs on a worker thread. Text deltas go into a
locked buffer that a UI timer drains (no cross-thread call per token). Every
structural change (new response segment, tool line, notice, finish) is
marshalled with `call_from_thread`, so widgets are only touched on the UI loop.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.theme import Theme
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import DirectoryTree, Footer, Header, Static, TextArea

from architect.agent import Agent, AgentCallbacks, RunOutcome
from architect.config import ConfigFile, Settings
from architect.conversation import Conversation
from architect.files import AccessPolicy, IgnoreMatcher, ProjectFiles
from architect.llm import LLM, ChatClient, ToolCall, make_client
from architect.logs import log
from architect.prompts import SYSTEM_PROMPT
from architect.references import current_query, expand_references, match_files
from architect.sessions import DisplayItem, LoadedSession, Session, SessionStore
from architect.tools.executor import ToolExecutor, ToolResult
from architect.ui.banner import HELP_TEXT, banner
from architect.ui.question_modal import QuestionModal
from architect.ui.widgets import ChatInput, CopyableMarkdown, CopyableStatic, ProjectTree

THEME = Theme(
    name="architect",
    primary="#6cd064",
    secondary="#3f8f3a",
    accent="#6cd064",
    success="#6cd064",
    warning="#e5c07b",
    error="#ff6b6b",
    foreground="#6cd064",
    background="#000000",
    surface="#000000",
    panel="#000000",
    dark=True,
)

SPINNER_FRAMES = "⣾⣽⣻⢿⡿⣟⣯⣷"
TICK_SECONDS = 0.08
MAX_COMPLETIONS = 5
COMMANDS = {"/new": "clear", "/clear": "clear", "/help": "help", "/quit": "quit", "/exit": "quit"}


class StreamBuffer:
    """Text deltas written by the worker thread, drained by the UI timer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._content: list[str] = []
        self._thinking: list[str] = []

    def add_content(self, delta: str) -> None:
        with self._lock:
            self._content.append(delta)

    def add_thinking(self, delta: str) -> None:
        with self._lock:
            self._thinking.append(delta)

    def take(self) -> tuple[str, str]:
        with self._lock:
            content, thinking = "".join(self._content), "".join(self._thinking)
            self._content.clear()
            self._thinking.clear()
        return content, thinking


@dataclass
class Segment:
    """One streamed model response on screen."""

    spinner: Static
    thinking: CopyableStatic
    body: CopyableMarkdown
    has_content: bool = False
    has_thinking: bool = False


class ArchitectApp(App[None]):
    TITLE = "ARCHITECT"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "interrupt", "Cancel / Quit", show=False, priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Files"),
        Binding("ctrl+l", "clear_chat", "New chat"),
    ]

    is_streaming: reactive[bool] = reactive(False)

    def __init__(
        self,
        *,
        settings: Settings,
        root: Path,
        store: SessionStore,
        config_file: ConfigFile | None = None,
        resume: LoadedSession | None = None,
        client: ChatClient | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.root = root.resolve()
        self.store = store
        self.config_file = config_file
        self.startup_warnings = list(warnings or [])
        self.quit_hint: str | None = None
        self.register_theme(THEME)

        self.policy = AccessPolicy(self.root, IgnoreMatcher(self.root), settings.allow_secret_files)
        self.files = ProjectFiles(self.root, self.policy)
        self._buffer = StreamBuffer()
        self._segment: Segment | None = None
        self._last_tool_line: Static | None = None
        self._last_tool_text = ""
        self._status = "thinking"
        self._spinner_index = 0
        self._tick_timer: Timer | None = None
        self._ctrl_c_armed = False
        self._banner: Static | None = None
        self._active_theme = settings.theme

        if resume is not None:
            session = Session(resume.path, resume.id)
            conversation = resume.conversation
            self._resume_display: list[DisplayItem] = resume.display
        else:
            session = store.create(
                root_path=self.root, model=settings.model, base_url=settings.base_url
            )
            conversation = Conversation()
            self._resume_display = []

        self.agent = Agent(
            settings=settings,
            llm=LLM(client if client is not None else make_client(settings), settings),
            conversation=conversation,
            session=session,
            make_executor=self._make_executor,
            callbacks=AgentCallbacks(
                on_status=lambda text: self._from_worker(self._set_status, text),
                on_segment=lambda: self._from_worker(self._start_segment),
                on_content=self._buffer.add_content,
                on_thinking=self._buffer.add_thinking,
                on_tool_call=lambda call, desc: self._from_worker(self._show_tool_call, desc),
                on_tool_result=self._tool_result_from_worker,
                on_notice=lambda text: self._from_worker(self._show_notice, text),
            ),
        )

    # ── layout ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Vertical(id="sidebar"):
                yield ProjectTree(self.root, policy=self.policy, id="tree")
            with Vertical(id="main"):
                yield VerticalScroll(id="chat")
                yield Static("", id="completion-hint", markup=False)
                with Horizontal(id="input-row"):
                    yield ChatInput(
                        id="msg-input",
                        complete=self._complete,
                        placeholder="Ask about your code… (@file to attach, /help)",
                    )
        yield Footer()

    def on_mount(self) -> None:
        if self.get_theme(self.settings.theme) is not None:
            self.theme = self.settings.theme
        else:
            self.theme = "architect"
        self._active_theme = self.theme
        self.theme_changed_signal.subscribe(self, self._on_theme_changed)
        self.query_one("#sidebar").display = self.settings.show_sidebar
        if self._resume_display:
            self._render_display(self._resume_display)
        else:
            self._show_banner()
        for warning in self.startup_warnings:
            self.notify(warning, severity="warning", timeout=6)
        self._update_status()
        self._refresh_files()
        self.chat_input.focus()

    @property
    def chat(self) -> VerticalScroll:
        return self.query_one("#chat", VerticalScroll)

    @property
    def chat_input(self) -> ChatInput:
        return self.query_one("#msg-input", ChatInput)

    @property
    def completion_hint(self) -> Static:
        return self.query_one("#completion-hint", Static)

    # ── thread bridge ───────────────────────────────────────────────────

    def _call_ui(self, callback: Callable[..., Any], *args: Any) -> bool:
        """Run `callback` on the UI loop and wait for it. False if the app is gone."""
        try:
            self.call_from_thread(callback, *args)
            return True
        except RuntimeError:
            return False

    def _from_worker(self, callback: Callable[..., Any], *args: Any) -> None:
        self._call_ui(callback, *args)

    def _tool_result_from_worker(self, call: ToolCall, result: ToolResult) -> None:
        self._from_worker(self._show_tool_result, result)

    def _make_executor(self, cancelled: threading.Event) -> ToolExecutor:
        return ToolExecutor(
            policy=self.policy,
            files=self.files,
            cancelled=cancelled,
            ask=lambda question, options: self._ask_user(question, options, cancelled),
        )

    def _ask_user(
        self, question: str, options: list[str], cancelled: threading.Event
    ) -> str | None:
        """Blocking (worker thread): show the question modal and wait for an answer."""
        answered = threading.Event()
        box: list[str | None] = [None]

        def done(answer: str | None) -> None:
            box[0] = answer
            answered.set()

        def show() -> None:
            self.push_screen(QuestionModal(question, options), done)

        if not self._call_ui(show):
            return None
        while not answered.wait(0.1):
            if cancelled.is_set():
                self._from_worker(self._dismiss_question)
                answered.wait(1.0)
                return None
        return box[0]

    def _dismiss_question(self) -> None:
        if isinstance(self.screen, QuestionModal):
            self.screen.dismiss(None)

    # ── sending ─────────────────────────────────────────────────────────

    @on(ChatInput.Submitted)
    def _on_submitted(self, event: ChatInput.Submitted) -> None:
        self.submit(event.text)

    def submit(self, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        if self.is_streaming:
            self.notify("Still answering — press Esc to cancel first.", severity="warning")
            return
        self.chat_input.load_text("")
        self.completion_hint.update("")

        command = COMMANDS.get(text.lower())
        if command == "clear":
            self.action_clear_chat()
            return
        if command == "help":
            self._hide_banner()
            self._mount(CopyableStatic(HELP_TEXT, classes="message"))
            return
        if command == "quit":
            self._quit()
            return

        prompt, attachments = expand_references(text, self.policy, self.files.snapshot())
        self._hide_banner()
        self._mount(CopyableStatic(_quote(text), classes="message user"))
        for skipped in (a for a in attachments if not a.attached):
            self._mount(
                Static(
                    f"⚠ @{skipped.ref} not attached: {skipped.note}", classes="note", markup=False
                )
            )
        refs = [a.ref for a in attachments if a.attached]

        self.is_streaming = True
        self._status = "thinking"
        self._tick_timer = self.set_interval(TICK_SECONDS, self._tick)
        self._run_agent(prompt, text, refs)

    @work(thread=True, exclusive=True, group="agent", exit_on_error=False)
    def _run_agent(self, prompt: str, display: str, refs: list[str]) -> None:
        try:
            outcome = self.agent.run(prompt, display, refs)
        except Exception as exc:  # a bug, not an API error: keep the UI usable
            log.exception("agent_crashed")
            outcome = RunOutcome(
                status="error", error=f"**Internal error:** {type(exc).__name__}: {exc}"
            )
        self._from_worker(self._finish_run, outcome)

    # ── streaming view (UI thread) ──────────────────────────────────────

    def _mount(self, widget: Widget) -> None:
        follow = self._at_bottom()
        self.chat.mount(widget)
        if follow:
            self.call_after_refresh(self.chat.scroll_end, animate=False)

    def _at_bottom(self) -> bool:
        chat = self.chat
        return chat.max_scroll_y == 0 or chat.scroll_y >= chat.max_scroll_y - 2

    async def _start_segment(self) -> None:
        await self._flush_stream()
        self._finalize_segment()
        segment = Segment(
            spinner=Static("", classes="spinner"),
            thinking=CopyableStatic("", classes="thinking"),
            body=CopyableMarkdown("", classes="message"),
        )
        segment.thinking.display = False
        segment.body.display = False
        self._mount(segment.spinner)
        self._mount(segment.thinking)
        self._mount(segment.body)
        self._segment = segment
        self._render_spinner()

    def _finalize_segment(self) -> None:
        segment = self._segment
        if segment is None:
            return
        segment.spinner.remove()
        if not segment.has_thinking:
            segment.thinking.remove()
        if not segment.has_content:
            segment.body.remove()
        self._segment = None

    async def _flush_stream(self) -> None:
        content, thinking = self._buffer.take()
        segment = self._segment
        if segment is None or not (content or thinking):
            return
        follow = self._at_bottom()
        if thinking:
            segment.has_thinking = True
            segment.thinking.display = True
            segment.thinking.set_text(segment.thinking.raw + thinking)
        if content:
            if not segment.has_content:
                segment.has_content = True
                segment.body.display = True
                segment.spinner.display = False
            await segment.body.append_text(content)
        if follow:
            self.call_after_refresh(self.chat.scroll_end, animate=False)

    async def _tick(self) -> None:
        await self._flush_stream()
        self._spinner_index += 1
        self._render_spinner()

    def _render_spinner(self) -> None:
        if self._segment is not None and self._segment.spinner.display:
            frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            self._segment.spinner.update(f"{frame} {self._status}…")

    def _set_status(self, text: str) -> None:
        self._status = text
        self._render_spinner()

    async def _show_tool_call(self, description: str) -> None:
        await self._flush_stream()
        self._finalize_segment()
        self._last_tool_text = f"→ {description}"
        line = Static(self._last_tool_text, classes="tool-line", markup=False)
        self._last_tool_line = line
        self._mount(line)

    def _show_tool_result(self, result: ToolResult) -> None:
        line = self._last_tool_line
        if line is None:
            return
        text = self._last_tool_text
        if result.is_error:
            line.add_class("error")
            detail = result.content.splitlines()[0] if result.content else "failed"
            line.update(f"{text}  ✗ {detail[:100]}")
        else:
            line.update(f"{text}  ✓")

    def _show_notice(self, text: str) -> None:
        self._mount(Static(f"⚠ {text}", classes="note", markup=False))

    async def _finish_run(self, outcome: RunOutcome) -> None:
        await self._flush_stream()
        self._finalize_segment()
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None
        if outcome.status == "cancelled":
            self._mount(Static("cancelled", classes="note", markup=False))
        elif outcome.status == "error":
            self._mount(
                CopyableMarkdown(outcome.error or "Unknown error", classes="message error-message")
            )
        elif not outcome.text.strip():
            self._mount(
                Static("(the model returned an empty response)", classes="note", markup=False)
            )
        self.is_streaming = False
        self._update_status()
        self.files.invalidate()
        self._refresh_files()
        self.chat_input.focus()

    # ── transcript helpers ──────────────────────────────────────────────

    def _render_display(self, items: list[DisplayItem]) -> None:
        for item in items:
            if item.kind == "user":
                self.chat.mount(CopyableStatic(_quote(item.text), classes="message user"))
            elif item.kind == "assistant":
                if item.thinking:
                    self.chat.mount(CopyableStatic(item.thinking, classes="thinking"))
                if item.text:
                    self.chat.mount(CopyableMarkdown(item.text, classes="message"))
            elif item.kind == "tool":
                self.chat.mount(Static(f"→ {item.text}", classes="tool-line", markup=False))
        self.chat.mount(
            Static(f"resumed session {self.agent.session.id}", classes="note", markup=False)
        )
        self.call_after_refresh(self.chat.scroll_end, animate=False)

    def _show_banner(self) -> None:
        self._banner = Static(self._banner_text(), classes="message", markup=False)
        self.chat.mount(self._banner)

    def _banner_text(self) -> str:
        width = self.size.width if self.is_mounted else 120
        return banner(width, self.settings.model, str(self.root))

    def _hide_banner(self) -> None:
        if self._banner is not None:
            self._banner.remove()
            self._banner = None

    def on_resize(self) -> None:
        if self._banner is not None:
            self._banner.update(self._banner_text())

    def _update_status(self) -> None:
        estimate = self.agent.conversation.estimated_tokens(SYSTEM_PROMPT)
        parts = [self.settings.model, f"ctx ~{estimate:,}/{self.settings.max_context_tokens:,}"]
        if self.agent.total_usage is not None:
            parts.append(f"{self.agent.total_usage.total:,} tokens used")
        parts.append(f"session {self.agent.session.id}")
        self.sub_title = " · ".join(parts)

    # ── files & completion ──────────────────────────────────────────────

    @work(thread=True, exclusive=True, group="files", exit_on_error=False)
    def _refresh_files(self) -> None:
        self.files.refresh()

    def _completion(self) -> tuple[tuple[int, int], tuple[int, int], list[str]] | None:
        row, column = self.chat_input.cursor_location
        line = self.chat_input.get_text_range((row, 0), (row, column))
        query = current_query(line)
        if query is None:
            return None
        start, text = query
        return (
            (row, start),
            (row, column),
            match_files(text, self.files.snapshot(), MAX_COMPLETIONS),
        )

    def _complete(self) -> bool:
        found = self._completion()
        if found is None or not found[2]:
            return False
        start, end, matches = found
        replacement = f"@{matches[0]} "
        self.chat_input.replace(replacement, start, end)
        self.chat_input.move_cursor((start[0], start[1] + len(replacement)))
        return True

    @on(TextArea.Changed, "#msg-input")
    @on(TextArea.SelectionChanged, "#msg-input")
    def _update_completion_hint(self) -> None:
        found = self._completion()
        if found is None:
            self.completion_hint.update("")
        elif found[2]:
            self.completion_hint.update("Tab: " + "  ".join(f"@{m}" for m in found[2]))
        elif not self.files.snapshot():
            self.completion_hint.update("indexing files…")
        else:
            self.completion_hint.update("no matching files")

    @on(DirectoryTree.FileSelected)
    def _on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        rel = self.policy.relative(Path(event.path)) or str(event.path)
        line = self.chat_input.get_text_range(
            (self.chat_input.cursor_location[0], 0), self.chat_input.cursor_location
        )
        prefix = "" if not line or line.endswith((" ", "\n")) else " "
        self.chat_input.insert(f"{prefix}@{rel} ")
        self.chat_input.focus()

    # ── actions ─────────────────────────────────────────────────────────

    def action_cancel(self) -> None:
        if self.is_streaming:
            self.agent.cancel()
            self._set_status("cancelling")

    def action_interrupt(self) -> None:
        if self.is_streaming:
            self.action_cancel()
            return
        if self._ctrl_c_armed:
            self._quit()
            return
        self._ctrl_c_armed = True
        self.notify("Press Ctrl+C again to quit", timeout=1.5)
        self.set_timer(1.5, self._disarm_ctrl_c)

    def _disarm_ctrl_c(self) -> None:
        self._ctrl_c_armed = False

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display
        if self.config_file is not None:
            self.config_file.update(show_sidebar=bool(sidebar.display))

    def action_clear_chat(self) -> None:
        if self.is_streaming:
            self.notify(
                "Cancel the current response before starting a new chat.", severity="warning"
            )
            return
        if len(self.agent.conversation) or self.agent.conversation.summary:
            self.agent.clear()
        self.chat.remove_children()
        self._banner = None
        self._segment = None
        self._show_banner()
        self._update_status()

    async def action_quit(self) -> None:
        self._quit()

    def on_unmount(self) -> None:
        # However the app exits, stop the worker: an open question or a stream
        # would otherwise keep its thread (and the process) alive.
        self.agent.cancel()

    def _quit(self) -> None:
        if self.is_streaming:
            self.agent.cancel()
        if self.agent.session.exists:
            self.quit_hint = f"Resume with: architect --resume {self.agent.session.id}"
        self.exit()

    def _on_theme_changed(self, theme: Theme) -> None:
        # The signal also fires for the theme applied at startup; only persist real changes.
        if theme.name == self._active_theme:
            return
        self._active_theme = theme.name
        if self.config_file is not None:
            self.config_file.update(theme=theme.name)


def _quote(text: str) -> str:
    return "\n".join(f"› {line}" if line else "›" for line in text.splitlines())
