"""Small widgets used by the chat UI."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from textual import events
from textual.message import Message
from textual.widgets import DirectoryTree, Markdown, Static, TextArea

from architect.files import AccessPolicy


class CopyableMarkdown(Markdown):
    """Markdown message; clicking copies its raw text."""

    def __init__(self, markdown: str = "", *, classes: str | None = None) -> None:
        super().__init__(markdown, classes=classes)
        self.raw = markdown

    def set_text(self, markdown: str) -> None:
        self.raw = markdown
        self.update(markdown)

    async def append_text(self, delta: str) -> None:
        self.raw += delta
        await self.append(delta)

    def on_click(self, event: events.Click) -> None:
        if (
            event.chain == 2 and self.raw
        ):  # double-click copies, single click keeps selection usable
            event.stop()
            self.app.copy_to_clipboard(self.raw)
            self.app.notify("Copied message to clipboard", timeout=1.5)


class CopyableStatic(Static):
    """Plain-text message (no markup parsing); double-click copies."""

    def __init__(self, text: str = "", *, classes: str | None = None) -> None:
        super().__init__(text, classes=classes, markup=False)
        self.raw = text

    def set_text(self, text: str) -> None:
        self.raw = text
        self.update(text)

    def on_click(self, event: events.Click) -> None:
        if event.chain == 2 and self.raw:
            event.stop()
            self.app.copy_to_clipboard(self.raw)
            self.app.notify("Copied to clipboard", timeout=1.5)


class ChatInput(TextArea):
    """Multi-line input: Enter sends, Shift+Enter / Ctrl+J / Alt+Enter add a newline."""

    NEWLINE_KEYS = frozenset({"shift+enter", "ctrl+j", "alt+enter"})

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, *, complete: Callable[[], bool] | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.complete = complete

    def on_key(self, event: events.Key) -> None:
        # Runs before TextArea's own key handling; prevent_default() skips it.
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
        elif event.key in self.NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
        elif event.key == "tab" and self.complete is not None and self.complete():
            event.stop()
            event.prevent_default()


class ProjectTree(DirectoryTree):
    """File tree that hides `.git` and gitignored paths (secrets stay visible
    but are refused when referenced, with an explanation)."""

    def __init__(self, path: str | Path, *, policy: AccessPolicy, id: str | None = None) -> None:
        super().__init__(path, id=id)
        self.policy = policy

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        visible: list[Path] = []
        for path in paths:
            rel = self.policy.relative(path)
            if rel is None or not self.policy.ignore.is_ignored(rel, is_dir=path.is_dir()):
                visible.append(path)
        return visible
