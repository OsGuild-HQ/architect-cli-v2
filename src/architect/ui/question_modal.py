"""Modal used by the `question` tool."""

from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown


class QuestionModal(ModalScreen[str | None]):
    """Ask the user a question; dismisses with the answer or None (skipped)."""

    BINDINGS = [Binding("escape", "skip", "Skip", show=False)]

    def __init__(self, question: str, options: list[str]) -> None:
        super().__init__()
        self.question = question
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="question-dialog"):
            yield Label("The assistant has a question", id="question-title")
            yield Markdown(self.question, id="question-body")
            if self.options:
                with Vertical(id="question-options"):
                    for index, option in enumerate(self.options, start=1):
                        yield Button(
                            f"{index}. {option}",
                            id=f"option-{index}",
                            classes="question-option",
                        )
            placeholder = "Or type an answer" if self.options else "Type your answer"
            yield Input(
                placeholder=f"{placeholder} — Enter to submit, Esc to skip", id="question-input"
            )
            with Horizontal(id="question-actions"):
                yield Button("Submit", id="question-submit", variant="primary")
                yield Button("Skip", id="question-skip")

    def on_mount(self) -> None:
        # With options, focus the first one so number keys and Enter pick answers;
        # Tab reaches the free-text box. Without options, type straight away.
        if self.options:
            self.query_one("#option-1", Button).focus()
        else:
            self.query_one("#question-input", Input).focus()

    def on_key(self, event: events.Key) -> None:
        # Number keys pick an option unless the user is typing in the text box.
        if isinstance(self.focused, Input) or not event.character or not event.character.isdigit():
            return
        index = int(event.character)
        if 1 <= index <= len(self.options):
            event.stop()
            self.dismiss(self.options[index - 1])

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("option-"):
            self.dismiss(self.options[int(button_id.removeprefix("option-")) - 1])
        elif button_id == "question-submit":
            self._submit()
        elif button_id == "question-skip":
            self.dismiss(None)

    @on(Input.Submitted, "#question-input")
    def _submitted(self) -> None:
        self._submit()

    def _submit(self) -> None:
        text = self.query_one("#question-input", Input).value.strip()
        self.dismiss(text or None)

    def action_skip(self) -> None:
        self.dismiss(None)
