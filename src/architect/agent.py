"""The agent loop: model turn → tool calls → model turn … → final answer.

Runs on a worker thread. It never touches the UI directly; it reports through
`AgentCallbacks`, which the UI marshals onto its event loop.

Invariants that keep sessions valid:
- Every message added to the conversation is written to the session file, in
  the same order, so replaying the file rebuilds the same history.
- An assistant tool-call message and its results form a group. If a run stops
  mid-group (cancel or error), the group is rolled back in memory and a
  ``rollback`` record is written, so the history never holds a half group.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from architect.compaction import summarize
from architect.config import Settings
from architect.conversation import Conversation, Message
from architect.llm import LLM, ToolCall, Usage, describe_api_error
from architect.logs import log
from architect.prompts import SYSTEM_PROMPT
from architect.sessions import Session, describe_call
from architect.tools.executor import ToolExecutor, ToolResult


def _ignore(*_: Any) -> None:
    return None


@dataclass
class AgentCallbacks:
    """Hooks called from the worker thread. All default to no-ops."""

    on_status: Callable[[str], None] = _ignore  # "thinking" | "compacting" | "running tools"
    on_segment: Callable[[], None] = _ignore  # a new model response is about to stream
    on_content: Callable[[str], None] = _ignore
    on_thinking: Callable[[str], None] = _ignore
    on_tool_call: Callable[[ToolCall, str], None] = _ignore  # call, one-line description
    on_tool_result: Callable[[ToolCall, ToolResult], None] = _ignore
    on_notice: Callable[[str], None] = _ignore


Status = Literal["done", "cancelled", "error"]


@dataclass
class RunOutcome:
    status: Status
    text: str = ""
    error: str | None = None
    usage: Usage | None = None
    hit_tool_limit: bool = False
    compacted: bool = False
    tool_calls: int = 0
    notices: list[str] = field(default_factory=list)


ExecutorFactory = Callable[[threading.Event], ToolExecutor]


class Agent:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLM,
        conversation: Conversation,
        session: Session,
        make_executor: ExecutorFactory,
        callbacks: AgentCallbacks | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.conversation = conversation
        self.session = session
        self.make_executor = make_executor
        self.callbacks = callbacks or AgentCallbacks()
        self.system_prompt = system_prompt
        self.total_usage: Usage | None = None
        self._cancelled = threading.Event()
        self._open_group: int | None = None

    # ── control ─────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Request cancellation; safe to call from any thread."""
        self._cancelled.set()
        self.llm.close_active_stream()

    @property
    def cancel_requested(self) -> bool:
        return self._cancelled.is_set()

    # ── history helpers ─────────────────────────────────────────────────

    def _commit(self, message: Message, **record_extra: Any) -> None:
        self.conversation.append(message)
        self.session.message(message, **record_extra)

    def _rollback_open_group(self) -> None:
        if self._open_group is None:
            return
        length = self._open_group
        self._open_group = None
        self.conversation.truncate(length)
        self.session.write({"type": "rollback", "message_count": length})
        log.info("tool_group_rolled_back session=%s length=%d", self.session.id, length)

    def clear(self) -> None:
        self.conversation.clear()
        self.session.write({"type": "clear"})

    # ── the loop ────────────────────────────────────────────────────────

    def run(self, prompt: str, display: str | None = None, refs: Sequence[str] = ()) -> RunOutcome:
        cancelled = threading.Event()
        self._cancelled = cancelled
        self._open_group = None
        cb = self.callbacks
        outcome = RunOutcome(status="done")

        self._commit(
            {"role": "user", "content": prompt},
            display_content=display if display is not None else prompt,
            file_refs=list(refs),
        )
        try:
            outcome.compacted = self._maybe_compact(cancelled, outcome)
            executor = self.make_executor(cancelled)
            max_iterations = max(0, self.settings.max_tool_iterations)
            for iteration in range(max_iterations + 1):
                if cancelled.is_set():
                    return self._finish_cancelled(outcome)
                final = iteration == max_iterations
                use_tools = (
                    self.settings.tools_enabled
                    and self.llm.tools_supported is not False
                    and not final
                )
                cb.on_segment()
                cb.on_status("thinking")
                result = self.llm.stream(
                    self.conversation.api_messages(self.system_prompt),
                    use_tools=use_tools,
                    on_content=cb.on_content,
                    on_thinking=cb.on_thinking,
                    cancelled=cancelled,
                )
                if result.usage is not None:
                    outcome.usage = (
                        result.usage if outcome.usage is None else outcome.usage + result.usage
                    )
                if result.tools_dropped:
                    self._notice(
                        outcome, "This model doesn't support tools — continuing without them."
                    )
                if result.cancelled or cancelled.is_set():
                    return self._finish_cancelled(outcome)

                if result.tool_calls and use_tools and not result.tools_dropped:
                    self.llm.tools_supported = True
                    self._run_tools(
                        result.content, result.thinking, result.tool_calls, executor, outcome
                    )
                    if cancelled.is_set():
                        return self._finish_cancelled(outcome)
                    continue

                self._commit(
                    {"role": "assistant", "content": result.content}, thinking=result.thinking
                )
                outcome.text = result.content
                outcome.hit_tool_limit = final and max_iterations > 0 and outcome.tool_calls > 0
                if outcome.hit_tool_limit:
                    self._notice(
                        outcome,
                        f"Reached the limit of {max_iterations} tool rounds; "
                        "the model was asked to answer without more tools.",
                    )
                return self._finish(outcome)
            return self._finish(outcome)  # pragma: no cover - loop always returns
        except Exception as exc:
            if cancelled.is_set():
                return self._finish_cancelled(outcome)
            self._rollback_open_group()
            message = describe_api_error(exc, self.settings.base_url)
            log.exception("run_failed session=%s", self.session.id)
            self.session.write({"type": "event", "name": "error", "message": message})
            outcome.status = "error"
            outcome.error = message
            return self._finish(outcome)

    def _run_tools(
        self,
        content: str,
        thinking: str,
        calls: list[ToolCall],
        executor: ToolExecutor,
        outcome: RunOutcome,
    ) -> None:
        cb = self.callbacks
        self._open_group = len(self.conversation)
        self._commit(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [call.to_message() for call in calls],
            },
            thinking=thinking,
        )
        cb.on_status("running tools")
        for call in calls:
            if executor.cancelled.is_set():
                self._rollback_open_group()
                return
            cb.on_tool_call(call, describe_call(call.name, call.arguments))
            args, error = call.parse_arguments()
            result = (
                ToolResult(error, is_error=True) if error else executor.execute(call.name, args)
            )
            if executor.cancelled.is_set():
                self._rollback_open_group()
                return
            cb.on_tool_result(call, result)
            self._commit(
                {"role": "tool", "tool_call_id": call.id, "content": result.content},
                name=call.name,
                is_error=result.is_error,
            )
            outcome.tool_calls += 1
            log.info(
                "tool_call session=%s name=%s error=%s chars=%d",
                self.session.id,
                call.name,
                result.is_error,
                len(result.content),
            )
        self._open_group = None

    def _maybe_compact(self, cancelled: threading.Event, outcome: RunOutcome) -> bool:
        s = self.settings
        if (
            self.conversation.estimated_tokens(self.system_prompt)
            <= s.max_context_tokens * s.compact_trigger_ratio
        ):
            return False
        covered = self.conversation.compaction_split(
            int(s.max_context_tokens * s.compact_keep_recent_ratio)
        )
        if not covered:
            return False
        self.callbacks.on_status("compacting")
        older = self.conversation.messages[:covered]
        try:
            summary = summarize(self.llm.client, s, older, self.conversation.summary)
        except Exception as exc:
            if cancelled.is_set():
                return False
            log.warning("compaction_failed session=%s error=%s", self.session.id, exc)
            self.session.write({"type": "event", "name": "compaction_error", "message": str(exc)})
            self._notice(outcome, f"Couldn't compact the conversation ({exc}); sending it in full.")
            return False
        if cancelled.is_set():
            return False
        self.conversation.apply_compaction(summary, covered)
        self.session.write(
            {"type": "compaction", "summary": summary, "covered_message_count": covered}
        )
        log.info(
            "compaction session=%s covered=%d chars=%d", self.session.id, covered, len(summary)
        )
        return True

    # ── outcomes ────────────────────────────────────────────────────────

    def _notice(self, outcome: RunOutcome, text: str) -> None:
        outcome.notices.append(text)
        self.callbacks.on_notice(text)

    def _finish_cancelled(self, outcome: RunOutcome) -> RunOutcome:
        self._rollback_open_group()
        self.session.write({"type": "event", "name": "cancelled"})
        outcome.status = "cancelled"
        return self._finish(outcome)

    def _finish(self, outcome: RunOutcome) -> RunOutcome:
        if outcome.usage is not None:
            self.total_usage = (
                outcome.usage if self.total_usage is None else self.total_usage + outcome.usage
            )
        return outcome
