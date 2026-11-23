"""Streaming chat completions against any OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai

from architect.config import Settings
from architect.conversation import Message
from architect.logs import log
from architect.tools.schemas import TOOL_SCHEMAS


class ChatClient(Protocol):
    """The slice of `openai.OpenAI` we use; tests provide a fake."""

    @property
    def chat(self) -> Any: ...


def make_client(settings: Settings) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.request_timeout,
        max_retries=2,
    )


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt + other.prompt,
            self.completion + other.completion,
            self.total + other.total,
        )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""

    def parse_arguments(self) -> tuple[dict[str, Any], str | None]:
        """Returns (args, error). Malformed JSON is reported back to the model."""
        if not self.arguments.strip():
            return {}, None
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            return {}, f"error: tool arguments are not valid JSON ({exc.msg})"
        if not isinstance(parsed, dict):
            return {}, "error: tool arguments must be a JSON object"
        return parsed, None

    def to_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class StreamResult:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    cancelled: bool = False
    tools_dropped: bool = False


class StreamCancelledError(Exception):
    pass


def _thinking_delta(delta: Any) -> str:
    for attr in ("reasoning_content", "reasoning", "thinking"):
        value = getattr(delta, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _mentions(exc: Exception, *words: str) -> bool:
    text = str(exc).lower()
    return any(word in text for word in words)


class LLM:
    """Wraps a client with capability fallbacks and cancellable streaming.

    Remembers across calls whether the backend rejected tools or
    `stream_options`, so the fallback costs one failed request at most.
    """

    def __init__(self, client: ChatClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.tools_supported: bool | None = None
        self.usage_supported = True
        self._active: Any = None
        self._active_lock = threading.Lock()

    def close_active_stream(self) -> None:
        """Abort an in-flight stream from another thread (used by cancel)."""
        with self._active_lock:
            stream = self._active
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # closing a half-read HTTP response may raise
                log.debug("stream_close_failed", exc_info=True)

    def _create(self, messages: list[Message], use_tools: bool) -> tuple[Any, bool]:
        """Start a stream, retrying without unsupported options. Returns (stream, tools_dropped)."""
        tools_dropped = False
        while True:
            kwargs: dict[str, Any] = {
                "model": self.settings.model,
                "messages": messages,
                "max_tokens": self.settings.max_tokens,
                "stream": True,
            }
            if self.usage_supported:
                kwargs["stream_options"] = {"include_usage": True}
            if use_tools:
                kwargs["tools"] = TOOL_SCHEMAS
                kwargs["tool_choice"] = "auto"
            try:
                return self.client.chat.completions.create(**kwargs), tools_dropped
            except openai.BadRequestError as exc:
                if self.usage_supported and _mentions(exc, "stream_options", "include_usage"):
                    log.warning("backend_rejected_stream_options error=%s", exc)
                    self.usage_supported = False
                    continue
                if use_tools and _mentions(exc, "tool", "function"):
                    log.warning(
                        "backend_rejected_tools model=%s error=%s", self.settings.model, exc
                    )
                    self.tools_supported = False
                    use_tools = False
                    tools_dropped = True
                    continue
                raise

    def stream(
        self,
        messages: list[Message],
        *,
        use_tools: bool,
        on_content: Callable[[str], None],
        on_thinking: Callable[[str], None],
        cancelled: threading.Event,
    ) -> StreamResult:
        stream, tools_dropped = self._create(messages, use_tools)
        with self._active_lock:
            self._active = stream
        result = StreamResult(tools_dropped=tools_dropped)
        slots: dict[int, ToolCall] = {}
        try:
            for chunk in stream:
                if cancelled.is_set():
                    result.cancelled = True
                    break
                if getattr(chunk, "usage", None) is not None:
                    u = chunk.usage
                    result.usage = Usage(
                        u.prompt_tokens or 0, u.completion_tokens or 0, u.total_tokens or 0
                    )
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    result.finish_reason = choice.finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                thinking = _thinking_delta(delta)
                if thinking:
                    result.thinking += thinking
                    on_thinking(thinking)
                if delta.content:
                    result.content += delta.content
                    on_content(delta.content)
                for tc in delta.tool_calls or []:
                    slot = slots.setdefault(tc.index, ToolCall(id="", name=""))
                    if tc.id:
                        slot.id = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            slot.name = tc.function.name
                        if tc.function.arguments:
                            slot.arguments += tc.function.arguments
        except Exception:
            if cancelled.is_set():
                result.cancelled = True  # closing the stream to cancel raises here
            else:
                raise
        finally:
            with self._active_lock:
                self._active = None
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    log.debug("stream_close_failed", exc_info=True)

        # Detect tool calls by content, not finish_reason: some backends report
        # "stop" even when they return tool calls.
        for index in sorted(slots):
            call = slots[index]
            if not call.name:
                continue
            if not call.id:
                call.id = f"call_{uuid.uuid4().hex[:12]}"
            result.tool_calls.append(call)
        return result


def describe_api_error(exc: Exception, base_url: str) -> str:
    if isinstance(exc, openai.AuthenticationError):
        return "**Authentication error** — check `OPENAI_API_KEY`."
    if isinstance(exc, openai.RateLimitError):
        return "**Rate limit exceeded.** Wait a moment and try again."
    if isinstance(exc, openai.APITimeoutError):
        return f"**Request timed out** talking to {base_url}."
    if isinstance(exc, openai.APIConnectionError):
        return f"**Connection error.** Could not reach {base_url}."
    if isinstance(exc, openai.NotFoundError):
        return f"**Not found** — is the model name right? ({exc})"
    if isinstance(exc, openai.APIStatusError):
        return f"**API error {exc.status_code}:** {exc.message}"
    return f"**Unexpected error:** {type(exc).__name__}: {exc}"
