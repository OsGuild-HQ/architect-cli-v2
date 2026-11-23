"""Chat history with token estimates, repair, and turn-aware compaction.

Messages are plain dicts in OpenAI chat format. The list kept here mirrors the
session file exactly (every append is also written there), which is what lets
`covered_message_count` and rollbacks be replayed on resume.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from typing import Any

Message = dict[str, Any]


def estimate_tokens(value: object) -> int:
    """Rough token count (4 chars per token). Accepts None and non-strings."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value) // 4
    if isinstance(value, list):  # content parts
        return sum(
            estimate_tokens(part.get("text") if isinstance(part, dict) else part) for part in value
        )
    return len(str(value)) // 4


def message_tokens(message: Message) -> int:
    total = estimate_tokens(message.get("content"))
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        total += estimate_tokens(function.get("name")) + estimate_tokens(function.get("arguments"))
    return total + 4  # per-message overhead


def tool_call_ids(message: Message) -> list[str]:
    calls = message.get("tool_calls") or []
    return [str(c.get("id") or "") for c in calls if isinstance(c, dict)]


def repair(messages: Iterable[Message]) -> list[Message]:
    """Return a copy that every OpenAI-compatible API will accept.

    - An assistant message with `tool_calls` must be followed by a result for
      every call id. Incomplete groups (e.g. interrupted by a cancel) are
      dropped; any text the assistant wrote is kept as a plain message.
    - Tool results not answering the immediately preceding tool calls are dropped.
    - `content: None` is only allowed alongside tool calls.
    """
    source = list(messages)
    result: list[Message] = []
    i = 0
    while i < len(source):
        message = dict(source[i])
        role = message.get("role")
        if role == "tool":
            i += 1  # orphan: its tool-call message was dropped or never existed
            continue
        if role == "assistant" and message.get("tool_calls"):
            ids = tool_call_ids(message)
            j = i + 1
            results: dict[str, Message] = {}
            while j < len(source) and source[j].get("role") == "tool":
                tool_id = str(source[j].get("tool_call_id") or "")
                if tool_id in ids and tool_id not in results:
                    results[tool_id] = dict(source[j])
                j += 1
            if ids and all(ids) and len(set(ids)) == len(ids) and set(results) == set(ids):
                result.append(message)
                result.extend(results[tool_id] for tool_id in ids)
            elif (message.get("content") or "").strip():
                message.pop("tool_calls", None)
                result.append(message)
            i = j
            continue
        if message.get("content") is None:
            message["content"] = ""
        result.append(message)
        i += 1
    return result


class Conversation:
    """Thread-safe message history plus a running summary of compacted turns."""

    def __init__(self, messages: Iterable[Message] = (), summary: str = "") -> None:
        self._lock = threading.RLock()
        self._messages: list[Message] = [dict(m) for m in messages]
        self.summary = summary

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    @property
    def messages(self) -> list[Message]:
        with self._lock:
            return [dict(m) for m in self._messages]

    def append(self, message: Message) -> None:
        with self._lock:
            self._messages.append(dict(message))

    def truncate(self, length: int) -> None:
        with self._lock:
            del self._messages[max(0, length) :]

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self.summary = ""

    def drop_first(self, count: int) -> None:
        with self._lock:
            del self._messages[: max(0, count)]

    def estimated_tokens(self, system_prompt: str = "") -> int:
        with self._lock:
            return (
                estimate_tokens(system_prompt)
                + estimate_tokens(self.summary)
                + sum(message_tokens(m) for m in self._messages)
            )

    def api_messages(self, system_prompt: str) -> list[Message]:
        """System prompt + summary + repaired history, ready to send."""
        with self._lock:
            history = repair(self._messages)
            summary = self.summary
        head: list[Message] = [{"role": "system", "content": system_prompt}]
        if summary:
            head.append({"role": "system", "content": f"Conversation summary so far:\n\n{summary}"})
        return head + history

    def compaction_split(self, keep_tokens: int) -> int | None:
        """How many leading messages to summarize, or None if nothing can go.

        Splits only where a user message starts a turn, so an assistant's tool
        calls and their results always stay together. The latest turn is kept
        even if it alone exceeds `keep_tokens`.
        """
        with self._lock:
            messages = list(self._messages)
        turn_starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not turn_starts:
            return None
        keep_from = turn_starts[-1]
        kept = sum(message_tokens(m) for m in messages[keep_from:])
        for start in reversed(turn_starts[:-1]):
            turn = sum(message_tokens(m) for m in messages[start:keep_from])
            if kept + turn > keep_tokens:
                break
            kept += turn
            keep_from = start
        return keep_from or None

    def apply_compaction(self, summary: str, covered: int) -> None:
        with self._lock:
            self.summary = summary
            del self._messages[:covered]


def transcript(messages: Iterable[Message], tool_result_chars: int = 2000) -> str:
    """Readable transcript for the summarizer, with long tool output clipped."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "tool":
            if len(content) > tool_result_chars:
                content = content[:tool_result_chars] + "\n…(clipped)"
            lines.append(f"tool result:\n{content}")
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.append(f"{role} called {function.get('name')}({function.get('arguments')})")
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)
