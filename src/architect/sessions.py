"""Session persistence as append-only JSONL.

Record types (one JSON object per line):

- ``meta``        session id, project root, model, base URL
- ``message``     a chat message (user / assistant / tool), mirrored into history
- ``compaction``  summary + ``covered_message_count`` leading messages it replaces
- ``rollback``    history truncated to ``message_count`` (cancelled tool group)
- ``clear``       history and summary reset (``/new``)
- ``event``       informational (errors, cancels); not replayed into history

Replaying these in order reproduces the in-memory history exactly.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from architect.conversation import Conversation, Message
from architect.logs import log


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


class SessionNotFoundError(LookupError):
    pass


class AmbiguousSessionError(LookupError):
    def __init__(self, query: str, matches: list[str]) -> None:
        super().__init__(f"Session id {query!r} matches several sessions: {', '.join(matches)}")
        self.matches = matches


@dataclass
class DisplayItem:
    """What the transcript view shows; independent of what the model sees."""

    kind: str  # "user" | "assistant" | "tool" | "note"
    text: str
    thinking: str = ""


@dataclass
class LoadedSession:
    id: str
    path: Path
    meta: dict[str, Any]
    conversation: Conversation
    display: list[DisplayItem] = field(default_factory=list)


@dataclass(frozen=True)
class SessionInfo:
    id: str
    path: Path
    created_at: str
    root_path: str
    title: str


class Session:
    """A session file that is created on the first write."""

    def __init__(self, path: Path, session_id: str, meta: dict[str, Any] | None = None) -> None:
        self.path = path
        self.id = session_id
        self._meta = meta
        self._lock = threading.Lock()

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def write(self, record: dict[str, Any]) -> None:
        record = {"created_at": utc_now(), **record}
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    if self._meta is not None:
                        meta = {"type": "meta", "session_id": self.id, **self._meta}
                        handle.write(
                            json.dumps({"created_at": record["created_at"], **meta}) + "\n"
                        )
                        self._meta = None
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                log.error("session_write_failed path=%s error=%s", self.path, exc)

    def message(self, message: Message, **extra: Any) -> None:
        self.write({"type": "message", **message, **extra})


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def create(self, *, root_path: Path, model: str, base_url: str) -> Session:
        session_id = new_session_id()
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        path = self.directory / f"{stamp}-{session_id}.jsonl"
        meta = {"root_path": str(root_path), "model": model, "base_url": base_url}
        return Session(path, session_id, meta=meta)

    def _files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(self.directory.glob("*.jsonl"))

    @staticmethod
    def id_from_path(path: Path) -> str:
        return path.stem.rsplit("-", 1)[-1]

    def find(self, query: str) -> Path:
        """Exact id first, then unique prefix. Raises on none/ambiguous."""
        query = query.strip()
        files = self._files()
        exact = [p for p in files if self.id_from_path(p) == query]
        if exact:
            return exact[-1]
        prefixed = [p for p in files if query and self.id_from_path(p).startswith(query)]
        if len(prefixed) == 1:
            return prefixed[0]
        if prefixed:
            raise AmbiguousSessionError(query, [self.id_from_path(p) for p in prefixed])
        raise SessionNotFoundError(f"No session with id {query!r} in {self.directory}")

    def list(self, root_path: Path | None = None, limit: int = 20) -> list[SessionInfo]:
        infos: list[SessionInfo] = []
        for path in reversed(self._files()):
            info = self._info(path)
            if info is None:
                continue
            if root_path is not None and info.root_path != str(root_path):
                continue
            infos.append(info)
            if len(infos) >= limit:
                break
        return infos

    def latest(self, root_path: Path) -> Path | None:
        found = self.list(root_path, limit=1)
        return found[0].path if found else None

    def _info(self, path: Path) -> SessionInfo | None:
        meta: dict[str, Any] = {}
        title = ""
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("type") == "meta":
                        meta = record
                    elif record.get("type") == "message" and record.get("role") == "user":
                        title = str(record.get("display_content") or record.get("content") or "")
                        break
        except OSError:
            return None
        if not title:
            return None  # a session nobody typed in is not worth listing
        return SessionInfo(
            id=str(meta.get("session_id") or self.id_from_path(path)),
            path=path,
            created_at=str(meta.get("created_at") or ""),
            root_path=str(meta.get("root_path") or ""),
            title=" ".join(title.split())[:80],
        )

    @staticmethod
    def load(path: Path) -> LoadedSession:
        """Replay a session file into history and a display transcript."""
        lines = path.read_text(encoding="utf-8").splitlines()
        conversation = Conversation()
        display: list[DisplayItem] = []
        meta: dict[str, Any] = {}
        for line_no, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("session_bad_line path=%s line=%d", path, line_no)
                continue
            if not isinstance(record, dict):
                continue
            kind = record.get("type")
            if kind == "meta":
                meta = record
            elif kind == "message":
                _replay_message(record, conversation, display)
            elif kind == "compaction":
                conversation.apply_compaction(
                    str(record.get("summary") or ""),
                    int(record.get("covered_message_count") or 0),
                )
            elif kind == "rollback":
                conversation.truncate(int(record.get("message_count") or 0))
            elif kind == "clear":
                conversation.clear()
                display.clear()
        session_id = str(meta.get("session_id") or SessionStore.id_from_path(path))
        return LoadedSession(session_id, path, meta, conversation, display)


def _replay_message(
    record: dict[str, Any], conversation: Conversation, display: list[DisplayItem]
) -> None:
    role = record.get("role")
    if role == "user":
        content = str(record.get("content") or "")
        conversation.append({"role": "user", "content": content})
        display.append(DisplayItem("user", str(record.get("display_content") or content)))
    elif role == "assistant":
        content = record.get("content")
        tool_calls = record.get("tool_calls") or None
        message: Message = {"role": "assistant", "content": content if content else None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        else:
            message["content"] = content or ""
        conversation.append(message)
        thinking = str(record.get("thinking") or "")
        if content or thinking or not tool_calls:
            display.append(DisplayItem("assistant", str(content or ""), thinking))
        for call in tool_calls or []:
            function = call.get("function") or {}
            display.append(
                DisplayItem("tool", describe_call(function.get("name"), function.get("arguments")))
            )
    elif role == "tool":
        conversation.append(
            {
                "role": "tool",
                "tool_call_id": str(record.get("tool_call_id") or ""),
                "content": str(record.get("content") or ""),
            }
        )


def describe_call(name: object, arguments: object) -> str:
    """Short one-line description of a tool call, e.g. `read src/app.py:1-400`."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else {}
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    name = str(name or "?")
    if name == "read":
        detail = str(args.get("path", ""))
        if args.get("offset") or args.get("limit"):
            detail += f" (offset {args.get('offset', 0)}, limit {args.get('limit', 400)})"
    elif name == "glob":
        detail = str(args.get("pattern", ""))
    elif name == "grep":
        detail = repr(args.get("pattern", ""))
        if args.get("include"):
            detail += f" in {args['include']}"
    elif name == "question":
        detail = str(args.get("question", ""))[:60]
    else:
        detail = ""
    return f"{name} {detail}".strip()
