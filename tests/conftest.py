"""Shared fixtures: a scripted fake OpenAI client and a throwaway project."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import openai
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from architect.agent import Agent, AgentCallbacks
from architect.config import Settings
from architect.conversation import Conversation
from architect.files import AccessPolicy, IgnoreMatcher, ProjectFiles
from architect.llm import LLM
from architect.sessions import Session, SessionStore
from architect.tools.executor import QuestionAsker, ToolExecutor

# ── chunk builders ──────────────────────────────────────────────────────


def _chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, int] | None = None):
    data: dict[str, Any] = {
        "id": "chunk",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        data["choices"] = []
        data["usage"] = usage
    return ChatCompletionChunk.model_validate(data)


def text_reply(text: str, *, thinking: str = "", usage: bool = True) -> list[Any]:
    chunks = []
    if thinking:
        chunks.append(_chunk({"reasoning": thinking}))
    for i in range(0, len(text), 5):
        chunks.append(_chunk({"content": text[i : i + 5]}))
    chunks.append(_chunk({}, finish="stop"))
    if usage:
        chunks.append(
            _chunk({}, usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        )
    return chunks


def tool_reply(
    *calls: tuple[str, str, str], finish: str = "tool_calls", content: str = ""
) -> list[Any]:
    """calls: (id, name, json-arguments). Arguments are streamed in two pieces."""
    chunks = []
    if content:
        chunks.append(_chunk({"content": content}))
    for index, (call_id, name, args) in enumerate(calls):
        half = len(args) // 2
        chunks.append(
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args[:half]},
                        }
                    ]
                }
            )
        )
        chunks.append(
            _chunk({"tool_calls": [{"index": index, "function": {"arguments": args[half:]}}]})
        )
    chunks.append(_chunk({}, finish=finish))
    return chunks


def completion(text: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
    )


class FakeStream:
    def __init__(
        self, chunks: list[Any], before_chunk: Callable[[int], None] | None = None
    ) -> None:
        self.chunks = chunks
        self.closed = False
        self.before_chunk = before_chunk

    def __iter__(self) -> Iterator[Any]:
        for i, chunk in enumerate(self.chunks):
            if self.closed:
                raise RuntimeError("stream closed")
            if self.before_chunk:
                self.before_chunk(i)
            yield chunk

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(f"unexpected API call: {kwargs.get('messages')}")
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, FakeStream):
            item = item(kwargs)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, list):
            return FakeStream(item)
        return item


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.completions = FakeCompletions(script)

        class _Chat:
            completions = self.completions

        self.chat = _Chat()

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


def _http() -> Any:
    """The HTTP library the installed openai SDK uses (httpx2 for SDK 3.x)."""
    try:
        import httpx2

        return httpx2
    except ImportError:  # pragma: no cover - openai 2.x
        import httpx  # pyright: ignore[reportMissingImports]

        return httpx


def bad_request(message: str) -> openai.BadRequestError:
    http = _http()
    request = http.Request("POST", "http://test/v1/chat/completions")
    response = http.Response(400, request=request, json={"error": {"message": message}})
    return openai.BadRequestError(message, response=response, body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_http().Request("POST", "http://test/v1"))


# ── project fixtures ────────────────────────────────────────────────────


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "main.py").write_text("print('hello')\n")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("def run():\n    return 42\n")
    (root / "README.md").write_text("# Demo\n")
    (root / ".env").write_text("API_KEY=sk-secret\n")
    (root / ".env.example").write_text("API_KEY=\n")
    (root / ".gitignore").write_text("build/\n*.log\n!keep.log\n")
    (root / "build").mkdir()
    (root / "build" / "out.py").write_text("x = 1\n")
    (root / "debug.log").write_text("noise\n")
    (root / "keep.log").write_text("kept\n")
    return root.resolve()


@pytest.fixture
def git_project(project: Path) -> Path:
    try:
        subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    return project


@pytest.fixture
def policy(project: Path) -> AccessPolicy:
    return AccessPolicy(project, IgnoreMatcher(project))


@pytest.fixture
def files(project: Path, policy: AccessPolicy) -> ProjectFiles:
    pf = ProjectFiles(project, policy)
    pf.refresh()
    return pf


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


@pytest.fixture
def settings() -> Settings:
    return Settings(model="fake", base_url="http://test/v1", api_key="k", max_tool_iterations=4)


class AgentHarness:
    def __init__(
        self,
        *,
        client: FakeClient,
        settings: Settings,
        project: Path,
        policy: AccessPolicy,
        files: ProjectFiles,
        session: Session,
        ask: QuestionAsker | None = None,
        callbacks: AgentCallbacks | None = None,
    ) -> None:
        self.client = client
        self.session = session
        self.llm = LLM(client, settings)
        self.agent = Agent(
            settings=settings,
            llm=self.llm,
            conversation=Conversation(),
            session=session,
            make_executor=lambda ev: ToolExecutor(
                policy=policy, files=files, cancelled=ev, ask=ask
            ),
            callbacks=callbacks,
        )


@pytest.fixture
def make_agent(
    settings: Settings,
    project: Path,
    policy: AccessPolicy,
    files: ProjectFiles,
    store: SessionStore,
) -> Callable[..., AgentHarness]:
    def factory(script: list[Any], **kwargs: Any) -> AgentHarness:
        session = store.create(root_path=project, model=settings.model, base_url=settings.base_url)
        return AgentHarness(
            client=FakeClient(script),
            settings=kwargs.pop("settings", settings),
            project=project,
            policy=policy,
            files=files,
            session=session,
            **kwargs,
        )

    return factory


@pytest.fixture
def event() -> threading.Event:
    return threading.Event()
