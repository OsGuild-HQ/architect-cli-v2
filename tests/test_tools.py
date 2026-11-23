from __future__ import annotations

import threading
from pathlib import Path

import pytest

from architect.files import AccessPolicy, ProjectFiles
from architect.tools.executor import ToolExecutor, ToolLimits


@pytest.fixture
def executor(policy: AccessPolicy, files: ProjectFiles, event: threading.Event) -> ToolExecutor:
    return ToolExecutor(
        policy=policy, files=files, cancelled=event, ask=lambda q, o: o[0] if o else "free text"
    )


def test_read_basic(executor: ToolExecutor) -> None:
    result = executor.execute("read", {"path": "src/app.py"})
    assert not result.is_error
    assert result.content.startswith("src/app.py (lines 1-2 of 2)")
    assert "     2      return 42" in result.content


def test_read_refuses_secrets_and_ignored(executor: ToolExecutor) -> None:
    secret = executor.execute("read", {"path": ".env"})
    assert secret.is_error
    assert "sk-secret" not in secret.content
    assert "secret" in secret.content
    assert executor.execute("read", {"path": "build/out.py"}).is_error
    assert executor.execute("read", {"path": "../../etc/passwd"}).is_error


def test_read_offset_on_file_larger_than_old_limit(executor: ToolExecutor, project: Path) -> None:
    """v1 refused any file over 80 KB, even when asked for a small window."""
    (project / "big.py").write_text("".join(f"line {i}\n" for i in range(20_000)))  # ~200 KB
    result = executor.execute("read", {"path": "big.py", "offset": 15_000, "limit": 3})
    assert not result.is_error
    assert "lines 15001-15003 of 20000" in result.content
    assert "line 15000" in result.content
    assert "offset=15003" in result.content


def test_read_output_cap(
    policy: AccessPolicy, files: ProjectFiles, event: threading.Event, project: Path
) -> None:
    (project / "wide.txt").write_text(("y" * 100 + "\n") * 50)
    ex = ToolExecutor(
        policy=policy, files=files, cancelled=event, limits=ToolLimits(read_max_output_chars=500)
    )
    result = ex.execute("read", {"path": "wide.txt"})
    assert "more available" in result.content
    assert len(result.content) < 800


def test_read_bad_args(executor: ToolExecutor) -> None:
    assert executor.execute("read", {}).is_error
    bad = executor.execute("read", {"path": "main.py", "offset": "abc"})
    assert bad.is_error and "integer" in bad.content
    assert executor.execute("read", {"path": "main.py", "offset": 99}).is_error


def test_glob(executor: ToolExecutor) -> None:
    result = executor.execute("glob", {"pattern": "**/*.py"})
    assert result.content.splitlines() == ["main.py", "src/app.py"]
    assert ".env" not in executor.execute("glob", {"pattern": "*"}).content.splitlines()
    assert "no files" in executor.execute("glob", {"pattern": "*.rs"}).content


def test_grep(executor: ToolExecutor) -> None:
    result = executor.execute("grep", {"pattern": r"return \d+"})
    assert result.content == "src/app.py:2:    return 42"
    only_md = executor.execute("grep", {"pattern": "Demo", "include": "*.md"})
    assert only_md.content == "README.md:1:# Demo"
    assert executor.execute("grep", {"pattern": "("}).is_error


def test_grep_secret_only_in_example(executor: ToolExecutor) -> None:
    hits = executor.execute("grep", {"pattern": "API_KEY"}).content.splitlines()
    assert hits == [".env.example:1:API_KEY="]


def test_grep_max_results(executor: ToolExecutor, project: Path, files: ProjectFiles) -> None:
    (project / "many.txt").write_text("hit\n" * 10)
    files.invalidate()
    result = executor.execute("grep", {"pattern": "hit", "max_results": 3})
    assert result.content.count("many.txt:") == 3
    assert "truncated at 3" in result.content


def test_grep_catastrophic_regex_times_out(
    policy: AccessPolicy, files: ProjectFiles, event: threading.Event, project: Path
) -> None:
    (project / "evil.txt").write_text("a " * 3000 + "!\n")
    files.invalidate()
    ex = ToolExecutor(
        policy=policy, files=files, cancelled=event, limits=ToolLimits(grep_search_timeout=0.2)
    )
    result = ex.execute("grep", {"pattern": r"(\w+\s?)+$", "include": "evil.txt"})
    assert "too slow" in result.content


def test_question(executor: ToolExecutor) -> None:
    assert executor.execute("question", {"question": "A or B?", "options": ["A", "B"]}).content == (
        "User answer: A"
    )
    assert executor.execute("question", {"question": "free?"}).content == "User answer: free text"


def test_question_skipped_and_unavailable(
    policy: AccessPolicy, files: ProjectFiles, event: threading.Event
) -> None:
    skipped = ToolExecutor(policy=policy, files=files, cancelled=event, ask=lambda q, o: None)
    assert "skipped" in skipped.execute("question", {"question": "?"}).content
    none = ToolExecutor(policy=policy, files=files, cancelled=event)
    assert none.execute("question", {"question": "?"}).is_error


def test_unknown_tool_and_cancelled(executor: ToolExecutor, event: threading.Event) -> None:
    assert executor.execute("rm", {}).is_error
    event.set()
    assert "cancelled" in executor.execute("read", {"path": "main.py"}).content
