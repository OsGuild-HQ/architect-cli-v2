from __future__ import annotations

from pathlib import Path

from architect.files import AccessPolicy, ProjectFiles
from architect.references import current_query, expand_references, find_refs, match_files


def test_find_refs_trims_punctuation_for_known_files() -> None:
    assert find_refs("see @main.py, and @src/app.py.", known=["main.py", "src/app.py"]) == [
        "main.py",
        "src/app.py",
    ]
    assert find_refs("email me@example.com") == []
    assert find_refs("@a @a @b") == ["a", "b"]


def test_current_query() -> None:
    assert current_query("look at @sr") == (8, "sr")
    assert current_query("@") == (0, "")
    assert current_query("hi @") == (3, "")
    assert current_query("me@example") is None
    assert current_query("@main.py done") is None


def test_match_files_ranking() -> None:
    files = ["docs/app_notes.md", "src/app.py", "app.py", "lib/happy.py"]
    assert match_files("app", files)[:2] == ["app.py", "src/app.py"]
    assert match_files("", files, limit=2) == files[:2]
    assert match_files("zzz", files) == []


def test_expand_attaches_and_reports(policy: AccessPolicy, files: ProjectFiles) -> None:
    text, attachments = expand_references(
        "explain @src/app.py and @.env and @missing.py", policy, files.snapshot()
    )
    assert "def run()" in text
    assert "```python" in text
    assert text.endswith("explain @src/app.py and @.env and @missing.py")
    assert "API_KEY" not in text
    by_ref = {a.ref: a for a in attachments}
    assert by_ref["src/app.py"].attached
    assert not by_ref[".env"].attached
    assert "secret" in by_ref[".env"].note
    assert "missing.py" not in by_ref  # not a file → plain text


def test_expand_truncates_without_dropping(
    policy: AccessPolicy, files: ProjectFiles, project: Path
) -> None:
    (project / "big.md").write_bytes(("x" * 99 + "é" + "y" * 50).encode())
    text, attachments = expand_references("@big.md", policy, files.ensure_fresh(), max_bytes=100)
    assert attachments[0].attached
    assert attachments[0].note == "truncated"
    assert "x" * 99 in text
    assert "truncated" in text


def test_expand_uses_longer_fence_when_file_has_fences(
    policy: AccessPolicy, files: ProjectFiles, project: Path
) -> None:
    (project / "doc.md").write_text("```py\nx\n```\n")
    text, _ = expand_references("@doc.md", policy, files.ensure_fresh())
    assert "````markdown" in text
