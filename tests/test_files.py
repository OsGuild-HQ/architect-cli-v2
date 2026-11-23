from __future__ import annotations

import os
from pathlib import Path

import pytest

from architect.files import (
    AccessPolicy,
    IgnoreMatcher,
    ProjectFiles,
    glob_match,
    is_secret,
    read_text_prefix,
)


@pytest.mark.parametrize(
    ("rel", "pattern", "expected"),
    [
        ("main.py", "**/*.py", True),  # v1 bug: fnmatch missed top-level files
        ("src/app.py", "**/*.py", True),
        ("src/app.py", "src/**/*.py", True),
        ("src/app.py", "*.py", True),  # no slash → matches basename anywhere
        ("main.py", "src/*.py", False),
        ("README.md", "README*", True),
        ("docs/README.md", "README*", True),
        ("src/app.py", "./src/app.py", True),
    ],
)
def test_glob_match(rel: str, pattern: str, expected: bool) -> None:
    assert glob_match(rel, pattern) is expected


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        (".env", True),
        (".env.local", True),
        ("config/prod.env", True),
        (".env.example", False),
        ("certs/server.pem", True),
        ("deploy/id_ed25519", True),
        ("home/.ssh/config", True),
        ("terraform.tfstate", True),
        ("src/env.py", False),
        ("keyboard.py", False),
        ("SECRETS.yaml", True),
    ],
)
def test_is_secret(rel: str, expected: bool) -> None:
    assert is_secret(rel) is expected


def test_secret_allow_list() -> None:
    assert is_secret("config/dev.env")
    assert not is_secret("config/dev.env", allow=["config/dev.env"])


def test_ignore_matcher_semantics(project: Path) -> None:
    ignore = IgnoreMatcher(project)
    assert ignore.is_ignored("build/out.py")
    assert ignore.is_ignored("build", is_dir=True)
    assert ignore.is_ignored("debug.log")
    assert not ignore.is_ignored("keep.log")  # negation
    assert not ignore.is_ignored("main.py")
    assert ignore.is_ignored(".git/config")


def test_nested_gitignore(project: Path) -> None:
    (project / "src" / ".gitignore").write_text("generated.py\n")
    (project / "src" / "generated.py").write_text("")
    ignore = IgnoreMatcher(project)
    assert ignore.is_ignored("src/generated.py")
    assert not ignore.is_ignored("generated.py")


def test_anchored_pattern(project: Path) -> None:
    (project / ".gitignore").write_text("/out\n")
    ignore = IgnoreMatcher(project)
    assert ignore.is_ignored("out", is_dir=True)
    assert ignore.is_ignored("out/a.txt")
    assert not ignore.is_ignored("src/out/a.txt")


def test_access_policy(policy: AccessPolicy, project: Path) -> None:
    assert policy.check("src/app.py").allowed
    assert policy.check("./src/../main.py").rel == "main.py"
    assert not policy.check("../outside.txt").allowed
    assert "outside" in (policy.check("/etc/hosts").reason or "")
    assert "secret" in (policy.check(".env").reason or "")
    assert "gitignore" in (policy.check("build/out.py").reason or "")
    assert policy.check(".env.example").allowed


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlink_to_secret_is_refused(policy: AccessPolicy, project: Path) -> None:
    (project / "notes.txt").symlink_to(project / ".env")
    assert not policy.check("notes.txt").allowed


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlink_escaping_root_is_refused(
    policy: AccessPolicy, project: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    (project / "link.txt").symlink_to(outside)
    assert policy.check("link.txt").reason == "outside the project"


def test_read_text_prefix_multibyte_cut(tmp_path: Path) -> None:
    """v1 bug: a multi-byte char straddling the cut made the whole file vanish."""
    path = tmp_path / "big.txt"
    path.write_bytes(("a" * 99 + "é" + "tail").encode())
    read = read_text_prefix(path, 100)
    assert read is not None
    assert read.truncated
    assert read.text == "a" * 99


def test_read_text_prefix_rejects_binary_and_bad_utf8(tmp_path: Path) -> None:
    (tmp_path / "bin").write_bytes(b"abc\x00def")
    (tmp_path / "latin1").write_bytes("café".encode("latin-1") + b" more text")
    assert read_text_prefix(tmp_path / "bin", 100) is None
    assert read_text_prefix(tmp_path / "latin1", 100) is None


def test_project_files_walk_filters(files: ProjectFiles) -> None:
    listed = files.snapshot()
    assert "main.py" in listed
    assert "src/app.py" in listed
    assert ".env" not in listed
    assert ".env.example" in listed
    assert "build/out.py" not in listed
    assert "debug.log" not in listed
    assert "keep.log" in listed


def test_project_files_git(git_project: Path) -> None:
    policy = AccessPolicy(git_project, IgnoreMatcher(git_project))
    pf = ProjectFiles(git_project, policy)
    listed = pf.refresh()
    assert pf.source == "git"
    assert "main.py" in listed
    assert ".env" not in listed
    assert "build/out.py" not in listed


def test_project_files_refresh_sees_new_files(files: ProjectFiles, project: Path) -> None:
    (project / "new.py").write_text("")
    assert "new.py" not in files.snapshot()
    files.invalidate()
    assert "new.py" in files.ensure_fresh()


def test_walk_skips_heavy_dirs(project: Path, policy: AccessPolicy) -> None:
    (project / "node_modules" / "pkg").mkdir(parents=True)
    (project / "node_modules" / "pkg" / "index.js").write_text("")
    (project / ".venv").mkdir()
    (project / ".venv" / "x.py").write_text("")
    listed = ProjectFiles(project, policy).refresh()
    assert not any(p.startswith(("node_modules/", ".venv/")) for p in listed)
