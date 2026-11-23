from __future__ import annotations

from pathlib import Path

import pytest

from architect import cli
from architect.config import sessions_dir
from architect.sessions import SessionStore


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.startswith("architect ")


def test_help_is_not_treated_as_a_path(capsys: pytest.CaptureFixture[str]) -> None:
    """v1 treated `--help` as a project directory."""
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    assert "--resume" in capsys.readouterr().out


def test_missing_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main([str(tmp_path / "nope")])
    assert info.value.code == 2
    assert "no such directory" in capsys.readouterr().err


def test_file_instead_of_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "f.txt").write_text("")
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "f.txt")])
    assert "not a directory" in capsys.readouterr().err


def test_unknown_resume_id(capsys: pytest.CaptureFixture[str]) -> None:
    """v1 silently started a new session under the unknown id."""
    with pytest.raises(SystemExit) as info:
        cli.main(["--resume", "deadbeef"])
    assert info.value.code == 2
    assert "No session with id" in capsys.readouterr().err


def test_continue_without_sessions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--continue", str(tmp_path)])
    assert "no previous session" in capsys.readouterr().err


def test_resume_and_continue_are_exclusive(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--continue", "--resume", "abc"])
    assert "not allowed with" in capsys.readouterr().err


def test_list_sessions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--list-sessions"])
    assert "No sessions" in capsys.readouterr().out

    store = SessionStore(sessions_dir())
    session = store.create(root_path=tmp_path.resolve(), model="m", base_url="u")
    session.message({"role": "user", "content": "how does auth work?"})
    cli.main(["--list-sessions"])
    out = capsys.readouterr().out
    assert session.id in out
    assert "how does auth work?" in out
