"""Command-line entry point: `architect [path] [options]`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from architect import __version__
from architect.config import ConfigFile, build_settings, config_path, log_path, sessions_dir
from architect.logs import log, setup_logging
from architect.sessions import (
    AmbiguousSessionError,
    LoadedSession,
    SessionNotFoundError,
    SessionStore,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="architect",
        description="Terminal AI coding assistant for any OpenAI-compatible API.",
        epilog=(
            "Environment: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL. "
            f"Config file: {config_path()}"
        ),
    )
    parser.add_argument("path", nargs="?", help="project directory (default: current directory)")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "-c", "--continue", dest="continue_latest", action="store_true",
        help="resume the most recent session for this project",
    )  # fmt: skip
    resume.add_argument(
        "-r", "--resume", metavar="ID", help="resume a session by id (or unique prefix)"
    )
    resume.add_argument(
        "--list-sessions", action="store_true", help="list recent sessions and exit"
    )
    parser.add_argument("-m", "--model", help="model name (overrides OPENAI_MODEL and config)")
    parser.add_argument(
        "--no-tools", action="store_true", help="disable read/glob/grep/question tools"
    )
    parser.add_argument("--version", action="version", version=f"architect {__version__}")
    return parser


def _fail(message: str, code: int = 2) -> None:
    print(f"architect: {message}", file=sys.stderr)
    raise SystemExit(code)


def _resolve_root(raw: str | None) -> Path:
    root = Path(raw or ".").expanduser()
    if not root.exists():
        _fail(f"no such directory: {root}")
    if not root.is_dir():
        _fail(f"not a directory: {root}")
    return root.resolve()


def _print_sessions(store: SessionStore, root: Path | None) -> None:
    sessions = store.list(root, limit=30)
    if not sessions:
        scope = f" for {root}" if root else ""
        print(f"No sessions{scope}.")
        return
    for info in sessions:
        where = "" if root else f"  [{info.root_path}]"
        print(f"{info.id}  {info.created_at:<20}  {info.title}{where}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(log_path())
    store = SessionStore(sessions_dir())

    if args.list_sessions:
        _print_sessions(store, _resolve_root(args.path) if args.path else None)
        return

    resume: LoadedSession | None = None
    root: Path | None = _resolve_root(args.path) if args.path else None
    if args.resume:
        try:
            resume = store.load(store.find(args.resume))
        except AmbiguousSessionError as exc:
            _fail(f"{exc}. Use more characters of the id.")
        except SessionNotFoundError as exc:
            _fail(f"{exc}. Run `architect --list-sessions` to see available ids.")
        except OSError as exc:
            _fail(f"could not read session: {exc}")
        assert resume is not None
        if root is None:
            saved_root = Path(str(resume.meta.get("root_path") or "."))
            root = saved_root if saved_root.is_dir() else _resolve_root(".")
    root = root or _resolve_root(".")
    if args.continue_latest:
        latest = store.latest(root)
        if latest is None:
            _fail(f"no previous session for {root}")
        else:
            resume = store.load(latest)

    config_file = ConfigFile(config_path())
    file_data = config_file.load()
    overrides = {"model": args.model, "tools_enabled": False if args.no_tools else None}
    settings, warnings = build_settings(file_data, os.environ, overrides)
    if config_file.error:
        warnings.insert(0, config_file.error)
    for warning in warnings:
        log.warning("config %s", warning)

    from architect.ui.app import ArchitectApp  # imported late: keeps --help/--version fast

    log.info(
        "start root=%s model=%s base_url=%s resume=%s",
        root,
        settings.model,
        settings.base_url,
        resume.id if resume else None,
    )
    app = ArchitectApp(
        settings=settings,
        root=root,
        store=store,
        config_file=config_file,
        resume=resume,
        warnings=warnings,
    )
    app.run()
    if app.quit_hint:
        print(app.quit_hint)


if __name__ == "__main__":  # pragma: no cover
    main()
