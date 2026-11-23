"""Executes the model's tool calls against the project, read-only."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import regex

from architect.files import BINARY_SNIFF_BYTES, AccessPolicy, ProjectFiles, glob_match
from architect.logs import log

# Blocking callback used by the `question` tool. Returns None when skipped.
QuestionAsker = Callable[[str, list[str]], str | None]


@dataclass(frozen=True)
class ToolLimits:
    read_default_lines: int = 400
    read_max_lines: int = 1000
    read_max_file_bytes: int = 2_000_000
    read_max_output_chars: int = 60_000
    glob_max_results: int = 200
    grep_default_results: int = 50
    grep_max_results: int = 200
    grep_max_file_bytes: int = 1_000_000
    grep_line_display_chars: int = 300
    grep_max_line_chars: int = 10_000
    grep_search_timeout: float = 0.5
    grep_total_timeout: float = 10.0
    question_max_options: int = 6


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


class ToolArgumentError(ValueError):
    pass


def _int_arg(args: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    raw = args.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ToolArgumentError(f"'{key}' must be an integer, got {raw!r}") from None
    return max(low, min(high, value))


def _str_arg(args: dict[str, Any], key: str, required: bool = False) -> str:
    raw = args.get(key)
    value = "" if raw is None else str(raw)
    if required and not value.strip():
        raise ToolArgumentError(f"'{key}' is required")
    return value


class ToolExecutor:
    def __init__(
        self,
        *,
        policy: AccessPolicy,
        files: ProjectFiles,
        cancelled: threading.Event,
        ask: QuestionAsker | None = None,
        limits: ToolLimits | None = None,
    ) -> None:
        self.policy = policy
        self.files = files
        self.cancelled = cancelled
        self.ask = ask
        self.limits = limits or ToolLimits()

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        if self.cancelled.is_set():
            return ToolResult("cancelled by the user", is_error=True)
        handler = {
            "read": self._read,
            "glob": self._glob,
            "grep": self._grep,
            "question": self._question,
        }.get(name)
        if handler is None:
            return ToolResult(f"error: unknown tool {name!r}", is_error=True)
        try:
            return handler(args)
        except ToolArgumentError as exc:
            return ToolResult(f"error: {exc}", is_error=True)
        except Exception as exc:
            log.exception("tool_error tool=%s", name)
            return ToolResult(f"error: {type(exc).__name__}: {exc}", is_error=True)

    # ── read ────────────────────────────────────────────────────────────

    def _read(self, args: dict[str, Any]) -> ToolResult:
        limits = self.limits
        ref = _str_arg(args, "path", required=True).strip()
        offset = _int_arg(args, "offset", 0, 0, 10_000_000)
        limit = _int_arg(args, "limit", limits.read_default_lines, 1, limits.read_max_lines)

        access = self.policy.check(ref)
        if access.path is None:
            return ToolResult(f"error: cannot read {ref!r}: {access.reason}", is_error=True)
        path = access.path
        if not path.is_file():
            return ToolResult(f"error: {ref!r} is not a file in the project", is_error=True)
        size = path.stat().st_size
        if size > limits.read_max_file_bytes:
            return ToolResult(
                f"error: {access.rel!r} is {size:,} bytes (limit {limits.read_max_file_bytes:,}); "
                "use grep to find the relevant lines instead",
                is_error=True,
            )
        raw = path.read_bytes()
        if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
            return ToolResult(f"error: {access.rel!r} is a binary file", is_error=True)
        lines = raw.decode("utf-8", errors="replace").splitlines()

        total = len(lines)
        if total == 0:
            return ToolResult(f"{access.rel} (empty file)")
        if offset >= total:
            return ToolResult(
                f"error: offset {offset} is past the end of {access.rel!r} ({total} lines)",
                is_error=True,
            )
        body: list[str] = []
        used = 0
        end = offset
        for index in range(offset, min(total, offset + limit)):
            line = f"{index + 1:>6}  {lines[index]}"
            if body and used + len(line) + 1 > limits.read_max_output_chars:
                break
            body.append(line)
            used += len(line) + 1
            end = index + 1
        header = f"{access.rel} (lines {offset + 1}-{end} of {total})"
        if end < total:
            header += f" — more available; call read with offset={end}"
        return ToolResult(header + "\n" + "\n".join(body))

    # ── glob ────────────────────────────────────────────────────────────

    def _glob(self, args: dict[str, Any]) -> ToolResult:
        pattern = _str_arg(args, "pattern", required=True).strip()
        cap = self.limits.glob_max_results
        matches: list[str] = []
        for ref in self.files.ensure_fresh():
            if glob_match(ref, pattern):
                matches.append(ref)
                if len(matches) > cap:
                    break
        if not matches:
            return ToolResult(f"(no files match {pattern!r})")
        if len(matches) > cap:
            return ToolResult("\n".join(matches[:cap]) + f"\n\n(truncated at {cap} matches)")
        return ToolResult("\n".join(matches))

    # ── grep ────────────────────────────────────────────────────────────

    def _grep(self, args: dict[str, Any]) -> ToolResult:
        limits = self.limits
        pattern = _str_arg(args, "pattern", required=True)
        include = _str_arg(args, "include").strip()
        max_results = _int_arg(
            args, "max_results", limits.grep_default_results, 1, limits.grep_max_results
        )
        try:
            compiled = regex.compile(pattern)
        except regex.error as exc:
            return ToolResult(f"error: invalid regex {pattern!r}: {exc}", is_error=True)

        files = self.files.ensure_fresh()
        if include:
            files = [f for f in files if glob_match(f, include)]
        deadline = time.monotonic() + limits.grep_total_timeout
        hits: list[str] = []
        stopped: str | None = None

        for ref in files:
            if self.cancelled.is_set():
                stopped = "cancelled"
                break
            if len(hits) >= max_results:
                stopped = f"truncated at {max_results} matches"
                break
            access = self.policy.check(ref)
            if access.path is None:
                continue
            try:
                if access.path.stat().st_size > limits.grep_max_file_bytes:
                    continue
                raw = access.path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
                continue
            for line_no, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if len(line) > limits.grep_max_line_chars:
                    continue  # minified/generated; slow to search and useless to show
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stopped = f"search stopped after {limits.grep_total_timeout:.0f}s"
                    break
                try:
                    found = compiled.search(
                        line, timeout=min(limits.grep_search_timeout, remaining)
                    )
                except TimeoutError:
                    stopped = "pattern too slow (catastrophic backtracking?); simplify the regex"
                    break
                if found:
                    shown = line.rstrip()
                    if len(shown) > limits.grep_line_display_chars:
                        shown = shown[: limits.grep_line_display_chars] + "…"
                    hits.append(f"{ref}:{line_no}:{shown}")
                    if len(hits) >= max_results:
                        break
            if stopped:
                break

        if len(hits) >= max_results and stopped is None:
            stopped = f"truncated at {max_results} matches"
        body = "\n".join(hits) if hits else f"(no matches for {pattern!r})"
        return ToolResult(body + (f"\n\n({stopped})" if stopped else ""))

    # ── question ────────────────────────────────────────────────────────

    def _question(self, args: dict[str, Any]) -> ToolResult:
        text = _str_arg(args, "question", required=True).strip()
        raw_options = args.get("options") or []
        options = (
            [str(o).strip() for o in raw_options if str(o).strip()]
            if isinstance(raw_options, list)
            else []
        )[: self.limits.question_max_options]
        if self.ask is None:
            return ToolResult("error: the question tool is not available here", is_error=True)
        answer = self.ask(text, options)
        if self.cancelled.is_set():
            return ToolResult("cancelled by the user", is_error=True)
        if answer is None or not answer.strip():
            return ToolResult("(user skipped the question)")
        return ToolResult(f"User answer: {answer.strip()}")
