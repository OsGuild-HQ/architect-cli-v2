"""Project file access: listing, ignore rules, secret protection, safe reads.

Every path the model or the user can reach goes through `AccessPolicy.check`,
so the tree, `@file` references and the tools all agree on what is visible.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

import pathspec

from architect.logs import log

EXT_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "jsx",
    ".tsx": "tsx", ".rs": "rust", ".go": "go", ".c": "c", ".cpp": "cpp", ".h": "c",
    ".hpp": "cpp", ".cs": "csharp", ".fs": "fsharp", ".java": "java", ".scala": "scala",
    ".rb": "ruby", ".php": "php", ".sh": "bash", ".zsh": "bash", ".lua": "lua",
    ".dart": "dart", ".ex": "elixir", ".exs": "elixir", ".r": "r", ".jl": "julia",
    ".m": "objective-c", ".md": "markdown", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml", ".toml": "toml", ".xml": "xml", ".ini": "ini", ".lock": "text",
    ".csv": "csv", ".txt": "text", ".sql": "sql", ".html": "html", ".css": "css",
    ".scss": "scss", ".vue": "vue", ".svelte": "svelte", ".tf": "hcl", ".kt": "kotlin",
    ".swift": "swift", ".proto": "protobuf", ".graphql": "graphql", ".gql": "graphql",
    ".gradle": "gradle",
}  # fmt: skip

# Directories never worth walking when git is unavailable.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", "__pycache__",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox", ".idea",
        ".next", ".nuxt", "dist", "build", "target", ".gradle", ".terraform",
    }
)  # fmt: skip

# File names that usually hold credentials. Matched case-insensitively against
# the file name only.
SECRET_NAME_PATTERNS: tuple[str, ...] = (
    ".env", ".env.*", "*.env", ".envrc", ".netrc", "_netrc", ".npmrc", ".pypirc",
    ".pgpass", ".htpasswd", ".git-credentials", "credentials", "credentials.json",
    "secrets.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.kdbx",
    "*.tfstate", "*.tfstate.*", "*.tfvars", "id_rsa*", "id_dsa*", "id_ecdsa*",
    "id_ed25519*", "service-account*.json",
)  # fmt: skip

# Any path inside these directories is treated as secret.
SECRET_DIRS: frozenset[str] = frozenset({".ssh", ".aws", ".gnupg", ".kube", ".docker"})

# Template files that look like secrets by name but are meant to be shared.
SAFE_SUFFIXES: tuple[str, ...] = (".example", ".sample", ".template", ".dist", ".tmpl")

BINARY_SNIFF_BYTES = 8192


def language_for(path: str | Path) -> str:
    return EXT_MAP.get(PurePosixPath(str(path)).suffix.lower(), "")


def glob_match(rel: str, pattern: str) -> bool:
    """Match a repo-relative POSIX path against a glob.

    `**` spans directories (so `**/*.py` also matches `main.py`). A pattern
    without `/` is matched against the file name too, like `.gitignore`.
    """
    pattern = pattern.strip().removeprefix("./")
    if not pattern:
        return False
    path = PurePosixPath(rel)
    if path.full_match(pattern):
        return True
    return "/" not in pattern and fnmatchcase(path.name, pattern)


def is_secret(rel: str, allow: Sequence[str] = ()) -> bool:
    """True when a repo-relative path looks like it holds credentials."""
    path = PurePosixPath(rel)
    if any(glob_match(rel, pattern) for pattern in allow):
        return False
    name = path.name.lower()
    if any(part.lower() in SECRET_DIRS for part in path.parts[:-1]):
        return True
    if name.endswith(SAFE_SUFFIXES):
        return False
    return any(fnmatchcase(name, pattern) for pattern in SECRET_NAME_PATTERNS)


class IgnoreMatcher:
    """Gitignore semantics (negation, anchoring, nested files) via pathspec.

    `.gitignore` files are loaded lazily per directory and cached. Call
    `reset()` to pick up edits.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._specs: dict[str, pathspec.GitIgnoreSpec | None] = {}

    def reset(self) -> None:
        with self._lock:
            self._specs.clear()

    def _spec_for(self, directory: str) -> pathspec.GitIgnoreSpec | None:
        with self._lock:
            if directory in self._specs:
                return self._specs[directory]
        lines: list[str] = []
        base = self.root / directory if directory else self.root
        sources = [base / ".gitignore"]
        if not directory:
            sources.insert(0, self.root / ".git" / "info" / "exclude")
        for source in sources:
            try:
                lines.extend(source.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                continue
        spec = pathspec.GitIgnoreSpec.from_lines(lines) if lines else None
        with self._lock:
            self._specs[directory] = spec
        return spec

    def _decide(self, rel: str, is_dir: bool) -> bool | None:
        """Last matching rule across the .gitignore chain, or None if no rule matched."""
        parts = PurePosixPath(rel).parts
        decision: bool | None = None
        for depth in range(len(parts)):
            directory = "/".join(parts[:depth])
            spec = self._spec_for(directory)
            if spec is None:
                continue
            sub = "/".join(parts[depth:]) + ("/" if is_dir else "")
            result = spec.check_file(sub)
            if result.include is not None:
                decision = result.include
        return decision

    def is_ignored(self, rel: str, is_dir: bool = False) -> bool:
        rel = rel.strip("/")
        if not rel:
            return False
        parts = PurePosixPath(rel).parts
        if ".git" in parts:
            return True
        # Git cannot re-include a file whose parent directory is excluded.
        for depth in range(1, len(parts)):
            if self._decide("/".join(parts[:depth]), is_dir=True):
                return True
        return bool(self._decide(rel, is_dir))


@dataclass(frozen=True)
class Access:
    """Outcome of an access check. `path` is set only when access is allowed."""

    rel: str
    path: Path | None
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.path is not None


class AccessPolicy:
    def __init__(
        self, root: Path, ignore: IgnoreMatcher, allow_secrets: Sequence[str] = ()
    ) -> None:
        self.root = root.resolve()
        self.ignore = ignore
        self.allow_secrets = tuple(allow_secrets)

    def relative(self, path: Path) -> str | None:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return None

    def hidden_reason(self, rel: str, is_dir: bool = False) -> str | None:
        if self.ignore.is_ignored(rel, is_dir=is_dir):
            return "ignored by .gitignore"
        if not is_dir and is_secret(rel, self.allow_secrets):
            return "looks like a secret file (allow it with `allow_secret_files` in config)"
        return None

    def is_visible(self, rel: str) -> bool:
        return self.hidden_reason(rel) is None

    def check(self, ref: str) -> Access:
        """Resolve a user/model supplied path and decide whether it may be read."""
        ref = ref.strip()
        if not ref or "\x00" in ref:
            return Access(ref, None, "empty or invalid path")
        candidate = Path(ref)
        if not candidate.is_absolute():
            candidate = self.root / ref
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:
            return Access(ref, None, f"cannot resolve path: {exc}")
        rel = self.relative(resolved)
        if rel is None or rel == ".":
            return Access(ref, None, "outside the project")
        # Check both the name used and the symlink target, so a link named
        # `notes.txt` pointing at `.env` is still refused.
        lexical = self.relative(Path(os.path.normpath(candidate)))
        for name in (rel, lexical):
            if name is None or name == ".":
                continue
            reason = self.hidden_reason(name)
            if reason:
                return Access(rel, None, reason)
        return Access(rel, resolved)


@dataclass(frozen=True)
class TextRead:
    text: str
    truncated: bool
    size: int


def read_text_prefix(path: Path, max_bytes: int) -> TextRead | None:
    """Read up to `max_bytes` of UTF-8 text.

    Returns None for binary or non-UTF-8 files. A multi-byte character split
    by the size cut is dropped instead of making the whole read fail.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return None
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        return None
    truncated = len(data) > max_bytes
    data = data[:max_bytes]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Only a cut at the very end (at most 3 bytes of a 4-byte char) is acceptable.
        if truncated and exc.start >= len(data) - 3:
            text = data[: exc.start].decode("utf-8", errors="strict")
        else:
            return None
    return TextRead(text=text, truncated=truncated, size=size)


class ProjectFiles:
    """Cached, policy-filtered list of project files.

    `snapshot()` never blocks (safe on the UI thread); `refresh()` rescans and
    should run in a worker. `ensure_fresh()` rescans only when stale.
    """

    def __init__(
        self,
        root: Path,
        policy: AccessPolicy,
        *,
        ttl: float = 10.0,
        max_files: int = 50_000,
    ) -> None:
        self.root = root.resolve()
        self.policy = policy
        self.ttl = ttl
        self.max_files = max_files
        self._lock = threading.Lock()
        self._files: list[str] = []
        self._scanned_at: float | None = None
        self.truncated = False
        self.source = "none"

    def snapshot(self) -> list[str]:
        with self._lock:
            return self._files

    @property
    def is_stale(self) -> bool:
        with self._lock:
            return self._scanned_at is None or time.monotonic() - self._scanned_at > self.ttl

    def ensure_fresh(self) -> list[str]:
        return self.refresh() if self.is_stale else self.snapshot()

    def invalidate(self) -> None:
        with self._lock:
            self._scanned_at = None
        self.policy.ignore.reset()

    def refresh(self) -> list[str]:
        self.policy.ignore.reset()
        files = self._git_files()
        source = "git"
        if files is None:
            files = self._walk_files()
            source = "walk"
        truncated = len(files) > self.max_files
        visible = sorted(f for f in files[: self.max_files] if self.policy.is_visible(f))
        with self._lock:
            self._files = visible
            self._scanned_at = time.monotonic()
            self.truncated = truncated
            self.source = source
        return visible

    def _git_files(self) -> list[str] | None:
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=self.root,
                check=True,
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.info("git_ls_files_unavailable root=%s error=%s", self.root, exc)
            return None
        names = result.stdout.decode("utf-8", errors="replace").split("\0")
        # --cached lists tracked files that were deleted from the working tree.
        return [n for n in names if n and (self.root / n).is_file()]

    def _walk_files(self) -> list[str]:
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel_dir = os.path.relpath(dirpath, self.root)
            rel_dir = "" if rel_dir == "." else PurePosixPath(Path(rel_dir)).as_posix()
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in SKIP_DIRS
                and not self.policy.ignore.is_ignored(f"{rel_dir}/{d}" if rel_dir else d, True)
            )
            for name in sorted(filenames):
                found.append(f"{rel_dir}/{name}" if rel_dir else name)
                if len(found) > self.max_files:
                    return found
        return found


def filter_matches(files: Iterable[str], pattern: str) -> list[str]:
    return [f for f in files if glob_match(f, pattern)]
