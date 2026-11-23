"""Settings: defaults, then `config.json`, then environment, then CLI flags."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from architect.logs import log

APP_NAME = "architect"

DEFAULT_MODEL = "minimax-m2.5:cloud"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / APP_NAME


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP_NAME


def config_path() -> Path:
    return config_dir() / "config.json"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def log_path() -> Path:
    return data_dir() / "architect.log"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = field(default=DEFAULT_API_KEY, repr=False)
    max_tokens: int = 8192
    max_context_tokens: int = 64_000
    compact_trigger_ratio: float = 0.80
    compact_keep_recent_ratio: float = 0.30
    request_timeout: float = 120.0
    tools_enabled: bool = True
    max_tool_iterations: int = 8
    # Glob patterns (repo-relative) that may be read even though they look like
    # secrets, e.g. ["config/dev.env"]. Gitignored files stay hidden regardless.
    allow_secret_files: tuple[str, ...] = ()
    show_sidebar: bool = True
    theme: str = "architect"


# Keys that may be stored in config.json. `api_key` is deliberately absent:
# secrets come from the environment only.
FILE_KEYS: frozenset[str] = frozenset(f.name for f in fields(Settings)) - {"api_key"}

ENV_KEYS: Mapping[str, str] = {
    "OPENAI_MODEL": "model",
    "OPENAI_BASE_URL": "base_url",
    "OPENAI_API_KEY": "api_key",
}


def _coerce(name: str, value: Any, default: Any) -> Any:
    """Convert a config value to the type of its default, or raise ValueError."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be true or false")
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(f"{name} must be an integer")
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive")
        return result
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise ValueError(f"{name} must be a number")
        result = float(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive")
        return result
    if isinstance(default, tuple):
        if not isinstance(value, list | tuple) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{name} must be a list of strings")
        return tuple(value)
    if isinstance(default, str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()
    raise ValueError(f"{name} has unsupported type")  # pragma: no cover


def build_settings(
    file_data: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[Settings, list[str]]:
    """Merge config sources. Returns the settings and human-readable warnings."""
    defaults = Settings()
    values: dict[str, Any] = {}
    warnings: list[str] = []

    def apply(name: str, raw: Any, source: str) -> None:
        try:
            values[name] = _coerce(name, raw, getattr(defaults, name))
        except (TypeError, ValueError) as exc:
            warnings.append(f"Ignoring {source} value: {exc}")

    for key, raw in (file_data or {}).items():
        if key in FILE_KEYS:
            apply(key, raw, "config")
        else:
            warnings.append(f"Ignoring unknown config key {key!r}")

    for env_key, name in ENV_KEYS.items():
        raw = (env or {}).get(env_key)
        if raw:
            apply(name, raw, env_key)

    for key, raw in (overrides or {}).items():
        if raw is not None:
            apply(key, raw, f"--{key.replace('_', '-')}")

    settings = replace(defaults, **values)
    if settings.compact_keep_recent_ratio >= settings.compact_trigger_ratio:
        warnings.append(
            "compact_keep_recent_ratio must be below compact_trigger_ratio; using defaults"
        )
        settings = replace(
            settings,
            compact_trigger_ratio=defaults.compact_trigger_ratio,
            compact_keep_recent_ratio=defaults.compact_keep_recent_ratio,
        )
    return settings, warnings


class ConfigFile:
    """A JSON config file that keeps unknown keys and writes atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        self.error: str | None = None

    def load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raw = {}
        except (OSError, json.JSONDecodeError) as exc:
            self.error = f"Could not read {self.path}: {exc}"
            log.warning("config_read_failed path=%s error=%s", self.path, exc)
            raw = {}
        if not isinstance(raw, dict):
            self.error = f"{self.path} must contain a JSON object"
            raw = {}
        self.data = raw
        return dict(raw)

    def update(self, **changes: Any) -> None:
        """Write changed keys. A no-op when nothing actually changed."""
        changed = {k: v for k, v in changes.items() if self.data.get(k) != v}
        if not changed or self.error:
            # Never overwrite a file we could not parse; the user may be editing it.
            return
        self.data.update(changed)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".config-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            Path(tmp).replace(self.path)
        except OSError as exc:
            Path(tmp).unlink(missing_ok=True)
            log.warning("config_write_failed path=%s error=%s", self.path, exc)
