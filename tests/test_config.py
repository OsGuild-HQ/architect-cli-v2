from __future__ import annotations

import json
from pathlib import Path

from architect.config import ConfigFile, Settings, build_settings


def test_defaults() -> None:
    settings, warnings = build_settings()
    assert settings == Settings()
    assert warnings == []


def test_precedence_file_env_override() -> None:
    settings, _ = build_settings(
        {"model": "from-file", "max_tokens": 1000},
        {"OPENAI_MODEL": "from-env", "OPENAI_API_KEY": "sk-env"},
        {"model": "from-flag"},
    )
    assert settings.model == "from-flag"
    assert settings.max_tokens == 1000
    assert settings.api_key == "sk-env"


def test_api_key_never_read_from_file_and_hidden_in_repr() -> None:
    settings, warnings = build_settings({"api_key": "sk-file"})
    assert settings.api_key == "ollama"
    assert any("api_key" in w for w in warnings)
    assert "sk-" not in repr(build_settings(env={"OPENAI_API_KEY": "sk-x"})[0])


def test_invalid_values_warn_and_fall_back() -> None:
    settings, warnings = build_settings(
        {
            "max_context_tokens": "lots",
            "tools_enabled": "yes",
            "allow_secret_files": "x",
            "bogus": 1,
        }
    )
    assert settings.max_context_tokens == Settings().max_context_tokens
    assert settings.tools_enabled is True
    assert len(warnings) == 4


def test_ratio_sanity() -> None:
    settings, warnings = build_settings(
        {"compact_trigger_ratio": 0.5, "compact_keep_recent_ratio": 0.6}
    )
    assert settings.compact_trigger_ratio == Settings().compact_trigger_ratio
    assert warnings


def test_config_file_roundtrip_keeps_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"custom": 1, "theme": "a"}))
    cfg = ConfigFile(path)
    assert cfg.load() == {"custom": 1, "theme": "a"}
    cfg.update(theme="b", show_sidebar=False)
    assert json.loads(path.read_text()) == {"custom": 1, "theme": "b", "show_sidebar": False}
    assert not list(path.parent.glob(".config-*"))  # temp file cleaned up


def test_config_file_never_overwrites_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{oops")
    cfg = ConfigFile(path)
    assert cfg.load() == {}
    assert cfg.error
    cfg.update(theme="x")
    assert path.read_text() == "{oops"
