# Changelog

## 2.0.0

A rewrite of Architect CLI v1 (`osguild`). The command is now `architect`, and settings and
sessions moved to `~/.config/architect` and `~/.local/share/architect`. v1 sessions are not
imported.

### Fixed
- Crash when finishing a response in a resumed session that used tools (`len(None)`).
- Tool calls and results were dropped from history after each turn, so the model forgot what it
  had just read. They are now kept, and live and resumed sessions see the same history.
- Cancelling during tool calls, or compacting between a tool call and its result, could leave
  history the API rejects with a 400 on every later message. Incomplete tool groups are now
  rolled back, compaction only splits between turns, and history is repaired before each request.
- The `read` tool could read `.env` and other gitignored or secret files; `@file` attached them
  too. Both now go through one access policy.
- Compaction ran a blocking API call on the UI thread, freezing the app.
- Attaching a file over 80 KB failed silently when the size cut split a multi-byte character.
- `read` refused files over 80 KB even when asked for a small window.
- `glob("**/*.py")` missed top-level files.
- Tool calls were ignored when a backend reported `finish_reason: "stop"`, and the last tool
  round could end with an empty answer.
- Resuming re-sent messages a compaction had already summarized.
- The file list never refreshed during a session; outside git it walked `.venv`/`node_modules`
  on the UI thread.
- `--help` was treated as a project path; an unknown `--continue` id silently started a new
  session.
- Shift+Enter was documented but only Ctrl+J worked; text the model wrote before a tool call
  vanished from the screen.
- Token usage was never reported on OpenAI (`stream_options.include_usage` was not sent).
- A regex from the model could hang `grep` (catastrophic backtracking); searches now time out.

### Changed
- Split the single 2,000-line module into focused modules with a UI-free agent loop.
- Gitignore handling uses real gitignore semantics (negation, anchors, nested files).
- Cancelling now also closes the HTTP stream, so it takes effect immediately.
- Sidebar toggle moved from Cmd/Ctrl+R to Ctrl+B, and the command palette is back on Ctrl+P
  (Cmd shortcuts rarely reach terminal apps).
- Clicking a message no longer copies it (that broke text selection); double-click does.
- Default context budget raised from 20k to 64k tokens.

### Added
- `--continue` (latest session for the project), `--resume ID`, `--list-sessions`,
  `--model`, `--no-tools`, `--version`.
- `/help`, `/quit`, Ctrl+L new chat, number keys to pick an answer in the question dialog.
- Config validation with warnings shown at startup; the config file is written atomically
  and keeps unknown keys.
- A test suite (unit and Textual end-to-end against a fake API), ruff, pyright, and CI.
