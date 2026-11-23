# Architect CLI

A terminal AI coding assistant from the [Open-Source Guild](https://osguild.dev), built on
[Textual](https://textual.textualize.io). It talks to **any OpenAI-compatible API**
(Ollama, OpenAI, OpenRouter, vLLM, LM Studio, …), reads your project through a small set of
read-only tools, and keeps resumable sessions.

- Streaming answers, with the model's reasoning shown when the backend provides it
- `@path` references attach files to a message (Tab completes, or pick in the file tree)
- Read-only tools the model can call: `read`, `glob`, `grep`, and `question` (asks you)
- Sessions saved as JSONL; resume with `--continue` or `--resume ID`
- Long chats are compacted automatically into a running summary
- Gitignored files and secrets (`.env`, keys, credentials) are never sent to the model

## Getting started

You need [`uv`](https://docs.astral.sh/uv/) and a model to talk to. You don't need to install
Python yourself: uv downloads Python 3.13 when it needs it.

### 1. Install uv

```bash
# macOS
brew install uv

# macOS / Linux, without Homebrew
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`.

### 2. Install Architect

```bash
git clone https://github.com/sadeeqrabiu/architect-cli-v2.git
cd architect-cli-v2
uv tool install .
uv tool update-shell
```

`uv tool install .` builds the `architect` command into `~/.local/bin`. `uv tool update-shell`
adds that folder to your `PATH`. **Open a new terminal afterwards**, then check that it works:

```bash
architect --version
```

### 3. Connect a model

Architect works with any OpenAI-compatible API. Pick one option.

**Ollama (the default, nothing to configure).** Install [Ollama](https://ollama.com), then:

```bash
ollama serve                      # skip if the Ollama app is already running
ollama pull minimax-m2.5:cloud    # the default model
```

`:cloud` models run on ollama.com and need a free account (`ollama signin`). To stay fully
local, pull a local model instead, e.g. `ollama pull qwen2.5-coder:7b`, and run
`architect --model qwen2.5-coder:7b`.

**OpenAI, OpenRouter, or another hosted API.** Set three environment variables, e.g. in
`~/.zshrc`:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"    # or https://openrouter.ai/api/v1, …
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1-mini"
```

### 4. Start it in a project

```bash
cd path/to/your/project
architect
```

Or pass the folder: `architect path/to/your/project`. Ask a question, type `@` to attach a
file (Tab completes), and press Enter. `/help` lists everything; Ctrl+C twice quits.

When you quit, Architect prints the command to pick up where you left off
(`architect --resume <id>`); `architect --continue` resumes the latest session for the
current project.

### Updating

```bash
cd architect-cli-v2
git pull
uv tool install --force .
```

The installed command is a copy, so local code changes only take effect after this.

## Configuration

Architect reads the model settings from these environment variables:

| Variable | Default |
|---|---|
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` (local Ollama) |
| `OPENAI_API_KEY` | `ollama` |
| `OPENAI_MODEL` | `minimax-m2.5:cloud` |

Other settings live in `~/.config/architect/config.json` (respects `XDG_CONFIG_HOME`).
All keys are optional:

| Key | Default | Meaning |
|---|---|---|
| `model`, `base_url` | see above | Used when the matching env var is not set |
| `max_tokens` | `8192` | Max tokens per model response |
| `max_context_tokens` | `64000` | Context budget; compaction starts at 80% of it |
| `compact_trigger_ratio` / `compact_keep_recent_ratio` | `0.8` / `0.3` | When to compact, and how much recent history to keep verbatim |
| `request_timeout` | `120` | Seconds before an API request times out |
| `tools_enabled` | `true` | Let the model call the read-only tools |
| `max_tool_iterations` | `8` | Tool rounds per message before the model must answer |
| `allow_secret_files` | `[]` | Globs that may be read even though they look like secrets, e.g. `["config/dev.env"]` |
| `show_sidebar`, `theme` | `true`, `"architect"` | Saved automatically when you change them |

The API key is only ever read from `OPENAI_API_KEY`, never from the config file.
Precedence: defaults → config file → environment → command-line flags.

## Usage

```bash
architect                    # current directory
architect path/to/project
architect --continue         # resume the latest session for this project
architect --resume 3f9a      # resume by id (a unique prefix is enough)
architect --list-sessions
architect --model qwen3-coder --no-tools
```

| Key | Action |
|---|---|
| Enter | Send |
| Shift+Enter / Ctrl+J | New line (Shift+Enter only where the terminal reports it) |
| Tab | Complete `@file` |
| Esc | Cancel the current response |
| Ctrl+B | Toggle the file tree |
| Ctrl+L or `/new` | New chat |
| Ctrl+P | Command palette (themes, …) |
| Ctrl+C twice, Ctrl+Q, `/quit` | Quit |

Double-click a message to copy it. `/help` shows this inside the app.

### What the model can see

Everything goes through one access policy, shared by `@file`, the tools and the file tree:

- Only paths inside the project; symlinks are resolved and must stay inside.
- Gitignored paths (including nested `.gitignore` files and negations) are hidden.
- Secret-looking files are refused even when tracked: `.env*`, `*.pem`, `*.key`, `id_rsa*`,
  `.netrc`, `.npmrc`, `credentials*`, `*.tfstate`, anything under `.ssh/`, `.aws/`, … Templates
  such as `.env.example` are allowed. Use `allow_secret_files` for deliberate exceptions.

Remember that whatever *is* attached or read is sent to the configured endpoint, which may be a
hosted service (for example Ollama `:cloud` models).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: architect` | Open a new terminal after `uv tool update-shell`, or run `source ~/.zshenv`. |
| `uv run architect` says `Failed to spawn: architect` | `uv run` only works inside the `architect-cli-v2` folder. Use the installed `architect` command instead. |
| `Connection error. Could not reach http://localhost:11434/v1` | Ollama isn't running: start it with `ollama serve`, or set `OPENAI_BASE_URL` to your provider. |
| `Not found — is the model name right?` | Pull the model (`ollama pull <name>`) or fix `OPENAI_MODEL` / `--model`. |
| "This model doesn't support tools" | Nothing to do: Architect carries on without tools. Use a tool-capable model to let it read files itself. |
| The first launch takes a few seconds | That's a one-time setup after installing; later launches take under a second. |
| Something else | Check the log at `~/.local/share/architect/architect.log`. |

## Files

| Path | Contents |
|---|---|
| `~/.config/architect/config.json` | Settings |
| `~/.local/share/architect/sessions/*.jsonl` | Sessions |
| `~/.local/share/architect/architect.log` | Log |

## Develop

```bash
uv sync                 # app + dev tools into .venv
uv run architect        # run from source
uv run pytest -q        # tests (no network; the API is faked)
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

If you have [Task](https://taskfile.dev): `task check` runs everything CI runs.

```
src/architect/
  cli.py           argument parsing, startup
  config.py        settings sources and the config file
  files.py         access policy, gitignore, secrets, file listing, safe reads
  references.py    @file parsing, completion, attachment
  conversation.py  history, token estimates, repair, compaction split
  sessions.py      JSONL persistence and replay
  llm.py           streaming client with capability fallbacks
  compaction.py    summarizer
  agent.py         the tool loop (worker thread, no UI code)
  tools/           tool schemas and executor
  ui/              Textual app, widgets, question modal, styles
```

## License

MIT — see [LICENSE](LICENSE).
