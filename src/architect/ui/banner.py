"""Empty-state banner and help text."""

from __future__ import annotations

from architect import __version__

WIDE_ART = """\
 ██████╗ ███████╗ ██████╗ ██╗   ██╗██╗██╗     ██████╗
██╔═══██╗██╔════╝██╔════╝ ██║   ██║██║██║     ██╔══██╗
██║   ██║███████╗██║  ███╗██║   ██║██║██║     ██║  ██║
██║   ██║╚════██║██║   ██║██║   ██║██║██║     ██║  ██║
╚██████╔╝███████║╚██████╔╝╚██████╔╝██║███████╗██████╔╝
 ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝

 █████╗ ███████╗███████╗██╗███████╗████████╗ █████╗ ███╗   ██╗████████╗
██╔══██╗██╔════╝██╔════╝██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║╚══██╔══╝
███████║███████╗███████╗██║███████╗   ██║   ███████║██╔██╗ ██║   ██║
██╔══██║╚════██║╚════██║██║╚════██║   ██║   ██╔══██║██║╚██╗██║   ██║
██║  ██║███████║███████║██║███████║   ██║   ██║  ██║██║ ╚████║   ██║
╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝"""

NARROW_ART = """\
┌─┐┌─┐┌─╴╷ ╷╷╷  ╶┬┐
│ │└─┐│╶┐│ │││   ││
└─┘└─┘└─┘└─┘╵└─╴╶┴┘
┌─┐┌─┐┌─┐╷┌─┐╶┬╴┌─┐┌┐╷╶┬╴
├─┤└─┐└─┐│└─┐ │ ├─┤│└┤ │
╵ ╵└─┘└─┘╵└─┘ ╵ ╵ ╵╵ ╵ ╵"""

HELP_TEXT = """\
Keys
  Enter                 send
  Shift+Enter / Ctrl+J  new line (Shift+Enter needs a terminal that reports it)
  Tab                   complete @file
  Esc                   cancel the current response
  Ctrl+B                toggle the file tree
  Ctrl+L                new chat
  Ctrl+P                command palette (themes, …)
  Ctrl+C twice / Ctrl+Q quit

Commands
  /new  /clear          start a new chat
  /help                 show this help
  /quit                 quit

Files
  @path                 attach a file to your message (pick one in the tree to insert it)
  Gitignored files and secrets (.env, keys, credentials) are never sent to the model."""


def banner(width: int, model: str, root: str) -> str:
    art = WIDE_ART if width >= 80 else NARROW_ART
    return (
        f"\n{art}\n\n"
        f"Architect CLI v{__version__} — your AI open source assistant @ OSGuild.dev\n"
        f"model {model} · project {root}\n\n"
        f"{HELP_TEXT}\n"
    )
