"""`@path` references: parsing, completion, and expansion into prompt context."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from architect.files import AccessPolicy, language_for, read_text_prefix

REF_PATTERN = re.compile(r"(?:^|(?<=\s))@([^\s@]+)")
QUERY_PATTERN = re.compile(r"(?:^|(?<=\s))@([^\s@]*)$")
TRAILING_PUNCTUATION = ".,;:!?)]}'\""
MAX_ATTACH_BYTES = 80_000


def find_refs(text: str, known: Iterable[str] | None = None) -> list[str]:
    """Unique `@refs` in order. With `known`, trailing punctuation is trimmed
    when that turns an unknown ref into a known file (`see @a.py,` → `a.py`)."""
    known_set = set(known) if known is not None else None
    refs: list[str] = []
    for match in REF_PATTERN.finditer(text):
        ref = match.group(1)
        if known_set is not None and ref not in known_set:
            trimmed = ref.rstrip(TRAILING_PUNCTUATION)
            if trimmed in known_set:
                ref = trimmed
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def current_query(line_before_cursor: str) -> tuple[int, str] | None:
    """If the cursor ends an `@query` token, return (index of '@', query).

    A bare `@` yields an empty query so completion can list files right away.
    """
    match = QUERY_PATTERN.search(line_before_cursor)
    if match is None:
        return None
    return match.start(), match.group(1)


def match_files(query: str, files: Sequence[str], limit: int = 5) -> list[str]:
    """Rank files for completion: name prefix, then path prefix, then substring."""
    q = query.lower()
    if not q:
        return list(files[:limit])
    name_prefix: list[str] = []
    path_prefix: list[str] = []
    contains: list[str] = []
    for path in files:
        lower = path.lower()
        if q not in lower:
            continue
        name = lower.rsplit("/", 1)[-1]
        if name.startswith(q):
            name_prefix.append(path)
        elif lower.startswith(q):
            path_prefix.append(path)
        else:
            contains.append(path)
    ranked = sorted(name_prefix, key=len) + sorted(path_prefix, key=len) + sorted(contains, key=len)
    return ranked[:limit]


@dataclass(frozen=True)
class Attachment:
    ref: str
    attached: bool
    note: str = ""


def _fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`{3,}", text)), default=2)
    return "`" * max(3, longest + 1)


def expand_references(
    text: str,
    policy: AccessPolicy,
    known_files: Iterable[str],
    max_bytes: int = MAX_ATTACH_BYTES,
) -> tuple[str, list[Attachment]]:
    """Prepend the contents of referenced files to `text`.

    Refs that are not project files are left alone (they may be e-mail
    addresses or decorators). Refs to files the policy hides are reported so
    the UI can tell the user why nothing was attached.
    """
    known = set(known_files)
    blocks: list[str] = []
    attachments: list[Attachment] = []
    for ref in find_refs(text, known):
        access = policy.check(ref)
        if access.path is None:
            # Only report refs that name a real file; anything else is just text.
            if (policy.root / ref).is_file():
                attachments.append(Attachment(ref, False, access.reason or "not readable"))
            continue
        if not access.path.is_file():
            continue
        read = read_text_prefix(access.path, max_bytes)
        if read is None:
            attachments.append(Attachment(ref, False, "binary or not UTF-8 text"))
            continue
        fence = _fence_for(read.text)
        note = f"\n\n*(truncated — showing first {max_bytes:,} of {read.size:,} bytes)*"
        blocks.append(
            f"Here is `{access.rel}`:\n\n{fence}{language_for(access.rel)}\n{read.text}\n{fence}"
            f"{note if read.truncated else ''}\n\n"
        )
        attachments.append(
            Attachment(access.rel, True, "truncated" if read.truncated else ""),
        )
    return "".join(blocks) + text, attachments
