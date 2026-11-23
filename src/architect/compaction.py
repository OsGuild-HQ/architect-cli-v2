"""Summarize older turns so long conversations fit the context budget."""

from __future__ import annotations

from collections.abc import Sequence

from architect.config import Settings
from architect.conversation import Message, transcript
from architect.llm import ChatClient
from architect.prompts import COMPACTION_PROMPT


def summarize(
    client: ChatClient,
    settings: Settings,
    messages: Sequence[Message],
    prior_summary: str = "",
) -> str:
    """Return a summary of `messages` merged with `prior_summary`.

    Raises on API errors; the caller decides whether to continue uncompacted.
    """
    body = transcript(messages)
    if prior_summary:
        body = f"Existing summary:\n{prior_summary}\n\nNew transcript:\n{body}"
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": COMPACTION_PROMPT},
            {"role": "user", "content": body},
        ],
        max_tokens=min(settings.max_tokens, 2048),
    )
    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        raise ValueError("the model returned an empty summary")
    return summary
