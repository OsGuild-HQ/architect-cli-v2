"""OpenAI function-calling schemas for the read-only project tools."""

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read lines from a UTF-8 text file in the project. Path is repo-relative. "
                "Returns up to `limit` lines starting at `offset` (0-based), with line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative path, e.g. 'src/app.py'",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "First line to return (0-based). Default 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Maximum lines to return. Default 400.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "List project files matching a glob. `**` spans directories, so '**/*.py' "
                "finds every Python file including top-level ones. A pattern without '/' "
                "(e.g. 'README*') also matches file names in any directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Glob pattern."}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Regex search across project text files. Returns up to `max_results` matches "
                "as 'file:line:content'. `include` is a glob filter with the same rules as `glob`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."},
                    "include": {"type": "string", "description": "Optional glob, e.g. '*.py'."},
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Cap on matches. Default 50.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "question",
            "description": (
                "Ask the user a clarifying question and wait for the answer. Use sparingly, "
                "only when genuinely ambiguous. Provide 2-4 `options` when possible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask."},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional multiple-choice answers.",
                    },
                },
                "required": ["question"],
            },
        },
    },
]
