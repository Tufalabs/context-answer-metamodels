"""Shared WeirdChat rubric-judge prompt and parsing utilities."""

from __future__ import annotations
import json
import re
from typing import Any


_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


_ANY_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def render_transcript(messages: list[dict[str, str]]) -> str:
    """Render one transcript in the Docent-compatible text format used by WeirdChat."""
    rows = ["<|T0|>", "<|T0 blocks|>"]
    for index, message in enumerate(messages):
        role = message["role"]
        rows.extend(
            (
                f"<|T0B{index}; role: {role}|>",
                message["content"],
                f"</|T0B{index}; role: {role}|>",
            )
        )
    end_index = len(messages)
    rows.extend(
        (
            f"<|T0B{end_index}; role: user|>",
            "<|user_ends_chat|>",
            f"<|T0B{end_index}; role: user|>",
            "</|T0 blocks|>",
            "</|T0|>",
        )
    )
    return "\n".join(rows)


def parse_judge_response(text: str) -> dict[str, Any]:
    candidate = None
    for pattern in (_JSON_BLOCK_RE, _ANY_BLOCK_RE):
        blocks = pattern.findall(text)
        if blocks:
            candidate = blocks[-1].strip()
            break
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace is None:
            raise ValueError("judge response contains no JSON object")
        candidate = brace.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = json.loads(candidate, strict=False)
    if (
        not isinstance(parsed, dict)
        or not isinstance(parsed.get("match"), bool)
        or not isinstance(parsed.get("explanation"), str)
    ):
        raise ValueError("judge response violates the match/explanation contract")
    return {"match": parsed["match"], "explanation": parsed["explanation"]}
