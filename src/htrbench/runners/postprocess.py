"""Normalise raw model output into the plain-text transcription that gets scored.

Only *presentation* artefacts are removed (code fences, markdown emphasis, chat preambles).
Nothing that could be part of a transcription is touched; scoring normalisation lives in eval.normalize.
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*$")
_PREAMBLE = re.compile(
    r"^(here('s| is)|the (following|transcription)|transcription|transkription|sure|okay|ok)\b.*[:：]\s*$",
    re.IGNORECASE,
)
_TRAILER = re.compile(
    r"^(note|hinweis|i (have|was)|the (text|page|image)|this (page|text|image)|\*+note)\b.*", re.IGNORECASE
)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_EMPH = re.compile(r"(\*\*|__)(.+?)\1")
_TAGS = re.compile(r"</?(u|b|i|em|strong|s|del|sub|sup|br|p|span|div)\s*/?>", re.IGNORECASE)


def clean(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    lines = [ln for ln in lines if not _FENCE.match(ln.strip())]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and _PREAMBLE.match(lines[0].strip()) and len(lines) > 1:
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    # drop a trailing commentary block introduced by a blank line + note-like sentence
    if len(lines) > 2:
        for i in range(len(lines) - 1, 0, -1):
            if not lines[i - 1].strip() and _TRAILER.match(lines[i].strip()):
                lines = lines[: i - 1]
                break
    out = []
    for ln in lines:
        ln = _MD_HEADING.sub("", ln)
        ln = _EMPH.sub(r"\2", ln)
        ln = _TAGS.sub("", ln)
        out.append(ln.rstrip())
    result = "\n".join(out).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result

