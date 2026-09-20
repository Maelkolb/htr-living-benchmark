"""D3, diplomatic fidelity, and the numeral facet of D4.

Three questions a character error rate cannot tell apart from ordinary misreadings, each answered on a character
alignment of matched line pairs:

* **glyphs** - does the system keep the period letter forms the reference records (long s, abbreviation hooks, ÿ,
  nasal strokes, superscript e), and does it refrain from inventing them? Tier L0; precision and recall over the
  glyphs listed below (the recall is the HCPR of Levchenko 2025). Only meaningful where the reference is diplomatic.
* **punctuation** - is a punctuation mark of the reference still in its place? Tier L2.
* **capitals** - of the reference capitals the system read as the same letter, how many did it lower-case? Tier L1.
  A capital read as a different letter is a misreading and counts for neither.

**Numerals** are tokens that contain a digit, compared as multisets on tier L1.

All functions take normalised strings and are pure.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

from rapidfuzz.distance import Levenshtein

from .metrics import strip_token

_COMBINING = ("ͤ", "̅", "̄")  # combining small e (uͤ), overline and macron (nasal strokes)
PERIOD_GLYPHS = frozenset("ſʆÿŸæœꝛꝑꝓꝗꝙꝰꝝﬅﬆ") | frozenset(_COMBINING)


def is_period_glyph(ch: str) -> bool:
    """A listed glyph, or a precomposed letter that carries one of the combining marks (ū)."""
    if ch in PERIOD_GLYPHS:
        return True
    return ord(ch) > 127 and any(m in unicodedata.normalize("NFD", ch) for m in _COMBINING)


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def glyph_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """(period glyphs in the reference, of which kept at their aligned place, period glyphs in the hypothesis)."""
    n_ref = sum(is_period_glyph(c) for c in ref)
    n_hyp = sum(is_period_glyph(c) for c in hyp)
    if not n_ref and not n_hyp:
        return 0, 0, 0
    kept = 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag == "equal":
            kept += sum(is_period_glyph(c) for c in ref[i1:i2])
        elif tag == "replace":
            kept += sum(1 for rc, hc in zip(ref[i1:i2], hyp[j1:j2], strict=True) if is_period_glyph(rc) and rc == hc)
    return n_ref, kept, n_hyp


def punct_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(punctuation and symbol characters in the reference, of which kept at their aligned place)."""
    n = sum(_is_punct(c) for c in ref)
    if not n:
        return 0, 0
    lost = sum(1 for op in Levenshtein.editops(ref, hyp) if op.tag in ("replace", "delete") and _is_punct(ref[op.src_pos]))
    return n, n - lost


def case_counts(ref: str, hyp: str) -> tuple[int, int]:
    """(reference capitals read as the same letter, of which in lower case)."""
    read = lowered = 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag == "equal":
            read += sum(1 for c in ref[i1:i2] if c.isalpha() and c.isupper())
        elif tag == "replace":
            for rc, hc in zip(ref[i1:i2], hyp[j1:j2], strict=True):
                if rc.isalpha() and rc.isupper() and hc == rc.lower():
                    read += 1
                    lowered += 1
    return read, lowered


def numeral_tokens(text: str) -> list[str]:
    return [t for t in (strip_token(w) for w in text.split()) if any(ch.isdigit() for ch in t)]


def numeral_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """(numerals in the reference, multiset overlap, numerals in the hypothesis)."""
    r, h = Counter(numeral_tokens(ref)), Counter(numeral_tokens(hyp))
    return sum(r.values()), sum((r & h).values()), sum(h.values())
