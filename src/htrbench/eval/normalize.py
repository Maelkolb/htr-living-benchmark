"""Cumulative normalisation tiers: what two transcriptions may differ in and still count as the same reading.

    L0  raw: only Unicode form, line structure and whitespace are neutralised. Glyph fidelity is measured here.
    L1  the reading: period glyphs and editorial marks that carry no reading difference are levelled
        (ſ = s, ß = ss, uͤ = ü, [word] = word). Every accuracy figure is computed here.
    L2  graphemic: case-folded, u/v and i/j merged. Punctuation fidelity is measured here.


A tier is a set of named steps. ``normalize`` applies them in the fixed order ``ORDER``, not in the order the
tier lists them, so that

* ``line_strip`` and ``dehyphenate`` run **before** ``newline_to_space`` (otherwise line-final hyphens
  could not be joined any more),
* ``uncertainty_marks`` runs while line structure is still intact,
* ``ligatures`` runs before ``long_s`` (so the U+FB05 ligature expands to ``ſt`` and then to ``st``),
* character-level maps (``long_s``, ``sharp_s``, ``combining_e``) run before ``lowercase``,
* ``lowercase`` runs before the graphemic merges ``uv``/``ij``,
* ``whitespace`` always runs **last** (removals may leave double spaces).

"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

# --------------------------------------------------------------------------- tiers (cumulative)
_L0 = ["nfc", "line_strip", "markdown", "newline_to_space", "whitespace"]
_L1 = _L0 + ["dehyphenate", "uncertainty_marks", "long_s", "sharp_s", "combining_e", "ligatures"]
_L2 = _L1 + ["lowercase", "uv", "ij"]

TIERS: dict[str, list[str]] = {"L0": _L0, "L1": _L1, "L2": _L2}

#: Canonical application order (see module docstring). Every step name must appear exactly once.
ORDER: list[str] = [
    "nfc",
    "line_strip",
    "markdown",
    "dehyphenate",
    "uncertainty_marks",
    "newline_to_space",
    "ligatures",
    "long_s",
    "sharp_s",
    "combining_e",
    "lowercase",
    "uv",
    "ij",
    "whitespace",
]

# --------------------------------------------------------------------------- step implementations
_HYPHEN_BREAK = re.compile(r"(?<=\S)[-¬⸗=]\n")  # -, ¬, ⸗, = at end of a line
_UNCERTAIN = re.compile(r"\[\?\]|\[\.\.\.\]|\[…\]|\(\?\)")  # [?] [...] […] (?)
_BRACKET_WORD = re.compile(r"\[([^\[\]\s]+)\]")  # [word] -> word
_ANGLE_WORD = re.compile(r"⟨([^⟨⟩\s]+)⟩")  # ⟨word⟩ -> word
_COMB_E = re.compile("[aouAOU]ͤ")  # base vowel + COMBINING LATIN SMALL LETTER E
_COMB_E_MAP = {
    "aͤ": "ä",
    "oͤ": "ö",
    "uͤ": "ü",
    "Aͤ": "Ä",
    "Oͤ": "Ö",
    "Uͤ": "Ü",
}
_LIGATURES = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ſt",  # ﬅ -> ſt (then long_s -> st)
        "ﬆ": "st",
    }
)


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t)


def _line_strip(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(ln.strip() for ln in t.split("\n"))


def _newline_to_space(t: str) -> str:
    return t.replace("\n", " ")


_MD_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")  # | :--- | ---: |
_MD_RULE = re.compile(r"^\s*([-*_]\s*){3,}$")  # *** / --- horizontal rules


def _markdown(t: str) -> str:
    """Drop markdown table separator rows and horizontal rules; turn cell pipes into spaces (systems that emit
    tables as markdown are not charged for the syntax; references contain none)."""
    out = []
    for ln in t.split("\n"):
        if _MD_SEP.match(ln) or _MD_RULE.match(ln):
            continue
        if "|" in ln:
            ln = re.sub(r"\s*\|\s*", " ", ln).strip()
        out.append(ln)
    return "\n".join(out)


def _whitespace(t: str) -> str:
    return " ".join(t.split())


def _dehyphenate(t: str) -> str:
    return _HYPHEN_BREAK.sub("", t)


def _uncertainty_marks(t: str) -> str:
    t = _UNCERTAIN.sub("", t)
    t = _BRACKET_WORD.sub(r"\1", t)
    return _ANGLE_WORD.sub(r"\1", t)


def _long_s(t: str) -> str:
    return t.replace("ſ", "s").replace("ʃ", "s")  # ʃ (U+0283) is used as long s in some hand-made transcriptions


def _sharp_s(t: str) -> str:
    return t.replace("ß", "ss").replace("ẞ", "SS")


def _combining_e(t: str) -> str:
    return _COMB_E.sub(lambda m: _COMB_E_MAP[m.group(0)], t)


def _ligatures(t: str) -> str:
    return t.translate(_LIGATURES)


def _lowercase(t: str) -> str:
    return t.lower()


def _uv(t: str) -> str:
    return t.replace("v", "u").replace("V", "U")


def _ij(t: str) -> str:
    return t.replace("j", "i").replace("J", "I")


STEPS: dict[str, Callable[[str], str]] = {
    "nfc": _nfc,
    "line_strip": _line_strip,
    "markdown": _markdown,
    "newline_to_space": _newline_to_space,
    "whitespace": _whitespace,
    "dehyphenate": _dehyphenate,
    "uncertainty_marks": _uncertainty_marks,
    "long_s": _long_s,
    "sharp_s": _sharp_s,
    "combining_e": _combining_e,
    "ligatures": _ligatures,
    "lowercase": _lowercase,
    "uv": _uv,
    "ij": _ij,
}


assert set(ORDER) == set(STEPS) == {s for steps in TIERS.values() for s in steps}
assert len(ORDER) == len(set(ORDER))


# --------------------------------------------------------------------------- public API
def steps_for(tier: str) -> list[str]:
    """Steps of ``tier`` in canonical application order (deduplicated)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; known: {', '.join(TIERS)}")
    wanted = set(TIERS[tier])
    return [s for s in ORDER if s in wanted]


def normalize(text: str, tier: str) -> str:
    t = text or ""
    for name in steps_for(tier):
        t = STEPS[name](t)
    return t

