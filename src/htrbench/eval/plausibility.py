"""Misread words that are words: the errors nobody sees.

A fluent reader that turns *Harzburg* into *Herzberg* leaves nothing for a proof-reader to stumble over; a line
engine's letter salad does. Reference and hypothesis are aligned word by word, and every substitution that is more
than a change of punctuation or case is checked against a lexicon: is the word the system wrote a real word?

The lexicon is wordfreq, restricted to the languages curated for the page (an English word on a German page is not a
plausible error), plus the vocabulary of all reference texts, so that period spellings count as words. Numbers
always count: a wrong number is a plausible number. Input strings are tier-L1 normalised.
"""

from __future__ import annotations

from collections.abc import Iterable

import wordfreq
from rapidfuzz.distance import Levenshtein

from .metrics import strip_token

#: minimum Zipf frequency for a wordfreq entry to count as a word
ZIPF_MIN: dict[str, float] = {"de": 2.0, "fr": 2.5, "en": 3.0}


class Lexicon:
    def __init__(self, reference_vocabulary: Iterable[str] = ()):
        self.reference_vocabulary = {self.key(w) for w in reference_vocabulary} - {""}
        self._wordfreq_hits: dict[tuple[str, tuple[str, ...] | None], bool] = {}

    @staticmethod
    def key(token: str) -> str:
        return strip_token(token).lower()

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> Lexicon:
        return cls(w for t in texts for w in t.split())

    def is_word(self, token: str, langs: tuple[str, ...] | None = None) -> bool:
        """``langs`` restricts the wordfreq lookup to the page's languages; None looks in all of ``ZIPF_MIN``."""
        w = self.key(token)
        if not w:
            return False
        if w.isdigit():
            return True
        if not any(ch.isalpha() for ch in w):
            return False
        if w in self.reference_vocabulary:
            return True
        hit = self._wordfreq_hits.get((w, langs))
        if hit is None:
            hit = any(wordfreq.zipf_frequency(w, lang) >= z for lang, z in ZIPF_MIN.items() if not langs or lang in langs)
            self._wordfreq_hits[(w, langs)] = hit
        return hit


def substitution_counts(ref: str, hyp: str, lex: Lexicon, langs: tuple[str, ...] | None = None) -> tuple[int, int]:
    """(word substitutions, of which the hypothesis token is a real word)."""
    r, h = ref.split(), hyp.split()
    n_sub = real = 0
    for op in Levenshtein.editops(r, h):
        if op.tag != "replace":
            continue
        ref_token, hyp_token = r[op.src_pos], h[op.dest_pos]
        if Lexicon.key(ref_token) == Lexicon.key(hyp_token):
            continue  # punctuation or case only
        n_sub += 1
        real += lex.is_word(hyp_token, langs)
    return n_sub, real
