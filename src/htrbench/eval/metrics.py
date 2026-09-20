"""Error rates on already-normalised strings (eval/normalize.py)."""

from __future__ import annotations

from .alignment import char_edit_counts, word_edit_counts

_TOKEN_PUNCT = ".,;:!?\"'()[]{}<>«»„“”‚‘’-–—/\\|*+=_~^`§$%&#@"


def _rate(edits: int, n_ref: int, hyp_nonempty: bool) -> float:
    """edits / n_ref; for an empty reference 0 if the hypothesis is empty too, else 1."""
    if n_ref == 0:
        return 1.0 if hyp_nonempty else 0.0
    return edits / n_ref


def cer(ref: str, hyp: str) -> float:
    return _rate(char_edit_counts(ref, hyp).n_edits, len(ref), bool(hyp))


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return _rate(word_edit_counts(r, h).n_edits, len(r), bool(h))


def strip_token(token: str) -> str:
    """A whitespace token without the punctuation around it."""
    return token.strip(_TOKEN_PUNCT)
