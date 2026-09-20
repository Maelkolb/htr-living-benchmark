"""Levenshtein edit counts (rapidfuzz).

The reference is the source and the hypothesis the destination: an insertion is a hypothesis element without a
reference counterpart, a deletion a reference element the hypothesis lacks.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class EditCounts:
    ins: int
    dele: int
    sub: int

    @property
    def n_edits(self) -> int:
        return self.ins + self.dele + self.sub


def _counts(ref: Sequence[Hashable], hyp: Sequence[Hashable]) -> EditCounts:
    ins = dele = sub = 0
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "insert":
            ins += 1
        elif op.tag == "delete":
            dele += 1
        else:
            sub += 1
    return EditCounts(ins=ins, dele=dele, sub=sub)


def char_edit_counts(ref: str, hyp: str) -> EditCounts:
    return _counts(ref, hyp)


def word_edit_counts(ref_tokens: list[str], hyp_tokens: list[str]) -> EditCounts:
    return _counts(ref_tokens, hyp_tokens)
