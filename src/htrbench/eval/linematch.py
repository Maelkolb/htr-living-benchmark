"""Line matching: which part of a system's output belongs to which reference line.

The five scores need different units. D1 and D2 compare whole pages and never come here. Fidelity, numerals, the
real-word share and the judge work on lines, and for them a line engine (one output line per segmented baseline) and
a page reader (whatever line breaks it chose) have to be brought onto the same unit, the reference line,
independent of reading order:

1. **assignment** - a one-to-one assignment (Hungarian algorithm) that maximises character similarity, with a mild
   preference for similar relative positions on the page; pairs below ``ACCEPT`` are rejected;
2. **merges** - a reference line left unmatched is searched inside neighbouring output lines (a system joined two
   physical lines, or the edition sets a marginal number on a line of its own). The best-aligned span becomes its
   text and is taken away from the line it was cut out of;
3. **splits** - a matched line whose output has an unmatched neighbour is re-scored against the two joined, and a
   line still missing is re-assembled from two or three consecutive unused output lines (a system broke one physical
   line in two or three);
4. what remains is **missing** (a reference line with no text) or **extra** (an output line with no reference).

Similarity is computed on tier-L1 text; the merge search works on the raw lines so that spans can be cut exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .normalize import normalize

ACCEPT = 0.35  # minimum character similarity for an assignment pair
ORDER_WEIGHT = 0.15  # cost of a full-page displacement between reference and hypothesis positions
MERGE_ACCEPT = 0.60  # minimum partial-alignment similarity for a line found inside a longer hypothesis line
MIN_FUZZY_CHARS = 4  # shorter reference lines are only found inside a neighbour as an exact whole token


@dataclass
class LinePair:
    gt_idx: int
    gt: str  # raw reference line
    hyp: str  # raw hypothesis text that corresponds to it ("" if missing)
    hyp_idx: int | None
    sim: float  # similarity that produced the match
    kind: str  # match | merged | split | missing


@dataclass
class LineMatch:
    pairs: list[LinePair]
    extra_hyp: list[int]  # hypothesis line indices without a reference line
    n_gt: int

    @property
    def n_matched(self) -> int:
        return sum(1 for p in self.pairs if p.kind != "missing")


def split_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]


def _sim_matrix(a: list[str], b: list[str]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(process.cdist(a, b, scorer=Levenshtein.normalized_similarity, workers=1), dtype=float)


def _token_span(needle: str, hay: str) -> tuple[int, int] | None:
    """Span of `needle` as a whole whitespace-delimited token (punctuation-insensitive) inside `hay`."""
    key = needle.strip().strip(".,;:")
    if not key:
        return None
    for m in re.finditer(r"\S+", hay):
        if m.group(0).strip(".,;:") == key:
            return m.start(), m.end()
    return None


def _cut(s: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given [start, end) spans from s and tidy whitespace."""
    if not spans:
        return s
    out, last = [], 0
    for a, b in sorted(spans):
        a, b = max(a, last), max(b, last)
        out.append(s[last:a])
        last = b
    out.append(s[last:])
    return re.sub(r"\s+", " ", "".join(out)).strip()


def match_lines(gt_lines: list[str], hyp_lines: list[str], accept: float = ACCEPT, order_weight: float = ORDER_WEIGHT) -> LineMatch:
    """Raw lines in, raw text out; see the module docstring for the four steps."""
    n, m = len(gt_lines), len(hyp_lines)
    pairs: list[LinePair] = [LinePair(i, gt_lines[i], "", None, 0.0, "missing") for i in range(n)]
    if n == 0 or m == 0:
        return LineMatch(pairs, list(range(m)), n)
    g1 = [normalize(x, "L1") for x in gt_lines]
    h1 = [normalize(x, "L1") for x in hyp_lines]
    sim = _sim_matrix(g1, h1)
    pos = np.abs(np.arange(n)[:, None] / max(n, 1) - np.arange(m)[None, :] / max(m, 1))
    cost = (1.0 - sim) + order_weight * pos
    rows, cols = linear_sum_assignment(cost)
    used: set[int] = set()
    for i, j in zip(rows, cols, strict=True):
        if sim[i, j] >= accept:
            pairs[i] = LinePair(i, gt_lines[i], hyp_lines[j], int(j), float(sim[i, j]), "match")
            used.add(int(j))

    # 2. merges: an unmatched reference line found inside a neighbour's hypothesis line (or an unused line).
    #    A span may only be cut out of a hypothesis line that already belongs to a matched reference line if its
    #    owner loses no more than 0.02 similarity by giving it away, so an exact match can shed a character at most.
    hyp_of_gt = {p.gt_idx: p.hyp_idx for p in pairs if p.hyp_idx is not None}
    owner_of_hyp = {p.hyp_idx: p.gt_idx for p in pairs if p.kind == "match" and p.hyp_idx is not None}
    cuts: dict[int, list[tuple[int, int]]] = {}  # hyp idx -> spans given away to merged reference lines

    def owner_keeps(j: int, span: tuple[int, int]) -> bool:
        g = owner_of_hyp.get(j)
        if g is None:
            return True
        before = Levenshtein.normalized_similarity(g1[g], normalize(_cut(hyp_lines[j], cuts.get(j, [])), "L1"))
        after = Levenshtein.normalized_similarity(g1[g], normalize(_cut(hyp_lines[j], cuts.get(j, []) + [span]), "L1"))
        return after >= before - 0.02

    def overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
        return any(span[0] < b and a < span[1] for a, b in taken)

    # A hypothesis line may hold several reference lines (a page reader that joins three lines into one): the search runs in
    # rounds, a merge makes the hypothesis line a neighbour candidate for the next missing line, and a span already given
    # away is never given twice.
    for _round in range(4):
        changed = False
        for p in pairs:
            if p.kind != "missing":
                continue
            raw_g = p.gt.strip()
            cands: list[int] = []
            for k in (p.gt_idx - 1, p.gt_idx + 1, p.gt_idx - 2, p.gt_idx + 2):
                j = hyp_of_gt.get(k)
                if j is not None and j not in cands:
                    cands.append(j)
            cands += [j for j in range(m) if j not in used and j not in cands]
            best, best_j, best_span = 0.0, None, None
            for j in cands:
                hay = hyp_lines[j]
                if len(hay) <= len(raw_g):
                    continue
                if len(raw_g) < MIN_FUZZY_CHARS:
                    span = _token_span(raw_g, hay)
                    if span is not None and not overlaps(span, cuts.get(j, [])) and owner_keeps(j, span):
                        best, best_j, best_span = 1.0, j, span
                        break
                    continue
                al = fuzz.partial_ratio_alignment(raw_g, hay)
                if al is not None and al.score / 100.0 > best:
                    span = (al.dest_start, al.dest_end)
                    if not overlaps(span, cuts.get(j, [])) and owner_keeps(j, span):
                        best, best_j, best_span = al.score / 100.0, j, span
            if best_j is not None and best >= MERGE_ACCEPT and best_span is not None:
                s, e = best_span
                pairs[p.gt_idx] = LinePair(p.gt_idx, p.gt, hyp_lines[best_j][s:e].strip() or hyp_lines[best_j], best_j, best, "merged")
                cuts.setdefault(best_j, []).append((s, e))
                used.add(best_j)
                hyp_of_gt[p.gt_idx] = best_j
                changed = True
        if not changed:
            break
    # give the cut spans away: the line that "owns" a hypothesis line keeps the rest of it
    for p in pairs:
        if p.kind == "match" and p.hyp_idx in cuts:
            rest = _cut(hyp_lines[p.hyp_idx], cuts[p.hyp_idx])
            pairs[p.gt_idx] = LinePair(p.gt_idx, p.gt, rest, p.hyp_idx, p.sim, "match")

    # 3. splits: a matched pair whose hypothesis has an unused neighbour that improves the match when concatenated
    for p in pairs:
        if p.kind != "match" or p.hyp_idx is None:
            continue
        for j2 in (p.hyp_idx + 1, p.hyp_idx - 1):
            if j2 < 0 or j2 >= m or j2 in used:
                continue
            first, second = (p.hyp, hyp_lines[j2]) if j2 > p.hyp_idx else (hyp_lines[j2], p.hyp)
            cand_raw = (first + " " + second).strip()
            s2 = Levenshtein.normalized_similarity(g1[p.gt_idx], normalize(cand_raw, "L1"))
            if s2 > p.sim + 0.10:
                pairs[p.gt_idx] = LinePair(p.gt_idx, p.gt, cand_raw, p.hyp_idx, float(s2), "split")
                used.add(j2)
                break
    # 3b. a reference line still missing may have been emitted as 2-3 consecutive (unused) hypothesis lines
    for p in pairs:
        if p.kind != "missing" or len(g1[p.gt_idx]) < 12:
            continue
        best, best_js = 0.0, None
        for j in range(m):
            if j in used:
                continue
            for w in (2, 3):
                js = list(range(j, j + w))
                if js[-1] >= m or any(k in used for k in js):
                    break
                cand = " ".join(hyp_lines[k] for k in js)
                s2 = Levenshtein.normalized_similarity(g1[p.gt_idx], normalize(cand, "L1"))
                if s2 > best:
                    best, best_js = s2, js
        if best_js and best >= accept:
            pairs[p.gt_idx] = LinePair(p.gt_idx, p.gt, " ".join(hyp_lines[k] for k in best_js), best_js[0], float(best), "split")
            used.update(best_js)
    extra = [j for j in range(m) if j not in used]
    return LineMatch(pairs, extra, n)


def normalized_pairs(lm: LineMatch, tier: str) -> list[tuple[str, str]]:
    """(reference, hypothesis) per reference line at ``tier``. A missing line has an empty hypothesis; reference lines
    that normalise to nothing are left out."""
    out = []
    for p in lm.pairs:
        ref = normalize(p.gt, tier)
        if ref:
            out.append((ref, normalize(p.hyp, tier) if p.kind != "missing" else ""))
    return out
