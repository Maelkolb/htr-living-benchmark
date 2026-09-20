"""Score runs: one row of counts per page and run (``schema.Score``), written to ``results/scores.parquet``.

Scoring only counts. Every rate, score and confidence interval is formed from sums of these counts in
``eval/aggregate.py``, so a figure can always be traced back to pages.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from .. import paths
from ..io import read_jsonl
from ..runners.base import last_predictions
from ..schema import Page, Prediction, RunMeta, Score
from . import fidelity
from .alignment import char_edit_counts, word_edit_counts
from .linematch import match_lines, normalized_pairs, split_lines
from .normalize import normalize
from .plausibility import Lexicon, substitution_counts

log = logging.getLogger("htrbench.score")


@lru_cache(maxsize=1)
def pages() -> dict[str, Page]:
    return {p.page_id: p for p in read_jsonl(paths.MANIFEST, Page)}


@lru_cache(maxsize=1)
def lexicon() -> Lexicon:
    return Lexicon.from_texts(normalize(p.gt_text, "L1") for p in pages().values())


def load_predictions(run_id: str) -> dict[str, Prediction]:
    return last_predictions(paths.RUNS / run_id / "predictions.jsonl")


def score_page(pred: Prediction, page: Page, meta: RunMeta, lex: Lexicon) -> Score:
    failed = pred.error is not None or not normalize(pred.text, "L0")
    hyp_text = "" if failed else pred.text

    ref, hyp = normalize(page.gt_text, "L1"), normalize(hyp_text, "L1")
    chars = char_edit_counts(ref, hyp)
    words = word_edit_counts(ref.split(), hyp.split())

    lm = match_lines(split_lines(page.gt_text), split_lines(hyp_text))
    l0, l1, l2 = (normalized_pairs(lm, tier) for tier in ("L0", "L1", "L2"))
    langs = tuple(page.facets.languages)

    def total(fn, pairs, *args) -> list[int]:
        return [sum(col) for col in zip(*(fn(r, h, *args) for r, h in pairs), strict=True)] if pairs else []

    n_sub, real_sub = total(substitution_counts, l1, lex, langs) or (0, 0)
    n_glyph_ref, glyph_kept, n_glyph_hyp = total(fidelity.glyph_counts, l0) or (0, 0, 0)
    n_punct_ref, punct_kept = total(fidelity.punct_counts, l2) or (0, 0)
    n_caps_read, caps_lowered = total(fidelity.case_counts, l1) or (0, 0)
    n_num_ref, num_kept, n_num_hyp = total(fidelity.numeral_counts, l1) or (0, 0, 0)

    return Score(
        run_id=meta.run_id, system_id=meta.system_id, page_id=page.page_id, corpus=page.corpus, error=failed,
        n_ref_chars=len(ref), n_ref_words=len(ref.split()), ins=chars.ins, dele=chars.dele, sub=chars.sub, word_edits=words.n_edits,
        n_lines_ref=lm.n_gt, n_lines_matched=lm.n_matched, n_sub_words=n_sub, real_word_sub=real_sub,
        n_glyph_ref=n_glyph_ref, glyph_kept=glyph_kept, n_glyph_hyp=n_glyph_hyp, n_punct_ref=n_punct_ref, punct_kept=punct_kept,
        n_caps_read=n_caps_read, caps_lowered=caps_lowered, n_num_ref=n_num_ref, num_kept=num_kept, n_num_hyp=n_num_hyp,
    )


def score_run(run_id: str) -> list[Score]:
    meta = RunMeta.model_validate_json((paths.RUNS / run_id / "meta.json").read_text(encoding="utf-8"))
    known, lex = pages(), lexicon()
    preds = load_predictions(run_id)
    unknown = sorted(set(preds) - set(known))
    if unknown:
        log.warning("%s: %d predictions for pages outside the manifest are ignored", run_id, len(unknown))
    rows = [score_page(pred, known[pid], meta, lex) for pid, pred in preds.items() if pid in known]
    log.info("scored %s: %d pages", run_id, len(rows))
    return rows


def score_all() -> pd.DataFrame:
    rows: list[Score] = []
    for run_dir in sorted(paths.RUNS.iterdir()):
        if (run_dir / "predictions.jsonl").exists() and (run_dir / "meta.json").exists():
            rows.extend(score_run(run_dir.name))
    df = pd.DataFrame([r.model_dump() for r in rows])
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.SCORES, index=False)
    log.info("%d rows from %d runs -> %s", len(df), df["run_id"].nunique(), paths.SCORES)
    return df
