"""One page through score_page and text_scores."""

import pandas as pd
import pytest

from htrbench.eval.aggregate import text_scores
from htrbench.eval.plausibility import Lexicon
from htrbench.eval.score import score_page
from htrbench.schema import Facets, Page, Prediction, RunMeta

REFERENCE = "Concʆ: Sub Consuetis clausu[-]\nſeeʆ geweſte Köchin außgeſaget\nden 25. Januar 1739"
FACETS = Facets(genre="protocol", period_bin="1700-1750", script="kurrent", languages=["de", "la"], layout_class="single_block",
                condition="clean", source_type="archival_scan")
PAGE = Page(page_id="p1", corpus="abp", image_path="pages/abp/p1.jpg", width=100, height=100, sha256="", gt_text=REFERENCE,
            gt_source="test", n_lines_gt=3, facets=FACETS)
META = RunMeta(run_id="sys__r0", system_id="sys", model_id="m", provider="p", prompt_id="page_v1", prompt_sha256="", hardware="api",
               harness_version="0", started_at="")
LEXICON = Lexicon.from_texts([REFERENCE])


def score(text: str, error: str | None = None):
    return score_page(Prediction(page_id="p1", text=text, error=error), PAGE, META, LEXICON)


def test_the_reference_scores_perfectly_against_itself():
    s = score(REFERENCE)
    assert (s.ins, s.dele, s.sub, s.word_edits) == (0, 0, 0, 0)
    assert s.n_lines_matched == s.n_lines_ref == 3
    assert s.glyph_kept == s.n_glyph_ref == s.n_glyph_hyp > 0
    assert s.punct_kept == s.n_punct_ref and s.caps_lowered == 0 and s.num_kept == s.n_num_ref == 2


def test_line_breaks_do_not_count_but_reading_order_does():
    assert score(REFERENCE.replace("\n", " ")).dele == 0
    lines = REFERENCE.split("\n")
    rotated = score("\n".join(lines[1:] + lines[:1]))  # the page is one string, so its order is part of it
    assert rotated.ins + rotated.dele + rotated.sub > 0


def test_a_modernising_reader_loses_the_glyphs_but_not_the_characters():
    s = score("Concl: Sub Consuetis clausu\nseel geweste Köchin außgesaget\nden 25. Januar 1739")
    assert s.glyph_kept == 0 and s.n_glyph_hyp == 0
    assert s.sub == 2  # the two abbreviation hooks read as l; ſ = s at tier L1


def test_a_refusal_loses_the_whole_page():
    s = score("", error="refusal")
    assert s.error and s.dele == s.n_ref_chars and s.word_edits == s.n_ref_words
    assert s.n_lines_matched == 0 and s.punct_kept == 0 and s.num_kept == 0


def test_a_runaway_page_is_capped_at_the_page():
    rows = pd.DataFrame([score(REFERENCE).model_dump(), score("x " * 2000).model_dump()])
    rows["page_id"] = ["p1", "p2"]
    assert text_scores(rows)["cer_page"] == pytest.approx(0.5)
