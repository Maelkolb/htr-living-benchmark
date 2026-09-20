"""The facets behind D3, D4 and D5."""

import math

import pytest

from htrbench import paths
from htrbench.eval import fidelity
from htrbench.eval.efficiency import efficiency, log_facet
from htrbench.eval.judge import kappa, parse_judge
from htrbench.eval.plausibility import Lexicon, substitution_counts
from htrbench.io import read_yaml

EFFICIENCY = read_yaml(paths.CONFIGS / "protocol.yaml")["efficiency"]  # the committed constants, not a copy


def test_glyphs_kept_modernised_and_invented():
    assert fidelity.glyph_counts("ſeeʆ geweſte", "ſeel geweste") == (3, 1, 1)
    assert fidelity.glyph_counts("dass", "daſs") == (0, 0, 1)  # an invented long s
    assert fidelity.glyph_counts("Bruͤder", "Brüder") == (1, 0, 0)


def test_punctuation_is_kept_at_its_place():
    assert fidelity.punct_counts("concl: ad 1. der erstere", "concl ad 1, der erstere") == (2, 0)
    assert fidelity.punct_counts("concl: ad 1.", "concl: ad 1.") == (2, 2)


def test_capitals_lowered_among_capitals_read():
    assert fidelity.case_counts("Der Wald bei Harzburg", "der Wald bei Marzburg") == (2, 1)  # H read as M counts for neither


def test_numerals_are_compared_as_multisets():
    assert fidelity.numeral_counts("den 25. Januar 1923, 25 fl.", "den 25. Januar 1928") == (3, 1, 2)


def test_real_word_substitutions():
    lex = Lexicon.from_texts(["Harzburg Herzberg entdekt"])
    assert substitution_counts("bei Harzburg am Harze", "bei Herzberg am Hxrze", lex, ("de",)) == (2, 1)
    assert substitution_counts("Am Harze.", "am Harze", lex, ("de",)) == (0, 0)  # case and punctuation are not word errors
    assert lex.is_word("1795") and not lex.is_word("—")


def test_wordfreq_is_restricted_to_the_languages_of_the_page():
    lex = Lexicon()
    assert lex.is_word("whereabouts", ("en",)) and not lex.is_word("whereabouts", ("de",))


def test_log_facet_anchors():
    assert log_facet(1.0, 1.0, 100.0) == 1.0 and log_facet(100.0, 1.0, 100.0) == 0.0
    assert log_facet(10.0, 1.0, 100.0) == pytest.approx(0.5)
    assert log_facet(0.2, 1.0, 100.0) == 1.0 and log_facet(500.0, 1.0, 100.0) == 0.0
    assert math.isnan(log_facet(math.nan, 1.0, 100.0))


def test_hosted_systems_pay_the_list_price_and_open_weights_the_electricity():
    hosted = efficiency(10.0, "api", 0.01, math.nan, 0.41, EFFICIENCY)
    assert hosted.price_basis == "list price" and hosted.usd_per_page == 0.01
    assert hosted.score(EFFICIENCY["weights"]) == pytest.approx(0.5 * 0.5 + 0.25 * 0.1 + 0.25 * (2 / 3))
    local = efficiency(10.0, "gpu_8gb", math.nan, 0.5, 0.41, EFFICIENCY)
    assert local.price_basis == "electricity" and local.usd_per_page == pytest.approx(0.5 / 1000 * 0.41)


def test_a_facet_that_is_missing_is_left_out_of_the_mean():
    no_price = efficiency(10.0, "gpu_8gb", math.nan, math.nan, 0.41, EFFICIENCY)
    assert no_price.score(EFFICIENCY["weights"]) == pytest.approx((0.5 * 0.5 + 0.25 * 0.8) / 0.75)


def test_judge_answers_are_mapped_onto_the_closed_lists():
    line = parse_judge('```json\n{"spans": [{"ref_span": "25", "hyp_span": "75", "type": "Number", "severity": 5}], "verdict": "Misleading."}\n```')
    assert line.verdict == "misleading"
    assert (line.spans[0].type, line.spans[0].severity) == ("number_error", 2)


def test_kappa():
    assert kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"]) == 0.0
