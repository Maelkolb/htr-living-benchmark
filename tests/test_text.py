"""Normalisation tiers, error rates and the post-processing of raw model output."""

from htrbench.eval.metrics import cer, strip_token, wer
from htrbench.eval.normalize import normalize, steps_for
from htrbench.runners.postprocess import clean


def test_tiers_are_cumulative():
    assert set(steps_for("L0")) < set(steps_for("L1")) < set(steps_for("L2"))


def test_a_ligature_is_expanded_before_the_long_s_is_levelled():
    assert normalize("ﬅur", "L1") == "stur"


def test_l0_keeps_the_glyphs_and_drops_the_line_structure():
    assert normalize("  Daſs  er\nkam ", "L0") == "Daſs er kam"


def test_l1_levels_glyphs_without_a_reading_difference():
    assert normalize("Daſs die Bruͤder", "L1") == "Dass die Brüder"
    assert normalize("Straße", "L1") == "Strasse"
    assert normalize("ein [?] Wort [unsicher]", "L1") == "ein Wort unsicher"


def test_l1_joins_words_broken_across_lines():
    assert normalize("Rats-\nprotokoll", "L1") == "Ratsprotokoll"
    assert normalize("Rats¬\nprotokoll", "L1") == "Ratsprotokoll"


def test_l2_folds_case_and_early_modern_graphemes():
    assert normalize("Vnd Jtem", "L2") == "und item"


def test_markdown_tables_are_not_charged_for_their_syntax():
    assert normalize("| Eiche | 12 |\n|---|---|\n| Buche | 7 |", "L0") == "Eiche 12 Buche 7"


def test_error_rates():
    assert cer("abcd", "abxd") == 0.25
    assert wer("a b c d", "a b d") == 0.25
    assert cer("", "") == 0.0 and cer("", "x") == 1.0


def test_strip_token():
    assert strip_token("„Harzburg“,") == "Harzburg"


def test_clean_removes_presentation_only():
    raw = "Here is the transcription:\n```\n**Concl:** Sub Consuetis\n```\n\nNote: the last word is unclear."
    assert clean(raw) == "Concl: Sub Consuetis"
