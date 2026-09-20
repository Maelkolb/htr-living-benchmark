"""The matcher brings page readers and line engines onto the reference line."""

from htrbench.eval.linematch import match_lines, normalized_pairs

REF = ["Caplan zu Sierning klagt ein", "wegen einer Schuld von 25 fl.", "Concl: ist zu bezahlen"]


def kinds(lm):
    return [p.kind for p in lm.pairs]


def test_identical_pages_match_line_by_line():
    lm = match_lines(REF, REF)
    assert kinds(lm) == ["match"] * 3 and not lm.extra_hyp


def test_reading_order_does_not_matter():
    lm = match_lines(REF, [REF[2], REF[0], REF[1]])
    assert [p.hyp for p in lm.pairs] == REF


def test_two_lines_joined_by_the_system_are_both_found():
    lm = match_lines(REF, [REF[0] + " " + REF[1], REF[2]])
    assert lm.n_matched == 3
    assert {"merged"} <= set(kinds(lm))
    assert [h for _, h in normalized_pairs(lm, "L1")] == REF


def test_a_line_broken_in_two_is_put_together():
    lm = match_lines(REF, ["Caplan zu Sierning", "klagt ein", REF[1], REF[2]])
    assert lm.pairs[0].kind == "split" and lm.pairs[0].hyp == REF[0]
    assert not lm.extra_hyp


def test_missing_and_extra_lines():
    lm = match_lines(REF, [REF[0], "Lorem ipsum dolor sit amet consectetur"])
    assert kinds(lm) == ["match", "missing", "missing"]
    assert lm.extra_hyp == [1]
    assert normalized_pairs(lm, "L1")[1] == (REF[1], "")


def test_empty_output():
    lm = match_lines(REF, [])
    assert lm.n_matched == 0 and len(lm.pairs) == 3
