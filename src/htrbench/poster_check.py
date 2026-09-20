"""Compare what the poster prints (``poster/numbers.yaml``) with ``results/tables`` and the data release.

The poster was laid out by hand in PowerPoint, so nothing generates it. This check is the link between the two: it
fails when a table value, rounded the way the poster prints it, differs from the printed value, when a quoted
reading or reference line is not the one the archived run holds, or when a ranking the poster asserts does not hold.
"""

from __future__ import annotations

import pandas as pd

from . import paths
from .eval.aggregate import DIMENSIONS
from .eval.judge import load_judged
from .eval.linematch import match_lines, split_lines
from .eval.metrics import cer, wer
from .eval.normalize import normalize
from .eval.score import load_predictions, pages
from .io import read_yaml


def _round(value: float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


class Report:
    def __init__(self) -> None:
        self.n = 0
        self.failures: list[str] = []

    def equal(self, what: str, got, printed) -> None:
        self.n += 1
        if got != printed:
            self.failures.append(f"{what}: tables give {got}, the poster prints {printed}")


def check() -> bool:
    spec = read_yaml(paths.ROOT / "poster" / "numbers.yaml")
    profile = pd.read_csv(paths.TABLES / "profile.csv").set_index("system_id")
    by_corpus = pd.read_csv(paths.TABLES / "profile_corpus.csv").set_index(["system_id", "corpus"])
    judge = pd.read_csv(paths.TABLES / "judge.csv").set_index("system_id")
    corpora = pd.read_csv(paths.TABLES / "corpora.csv").set_index("corpus")
    lineup = profile[profile["role"] == "lineup"]
    r = Report()

    c = spec["counts"]
    r.equal("pages", int(corpora["n_pages"].sum()), c["pages"])
    r.equal("corpora", len(corpora), c["corpora"])
    r.equal("reference lines", int(corpora["n_lines"].sum()), c["reference_lines"])
    r.equal("systems", len(profile), c["systems"])
    full = profile[(profile["family"] == "vlm_open_local") & (profile["n_pages"] == c["pages"])]
    r.equal("open-weight page readers on all pages", len(full), c["open_weight_page_readers_on_all_pages"])
    r.equal("best of them", full["cer_page"].idxmin(), c["best_open_weight_page_reader"])
    for corpus, (n_pages, n_lines) in c["corpus_cards"].items():
        r.equal(f"{corpus} card", [int(corpora.loc[corpus, "n_pages"]), int(corpora.loc[corpus, "n_lines"])], [n_pages, n_lines])

    for sid, printed in spec["scorecard"].items():
        for dim, value in zip(DIMENSIONS, printed, strict=True):
            r.equal(f"scorecard {sid} {dim}", _round(profile.loc[sid, dim], 2), _round(value, 2))
    for sid, rows in spec["corpus_matrix"].items():
        for corpus, printed in rows.items():
            for dim, value in zip(DIMENSIONS, printed, strict=True):
                r.equal(f"matrix {sid} {corpus} {dim}", _round(by_corpus.loc[(sid, corpus), dim], 2), _round(value, 2))

    for sid, (seconds, cents, hardware) in spec["efficiency_tile"].items():
        r.equal(f"D5 tile {sid} seconds", _round(profile.loc[sid, "s_per_page"], 1), _round(seconds, 1))
        r.equal(f"D5 tile {sid} hardware", profile.loc[sid, "hardware_class"], hardware)
        got = 100 * profile.loc[sid, "price_usd_per_page"]
        if cents == "<0.1":
            r.equal(f"D5 tile {sid} price below 0.1 cent", bool(got < 0.1), True)
        else:
            decimals = 0 if float(cents) >= 10 else 1
            r.equal(f"D5 tile {sid} cents", _round(got, decimals), _round(cents, decimals))

    for sid, pct in spec["judge_panel_misleading"].items():
        r.equal(f"judge panel {sid}", _round(100 * judge.loc[sid, "share_misleading"], 0), _round(pct, 0))

    for sid, column, printed, fmt in spec["statements"]:
        value = profile.loc[sid, column]
        decimals = int(fmt[-1])
        r.equal(f"{sid} {column}", _round(100 * value if fmt.startswith("pct") else value, decimals), _round(printed, decimals))

    for column, (highest, lowest) in spec["extremes"].items():
        r.equal(f"highest {column} in the line-up", lineup[column].idxmax(), highest)
        r.equal(f"lowest {column} in the line-up", lineup[column].idxmin(), lowest)

    for corpus, winner in spec["corpus_winners"].items():
        rows = by_corpus.xs(corpus, level="corpus")
        rows = rows[rows["n_pages"] == corpora.loc[corpus, "n_pages"]]
        r.equal(f"highest character accuracy on {corpus}", rows["char_accuracy"].idxmax(), winner)

    years = {}
    for page in pages().values():
        if page.year:
            years.setdefault(page.corpus, []).append(page.year)
    for corpus, (first, last) in spec["corpus_years"].items():
        dated = years.get(corpus)
        r.equal(f"{corpus} pages lie inside the printed range", bool(dated is None or first <= min(dated) and max(dated) <= last), True)

    judged = load_judged()
    for q in spec["quotes"]:
        page = pages()[q["page"]]
        pred = load_predictions(f"{q['system']}__r0")[q["page"]]
        pair = match_lines(split_lines(page.gt_text), split_lines(pred.text)).pairs[q["line"]]
        where = f"quote {q['system']} {q['page']} line {q['line']}"
        r.equal(f"{where} reference", pair.gt, q["ref"])
        r.equal(where, pair.hyp, q["text"])
        ref, hyp = normalize(pair.gt, "L1"), normalize(pair.hyp, "L1")
        if "cer" in q:  # like a page, a line cannot lose more than all of itself
            r.equal(f"{where} CER", _round(100 * min(cer(ref, hyp), 1.0), 0), _round(q["cer"], 0))
            r.equal(f"{where} WER", _round(100 * min(wer(ref, hyp), 1.0), 0), _round(q["wer"], 0))
        if "verdict" in q:
            row = judged[(judged["system_id"] == q["system"]) & (judged["page_id"] == q["page"]) & (judged["line_idx"] == q["line"])
                         & (judged["pass_idx"] == 0)]
            r.equal(f"{where} verdict", row["verdict"].iloc[0] if len(row) else None, q["verdict"])
            if "spans" in q:
                spans = [[s["hyp_span"], s["type"], s["severity"]] for s in row["spans"].iloc[0]]
                r.equal(f"{where} judged spans", spans, q["spans"])

    for failure in r.failures:
        print("MISMATCH", failure)
    print(f"{r.n - len(r.failures)} of {r.n} poster values match results/tables")
    return not r.failures
