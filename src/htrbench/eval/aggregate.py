"""From page counts to the tables in ``results/tables`` (columns: ``results/tables/README.md``).

    profile.csv         one row per system: the five scores, their facets and what the facets were computed from
    profile_corpus.csv  the same per system and corpus
    judge.csv           what the judge found, per line-up system
    corpora.csv         pages, lines and words per corpus
    runs.csv            the archived runs

Every rate is micro-averaged: sums over pages divided by sums over pages.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from .. import paths
from ..io import read_jsonl, read_yaml
from ..runners import pricing
from ..runners.registry import systems
from ..schema import Page, RunMeta, SystemSpec
from .bootstrap import bootstrap_ci
from .efficiency import efficiency
from .judge import judge_table, load_judged
from .linematch import split_lines
from .score import load_predictions

log = logging.getLogger("htrbench.aggregate")

#: Corpora whose reference records the period letter forms. Elsewhere the glyph facet is undefined: the Humboldt
#: edition and the post-edited references write a plain s, so a correctly read ſ would count as an invention.
DIPLOMATIC_CORPORA = ("abp",)
DIMENSIONS = ["char_accuracy", "word_accuracy", "fidelity", "integrity", "efficiency"]
SYSTEM_COLUMNS = ["system_id", "label", "family", "weights", "role"]


def _ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def _f1(kept: float, n_ref: float, n_hyp: float) -> float:
    if not n_ref and not n_hyp:
        return math.nan
    recall, precision = (kept / n_ref if n_ref else 0.0), (kept / n_hyp if n_hyp else 0.0)
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def _mean(facets: list[float]) -> float:
    """Mean of the facets that exist. A facet that could not be measured is left out, not counted as zero."""
    present = [v for v in facets if not math.isnan(v)]
    return sum(present) / len(present) if present else math.nan


def text_scores(pages: pd.DataFrame) -> dict:
    """D1 to D3, the numeral facet of D4 and two diagnostics, from the score rows of one system (all pages or one corpus)."""
    # A runaway output, a loop of thousands of characters, loses the page once and not several times over.
    capped = pages.assign(char_edits=(pages["ins"] + pages["dele"] + pages["sub"]).clip(upper=pages["n_ref_chars"]),
                          word_edits=pages["word_edits"].clip(upper=pages["n_ref_words"]))
    cer, cer_lo, cer_hi = bootstrap_ci(capped, "char_edits", "n_ref_chars")
    wer, wer_lo, wer_hi = bootstrap_ci(capped, "word_edits", "n_ref_words")
    total = pages.sum(numeric_only=True)
    diplomatic = pages[pages["corpus"].isin(DIPLOMATIC_CORPORA)].sum(numeric_only=True)
    glyph_f1 = _f1(diplomatic["glyph_kept"], diplomatic["n_glyph_ref"], diplomatic["n_glyph_hyp"])
    punct_kept = _ratio(total["punct_kept"], total["n_punct_ref"])
    caps_kept = 1 - _ratio(total["caps_lowered"], total["n_caps_read"])
    return {
        "n_pages": len(pages), "n_failed": int(pages["error"].sum()),
        "char_accuracy": 1 - cer, "word_accuracy": 1 - wer, "fidelity": _mean([glyph_f1, punct_kept, caps_kept]),
        "cer_page": cer, "cer_page_lo": cer_lo, "cer_page_hi": cer_hi, "wer_page": wer, "wer_page_lo": wer_lo, "wer_page_hi": wer_hi,
        "glyph_f1": glyph_f1, "punct_kept": punct_kept, "caps_kept": caps_kept,
        "numeral_f1": _f1(total["num_kept"], total["n_num_ref"], total["n_num_hyp"]),
        "line_recall": _ratio(total["n_lines_matched"], total["n_lines_ref"]),
        "real_word_share": _ratio(total["real_word_sub"], total["n_sub_words"]),
    }


def cost_basis(spec: SystemSpec, meta: RunMeta, preds: pd.DataFrame) -> dict:
    """What the price facet of D5 starts from, per page: USD for hosted systems, Wh for everything else."""
    elec = pricing.electricity()
    usd = wh = math.nan
    wh_basis = "not measured"
    if spec.hardware_class == "api":
        usd = float(preds["cost_usd"].mean())
    elif spec.hardware_class == "transkribus":
        usd = float(meta.params["credits_per_page"]) * pricing.transkribus()["eur_per_credit"] * elec["usd_per_eur"]
    elif preds["energy_wh"].notna().any():
        wh, wh_basis = float(preds["energy_wh"].mean()), "measured (GPU board power)"
    elif spec.hardware == "cpu_local":
        wh = elec["cpu_w_estimate"] * float(preds["latency_s"].mean()) / 3600
        wh_basis = f"estimated ({elec['cpu_w_estimate']} W x seconds)"
    return {"usd_per_page": usd, "wh_per_page": wh, "wh_basis": wh_basis}


def efficiency_scores(s_per_page: float, spec: SystemSpec, basis: dict, cfg: dict) -> dict:
    elec = pricing.electricity()
    e = efficiency(s_per_page, spec.hardware_class, basis["usd_per_page"], basis["wh_per_page"], elec["eur_per_kwh"] * elec["usd_per_eur"], cfg)
    return {"efficiency": e.score(cfg["weights"]), "efficiency_speed": e.speed, "efficiency_hardware": e.hardware,
            "efficiency_price": e.price, "s_per_page": s_per_page, "hardware_class": spec.hardware_class,
            "price_usd_per_page": e.usd_per_page, "price_basis": e.price_basis, "wh_per_page": basis["wh_per_page"], "wh_basis": basis["wh_basis"]}


def corpora_table(manifest: list[Page]) -> pd.DataFrame:
    rows = []
    for corpus in sorted({p.corpus for p in manifest}):
        pages = [p for p in manifest if p.corpus == corpus]
        rows.append({"corpus": corpus, "n_pages": len(pages), "n_lines": sum(p.n_lines_gt for p in pages),
                     "n_lines_scored": sum(len(split_lines(p.gt_text)) for p in pages),
                     "n_words": sum(len(p.gt_text.split()) for p in pages), "n_judge_sample_pages": sum(p.judge_sample for p in pages)})
    return pd.DataFrame(rows)


def _ordered(rows: list[dict], *first: str) -> pd.DataFrame:
    """Identification and the five scores first, then everything they were computed from."""
    df = pd.DataFrame(rows)
    lead = [*SYSTEM_COLUMNS, *first, "n_pages", "n_failed", *DIMENSIONS]
    return df[lead + [c for c in df.columns if c not in lead]]


def build_all(out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or paths.TABLES
    scores = pd.read_parquet(paths.SCORES)
    manifest = read_jsonl(paths.MANIFEST, Page)
    corpus_of = {p.page_id: p.corpus for p in manifest}
    eff_cfg = read_yaml(paths.CONFIGS / "protocol.yaml")["efficiency"]
    judge = judge_table(load_judged())
    misleading = dict(zip(judge["system_id"], judge["share_misleading_both_passes"], strict=True))

    repeats = sorted(r for r in scores["run_id"].unique() if not r.endswith("__r0"))
    if repeats:  # the tables describe one run per system; a repeat run is scored but not aggregated
        log.warning("run indices other than 0 are not aggregated: %s", ", ".join(repeats))

    profile, by_corpus, runs = [], [], []
    for sid, spec in systems().items():
        run_id = f"{sid}__r0"
        rows = scores[scores["run_id"] == run_id]
        if rows.empty:
            log.warning("no scores for %s", run_id)
            continue
        meta = RunMeta.model_validate_json((paths.RUNS / run_id / "meta.json").read_text(encoding="utf-8"))
        # Time, cost and energy come from the pages the system actually answered.
        answered = [p.model_dump() for p in load_predictions(run_id).values() if p.error is None and p.page_id in corpus_of]
        if not answered:
            log.warning("%s answered no page; skipped", run_id)
            continue
        preds = pd.DataFrame(answered)
        preds["corpus"] = preds["page_id"].map(corpus_of)
        basis = cost_basis(spec, meta, preds)
        head = {"system_id": sid, "label": spec.label, "family": spec.family, "weights": spec.weights, "role": spec.role}

        text = text_scores(rows)
        judge_misleading = misleading.get(sid, math.nan)
        profile.append({**head, **text, "integrity": _mean([1 - judge_misleading, text["numeral_f1"]]), "judge_misleading": judge_misleading,
                        **efficiency_scores(float(preds["latency_s"].median()), spec, basis, eff_cfg)})
        for corpus, corpus_rows in rows.groupby("corpus"):
            text = text_scores(corpus_rows)
            seconds = float(preds.loc[preds["corpus"] == corpus, "latency_s"].median())
            # The judge sample is too small to split by corpus, so integrity per corpus rests on the numerals alone.
            by_corpus.append({**head, "corpus": corpus, **text, "integrity": text["numeral_f1"],
                              **efficiency_scores(seconds, spec, basis, eff_cfg)})
        runs.append({"run_id": run_id, "system_id": sid, "model_id": meta.model_id, "provider": meta.provider, "hardware": meta.hardware,
                     "gpu_name": meta.gpu_name, "prompt_id": meta.prompt_id, "started_at": meta.started_at, "finished_at": meta.finished_at,
                     "n_pages": len(rows), "n_failed": int(rows["error"].sum()), "notes": meta.notes})

    tables = {"profile": _ordered(profile), "profile_corpus": _ordered(by_corpus, "corpus"), "judge": judge,
              "corpora": corpora_table(manifest), "runs": pd.DataFrame(runs)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")
    log.info("tables written to %s: %s", out_dir, {k: len(v) for k, v in tables.items()})
    return tables
