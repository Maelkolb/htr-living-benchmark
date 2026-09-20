"""The LLM judge behind D4: would a reader be misled?

The judge sees a reference line, the machine's line and the neighbouring reference lines, never the image. It lists
every difference with a type and a severity and gives the line a verdict: usable, needs_correction or misleading
(``prompts/judge_v1.txt``). Each line is judged twice at temperature 0.

Sample: pages flagged ``judge_sample`` in the manifest; matched lines of at least three words whose tier-L1 words
differ from the reference; up to ``lines_per_system`` per system, the corpora in turn, and the same reference lines
across systems wherever a system has them, so that systems are judged on the same material.

Judgements are stored in ``data/judge/<judge model>/<system_id>.jsonl``, one row per line and pass. They are part of
the data release: the judge costs money and is not deterministic, so the tables are built from the stored rows.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .. import paths
from ..io import read_jsonl, read_prompt, read_yaml
from ..runners.pricing import cost_usd
from ..runners.registry import lineup, systems
from ..schema import Page
from .linematch import match_lines, split_lines
from .normalize import normalize
from .score import load_predictions

log = logging.getLogger("htrbench.judge")

TYPES = ["garble", "spelling_or_normalisation", "abbreviation_expansion", "meaning_preserving_substitution",
         "meaning_changing_substitution", "name_error", "number_error", "omission", "hallucinated_insertion", "punctuation_case"]
VERDICTS = ["usable", "needs_correction", "misleading"]
_TYPE_ALIASES = {"normalisation": "spelling_or_normalisation", "spelling": "spelling_or_normalisation", "name": "name_error",
                 "number": "number_error", "insertion": "hallucinated_insertion", "punctuation": "punctuation_case"}


class JudgeSpan(BaseModel):
    ref_span: str = ""
    hyp_span: str = ""
    type: str
    severity: int = Field(default=0, ge=0, le=2)


class JudgeLine(BaseModel):
    """Response schema handed to the judge model."""

    spans: list[JudgeSpan] = Field(default_factory=list)
    verdict: Literal["usable", "needs_correction", "misleading"]
    note: str = ""


@dataclass(frozen=True)
class Sample:
    system_id: str
    page_id: str
    corpus: str
    line_idx: int
    ref: str
    hyp: str
    context_before: str
    context_after: str
    language: str
    genre: str


def config() -> dict:
    return read_yaml(paths.CONFIGS / "protocol.yaml")["judge"]


# --------------------------------------------------------------------------------------------------- sampling
def sample_lines(system_ids: list[str], n_per_system: int, seed: int) -> list[Sample]:
    pages = sorted((p for p in read_jsonl(paths.MANIFEST, Page) if p.judge_sample), key=lambda p: p.page_id)
    candidates: dict[str, dict[tuple[str, int], Sample]] = {}
    for sid in system_ids:
        preds = load_predictions(f"{sid}__r0")
        candidates[sid] = {}
        for page in pages:
            pred = preds.get(page.page_id)
            if pred is None or pred.error or not pred.text:
                continue
            refs = split_lines(page.gt_text)
            for pair in match_lines(refs, split_lines(pred.text)).pairs:
                ref_words = normalize(pair.gt, "L1").split()
                if pair.kind == "missing" or len(ref_words) < 3 or ref_words == normalize(pair.hyp, "L1").split():
                    continue
                i = pair.gt_idx
                candidates[sid][(page.page_id, i)] = Sample(
                    system_id=sid, page_id=page.page_id, corpus=page.corpus, line_idx=i, ref=pair.gt, hyp=pair.hyp,
                    context_before=refs[i - 1] if i > 0 else "", context_after=refs[i + 1] if i + 1 < len(refs) else "",
                    language=", ".join(page.facets.languages) or "German", genre=page.facets.genre)

    # One shared order of reference lines: shuffled within each corpus, lines that more systems have first, corpora in turn.
    keys = sorted({k for c in candidates.values() for k in c})
    corpus_of = {k: next(c[k].corpus for c in candidates.values() if k in c) for k in keys}
    by_corpus: dict[str, list] = {}
    for k in keys:
        by_corpus.setdefault(corpus_of[k], []).append(k)
    rng = random.Random(seed)
    for corpus in by_corpus:
        rng.shuffle(by_corpus[corpus])
        by_corpus[corpus].sort(key=lambda k: -sum(k in c for c in candidates.values()))
    order = []
    while any(by_corpus.values()):
        for corpus in sorted(by_corpus):
            if by_corpus[corpus]:
                order.append(by_corpus[corpus].pop(0))
    return [candidates[sid][k] for sid in system_ids for k in [k for k in order if k in candidates[sid]][:n_per_system]]


# --------------------------------------------------------------------------------------------------- the call
def judge_prompt(s: Sample, prompt_id: str) -> str:
    fields = {"context_before": s.context_before or "(start of page)", "context_after": s.context_after or "(end of page)",
              "ref": s.ref, "hyp": s.hyp or "(empty)", "language": s.language, "genre": s.genre}
    text = read_prompt(prompt_id)
    for name, value in fields.items():
        text = text.replace("{" + name + "}", value)
    return text.replace("{{", "{").replace("}}", "}")


def parse_judge(raw: str) -> JudgeLine:
    """The model's JSON, with unknown type and verdict labels mapped onto the closed lists."""
    data = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip())
    if isinstance(data, list):
        data = {"spans": data, "verdict": "needs_correction"}
    spans = []
    for sp in data.get("spans") or []:
        if not isinstance(sp, dict):
            continue
        label = str(sp.get("type", "")).strip().lower().replace(" ", "_").replace("-", "_")
        if label not in TYPES:
            label = _TYPE_ALIASES.get(label, "garble")
        try:
            severity = max(0, min(2, int(sp.get("severity", 0))))
        except (TypeError, ValueError):
            severity = 0
        spans.append(JudgeSpan(ref_span=str(sp.get("ref_span", "")), hyp_span=str(sp.get("hyp_span", "")), type=label, severity=severity))
    verdict = str(data.get("verdict", "needs_correction")).strip().lower().replace(" ", "_")
    if verdict not in VERDICTS:
        verdict = "misleading" if "mislead" in verdict else "usable" if "usable" in verdict else "needs_correction"
    return JudgeLine(spans=spans, verdict=verdict, note=str(data.get("note", ""))[:300])


def call_judge(client, model_id: str, prompt: str, retries: int = 4) -> tuple[JudgeLine, dict, str]:
    from google.genai import types

    from ..runners.gemini import safety_off

    cfg = types.GenerateContentConfig(
        temperature=0.0, max_output_tokens=1024, response_mime_type="application/json", response_schema=JudgeLine,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"), safety_settings=safety_off(),
    )
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model_id, contents=[prompt], config=cfg)
            break
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raw = resp.text or ""
    um = resp.usage_metadata
    usage = {"tokens_in": getattr(um, "prompt_token_count", None), "tokens_out": getattr(um, "candidates_token_count", None),
             "tokens_thinking": getattr(um, "thoughts_token_count", None)}
    return parse_judge(raw), usage, raw


def _out_path(model_id: str, system_id: str) -> Path:
    return paths.JUDGE / model_id / f"{system_id}.jsonl"


def _stored(model_id: str, system_id: str) -> list[dict]:
    p = _out_path(model_id, system_id)
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] if p.exists() else []


def run_judge(system_ids: list[str] | None = None, workers: int = 4, dry_run: bool = False, budget_usd: float = 8.0) -> None:
    cfg = config()
    model_id, prompt_id, passes = cfg["model_id"], cfg["prompt_id"], int(cfg["passes"])
    samples = sample_lines(system_ids or lineup(), int(cfg["lines_per_system"]), int(cfg["seed"]))
    todo = []
    for sid in {s.system_id for s in samples}:
        done = {(r["page_id"], r["line_idx"], r["pass_idx"]) for r in _stored(model_id, sid)}
        todo += [(s, k) for s in samples if s.system_id == sid for k in range(passes) if (s.page_id, s.line_idx, k) not in done]
    # pre-flight estimate for the budget guard only: about 4 characters a token, 20 of envelope, 180 out per line
    tokens_in = sum(len(judge_prompt(s, prompt_id)) // 4 + 20 for s, _ in todo)
    estimate = cost_usd(model_id, tokens_in, 180 * len(todo)) or 0.0
    print(f"judge {model_id}: {len(samples)} lines, {len(todo)} calls to make, about ${estimate:.2f}")
    if dry_run or not todo:
        return
    if estimate > budget_usd:
        raise SystemExit(f"estimated ${estimate:.2f} exceeds the budget of ${budget_usd:.2f}; raise --budget-usd to proceed")
    from ..runners.gemini import make_client

    client = make_client(120)

    def one(item: tuple[Sample, int]) -> dict:
        s, k = item
        t0 = time.perf_counter()
        line, usage, raw = call_judge(client, model_id, judge_prompt(s, prompt_id))
        return {**asdict(s), "pass_idx": k, "prompt_id": prompt_id, "model_id": model_id,
                "spans": [sp.model_dump() for sp in line.spans], "verdict": line.verdict, "note": line.note, **usage,
                "cost_usd": cost_usd(model_id, usage["tokens_in"], usage["tokens_out"], usage["tokens_thinking"]),
                "latency_s": time.perf_counter() - t0, "raw": raw[:4000]}

    spent, failed = 0.0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, it) for it in todo]), 1):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001 - a failed call is simply not stored; the next run makes it again
                failed += 1
                log.warning("judge call failed: %s", str(e)[:200])
                continue
            out = _out_path(model_id, row["system_id"])
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            spent += row["cost_usd"] or 0.0
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} calls, ${spent:.2f}, {failed} failed")


# --------------------------------------------------------------------------------------------------- evaluation
def load_judged(model_id: str | None = None) -> pd.DataFrame:
    """Every stored judgement of lines on manifest pages, one row per (system, page, line, pass)."""
    model_id = model_id or config()["model_id"]
    known = {p.page_id for p in read_jsonl(paths.MANIFEST, Page)}
    rows = {}
    for path in sorted((paths.JUDGE / model_id).glob("*.jsonl")):
        for r in _stored(model_id, path.stem):
            if r["page_id"] in known:
                rows[(r["system_id"], r["page_id"], r["line_idx"], r["pass_idx"])] = r
    return pd.DataFrame(rows.values())


def kappa(a: list, b: list) -> float:
    """Cohen's kappa of two equally long label lists."""
    n = len(a)
    if n == 0:
        return math.nan
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def judge_table(judged: pd.DataFrame) -> pd.DataFrame:
    """Per system: what the judge found per 100 reference words (first pass), the verdict shares (first pass), the share
    of misleading lines over both passes (the judge facet of D4) and the agreement between the passes."""
    if judged.empty:
        return pd.DataFrame()
    labels = {sid: s.label for sid, s in systems().items()}
    rows = []
    for sid, g in judged.groupby("system_id"):
        first = g[g["pass_idx"] == 0].set_index(["page_id", "line_idx"])
        second = g[g["pass_idx"] == 1].set_index(["page_id", "line_idx"])
        n_words = sum(len(str(ref).split()) for ref in first["ref"])
        row = {"system_id": sid, "label": labels.get(sid, sid), "n_lines": len(first), "n_ref_words": n_words}
        for t in TYPES:
            row[f"{t}_per_100w"] = 100 * sum(sp["type"] == t for spans in first["spans"] for sp in spans) / n_words
        for v in VERDICTS:
            row[f"share_{v}"] = float((first["verdict"] == v).mean())
        row["share_misleading_both_passes"] = float((g["verdict"] == "misleading").mean())
        common = first.index.intersection(second.index)
        row["kappa_verdict"] = kappa(list(first.loc[common, "verdict"]), list(second.loc[common, "verdict"]))
        rows.append(row)
    return pd.DataFrame(rows)
