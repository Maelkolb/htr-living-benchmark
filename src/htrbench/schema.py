"""Data contracts of the benchmark.

Modules exchange data only through these models, stored as JSONL or JSON. The data release
(``configs/data.yaml``) is written in the same models, so a change here is a change of the release format.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Corpus = Literal["abp", "humboldt", "forst", "laubmann"]
SystemFamily = Literal["specialised_line", "ocr_pipeline", "vlm_open_local", "vlm_api", "transkribus"]
Hardware = Literal["rtx5060_8gb", "cpu_local", "api", "modal_a10g", "modal_a100", "modal_h100", "transkribus_cloud"]
HardwareClass = Literal["cpu", "gpu_8gb", "gpu_24gb", "gpu_80gb", "api", "transkribus"]


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Facets(StrictModel):
    """Curated description of a page; assigned per corpus in release 1.0."""

    genre: str
    period_bin: str
    script: str
    languages: list[str]
    layout_class: str
    condition: str
    source_type: str
    provenance: str = ""


class Page(StrictModel):
    page_id: str
    corpus: Corpus
    image_path: str  # POSIX path relative to the data directory
    width: int
    height: int
    sha256: str  # of the image file
    gt_text: str  # reference transcription, one physical line per text line
    gt_source: str
    n_lines_gt: int
    facets: Facets
    year: int | None = None
    judge_sample: bool = False  # the judge's line sample is drawn from these pages
    notes: str = ""


class Prediction(StrictModel):
    """One system's output for one page."""

    page_id: str
    text: str  # after runners/postprocess.clean
    text_raw: str = ""
    latency_s: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_thinking: int | None = None
    cost_usd: float | None = None
    gpu_s: float | None = None
    energy_wh: float | None = None
    error: str | None = None  # refusal, empty output or failure: the page counts as unread
    created_at: str = Field(default_factory=utcnow_iso)


class RunMeta(StrictModel):
    run_id: str  # <system_id>__r<run_idx>
    system_id: str
    model_id: str
    provider: str
    prompt_id: str
    prompt_sha256: str
    hardware: Hardware
    harness_version: str
    started_at: str
    run_idx: int = 0
    params: dict = Field(default_factory=dict)
    gpu_name: str | None = None
    git_commit: str | None = None
    finished_at: str | None = None
    n_pages: int = 0
    n_errors: int = 0
    notes: str = ""


class Score(StrictModel):
    """Counts for one page of one run. Every rate in the tables is a ratio of sums of these counts."""

    run_id: str
    system_id: str
    page_id: str
    corpus: Corpus
    error: bool  # no usable output: the whole reference counts as deleted
    # whole page, tier L1: reference and output joined in reading order and aligned once
    n_ref_chars: int
    n_ref_words: int
    ins: int
    dele: int
    sub: int
    word_edits: int
    # matched lines (eval/linematch.py), the unit of the counts below
    n_lines_ref: int
    n_lines_matched: int
    # word substitutions on matched lines and how many of them produced a real word (eval/plausibility.py)
    n_sub_words: int
    real_word_sub: int
    # diplomatic fidelity and numerals (eval/fidelity.py)
    n_glyph_ref: int
    glyph_kept: int
    n_glyph_hyp: int
    n_punct_ref: int
    punct_kept: int
    n_caps_read: int
    caps_lowered: int
    n_num_ref: int
    num_kept: int
    n_num_hyp: int


class PriceEntry(StrictModel):
    model_id: str
    provider: str
    usd_per_m_in: float
    usd_per_m_out: float
    verified_on: str
    source_url: str = ""
    usd_per_m_thinking: float | None = None  # billed as output when absent
    notes: str = ""


class SystemSpec(StrictModel):
    system_id: str
    label: str
    runner: str  # key of runners.registry.RUNNERS
    model_id: str
    provider: str
    family: SystemFamily
    hardware: Hardware  # what the archived run used
    hardware_class: HardwareClass  # the smallest machine that runs one page (D5)
    weights: Literal["open", "closed"]
    role: Literal["lineup", "field"] = "field"
    price_key: str | None = None
    params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------
# Bundle contract: systems that run on a rented GPU read a zip of BundleTasks and return BundlePredictions.
class BundleTask(StrictModel):
    page_id: str
    image: str  # relative to the bundle root
    prompt_id: str
    prompt: str
    model: str


class BundlePrediction(StrictModel):
    page_id: str
    text: str
    latency_s: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None
    energy_wh: float | None = None


class BundleRunInfo(StrictModel):
    gpu: str | None = None
    platform: str | None = None
    torch_version: str | None = None
    model_revision: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    total_wall_s: float | None = None
    notes: str = ""
