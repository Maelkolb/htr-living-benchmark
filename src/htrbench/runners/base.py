"""Runner base class: resume, retries, concurrency, timing, cost and provenance.

A concrete runner implements ``predict_one(page, prompt)`` (API style, run in a thread pool) or overrides
``predict_batch(pages, prompt)`` (local models that batch on the GPU). The base class post-processes the text and
fills in the cost.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import __version__, paths
from ..io import append_jsonl, git_commit, read_jsonl, read_prompt, sha256_text, write_json
from ..schema import Page, Prediction, RunMeta, SystemSpec, utcnow_iso
from . import postprocess
from .power import gpu_name
from .pricing import cost_usd

log = logging.getLogger("htrbench.run")

PAGE_PROMPT = "page_v1"


class RunnerError(RuntimeError):
    """Retryable failure of a single prediction."""


class Refusal(RuntimeError):
    """Not retryable: the model refused, was safety-blocked or returned nothing."""

    label = "refusal"


def make_run_id(system_id: str, run_idx: int = 0) -> str:
    return f"{system_id}__r{run_idx}"


def load_pages() -> list[Page]:
    return read_jsonl(paths.MANIFEST, Page)


def last_predictions(path: Path) -> dict[str, Prediction]:
    """One prediction per page: a resumed run appends, so the last successful row of a page wins."""
    out: dict[str, Prediction] = {}
    for p in read_jsonl(path, Prediction):
        if p.page_id not in out or p.error is None:
            out[p.page_id] = p
    return out


def image_bytes(page: Page) -> tuple[bytes, str]:
    p = paths.data_file(page.image_path)
    return p.read_bytes(), "image/png" if p.suffix.lower() == ".png" else "image/jpeg"


class Runner:
    max_retries = 5
    backoff_base_s = 2.0

    def __init__(self, spec: SystemSpec, params: dict | None = None):
        self.spec = spec
        self.params = {**spec.params, **(params or {})}
        self.workers = int(self.params.get("workers", 1))
        self.max_retries = int(self.params.get("max_retries", type(self).max_retries))
        self.backoff_base_s = float(self.params.get("backoff_base_s", type(self).backoff_base_s))
        self.backoff_max_s = float(self.params.get("backoff_max_s", 60.0))
        self._ready = False

    def setup(self) -> None:
        """Load models or clients. Called once before the first prediction."""

    def teardown(self) -> None:
        """Release what ``setup`` acquired. Called once when the run ends."""

    def predict_one(self, page: Page, prompt: str) -> Prediction:
        raise NotImplementedError

    def predict_batch(self, pages: list[Page], prompt: str) -> list[Prediction]:
        """Thread pool over ``predict_one`` with retries. Local runners override this."""
        if self.workers <= 1:
            return [self._predict_with_retry(p, prompt) for p in pages]
        out: dict[str, Prediction] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = [ex.submit(self._predict_with_retry, p, prompt) for p in pages]
            for fut in as_completed(futures):
                pred = fut.result()
                out[pred.page_id] = pred
        return [out[p.page_id] for p in pages]

    def _predict_with_retry(self, page: Page, prompt: str) -> Prediction:
        last_err: str | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                pred = self.predict_one(page, prompt)
                if pred.latency_s == 0.0:
                    pred.latency_s = time.perf_counter() - t0
                return pred
            except Refusal as e:
                return Prediction(page_id=page.page_id, text="", text_raw=str(e), error=e.label, latency_s=time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"[:500]
                if attempt == self.max_retries - 1:
                    break
                pause = min(self.backoff_base_s * (2**attempt), self.backoff_max_s) + random.random()
                log.warning("%s attempt %d failed: %s (retry in %.1fs)", page.page_id, attempt + 1, last_err, pause)
                time.sleep(pause)
        return Prediction(page_id=page.page_id, text="", error=f"failed after retries: {last_err}")

    def run(self, pages: list[Page], run_idx: int = 0, resume: bool = True, budget_usd: float | None = None,
            notes: str = "") -> RunMeta:
        prompt_id = self.params.get("prompt_id", PAGE_PROMPT)
        prompt = read_prompt(prompt_id)
        run_id = make_run_id(self.spec.system_id, run_idx)
        out_dir = paths.RUNS / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path, meta_path = out_dir / "predictions.jsonl", out_dir / "meta.json"

        if not resume:
            pred_path.unlink(missing_ok=True)
        done: set[str] = set()
        if resume and pred_path.exists():
            done = {p.page_id for p in read_jsonl(pred_path, Prediction) if p.error is None}
        todo = [p for p in pages if p.page_id not in done]

        meta = RunMeta(
            run_id=run_id, system_id=self.spec.system_id, model_id=self.spec.model_id, provider=self.spec.provider,
            run_idx=run_idx, prompt_id=prompt_id, prompt_sha256=sha256_text(prompt), params=self.params,
            hardware=self.spec.hardware, gpu_name=gpu_name() if self.spec.hardware in ("rtx5060_8gb", "cpu_local") else None,
            git_commit=git_commit(), harness_version=__version__, started_at=utcnow_iso(), n_pages=len(pages), notes=notes,
        )
        if resume and meta_path.exists():
            old = RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
            meta.started_at = old.started_at
            meta.notes = " | ".join(x for x in (old.notes, notes) if x)
        write_json(meta_path, meta)
        log.info("run %s: %d pages (%d already done)", run_id, len(pages), len(done))

        if todo and not self._ready:
            self.setup()
            self._ready = True
        spent = 0.0
        batch = int(self.params.get("batch_size", 0)) or (self.workers * 4 if self.workers > 1 else 8)
        try:
            for i in range(0, len(todo), batch):
                for pred in self.predict_batch(todo[i : i + batch], prompt):
                    pred.text_raw = pred.text_raw or pred.text
                    if pred.error is None:
                        pred.text = postprocess.clean(pred.text)
                        if not pred.text.strip():
                            pred.error = "empty"
                    if pred.cost_usd is None and self.spec.price_key:
                        pred.cost_usd = cost_usd(self.spec.price_key, pred.tokens_in, pred.tokens_out, pred.tokens_thinking)
                    spent += pred.cost_usd or 0.0
                    append_jsonl(pred_path, pred)
                log.info("run %s: %d/%d done, spent $%.3f", run_id, min(i + batch, len(todo)), len(todo), spent)
                if budget_usd is not None and spent > budget_usd:
                    meta.notes = " | ".join(x for x in (meta.notes, f"aborted: budget of {budget_usd} USD exceeded") if x)
                    log.error("budget exceeded for %s (%.2f > %.2f)", run_id, spent, budget_usd)
                    break
        finally:
            self.teardown()
            meta.finished_at = utcnow_iso()
            meta.n_errors = sum(1 for p in last_predictions(pred_path).values() if p.error)
            write_json(meta_path, meta)
        return meta
