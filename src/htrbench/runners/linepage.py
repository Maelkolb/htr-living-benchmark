"""Turns a line recogniser into a page reader.

The page is segmented with Kraken (``data/layout/<page_id>.xml``, shared by every line engine and shipped with the
data release), the line polygons are cut out, recognised, and joined with newlines in the segmentation's reading
order. A subclass implements ``recognize_images`` and sets ``device``.

Wall time, GPU seconds and energy are measured per batch and divided among the pages in proportion to their
number of lines. Segmentation time is not part of a page's latency: the layout is a shared input, not a property
of the recogniser.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .. import paths
from ..layout.crop import make_line_crops
from ..layout.segment_kraken import read_lines, segment_page
from ..schema import Page, Prediction
from .power import PowerSampler

log = logging.getLogger("htrbench.linepage")


class LineModelMixin:
    device: str = "cpu"
    crop_pad: int = 8

    def recognize_images(self, images: list[Path]) -> list[str]:
        raise NotImplementedError

    def page_crops(self, page: Page) -> list[Path]:
        image = paths.data_file(page.image_path)
        xml = segment_page(image, paths.LAYOUT / f"{page.page_id}.xml")
        crops = make_line_crops(read_lines(xml, page), image, paths.CACHE / "line_crops" / page.page_id, pad=self.crop_pad)
        if not crops:
            raise RuntimeError("segmentation found no lines")
        return crops

    def _cuda_sync(self) -> None:
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.synchronize()

    def predict_batch(self, pages: list[Page], prompt: str) -> list[Prediction]:  # noqa: ARG002 - line engines take no prompt
        crops: dict[str, list[Path]] = {}
        errors: dict[str, str] = {}
        for page in pages:
            try:
                crops[page.page_id] = self.page_crops(page)
            except Exception as e:  # noqa: BLE001
                errors[page.page_id] = f"prepare: {type(e).__name__}: {e}"[:500]
        flat = [c for page in pages for c in crops.get(page.page_id, [])]

        texts: list[str] = []
        wall, energy = 0.0, None
        if flat:
            self._cuda_sync()
            t0 = time.perf_counter()
            try:
                with PowerSampler() as power:
                    texts = list(self.recognize_images(flat))
                    self._cuda_sync()
                if len(texts) != len(flat):
                    raise RuntimeError(f"recognize_images returned {len(texts)} texts for {len(flat)} images")
                energy = power.energy_wh if str(self.device).startswith("cuda") else None
            except Exception as e:  # noqa: BLE001
                log.error("batch failed: %s", e)
                errors.update({pid: f"recognize: {type(e).__name__}: {e}"[:500] for pid in crops})
                texts = []
            wall = time.perf_counter() - t0

        on_gpu = str(self.device).startswith("cuda")
        s_per_line = wall / len(flat) if flat else 0.0
        wh_per_line = energy / len(flat) if energy is not None and flat else None
        preds: list[Prediction] = []
        pos = 0
        for page in pages:
            n = len(crops.get(page.page_id, []))
            if page.page_id in errors:
                preds.append(Prediction(page_id=page.page_id, text="", error=errors[page.page_id], latency_s=s_per_line * n))
                continue
            lines, pos = texts[pos : pos + n], pos + n
            preds.append(Prediction(
                page_id=page.page_id, text="\n".join(s.strip() for s in lines), latency_s=s_per_line * n,
                gpu_s=s_per_line * n if on_gpu else None, energy_wh=wh_per_line * n if wh_per_line is not None else None,
            ))
        return preds
