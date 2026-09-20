"""The shared page layout: Kraken baseline segmentation (blla), one PAGE-XML per page in ``data/layout/``.

Kraken does not install next to the main torch stack, so it lives in its own environment
(``scripts/setup_kraken.ps1``) and is called as a subprocess:
``kraken -x -i <image> <out.xml> segment -bl``. Every line engine reads the same segmentation; the data release
ships the files the archived runs used, so a rerun does not depend on the Kraken version.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..schema import Page
from .pagexml import bbox_of, read_page, scale_factor

log = logging.getLogger("htrbench.segment")

_SCRIPTS = "Scripts" if sys.platform == "win32" else "bin"
_EXE = ".exe" if sys.platform == "win32" else ""
KRAKEN_EXE = Path(os.environ.get("HTRBENCH_KRAKEN_EXE", paths.ROOT / ".venv-kraken" / _SCRIPTS / f"kraken{_EXE}"))
KRAKEN_PYTHON = Path(os.environ.get("HTRBENCH_KRAKEN_PYTHON", paths.ROOT / ".venv-kraken" / _SCRIPTS / f"python{_EXE}"))
SEGMENT_TIMEOUT_S = 600


@dataclass(frozen=True)
class SegLine:
    idx: int  # position in the segmentation's reading order
    polygon: list[tuple[int, int]]  # page pixels


def segment_page(image_path: Path, out_xml: Path) -> Path:
    """Segment one page image; an existing PAGE-XML is reused."""
    if out_xml.exists() and out_xml.stat().st_size > 0:
        return out_xml
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not KRAKEN_EXE.exists():
        raise FileNotFoundError(f"kraken not found at {KRAKEN_EXE}: run scripts/setup_kraken.ps1 or set HTRBENCH_KRAKEN_EXE")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(KRAKEN_EXE), "-x", "-i", image_path.as_posix(), out_xml.as_posix(), "segment", "-bl"]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=SEGMENT_TIMEOUT_S,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if not out_xml.exists() or out_xml.stat().st_size == 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise RuntimeError(f"kraken segment failed (rc={proc.returncode}) for {image_path.name}: {tail}")
    log.info("segmented %s in %.1fs", image_path.name, time.perf_counter() - t0)
    return out_xml


def read_lines(xml_path: Path, page: Page) -> list[SegLine]:
    """Lines of a PAGE-XML in document order, scaled to the page's pixel size."""
    px = read_page(xml_path)
    sx, sy = scale_factor(px, page.width, page.height)
    out: list[SegLine] = []
    for idx, ln in enumerate(px.lines()):
        poly = [(int(round(x * sx)), int(round(y * sy))) for x, y in ln.polygon]
        if len(poly) < 3:
            x0, y0, x1, y1 = bbox_of(poly)
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        out.append(SegLine(idx=idx, polygon=poly))
    return out
