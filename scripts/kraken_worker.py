"""JSON-lines recognition worker, run inside .venv-kraken (kraken 7.x).

    python scripts/kraken_worker.py <model.mlmodel> [cpu|cuda] [bbox|baseline]

stdin : one JSON object per line: {"paths": ["abs/crop1.png", ...]}  (or {"cmd": "quit"})
stdout: {"ready": true, ...} once, then one {"texts": [...], "errors": [...]} per request.
Each crop is recognised as ONE line: a bbox Segmentation covering the whole image (falls back to a
straight-baseline BaselineLine with a rectangular boundary if the bbox path raises).
"""

from __future__ import annotations

import json
import logging
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from kraken import rpred  # noqa: E402
from kraken.containers import BaselineLine, BBoxLine, Segmentation  # noqa: E402
from kraken.lib import models  # noqa: E402
from PIL import Image  # noqa: E402


def _seg(kind: str, path: str, w: int, h: int) -> Segmentation:
    if kind == "bbox":
        return Segmentation(type="bbox", imagename=path, text_direction="horizontal-lr", script_detection=False,
                            lines=[BBoxLine(id="l0", bbox=(0, 0, w, h))])
    y = int(h * 0.75)
    return Segmentation(type="baselines", imagename=path, text_direction="horizontal-lr", script_detection=False,
                        lines=[BaselineLine(id="l0", baseline=[(0, y), (w, y)], boundary=[(0, 0), (w, 0), (w, h), (0, h)])])


def recognise(net, path: str, mode: str) -> tuple[str, str | None]:
    im = Image.open(path)
    im.load()
    w, h = im.size
    kinds = [mode] + [k for k in ("bbox", "baseline") if k != mode]
    err = None
    for kind in kinds:
        try:
            recs = list(rpred.rpred(net, im, _seg(kind, path, w, h), pad=16))
            return " ".join(r.prediction for r in recs).strip(), None
        except Exception as e:  # noqa: BLE001
            err = f"{kind}: {type(e).__name__}: {e}"
    return "", err


def main() -> None:
    model_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    mode = sys.argv[3] if len(sys.argv) > 3 else "bbox"
    net = models.load_any(model_path, device=device)
    out = sys.stdout
    out.write(json.dumps({"ready": True, "model": model_path, "seg_type": getattr(net, "seg_type", None),
                          "device": device, "mode": mode}) + "\n")
    out.flush()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        if req.get("cmd") == "quit":
            break
        texts, errors = [], []
        for p in req.get("paths", []):
            t, e = recognise(net, p, mode)
            texts.append(t)
            errors.append(e)
        out.write(json.dumps({"texts": texts, "errors": errors}, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
