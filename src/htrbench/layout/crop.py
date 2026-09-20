"""Line crops for the line engines: the line polygon on white, inside its padded bounding box."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .segment_kraken import SegLine


def crop_polygon(img: Image.Image, polygon: list[tuple[int, int]], pad: int = 8) -> Image.Image:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    width, height = img.size
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    x0, y0 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
    x1, y1 = min(max(xs) + pad, width), min(max(ys) + pad, height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("degenerate polygon")
    region = img.crop((x0, y0, x1, y1))
    mask = Image.new("L", region.size, 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in polygon], fill=255)
    mask = mask.filter(ImageFilter.MaxFilter(5))  # keep ascenders and descenders that touch the polygon edge
    white = Image.new(region.mode, region.size, 255 if region.mode == "L" else (255, 255, 255))
    return Image.composite(region, white, mask)


def make_line_crops(lines: list[SegLine], page_image: Path, out_dir: Path, pad: int = 8) -> list[Path]:
    """One PNG per line under ``out_dir`` (existing files are kept); degenerate polygons are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(page_image)
    img.load()
    out: list[Path] = []
    for ln in lines:
        dest = out_dir / f"line_{ln.idx:04d}.png"
        if not dest.exists():
            try:
                crop_polygon(img, ln.polygon, pad=pad).save(dest, format="PNG", optimize=True)
            except ValueError:
                continue
        out.append(dest)
    return out
