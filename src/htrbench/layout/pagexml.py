"""PAGE-XML reader for the two places the format appears: Kraken's segmentation and Transkribus exports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

_NAMESPACE = re.compile(r"\{(http://schema\.primaresearch\.org/PAGE/gts/pagecontent/[0-9-]+)\}")


@dataclass
class PxLine:
    id: str
    polygon: list[tuple[int, int]]
    text: str = ""


@dataclass
class PxRegion:
    id: str
    lines: list[PxLine] = field(default_factory=list)


@dataclass
class PxPage:
    image_filename: str | None
    image_width: int | None
    image_height: int | None
    regions: list[PxRegion]
    reading_order: list[str]  # region ids; empty when the file declares none
    creator: str | None = None

    def lines(self) -> list[PxLine]:
        """Lines in reading order: regions by the declared order (undeclared ones last), lines as they stand in the file."""
        rank = {rid: i for i, rid in enumerate(self.reading_order)}
        regions = sorted(self.regions, key=lambda r: rank.get(r.id, len(rank)))
        return [ln for r in regions for ln in r.lines]

    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines())


def parse_points(points: str | None) -> list[tuple[int, int]]:
    out = []
    for pair in (points or "").split():
        x, _, y = pair.partition(",")
        try:
            out.append((int(float(x)), int(float(y))))
        except ValueError:
            continue
    return out


def bbox_of(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _line_text(line, q, ns) -> str:
    """The line's transcription: the TextEquiv with index 0 if the indices are given, else the first one."""
    equivs = line.findall(q("TextEquiv"), ns)
    if not equivs:
        return ""
    chosen = next((e for e in equivs if e.get("index") == "0"), equivs[0])
    unicode_el = chosen.find(q("Unicode"), ns)
    return (unicode_el.text or "") if unicode_el is not None else ""


def _parse(root) -> PxPage:
    m = _NAMESPACE.match(root.tag)
    ns = {"p": m.group(1)} if m else {}
    q = (lambda tag: f"p:{tag}") if m else (lambda tag: tag)

    page = root.find(f".//{q('Page')}", ns)
    if page is None:
        raise ValueError("no <Page> element")
    refs = page.findall(f"{q('ReadingOrder')}//{q('RegionRefIndexed')}", ns)
    reading_order = [r.get("regionRef") for r in sorted(refs, key=lambda r: int(r.get("index", "0"))) if r.get("regionRef")]

    regions: list[PxRegion] = []
    for el in page:
        if not etree.QName(el).localname.endswith("Region"):
            continue
        region = PxRegion(id=el.get("id", ""))
        for line in el.findall(q("TextLine"), ns):
            coords = line.find(q("Coords"), ns)
            polygon = parse_points(coords.get("points") if coords is not None else None)
            if polygon:
                region.lines.append(PxLine(id=line.get("id", ""), polygon=polygon, text=_line_text(line, q, ns)))
        regions.append(region)

    creator = root.find(f".//{q('Metadata')}/{q('Creator')}", ns)
    width, height = page.get("imageWidth", ""), page.get("imageHeight", "")
    return PxPage(
        image_filename=page.get("imageFilename"),
        image_width=int(width) if width.isdigit() else None,
        image_height=int(height) if height.isdigit() else None,
        regions=regions, reading_order=reading_order,
        creator=(creator.text or "").strip() if creator is not None else None,
    )


def read_page(path: Path | str) -> PxPage:
    return _parse(etree.parse(str(path)).getroot())


def read_page_from_string(xml: bytes) -> PxPage:
    return _parse(etree.fromstring(xml))


def scale_factor(px: PxPage, actual_w: int, actual_h: int) -> tuple[float, float]:
    """Factors that map the XML's coordinates onto the actual image (1.0 when the XML declares no size)."""
    if px.image_width and px.image_height:
        return actual_w / px.image_width, actual_h / px.image_height
    return 1.0, 1.0
