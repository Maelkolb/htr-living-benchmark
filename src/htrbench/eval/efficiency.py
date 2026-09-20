"""D5, efficiency: what one page costs in time, hardware and money, as three absolute 0-1 facets.

    speed     seconds per page on a log scale between a floor (scores 1) and a ceiling (scores 0)
    hardware  the smallest machine that runs one page, from a fixed ladder (CPU 1 ... hosted 0.1)
    price     USD per page on a log scale between a floor (scores 1) and a ceiling (scores 0)

    efficiency = speed / 2 + (hardware + price) / 4

Anchors, ladder and weights: configs/protocol.yaml, explained in docs/METHODS.md, D5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

HOSTED = ("api", "transkribus")


@dataclass(frozen=True)
class Efficiency:
    speed: float
    hardware: float
    price: float
    usd_per_page: float
    price_basis: str

    def score(self, weights: dict[str, float]) -> float:
        """Weighted mean of the facets that could be computed."""
        parts = [(getattr(self, k), w) for k, w in weights.items() if not math.isnan(getattr(self, k))]
        total = sum(w for _, w in parts)
        return sum(v * w for v, w in parts) / total if total else math.nan


def log_facet(value: float, floor: float, ceiling: float) -> float:
    """1 at ``floor`` or below, 0 at ``ceiling`` or above, linear in log10 between them."""
    if value is None or math.isnan(value) or value <= 0:
        return math.nan
    span = math.log10(ceiling) - math.log10(floor)
    return max(0.0, min(1.0, (math.log10(ceiling) - math.log10(value)) / span))


def price_per_page(hardware_class: str, usd_per_page: float, wh_per_page: float, usd_per_kwh: float) -> tuple[float, str]:
    if hardware_class in HOSTED:
        return usd_per_page, "list price" if hardware_class == "api" else "credits"
    if not math.isnan(wh_per_page):
        return wh_per_page / 1000 * usd_per_kwh, "electricity"
    return math.nan, "not priced"


def efficiency(s_per_page: float, hardware_class: str, usd_per_page: float, wh_per_page: float, usd_per_kwh: float,
               cfg: dict) -> Efficiency:
    usd, basis = price_per_page(hardware_class, usd_per_page, wh_per_page, usd_per_kwh)
    return Efficiency(
        speed=log_facet(s_per_page, cfg["speed"]["s_floor"], cfg["speed"]["s_ceiling"]),
        hardware=float(cfg["hardware"][hardware_class]),
        price=log_facet(usd, cfg["price"]["usd_floor"], cfg["price"]["usd_ceiling"]),
        usd_per_page=usd, price_basis=basis,
    )
