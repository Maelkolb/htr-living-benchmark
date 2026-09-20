"""Prices from configs/prices.yaml: API list prices per token, Transkribus credits, household electricity."""

from __future__ import annotations

from functools import lru_cache

from .. import paths
from ..io import read_yaml
from ..schema import PriceEntry


@lru_cache(maxsize=1)
def _config() -> dict:
    return read_yaml(paths.CONFIGS / "prices.yaml")


@lru_cache(maxsize=1)
def price_table() -> dict[str, PriceEntry]:
    return {m["model_id"]: PriceEntry.model_validate(m) for m in _config()["models"]}


def electricity() -> dict:
    return _config()["electricity"]


def transkribus() -> dict:
    return _config()["transkribus"]


def cost_usd(price_key: str | None, tokens_in: int | None, tokens_out: int | None, tokens_thinking: int | None = None) -> float | None:
    p = price_table().get(price_key or "")
    if p is None:
        return None
    thinking_rate = p.usd_per_m_thinking if p.usd_per_m_thinking is not None else p.usd_per_m_out
    return ((tokens_in or 0) * p.usd_per_m_in + (tokens_out or 0) * p.usd_per_m_out + (tokens_thinking or 0) * thinking_rate) / 1e6
