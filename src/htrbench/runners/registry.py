"""system_id -> SystemSpec -> Runner. Runner modules are imported on demand so that optional dependencies stay optional."""

from __future__ import annotations

import importlib
from functools import lru_cache

from .. import paths
from ..io import read_yaml
from ..schema import SystemSpec

RUNNERS: dict[str, tuple[str, str]] = {
    "gemini": ("htrbench.runners.gemini", "GeminiRunner"),
    "anthropic": ("htrbench.runners.anthropic_", "AnthropicRunner"),
    "trocr": ("htrbench.runners.trocr", "TrOCRRunner"),
    "kraken_rec": ("htrbench.runners.kraken_rec", "KrakenRecRunner"),
    "bundle": ("htrbench.runners.bundle", "BundleRunner"),
    "transkribus": ("htrbench.runners.transkribus", "TranskribusRunner"),
}


@lru_cache(maxsize=1)
def systems() -> dict[str, SystemSpec]:
    specs = [SystemSpec.model_validate(s) for s in read_yaml(paths.CONFIGS / "systems.yaml")["systems"]]
    return {s.system_id: s for s in specs}


def spec(system_id: str) -> SystemSpec:
    try:
        return systems()[system_id]
    except KeyError:
        raise SystemExit(f"unknown system_id {system_id!r}; known: {', '.join(systems())}") from None


def lineup() -> list[str]:
    return [sid for sid, s in systems().items() if s.role == "lineup"]


def build(system_id: str, params: dict | None = None):
    s = spec(system_id)
    module, cls = RUNNERS[s.runner]
    return getattr(importlib.import_module(module), cls)(s, params)
