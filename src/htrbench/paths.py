"""Locations on disk. The repository holds code, configs and result tables; ``data/`` holds the data release."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    env = os.environ.get("HTRBENCH_ROOT")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").exists() and (p / "src" / "htrbench").exists():
            return p
    return Path.cwd()


ROOT = repo_root()
CONFIGS = ROOT / "configs"
PROMPTS = ROOT / "prompts"
BUNDLES = ROOT / "bundles"  # task bundles for systems that run elsewhere (not in the repository)
CACHE = ROOT / "cache"  # line crops cut for the line engines (not in the repository)

DATA = Path(os.environ.get("HTRBENCH_DATA") or ROOT / "data")
MANIFEST = DATA / "manifest.jsonl"
LAYOUT = DATA / "layout"  # the shared Kraken segmentation, one PAGE-XML per page
RUNS = DATA / "runs"
JUDGE = DATA / "judge"

RESULTS = ROOT / "results"
SCORES = RESULTS / "scores.parquet"
TABLES = RESULTS / "tables"


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def data_file(relative: str | Path) -> Path:
    """Absolute path of a file named relative to the data directory (``Page.image_path``)."""
    p = Path(relative)
    return p if p.is_absolute() else DATA / p
