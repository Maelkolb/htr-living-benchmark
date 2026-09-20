"""I/O helpers: JSONL of pydantic models, YAML, hashing, prompts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel

from . import paths


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.exists():
        return []
    out: list[T] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(model.model_validate_json(line))
    return out


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")


def append_jsonl(path: Path, row: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(row.model_dump_json() + "\n")


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_prompt(prompt_id: str) -> str:
    return (paths.PROMPTS / f"{prompt_id}.txt").read_text(encoding="utf-8").strip()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None
