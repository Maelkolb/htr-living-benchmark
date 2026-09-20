"""Bundle contract for systems that run on a rented GPU (``scripts/modal_runner.py``).

``export_bundle`` writes the pages, the prompt and a task list into ``bundles/<run_id>.zip``; the remote side returns
``predictions.jsonl`` (``BundlePrediction`` rows), ``run_info.json`` and the bundle's own ``bundle_meta.json``, and
``import_bundle`` turns them into a run.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image

from .. import __version__, paths
from ..io import read_prompt, sha256_text, write_json, write_jsonl
from ..schema import BundlePrediction, BundleRunInfo, BundleTask, Prediction, RunMeta, utcnow_iso
from . import postprocess
from .base import PAGE_PROMPT, Runner, load_pages, make_run_id
from .registry import spec as system_spec

#: longer side of the exported JPEGs; the remote runner works at this size, so nothing larger needs uploading
BUNDLE_MAX_SIDE = 1800


def export_bundle(system_id: str, run_idx: int = 0, limit: int | None = None) -> Path:
    sp = system_spec(system_id)
    pages = load_pages()[:limit] if limit else load_pages()
    prompt_id = sp.params.get("prompt_id", PAGE_PROMPT)
    prompt = read_prompt(prompt_id)
    name = make_run_id(system_id, run_idx)
    root = paths.BUNDLES / name
    (root / "images").mkdir(parents=True, exist_ok=True)
    rows: list[BundleTask] = []
    for page in pages:
        rel = f"images/{page.page_id}.jpg"
        if not (root / rel).exists():
            with Image.open(paths.data_file(page.image_path)) as im:
                im = im.convert("RGB")
                im.thumbnail((BUNDLE_MAX_SIDE, BUNDLE_MAX_SIDE), Image.Resampling.LANCZOS)
                im.save(root / rel, format="JPEG", quality=90)
        rows.append(BundleTask(page_id=page.page_id, image=rel, prompt_id=prompt_id, prompt=prompt, model=sp.model_id))
    write_jsonl(root / "tasks.jsonl", rows)
    write_json(root / "bundle_meta.json", {
        "bundle": name, "system_id": system_id, "model_id": sp.model_id, "run_idx": run_idx, "prompt_id": prompt_id,
        "prompt_sha256": sha256_text(prompt), "harness_version": __version__, "created_at": utcnow_iso(), "n_pages": len(rows),
    })
    zpath = paths.BUNDLES / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
    print(f"bundle {name}: {len(rows)} pages -> {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath


def _read_results(path: Path) -> tuple[list[BundlePrediction], BundleRunInfo | None, dict]:
    """Results as a directory or a zip: predictions.jsonl, run_info.json, bundle_meta.json."""
    files: dict[str, str] = {}
    if path.is_dir():
        files = {p.name: p.read_text(encoding="utf-8") for p in path.iterdir() if p.is_file()}
    else:
        with zipfile.ZipFile(path) as z:
            files = {n.rsplit("/", 1)[-1]: z.read(n).decode("utf-8") for n in z.namelist() if not n.endswith("/")}
    if "predictions.jsonl" not in files or "bundle_meta.json" not in files:
        raise SystemExit(f"{path}: predictions.jsonl and bundle_meta.json expected")
    preds = [BundlePrediction.model_validate_json(ln) for ln in files["predictions.jsonl"].splitlines() if ln.strip()]
    info = BundleRunInfo.model_validate_json(files["run_info.json"]) if "run_info.json" in files else None
    return preds, info, json.loads(files["bundle_meta.json"])


def _hardware(gpu: str | None) -> str:
    name = (gpu or "").upper()
    for key, hardware in (("H100", "modal_h100"), ("A100", "modal_a100"), ("A10", "modal_a10g")):
        if key in name:
            return hardware
    raise SystemExit(f"unknown GPU {gpu!r}: add it to the table above and to schema.Hardware")


def import_bundle(path: str | Path) -> RunMeta:
    path = Path(path)
    preds, info, bmeta = _read_results(path)
    sp = system_spec(bmeta["system_id"])
    run_id = make_run_id(sp.system_id, int(bmeta.get("run_idx", 0)))
    by_page = {p.page_id: p for p in preds}
    out: list[Prediction] = []
    for page in load_pages():  # manifest order; a page the bundle did not cover simply has no row
        p = by_page.get(page.page_id)
        if p is None:
            continue
        text = "" if p.error else postprocess.clean(p.text)
        error = p.error or (None if text.strip() else "empty")
        out.append(Prediction(page_id=page.page_id, text=text, text_raw=p.text or "", latency_s=p.latency_s or 0.0, gpu_s=p.latency_s,
                              tokens_in=p.tokens_in, tokens_out=p.tokens_out, energy_wh=p.energy_wh, error=error))
    out_dir = paths.RUNS / run_id
    write_jsonl(out_dir / "predictions.jsonl", out)
    meta = RunMeta(
        run_id=run_id, system_id=sp.system_id, model_id=sp.model_id, provider=sp.provider, run_idx=int(bmeta.get("run_idx", 0)),
        prompt_id=bmeta["prompt_id"], prompt_sha256=bmeta["prompt_sha256"], params={"run_info": info.model_dump() if info else None},
        hardware=_hardware(info.gpu if info else None), gpu_name=info.gpu if info else None, harness_version=__version__,  # type: ignore[arg-type]
        started_at=(info.started_at if info and info.started_at else utcnow_iso()), finished_at=(info.finished_at if info else utcnow_iso()),
        n_pages=len(out), n_errors=sum(1 for p in out if p.error), notes=f"imported from bundle results {path.name}",
    )
    write_json(out_dir / "meta.json", meta)
    print(f"imported {run_id}: {len(out)} pages, {meta.n_errors} errors, hardware {meta.hardware}")
    return meta


class BundleRunner(Runner):
    """``htrbench run`` for a bundle system: import the results if they are there, otherwise export the bundle."""

    def run(self, pages, run_idx=0, resume=True, budget_usd=None, notes=""):  # type: ignore[override]  # noqa: ARG002
        run_id = make_run_id(self.spec.system_id, run_idx)
        for results in (paths.BUNDLES / f"{run_id}.results.zip", paths.BUNDLES / f"{run_id}.results"):
            if results.exists():
                return import_bundle(results)
        zpath = export_bundle(self.spec.system_id, run_idx=run_idx)
        raise SystemExit(f"{run_id} runs elsewhere: process {zpath.name} with scripts/modal_runner.py, put the results at "
                         f"{paths.BUNDLES / (run_id + '.results')} and run this command again.")
