"""Transkribus models run in the web app: export the page images, import the PAGE-XML export as a run.

The text of a page is the text of its lines in the export's reading order. Transkribus reports no time per page, so
the wall time of the whole job (``--wall-s``) is divided by the pages it processed; credits likewise.
"""

from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path

from .. import __version__, paths
from ..io import write_json, write_jsonl
from ..layout.pagexml import PxPage, read_page, read_page_from_string
from ..schema import Page, Prediction, RunMeta, utcnow_iso
from .base import Runner, load_pages, make_run_id
from .registry import spec as system_spec

EXPORT_DIR = paths.BUNDLES / "transkribus"
_NOT_PAGES = ("mets.xml", "metadata.xml")


def export_images() -> Path:
    """Copy every page image to ``bundles/transkribus/images/`` under its page id, the name the importer matches on."""
    images = EXPORT_DIR / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    pages = load_pages()
    with (EXPORT_DIR / "pages.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["page_id", "corpus", "image"])
        for p in pages:
            src = paths.data_file(p.image_path)
            shutil.copyfile(src, images / f"{p.page_id}{src.suffix.lower()}")
            w.writerow([p.page_id, p.corpus, f"{p.page_id}{src.suffix.lower()}"])
    print(f"{len(pages)} page images -> {images}\n"
          "Upload them as one document, run the layout analysis and one text-recognition model, export as PAGE-XML,\n"
          "then: htrbench transkribus-import <export.zip> <system_id> --wall-s <seconds> --credits <credits> --pages-run <pages>")
    return EXPORT_DIR


def _iter_pagexml(path: Path):
    if path.is_dir():
        for p in sorted(path.rglob("*.xml")):
            if p.name.lower() not in _NOT_PAGES:
                yield p.stem, read_page(p)
        return
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            base = name.rsplit("/", 1)[-1]
            if base.lower().endswith(".xml") and base.lower() not in _NOT_PAGES:
                yield Path(base).stem, read_page_from_string(z.read(name))


def _page_id(stem: str, px: PxPage, known: dict[str, Page]) -> str | None:
    names = [stem] + ([Path(px.image_filename).stem] if px.image_filename else [])
    names += [re.sub(r"^\d{3,5}_", "", n) for n in names]  # the export prefixes each file with its page number
    return next((n for n in names if n in known), None)


def import_pagexml(path: str | Path, system_id: str, wall_s: float | None = None, credits_used: float | None = None,
                   pages_run: int | None = None) -> RunMeta:
    """``pages_run``: pages the web app processed in the job, when that is more than the pages imported here; it keeps
    seconds and credits per page at what was actually spent."""
    path = Path(path)
    sp = system_spec(system_id)
    known = {p.page_id: p for p in load_pages()}
    preds: dict[str, Prediction] = {}
    other_model: list[str] = []
    for stem, px in _iter_pagexml(path):
        pid = _page_id(stem, px, known)
        if pid is None or pid in preds:
            continue
        # A document that several models have run on exports the latest version of each page. The PAGE-XML Creator
        # names the model ("...:model_id=265149:..."), so pages last written by another model are left out.
        made_by = re.search(r"model_id=(\d+)", px.creator or "")
        if made_by and made_by.group(1) != sp.model_id:
            other_model.append(pid)
            continue
        text = px.text()
        preds[pid] = Prediction(page_id=pid, text=text, text_raw=text, error=None if text.strip() else "empty")
    if not preds:
        raise SystemExit(f"no PAGE-XML in {path} matches a page id of the manifest")
    if other_model:
        print(f"{len(other_model)} pages skipped, their latest version is another model's: {', '.join(sorted(other_model)[:6])} ...")
    n_run = int(pages_run) if pages_run else len(preds)
    if wall_s is not None:
        for p in preds.values():
            p.latency_s = float(wall_s) / n_run
    run_id = make_run_id(system_id)
    write_jsonl(paths.RUNS / run_id / "predictions.jsonl", preds.values())
    meta = RunMeta(
        run_id=run_id, system_id=system_id, model_id=sp.model_id, provider="transkribus", prompt_id="none", prompt_sha256="",
        params={"wall_s_total": wall_s, "credits_used": credits_used, "pages_run": n_run,
                "credits_per_page": float(credits_used) / n_run if credits_used is not None else None},
        hardware="transkribus_cloud", harness_version=__version__, started_at=utcnow_iso(), finished_at=utcnow_iso(),
        n_pages=len(preds), n_errors=sum(1 for p in preds.values() if p.error), notes=f"imported from Transkribus export {path.name}",
    )
    write_json(paths.RUNS / run_id / "meta.json", meta)
    print(f"imported {run_id}: {len(preds)} pages, {meta.n_errors} empty")
    return meta


class TranskribusRunner(Runner):
    def run(self, pages, run_idx=0, resume=True, budget_usd=None, notes=""):  # type: ignore[override]  # noqa: ARG002
        export_images()
        raise SystemExit(f"{self.spec.system_id} runs in the Transkribus web app; import its export with `htrbench transkribus-import`.")
