"""The htrbench command line. Commands import what they need when they run, so an optional dependency never breaks the rest."""

from __future__ import annotations

import logging
import sys

import typer

app = typer.Typer(help="Benchmark of handwritten text recognition on historical German documents.", no_args_is_help=True)
data_app = typer.Typer(help="The data release: page images, references, archived runs and judgements.", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _setup() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, the transcriptions are full of ſ and uͤ
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@data_app.command("fetch")
def data_fetch(source: str | None = typer.Option(None, "--from", help="a downloaded release zip, instead of Google Drive")):
    """Download the release named in configs/data.yaml, check it and unpack it into data/."""
    _setup()
    from .data import fetch

    fetch(source)


@data_app.command("verify")
def data_verify():
    """Check every file of data/ against the checksums of the release."""
    _setup()
    from .data import verify

    verify()


@data_app.command("pack")
def data_pack(release: str = typer.Argument(..., help="version of the new release, e.g. 1.1"), out_dir: str = "."):
    """Write SHA256SUMS and zip data/ as a new release (for maintainers)."""
    _setup()
    from pathlib import Path

    from .data import pack

    pack(release, Path(out_dir))


@app.command()
def run(system: str = typer.Argument(..., help="system_id of configs/systems.yaml"),
        run_idx: int = typer.Option(0, "--run-idx", "-r"),
        limit: int | None = typer.Option(None, help="only the first N pages"),
        budget_usd: float | None = typer.Option(None, help="stop when the run has cost more than this"),
        resume: bool = typer.Option(True, "--resume/--no-resume", help="reuse predictions already stored for this run"),
        notes: str = ""):
    """Run one system over the pages of the manifest; writes data/runs/<system>__r<idx>/."""
    _setup()
    from .data import require
    from .runners.base import load_pages
    from .runners.registry import build

    require()

    pages = load_pages()[:limit] if limit else load_pages()
    meta = build(system).run(pages, run_idx=run_idx, resume=resume, budget_usd=budget_usd, notes=notes)
    typer.echo(f"{meta.run_id}: {meta.n_pages} pages, {meta.n_errors} errors")


@app.command("bundle-export")
def bundle_export(system: str, run_idx: int = 0, limit: int | None = None):
    """Write bundles/<run_id>.zip for a system that runs on a rented GPU (scripts/modal_runner.py)."""
    _setup()
    from .data import require
    from .runners.bundle import export_bundle

    require()
    export_bundle(system, run_idx=run_idx, limit=limit)


@app.command("bundle-import")
def bundle_import(path: str):
    """Import the results of a bundle (directory or zip) as a run."""
    _setup()
    from .runners.bundle import import_bundle

    import_bundle(path)


@app.command("transkribus-export")
def transkribus_export():
    """Copy the page images to bundles/transkribus/images for upload to the Transkribus web app."""
    _setup()
    from .data import require
    from .runners.transkribus import export_images

    require()
    export_images()


@app.command("transkribus-import")
def transkribus_import(path: str, system: str,
                       wall_s: float | None = typer.Option(None, help="wall-clock seconds of the whole recognition job"),
                       credits: float | None = typer.Option(None, help="credits the job was charged"),
                       pages_run: int | None = typer.Option(None, help="pages the job processed, if more than are imported")):
    """Import a Transkribus PAGE-XML export (directory or zip) as the run of one system."""
    _setup()
    from .runners.transkribus import import_pagexml

    import_pagexml(path, system, wall_s=wall_s, credits_used=credits, pages_run=pages_run)


@app.command()
def judge(systems: str | None = typer.Option(None, help="comma-separated system ids; default: the line-up"),
          workers: int = 4, budget_usd: float = 8.0,
          dry_run: bool = typer.Option(False, help="print the number of calls and the cost estimate only")):
    """Have the LLM judge rate a sample of lines (configs/protocol.yaml); stored judgements are not repeated."""
    _setup()
    from .data import require
    from .eval.judge import run_judge

    require()

    run_judge(systems.split(",") if systems else None, workers=workers, dry_run=dry_run, budget_usd=budget_usd)


@app.command()
def score():
    """Score every run in data/runs; writes results/scores.parquet."""
    _setup()
    from .data import require
    from .eval.score import score_all

    require()
    score_all()


@app.command()
def aggregate():
    """Build results/tables from the scores and the stored judgements."""
    _setup()
    from .data import require
    from .eval.aggregate import build_all

    require()
    build_all()


@app.command("check-poster")
def check_poster():
    """Compare every number printed on the poster (poster/numbers.yaml) with results/tables."""
    _setup()
    from .poster_check import check

    raise typer.Exit(code=0 if check() else 1)


if __name__ == "__main__":
    app()
