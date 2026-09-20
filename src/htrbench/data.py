"""The data release: one zip on Google Drive, named and checksummed in ``configs/data.yaml``.

``fetch`` downloads, checks and unpacks it, ``verify`` re-checks the unpacked files, ``pack`` builds the zip of a
new release. What is inside: docs/DATASETS.md.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from . import paths
from .io import read_yaml, sha256_file

SUMS = "SHA256SUMS"
DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def config() -> dict:
    return read_yaml(paths.CONFIGS / "data.yaml")


def require() -> None:
    """Every command that reads the release calls this first."""
    if not paths.MANIFEST.exists():
        raise SystemExit(f"no data release in {paths.DATA}; run `htrbench data fetch` first (docs/DATASETS.md)")


def _download(file_id: str, dest: Path) -> None:
    with urllib.request.urlopen(DOWNLOAD_URL.format(file_id=file_id)) as resp, dest.open("wb") as fh:
        if "text/html" in resp.headers.get("Content-Type", ""):
            # Drive answers with a sign-in or warning page instead of the file when the caller may not read it.
            raise SystemExit(f"Google Drive did not hand out the file. Ask for access to {config()['drive_folder']}, "
                             f"download the zip by hand and run: htrbench data fetch --from <zip>")
        shutil.copyfileobj(resp, fh)


def fetch(source: str | None = None) -> None:
    cfg = config()
    if paths.MANIFEST.exists():
        raise SystemExit(f"{paths.DATA} already holds a release; remove it first or check it with `htrbench data verify`")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(source) if source else Path(tmp) / cfg["archive"]
        if not source:
            print(f"downloading {cfg['archive']} ({cfg['size_mb']} MB)")
            _download(cfg["drive_file_id"], archive)
        if sha256_file(archive) != cfg["sha256"]:
            raise SystemExit(f"{archive.name} is not release {cfg['release']}: its SHA-256 differs from configs/data.yaml")
        with zipfile.ZipFile(archive) as z:
            z.extractall(paths.DATA)
    verify()


def _listed() -> dict[str, str]:
    lines = (paths.DATA / SUMS).read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (ln.split("  ", 1) for ln in lines if ln.strip())}


def verify() -> None:
    listed = _listed()
    bad = [name for name, digest in listed.items() if not (paths.DATA / name).exists() or sha256_file(paths.DATA / name) != digest]
    if bad:
        raise SystemExit(f"{len(bad)} of {len(listed)} files are missing or altered, e.g. {', '.join(bad[:5])}")
    print(f"data release {config()['release']}: {len(listed)} files verified in {paths.DATA}")


def pack(release: str, out_dir: Path) -> Path:
    """Prints the lines that belong into configs/data.yaml."""
    files = sorted(p for p in paths.DATA.rglob("*") if p.is_file() and p.name != SUMS)
    sums = "".join(f"{sha256_file(p)}  {p.relative_to(paths.DATA).as_posix()}\n" for p in files)
    (paths.DATA / SUMS).write_text(sums, encoding="utf-8", newline="\n")
    archive = out_dir / f"htr-living-benchmark-data-{release}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [*files, paths.DATA / SUMS]:
            z.write(p, p.relative_to(paths.DATA).as_posix())
    print(f"release: \"{release}\"\narchive: {archive.name}\nsha256: {sha256_file(archive)}\nsize_mb: {archive.stat().st_size / 1e6:.0f}")
    return archive
