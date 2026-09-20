"""The bundle round trip: what `bundle-export` writes, `scripts/modal_runner.py` answers and `bundle-import` reads."""

import json
import zipfile

from PIL import Image
from test_scoring import FACETS, PAGE

from htrbench import paths
from htrbench.io import read_jsonl, write_jsonl
from htrbench.runners.bundle import export_bundle, import_bundle
from htrbench.schema import BundlePrediction, BundleRunInfo, BundleTask, Prediction, RunMeta

SYSTEM = "olmocr-2-7b"


def release(tmp_path, monkeypatch):
    """A two-page data release and an empty bundle directory, both under tmp_path."""
    data, bundles = tmp_path / "data", tmp_path / "bundles"
    (data / "pages").mkdir(parents=True)
    pages = [PAGE.model_copy(update={"page_id": pid, "image_path": f"pages/{pid}.png", "facets": FACETS}) for pid in ("p1", "p2")]
    for page in pages:
        Image.new("RGB", (40, 20), "white").save(data / page.image_path)
    write_jsonl(data / "manifest.jsonl", pages)
    for name, value in (("DATA", data), ("MANIFEST", data / "manifest.jsonl"), ("RUNS", data / "runs"), ("BUNDLES", bundles)):
        monkeypatch.setattr(paths, name, value)
    return pages


def answer(bundle_zip, results, text="a reading", error=None):
    """What the remote side returns: one prediction per task, plus the run info and the bundle's own metadata."""
    results.mkdir(parents=True)
    with zipfile.ZipFile(bundle_zip) as z:
        tasks = [BundleTask.model_validate_json(ln) for ln in z.read("tasks.jsonl").decode("utf-8").splitlines() if ln.strip()]
        (results / "bundle_meta.json").write_bytes(z.read("bundle_meta.json"))
    write_jsonl(results / "predictions.jsonl",
                [BundlePrediction(page_id=t.page_id, text=text, latency_s=2.5, energy_wh=1.5, error=error) for t in tasks])
    (results / "run_info.json").write_text(
        BundleRunInfo(gpu="NVIDIA A100-SXM4-80GB", platform="modal", started_at="2026-09-05T14:06:38Z",
                      finished_at="2026-09-05T14:14:37Z").model_dump_json(), encoding="utf-8")
    return tasks


def test_a_bundle_exports_the_pages_and_the_prompt(tmp_path, monkeypatch):
    release(tmp_path, monkeypatch)
    with zipfile.ZipFile(export_bundle(SYSTEM)) as z:
        assert set(z.namelist()) == {"tasks.jsonl", "bundle_meta.json", "images/p1.jpg", "images/p2.jpg"}
        tasks = [BundleTask.model_validate_json(ln) for ln in z.read("tasks.jsonl").decode("utf-8").splitlines() if ln.strip()]
        meta = json.loads(z.read("bundle_meta.json"))
    assert [t.page_id for t in tasks] == ["p1", "p2"]
    assert tasks[0].prompt.startswith("You are transcribing")  # the benchmark prompt travels with the bundle
    assert meta["system_id"] == SYSTEM and meta["prompt_id"] == "page_v1"


def test_the_results_come_back_as_a_run(tmp_path, monkeypatch):
    release(tmp_path, monkeypatch)
    zpath = export_bundle(SYSTEM)
    answer(zpath, tmp_path / "results")

    meta = import_bundle(tmp_path / "results")
    assert meta.run_id == f"{SYSTEM}__r0" and meta.hardware == "modal_a100" and meta.n_pages == 2 and meta.n_errors == 0
    assert meta.prompt_sha256 and meta.gpu_name == "NVIDIA A100-SXM4-80GB"

    stored = {p.page_id: p for p in read_jsonl(paths.RUNS / meta.run_id / "predictions.jsonl", Prediction)}
    assert set(stored) == {"p1", "p2"}
    assert stored["p1"].text == "a reading" and stored["p1"].energy_wh == 1.5 and stored["p1"].gpu_s == 2.5
    assert RunMeta.model_validate_json((paths.RUNS / meta.run_id / "meta.json").read_text(encoding="utf-8")).system_id == SYSTEM


def test_a_page_the_bundle_did_not_cover_simply_has_no_row(tmp_path, monkeypatch):
    release(tmp_path, monkeypatch)
    zpath = export_bundle(SYSTEM, limit=1)
    answer(zpath, tmp_path / "results")
    assert import_bundle(tmp_path / "results").n_pages == 1


def test_an_empty_answer_is_a_failed_page(tmp_path, monkeypatch):
    release(tmp_path, monkeypatch)
    zpath = export_bundle(SYSTEM)
    answer(zpath, tmp_path / "results", text="")
    assert import_bundle(tmp_path / "results").n_errors == 2
