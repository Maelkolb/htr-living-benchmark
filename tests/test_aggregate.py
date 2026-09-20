"""The steps between page counts and the five scores, and the consistency of the configs they read."""

import math

import pandas as pd
import pytest
from test_scoring import META, PAGE, score

from htrbench import paths
from htrbench.eval.aggregate import cost_basis, text_scores
from htrbench.eval.judge import judge_table
from htrbench.io import read_yaml
from htrbench.runners.pricing import electricity, price_table, transkribus
from htrbench.runners.registry import RUNNERS, systems
from htrbench.schema import RunMeta, SystemSpec


def spec(**over) -> SystemSpec:
    return SystemSpec.model_validate({"system_id": "s", "label": "S", "runner": "gemini", "model_id": "m", "provider": "p",
                                      "family": "vlm_api", "hardware": "api", "hardware_class": "api", "weights": "closed", **over})


def meta(**params) -> RunMeta:
    return META.model_copy(update={"params": params})


def preds(cost_usd=None, energy_wh=None, latency_s=1.0) -> pd.DataFrame:
    return pd.DataFrame({"cost_usd": [cost_usd, cost_usd], "energy_wh": [energy_wh, energy_wh], "latency_s": [latency_s, latency_s]})


# --------------------------------------------------------------------------------- what D5 is priced on
def test_a_hosted_model_is_priced_at_what_its_calls_cost():
    basis = cost_basis(spec(), meta(), preds(cost_usd=0.006))
    assert basis["usd_per_page"] == pytest.approx(0.006)
    assert math.isnan(basis["wh_per_page"]) and basis["wh_basis"] == "not measured"


def test_transkribus_is_priced_at_the_credits_it_was_charged():
    basis = cost_basis(spec(hardware="transkribus_cloud", hardware_class="transkribus"), meta(credits_per_page=1.0), preds())
    expected = 1.0 * transkribus()["eur_per_credit"] * electricity()["usd_per_eur"]
    assert basis["usd_per_page"] == pytest.approx(expected)
    assert basis["usd_per_page"] == pytest.approx(0.257, abs=5e-4)  # the 26 cents a page printed on the poster


def test_an_open_weight_model_is_priced_on_its_measured_energy():
    basis = cost_basis(spec(hardware="modal_a100", hardware_class="gpu_24gb"), meta(), preds(energy_wh=1.08))
    assert basis["wh_per_page"] == pytest.approx(1.08) and basis["wh_basis"].startswith("measured")
    assert math.isnan(basis["usd_per_page"])  # the rented GPU is the benchmark's cost, not the model's


def test_a_cpu_model_has_its_energy_estimated_from_its_seconds():
    watts = electricity()["cpu_w_estimate"]
    basis = cost_basis(spec(hardware="cpu_local", hardware_class="cpu"), meta(), preds(latency_s=3600.0))
    assert basis["wh_per_page"] == pytest.approx(watts) and basis["wh_basis"].startswith("estimated")


# --------------------------------------------------------------------------------- what D3 is measured on
def test_the_glyph_facet_exists_only_where_the_reference_records_letter_forms():
    row = score(PAGE.gt_text).model_dump()
    on_diplomatic = text_scores(pd.DataFrame([row]))
    off_diplomatic = text_scores(pd.DataFrame([{**row, "corpus": "humboldt"}]))
    assert on_diplomatic["glyph_f1"] == 1.0
    assert math.isnan(off_diplomatic["glyph_f1"])
    assert off_diplomatic["fidelity"] == pytest.approx((off_diplomatic["punct_kept"] + off_diplomatic["caps_kept"]) / 2)


# --------------------------------------------------------------------------------- what D4's judge facet counts
def judged_rows() -> pd.DataFrame:
    number_error = [{"ref_span": "25", "hyp_span": "75", "type": "number_error", "severity": 2}]
    return pd.DataFrame([
        # a line both passes call misleading, one they disagree on, and one judged only once
        {"system_id": "s", "page_id": "p", "line_idx": 0, "pass_idx": 0, "ref": "one two three", "spans": number_error, "verdict": "misleading"},
        {"system_id": "s", "page_id": "p", "line_idx": 0, "pass_idx": 1, "ref": "one two three", "spans": [], "verdict": "misleading"},
        {"system_id": "s", "page_id": "p", "line_idx": 1, "pass_idx": 0, "ref": "four five six", "spans": [], "verdict": "usable"},
        {"system_id": "s", "page_id": "p", "line_idx": 1, "pass_idx": 1, "ref": "four five six", "spans": [], "verdict": "misleading"},
        {"system_id": "s", "page_id": "p", "line_idx": 2, "pass_idx": 0, "ref": "seven eight nine", "spans": [], "verdict": "usable"},
    ])


def test_the_judge_table_counts_the_first_pass_and_scores_on_both():
    t = judge_table(judged_rows()).set_index("system_id").loc["s"]
    assert t["n_lines"] == 3 and t["n_ref_words"] == 9  # the first pass only
    assert t["number_error_per_100w"] == pytest.approx(100 / 9)
    assert t["share_misleading"] == pytest.approx(1 / 3)  # first pass
    assert t["share_misleading_both_passes"] == pytest.approx(3 / 5)  # every stored verdict; this is what D4 uses
    assert t["kappa_verdict"] < 1.0  # the two passes disagree on one of the two lines they share


def test_the_judge_table_survives_a_benchmark_that_was_never_judged():
    assert judge_table(pd.DataFrame()).empty


# --------------------------------------------------------------------------------- the configs hang together
def test_every_system_names_a_runner_a_hardware_class_and_a_price():
    protocol = read_yaml(paths.CONFIGS / "protocol.yaml")
    prices = price_table()
    for sid, s in systems().items():
        assert s.runner in RUNNERS, sid
        assert s.hardware_class in protocol["efficiency"]["hardware"], sid
        if s.hardware_class == "api":
            assert s.price_key in prices, sid  # without a price entry a hosted system silently loses its D5 price facet
    assert protocol["judge"]["model_id"] in prices
    assert (paths.PROMPTS / f"{protocol['judge']['prompt_id']}.txt").exists()


def test_the_line_up_is_the_eight_systems_the_poster_follows():
    lineup = [sid for sid, s in systems().items() if s.role == "lineup"]
    printed = read_yaml(paths.ROOT / "poster" / "numbers.yaml")["scorecard"]
    assert sorted(lineup) == sorted(printed)

# --------------------------------------------------------------------------------- the release round trip
def test_a_release_is_packed_with_checksums_that_catch_an_altered_file(tmp_path, monkeypatch):
    import zipfile

    from htrbench import data

    monkeypatch.setattr(paths, "DATA", tmp_path)
    (tmp_path / "runs").mkdir()
    (tmp_path / "manifest.jsonl").write_text("a page", encoding="utf-8")
    (tmp_path / "runs" / "meta.json").write_text("a run", encoding="utf-8")

    archive = data.pack("test", tmp_path)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {"manifest.jsonl", "runs/meta.json", "SHA256SUMS"}
    data.verify()

    (tmp_path / "manifest.jsonl").write_text("another page", encoding="utf-8")
    with pytest.raises(SystemExit):
        data.verify()
