"""End to end on the data release: archived runs -> scores -> tables -> every number printed on the poster.

Skipped when data/ is not there (`htrbench data fetch`).
"""

import pytest

from htrbench import paths

pytestmark = pytest.mark.skipif(not paths.MANIFEST.exists(), reason="data release not fetched")


def test_release_is_intact():
    from htrbench.data import verify

    verify()


def test_the_committed_prompts_are_the_ones_the_runs_used():
    """Each run recorded the SHA-256 of the prompt it was given; the Transkribus runs took none."""
    from htrbench.io import read_prompt, sha256_text
    from htrbench.schema import RunMeta

    committed = {p.stem: sha256_text(read_prompt(p.stem)) for p in paths.PROMPTS.glob("*.txt")}
    hashed = 0
    for meta_path in sorted(paths.RUNS.glob("*/meta.json")):
        meta = RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta.prompt_sha256:
            assert committed[meta.prompt_id] == meta.prompt_sha256, meta.run_id
            hashed += 1
    assert hashed == 18  # every run but the two Transkribus ones, which the web app ran without a prompt


def test_the_committed_scores_and_tables_are_what_the_release_produces(tmp_path, monkeypatch):
    """Score and aggregate the archived runs into a scratch directory and compare with what is committed."""
    import pandas as pd

    from htrbench import poster_check
    from htrbench.eval.aggregate import build_all
    from htrbench.eval.score import score_all

    committed_scores, committed_tables = paths.SCORES, paths.TABLES
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "SCORES", tmp_path / "scores.parquet")
    monkeypatch.setattr(paths, "TABLES", tmp_path / "tables")

    pd.testing.assert_frame_equal(score_all(), pd.read_parquet(committed_scores))
    for name, table in build_all().items():
        pd.testing.assert_frame_equal(table, pd.read_csv(committed_tables / f"{name}.csv"), check_dtype=False)
    assert poster_check.check()
