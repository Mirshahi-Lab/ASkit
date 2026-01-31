from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from askit.run_study.postprocessing import (
    _add_phecode_definitions,
    _calculate_corrected_pvalues,
)
from tests.conftest import make_config


def test_calculate_corrected_pvalues_bonferroni() -> None:
    config = make_config(alpha=0.05, correction="bonferroni")
    df = pl.DataFrame({"pval": [0.001, 0.01, 0.02, 0.5]})
    corrected = _calculate_corrected_pvalues(df, config)
    assert "significant_bonferroni" in corrected.columns
    assert corrected.get_column("significant_bonferroni").to_list() == [
        True,
        True,
        False,
        False,
    ]


def test_calculate_corrected_pvalues_none() -> None:
    config = make_config(correction="none")
    df = pl.DataFrame({"pval": [0.1, 0.2]})
    corrected = _calculate_corrected_pvalues(df, config)
    assert "significant_bonferroni" not in corrected.columns


def test_add_phecode_definitions_phewas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phecode_path = tmp_path / "phecodes.parquet"
    pl.DataFrame(
        {"phecode": ["001", "002"], "phenotype": ["Alpha", "Beta"]}
    ).write_parquet(phecode_path)

    monkeypatch.setattr(
        "askit.run_study.postprocessing.phecode_defs",
        {"1.2": phecode_path},
        raising=False,
    )

    config = make_config(phewas=True, flipwas=False)
    df = pl.DataFrame({"dependent": ["001", "002"], "predictor": ["x", "y"]})
    enriched = _add_phecode_definitions(df, config)
    assert "phenotype" in enriched.columns
    assert enriched.get_column("phenotype").to_list() == ["Alpha", "Beta"]


def test_add_phecode_definitions_flipwas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phecode_path = tmp_path / "phecodes.parquet"
    pl.DataFrame(
        {"phecode": ["001", "002"], "phenotype": ["Alpha", "Beta"]}
    ).write_parquet(phecode_path)

    monkeypatch.setattr(
        "askit.run_study.postprocessing.phecode_defs",
        {"1.2": phecode_path},
        raising=False,
    )

    config = make_config(phewas=False, flipwas=True)
    df = pl.DataFrame({"predictor": ["001", "002"], "dependent": ["x", "y"]})
    enriched = _add_phecode_definitions(df, config)
    assert "phenotype" in enriched.columns
    assert enriched.get_column("phenotype").to_list() == ["Alpha", "Beta"]
