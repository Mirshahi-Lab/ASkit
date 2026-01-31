from __future__ import annotations

from pathlib import Path

import pytest

from askit.run_study.config import StudyConfig
from tests.conftest import make_args, make_config


def test_validate_io_rejects_missing_input(tmp_path: Path) -> None:
    args = make_args(
        input_path=tmp_path / "missing.csv", output_path=tmp_path / "out.csv"
    )
    with pytest.raises(FileNotFoundError):
        StudyConfig.from_args(args)


def test_parse_column_list_by_name_and_index() -> None:
    config = make_config(
        predictors="pred",
        dependents="i:1",
        covariates=None,
        categorical_covariates=None,
    )
    assert config.predictor_columns == ["pred"]
    assert config.dependent_columns == ["dep"]


def test_parse_column_list_index_range() -> None:
    config = make_config(
        predictors="pred",
        dependents="i:1-3",
        covariates=None,
        categorical_covariates=None,
    )
    assert config.dependent_columns == ["dep", "age"]

    config = make_config(
        predictors="pred",
        dependents="i:5-.",
        covariates=None,
        categorical_covariates=None,
    )
    assert config.dependent_columns == ["race", "const_cov", "cat_cov"]


def test_parse_column_list_invalid_index() -> None:
    with pytest.raises(ValueError):
        make_config(
            predictors="pred",
            dependents="i:999",
            covariates=None,
            categorical_covariates=None,
        )
    with pytest.raises(ValueError):
        make_config(
            predictors="pred",
            dependents="i:1-999",
            covariates=None,
            categorical_covariates=None,
        )


def test_parse_column_list_invalid_name() -> None:
    with pytest.raises(ValueError):
        make_config(
            predictors="missing_col",
            dependents="dep",
            covariates=None,
            categorical_covariates=None,
        )


def test_parse_column_list_invalid_range_format() -> None:
    with pytest.raises(ValueError):
        make_config(
            predictors="pred",
            dependents="i:1-2-3",
            covariates=None,
            categorical_covariates=None,
        )


def test_parse_column_list_invalid_range_start() -> None:
    with pytest.raises(ValueError):
        make_config(
            predictors="pred",
            dependents="i:x-2",
            covariates=None,
            categorical_covariates=None,
        )


def test_unique_column_sets_enforced() -> None:
    with pytest.raises(ValueError):
        make_config(predictors="pred", dependents="pred")


def test_categorical_covariates_subset_enforced() -> None:
    with pytest.raises(ValueError):
        make_config(
            covariates="age,bmi",
            categorical_covariates="cat_cov",
        )


def test_read_data_updates_counts() -> None:
    config = make_config(
        predictors="pred",
        dependents="dep",
        covariates="age,sex",
        categorical_covariates=None,
    )
    data = config.read_data()
    assert config.included_row_count == 12
    assert config.included_column_count == 4
    assert data.collect().shape == (12, 4)
