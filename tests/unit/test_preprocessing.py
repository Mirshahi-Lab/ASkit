from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from askit.run_study.preprocessing import (
    _create_dummy_covariates,
    _drop_constant_covariates,
    _handle_missing_covariates,
    _limit_to_sex_specific,
    _write_temp_ipc_file,
    cleanup_ipc,
)
from tests.conftest import make_config


def _make_missing_covariate_data() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "age": [40, 41, 42, 43],
            "bmi": [25.1, None, 27.3, 28.4],
            "race": [1, 2, 1, 3],
        }
    ).lazy()


def test_limit_to_sex_specific_requires_sex_col() -> None:
    config = make_config()
    data = config.read_data()
    config.male_only = True
    config.included_columns = [col for col in config.included_columns if col != "sex"]
    with pytest.raises(ValueError):
        _limit_to_sex_specific(data, config)


def test_limit_to_sex_specific_male_only() -> None:
    config = make_config()
    data = config.read_data()
    config.female_code = "F"
    config.male_only = True
    original_rows = config.included_row_count
    filtered = _limit_to_sex_specific(data, config)
    assert config.included_row_count < original_rows
    assert filtered.collect().get_column("sex").to_list().count("F") == 0


def test_limit_to_sex_specific_female_only() -> None:
    config = make_config()
    data = config.read_data()
    config.female_code = "F"
    config.female_only = True
    original_rows = config.included_row_count
    filtered = _limit_to_sex_specific(data, config)
    assert config.included_row_count < original_rows
    assert all(val == "F" for val in filtered.collect().get_column("sex").to_list())


def test_handle_missing_covariates_fail() -> None:
    config = make_config(
        covariates="age,bmi,race",
        categorical_covariates=None,
        missing_covariates_operation="fail",
    )
    data = _make_missing_covariate_data()
    config.update_row_and_column_counts(data)
    with pytest.raises(ValueError):
        _handle_missing_covariates(data, config)


def test_handle_missing_covariates_drop() -> None:
    config = make_config(
        covariates="age,bmi,race",
        categorical_covariates=None,
        missing_covariates_operation="drop",
    )
    data = _make_missing_covariate_data()
    config.update_row_and_column_counts(data)
    original_rows = config.included_row_count
    cleaned = _handle_missing_covariates(data, config)
    assert config.included_row_count == original_rows - 1
    assert cleaned.collect().height == original_rows - 1


def test_handle_missing_covariates_impute() -> None:
    config = make_config(
        covariates="age,bmi,race",
        categorical_covariates=None,
        missing_covariates_operation="mean",
    )
    data = _make_missing_covariate_data()
    config.update_row_and_column_counts(data)
    cleaned = _handle_missing_covariates(data, config)
    null_counts = (
        cleaned.select(pl.col(config.covariate_columns).null_count())
        .collect()
        .to_dicts()[0]
    )
    assert all(count == 0 for count in null_counts.values())


def test_drop_constant_covariates() -> None:
    config = make_config(
        covariates="age,sex,bmi,race,const_cov",
        categorical_covariates=None,
    )
    data = config.read_data()
    cleaned = _drop_constant_covariates(data, config)
    assert "const_cov" not in cleaned.collect_schema().names()
    assert "const_cov" not in config.covariate_columns


def test_create_dummy_covariates() -> None:
    config = make_config(
        covariates="age,sex,bmi,race,cat_cov",
        categorical_covariates="cat_cov",
    )
    data = config.read_data()
    transformed = _create_dummy_covariates(data, config)
    schema_names = transformed.collect_schema().names()
    assert "cat_cov" not in schema_names
    dummy_cols = [name for name in schema_names if name.startswith("cat_cov_")]
    assert len(dummy_cols) == 2
    assert all(name in config.covariate_columns for name in dummy_cols)


def test_write_and_cleanup_ipc(tmp_path: Path) -> None:
    config = make_config()
    data = config.read_data()
    _write_temp_ipc_file(data, config)
    assert config.ipc_file
    assert Path(config.ipc_file).exists()
    cleanup_ipc(config.ipc_file)
    assert not Path(config.ipc_file).exists()
