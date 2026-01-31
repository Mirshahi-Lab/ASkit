from __future__ import annotations

from pathlib import Path

import polars as pl

from askit.run_study.analysis import (
    _check_case_counts,
    _drop_constant_covariates_for_regression,
    _get_output_schema,
    _run_single_regression,
    _validate_regression_input,
)
from askit.run_study.preprocessing import _write_temp_ipc_file, cleanup_ipc
from tests.conftest import make_config


def test_get_output_schema_logistic_and_linear() -> None:
    logistic_schema = _get_output_schema("logistic")
    assert "OR" in logistic_schema
    assert "cases" in logistic_schema
    linear_schema = _get_output_schema("linear")
    assert "n_observations" in linear_schema
    assert "OR" not in linear_schema


def test_check_case_counts_no_variation() -> None:
    df = pl.DataFrame({"dep": [1, 1, 1, 1], "x": [0, 1, 0, 1]})
    passed, reason, n_cases, n_controls, total_n = _check_case_counts(df, "dep", 2)
    assert passed is False
    assert "No variation" in reason
    assert n_cases == 4
    assert n_controls == 0
    assert total_n == 4


def test_check_case_counts_insufficient_cases_controls() -> None:
    df_cases = pl.DataFrame({"dep": [1, 1, 0, 0, 0], "x": [0, 1, 0, 1, 0]})
    passed, reason, n_cases, n_controls, _ = _check_case_counts(df_cases, "dep", 3)
    assert passed is False
    assert "Insufficient cases" in reason
    assert n_cases == 2
    assert n_controls == 3

    df_controls = pl.DataFrame({"dep": [1, 1, 1, 1, 0], "x": [0, 1, 0, 1, 0]})
    passed, reason, n_cases, n_controls, _ = _check_case_counts(df_controls, "dep", 2)
    assert passed is False
    assert "Insufficient controls" in reason
    assert n_cases == 4
    assert n_controls == 1


def test_validate_regression_input_min_case_count() -> None:
    config = make_config(
        min_case_count=50,
        model="linear",
        covariates=None,
        categorical_covariates=None,
    )
    df = pl.DataFrame({"pred": [1, 2], "dep": [3, 4]})
    output_schema = _get_output_schema("linear")
    updated = _validate_regression_input(df, "pred", "dep", config, output_schema)
    assert "Insufficient data" in updated["failed_reason"]


def test_drop_constant_covariates_for_regression() -> None:
    config = make_config(
        predictors="pred",
        dependents="dep",
        covariates="age,bmi,const_cov",
        categorical_covariates=None,
    )
    df = config.read_data().select(["pred", "dep", "age", "bmi", "const_cov"]).collect()
    cleaned = _drop_constant_covariates_for_regression(df, config)
    assert "const_cov" not in cleaned.columns
    assert "const_cov" in config.covariate_columns


def test_run_single_regression_linear_smoke(tmp_path: Path) -> None:
    config = make_config(
        model="linear",
        predictors="pred",
        dependents="age",
        covariates="bmi,race",
        categorical_covariates=None,
        output_path=tmp_path / "out.csv",
    )
    data = config.read_data()
    _write_temp_ipc_file(data, config)
    try:
        result = _run_single_regression("pred", "age", config, 1, 1)
        row = result.row(0, named=True)
        assert row["failed_reason"] == "nan"
        assert row["model"] == "linear"
        assert row["predictor"] == "pred"
        assert row["dependent"] == "age"
    finally:
        cleanup_ipc(config.ipc_file)


def test_run_single_regression_logistic_hybrid_switch(tmp_path: Path) -> None:
    config = make_config(
        model="logistic-hybrid",
        alpha=1.0,
        predictors="pred",
        dependents="dep",
        covariates="age,bmi",
        categorical_covariates=None,
        output_path=tmp_path / "out.csv",
    )
    data = config.read_data()
    _write_temp_ipc_file(data, config)
    try:
        result = _run_single_regression("pred", "dep", config, 1, 1)
        row = result.row(0, named=True)
        assert row["failed_reason"] == "nan"
        assert row["model"] == "logistic-firth"
    finally:
        cleanup_ipc(config.ipc_file)
