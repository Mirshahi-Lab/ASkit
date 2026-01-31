from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from askit.run_study.config import StudyConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture_path(name: str) -> Path:
    return FIXTURES_DIR / name


@pytest.fixture
def sample_csv_path() -> Path:
    return load_fixture_path("sample_study.csv")


def make_args(
    input_path: Path | None = None,
    output_path: Path | None = None,
    **overrides,
) -> Namespace:
    if input_path is None:
        input_path = load_fixture_path("sample_study.csv")
    if output_path is None:
        output_path = input_path.parent / "out.csv"
    defaults = {
        "dry_run": False,
        "input_file": input_path,
        "output_file": output_path,
        "predictors": "pred",
        "dependents": "dep",
        "covariates": "age,sex,bmi,race,const_cov,cat_cov",
        "categorical_covariates": "cat_cov",
        "null_values": None,
        "make_dirs": True,
        "num_workers": 1,
        "threads_per_worker": 1,
        "model": "linear",
        "max_iter": 25,
        "max_step": 5.0,
        "max_halfstep": 25,
        "gtol": 1e-4,
        "xtol": 1e-4,
        "no_intercept": False,
        "penalty_weight": 0.5,
        "alpha": 0.05,
        "correction": "bonferroni",
        "min_case_count": 2,
        "missing_covariates_operation": "fail",
        "phewas": False,
        "flipwas": False,
        "phecode_def": "1.2",
        "sex_col": "sex",
        "female_code": "1",
        "male_only": False,
        "female_only": False,
        "verbose": False,
        "quiet": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def make_config(
    input_path: Path | None = None,
    output_path: Path | None = None,
    **overrides,
) -> StudyConfig:
    return StudyConfig.from_args(make_args(input_path, output_path, **overrides))
