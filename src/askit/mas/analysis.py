import os
from typing import Any

import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from threadpoolctl import threadpool_limits

from .config import MASConfig
from .models import firth_regression, linear_regression, logistic_regression


def run_all_regressions(config: MASConfig) -> pl.DataFrame:
    """
    Run all regressions between predictors x dependents
     using the provided configuration.

    Parameters
    ----------
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.DataFrame
        The DataFrame containing the results.
    """
    targets = []
    for predictor in config.predictor_columns:
        for dependent in config.dependent_columns:
            targets.append((predictor, dependent))

    num_groups = len(targets)
    logger.info(
        f"Running {num_groups} regressions with "
        f"{len(config.predictor_columns)} predictors and "
        f"{len(config.dependent_columns)} dependents."
    )
    # Set POLARS_MAX_THREADS to config.num_threads (loky uses spawn)
    original_env = os.environ.get("POLARS_MAX_THREADS", None)
    os.environ["POLARS_MAX_THREADS"] = str(config.threads_per_worker)
    try:
        results = Parallel(
            n_jobs=config.num_workers,
            verbose=0,
            backend="loky",
            return_as="generator_unordered",
        )(
            delayed(_run_single_regression)(predictor, dependent, config, i, num_groups)
            for i, (predictor, dependent) in enumerate(targets, start=1)
        )
    finally:
        if original_env is None:
            del os.environ["POLARS_MAX_THREADS"]
        else:
            os.environ["POLARS_MAX_THREADS"] = original_env
    results_combined = pl.concat(results, how="diagonal_relaxed").sort("pval")  # type: ignore
    logger.success("All regressions completed!")
    return results_combined


def _run_single_regression(
    predictor: str, dependent: str, config: MASConfig, task_num: int, total_tasks: int
) -> pl.DataFrame:
    """
    Run a single regression between a predictor and a dependent variable.

    Parameters
    ----------
    predictor : str
        The predictor column name.
    dependent : str
        The dependent column name.
    config : MASConfig
        The MAS configuration object.
    task_num : int
        The current task number (for logging).
    total_tasks : int
        The total number of tasks (for logging).

    Returns
    -------
    pl.DataFrame
        The DataFrame containing the result of the regression.
    """
    # loky uses spawn, so we need to set up logging in each worker
    config.setup_logger()
    output_schema = _get_output_schema(config.model)
    df = (
        pl.scan_ipc(config.ipc_file, memory_map=True)
        .select([predictor, dependent, *config.covariate_columns])
        .drop_nulls([predictor, dependent])
        .collect()
    )
    output_schema = _validate_regression_input(
        df, predictor, dependent, config, output_schema
    )
    if output_schema.get("failed_reason", "nan") != "nan":
        logger.warning(
            f"Task {task_num}/{total_tasks}: "
            f"Skipping regression '{dependent}' ~ '{predictor}': "
            f"{output_schema['failed_reason']}"
        )
        result = pl.DataFrame(
            [output_schema], schema=list(output_schema.keys()), orient="row"
        )
        return result
    # Prepare data for regression
    df = _drop_constant_covariates_for_regression(df, config)
    col_names = df.schema.names()
    pred_col = col_names[0]
    dep_col = col_names[1]
    covariates = [c for c in col_names if c not in [pred_col, dep_col]]
    # Add the equation into the output schema
    output_schema["equation"] = f"{dep_col} ~ {pred_col} + {' + '.join(covariates)}"
    # Split into X and y
    X = df.select([pred_col, *covariates])
    y = df.get_column(dep_col).to_numpy()
    # Get the appropriate model
    match config.model:
        case "firth":
            model_func = firth_regression
        case "logistic":
            model_func = logistic_regression
        case "linear":
            model_func = linear_regression
        case "firth-hybrid":
            model_func = logistic_regression

    try:
        with threadpool_limits(limits=config.threads_per_worker):
            regression_result = model_func(X, y, config)
            if (
                config.model == "firth-hybrid"
                and regression_result["pval"] < config.alpha
            ):
                logger.debug(
                    f"Switching to Firth regression for {dependent} ~ {predictor}"
                )
                regression_result = firth_regression(X, y, config)
        output_schema.update(regression_result)
    except Exception as e:
        logger.error(f"Regression failed for {dependent} ~ {predictor}: {e}")
        output_schema["failed_reason"] = str(e)

    log_interval = _get_log_interval(total_tasks)
    if task_num % log_interval == 0 or task_num == total_tasks:
        logger.info(
            f"Progress: {task_num}/{total_tasks} ({(task_num / total_tasks):.2%})"
        )
    return pl.DataFrame(
        [output_schema], schema=list(output_schema.keys()), orient="row"
    )


def _validate_regression_input(
    df: pl.DataFrame,
    predictor: str,
    dependent: str,
    config: MASConfig,
    output_schema: dict,
) -> dict[str, Any]:
    """
    Validate the input data for regression.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing the data.
    predictor : str
        The predictor column name.
    dependent : str
        The dependent column name.
    config : MASConfig
        The MAS configuration object.
    output_schema : dict
        The output schema for the results DataFrame.

    Returns
    -------
    dict[str, Any]
        The output schema with updated values,
        including failure reason if validation fails.
    """
    # this handles both linear and logistic regression size checks
    output_schema.update(
        {
            "predictor": predictor,
            "dependent": dependent,
        }
    )
    if df.height < config.min_case_count:
        output_schema.update(
            {
                "failed_reason": (
                    f"Insufficient data: {df.height} rows (< {config.min_case_count})"
                )
            }
        )
        return output_schema
    # for the logistic regressions, need to check for minimum number of cases/controls
    if config.model in ["logistic", "firth", "firth-hybrid"]:
        passed, reason, n_cases, n_controls, total_n = _check_case_counts(
            df, dependent, config.min_case_count
        )
        output_schema.update(
            {
                "failed_reason": reason,
                "cases": n_cases,
                "controls": n_controls,
                "total_n": total_n,
            }
        )
    # Linear model needs no other checks
    else:
        output_schema.update({"n_observations": df.height})
    return output_schema


def _check_case_counts(
    df: pl.DataFrame, dependent: str, min_case_count: int
) -> tuple[bool, str, int, int, int]:
    """
    Check the number of cases and controls for logistic regression.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame containing the data.
    dependent : str
        The dependent column name.
    min_case_count : int
        The minimum number of cases and controls required.

    Returns
    -------
    tuple[bool, str, int, int, int]
        A tuple containing:
        - A boolean indicating if the check passed.
        - A failure reason string (empty if check passed).
        - The number of cases.
        - The number of controls.
        - The total number of observations.
    """
    n_rows = df.height
    n_cases = df.select(pl.col(dependent).sum()).item()
    n_controls = n_rows - n_cases
    if n_cases == 0 or n_controls == 0:
        return (
            False,
            "No variation in dependent variable (all cases or all controls)",
            n_cases,
            n_controls,
            n_rows,
        )
    elif n_cases < min_case_count:
        return (
            False,
            f"Insufficient cases: {n_cases} (< {min_case_count})",
            n_cases,
            n_controls,
            n_rows,
        )
    elif n_controls < min_case_count:
        return (
            False,
            f"Insufficient controls: {n_controls} (< {min_case_count})",
            n_cases,
            n_controls,
            n_rows,
        )
    else:
        return (True, "nan", n_cases, n_controls, n_rows)


def _drop_constant_covariates_for_regression(
    df: pl.DataFrame, config: MASConfig
) -> pl.DataFrame:
    """
    Drop covariate columns that are constant (no variance).
    This function is needed in addition to
    `askit.mas.preprocessing._drop_constant_covariates`
    because we do not want to modify the config's covariate_columns list here,
    and the input is a DataFrame, not a LazyFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.DataFrame
        The DataFrame with constant covariate columns dropped.
    """
    covariate_cols = config.covariate_columns
    if not covariate_cols:
        return df
    unique_counts = df.select(pl.col(covariate_cols).n_unique()).to_dicts()[0]
    constant_covariates = [col for col, count in unique_counts.items() if count <= 1]
    if constant_covariates:
        logger.debug(
            f"Dropping constant covariate columns: {', '.join(constant_covariates)}"
        )
        return df.drop(constant_covariates)
    return df


def _get_output_schema(model_type: str) -> dict:
    """
    Get the output schema for the results DataFrame based on model type.

    Parameters
    ----------
    model_type : str
        The type of model being used.

    Returns
    -------
    dict
        A dictionary defining the default schema of the results DataFrame.
    """
    base_schema = {
        "predictor": "nan",
        "dependent": "nan",
        "pval": float("nan"),
        "beta": float("nan"),
        "se": float("nan"),
        "beta_ci_low": float("nan"),
        "beta_ci_high": float("nan"),
        "failed_reason": "nan",
        "equation": "nan",
    }
    if model_type in ["logistic", "firth", "firth-hybrid"]:
        base_schema.update(
            {
                "OR": float("nan"),
                "OR_ci_low": float("nan"),
                "OR_ci_high": float("nan"),
                "cases": -9,
                "controls": -9,
                "total_n": -9,
                "converged": False,
            }
        )
    elif model_type == "linear":
        base_schema.update(
            {
                "n_observations": -9,
            }
        )
    return base_schema


def _get_log_interval(total: int) -> int:
    """Return how often to log progress based on total task count."""
    if total <= 10:
        return 1
    if total <= 50:
        return 5
    if total <= 100:
        return 10
    if total <= 200:
        return 20
    if total <= 300:
        return 30
    if total <= 400:
        return 40
    if total <= 500:
        return 50
    return 100
