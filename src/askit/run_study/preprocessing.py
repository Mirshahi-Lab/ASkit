import atexit
import os
import tempfile

import polars as pl
from loguru import logger

from askit.run_study.config import StudyConfig


def preprocess_input(config: StudyConfig) -> pl.LazyFrame:
    """Preprocess the input data according to the MAS configuration.

    Args:
        config (MASConfig): The MAS configuration object.

    Returns:
        pl.LazyFrame: The preprocessed data as a Polars LazyFrame.
    """
    # Load the data
    data = config.read_data()
    logger.info("Starting preprocessing of input data.")
    data = _limit_to_sex_specific(data, config)
    data = _handle_missing_covariates(data, config)
    data = _drop_constant_covariates(data, config)
    data = _create_dummy_covariates(data, config)
    _write_temp_ipc_file(data, config)
    logger.success("Preprocessing complete.")
    return data


def _limit_to_sex_specific(data: pl.LazyFrame, config: StudyConfig) -> pl.LazyFrame:
    """
    Limit the data to sex-specific rows if specified in the configuration.

    Parameters
    ----------
    data : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.LazyFrame
        The sex-specific data as a Polars
        LazyFrame if specified, otherwise the original data.

    Raises
    ------
    ValueError
        If the sex identifier column (--sex-col) is not in the included columns.
    """
    if not config.male_only and not config.female_only:
        return data
    original_row_count = config.included_row_count
    if config.sex_col not in config.included_columns:
        raise ValueError(f'Sex column "{config.sex_col}" not in included columns.')
    # Male Only
    if config.male_only:
        logger.info("Limiting analysis to male samples only.")
        sex_specific = data.filter(pl.col(config.sex_col) != config.female_code)
        config.update_row_and_column_counts(sex_specific)
    # Female Only
    else:
        logger.info("Limiting analysis to female samples only.")
        sex_specific = data.filter(pl.col(config.sex_col) == config.female_code)
        config.update_row_and_column_counts(sex_specific)
    if config.included_row_count <= original_row_count:
        logger.success(
            f"Dropped {original_row_count - config.included_row_count} rows "
            f"for {'male' if config.male_only else 'female'} specific analysis."
        )
    return sex_specific


def _handle_missing_covariates(data: pl.LazyFrame, config: StudyConfig) -> pl.LazyFrame:
    """
    Handle missing values in covariate columns by dropping rows with missing values.

    Parameters
    ----------
    data : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.LazyFrame
        The data with rows containing missing covariate values
        handled according to the specified operation.

    Raises
    ------
    ValueError
        If missing values are found in covariate
        columns and the operation is set to "fail".
    """
    if config.missing_covariates_operation == "fail":
        missing_counts = _count_missing_values(data, config.covariate_columns)
        cols_with_missing = {
            col: count for col, count in missing_counts.items() if count > 0
        }
        if cols_with_missing:
            raise ValueError(
                f"Missing values found in covariate columns: {cols_with_missing}"
            )
        return data
    elif config.missing_covariates_operation == "drop":
        original_row_count = config.included_row_count
        data_cleaned = data.drop_nulls(config.covariate_columns)
        config.update_row_and_column_counts(data_cleaned)
        logger.info(
            f"Dropped {original_row_count - config.included_row_count} "
            f"rows due to missing covariate values."
        )
        return data_cleaned
    else:
        data_cleaned = data.with_columns(
            pl.col(config.covariate_columns).fill_null(
                strategy=config.missing_covariates_operation
            )
        )
        return data_cleaned


def _count_missing_values(lf: pl.LazyFrame, columns: list[str]) -> dict[str, int]:
    """
    Count the number of missing values in specified columns.

    Parameters
    ----------
    data : pl.LazyFrame
        The input data as a Polars LazyFrame.
    columns : list[str]
        The list of column names to check for missing values.

    Returns
    -------
    dict[str, int]
        A dictionary mapping column names to their respective counts of missing values.
    """
    missing_counts = lf.select(pl.col(columns).null_count()).collect().to_dicts()[0]
    return missing_counts


def _drop_constant_covariates(data: pl.LazyFrame, config: StudyConfig) -> pl.LazyFrame:
    """
    Drop covariate columns that are constant (i.e., have the same value for all rows).

    Parameters
    ----------
    data : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.LazyFrame
        The data with constant covariate columns dropped.
    """
    constant_covariates = []
    unique_counts = (
        data.select(pl.col(config.covariate_columns).n_unique()).collect().to_dicts()[0]
    )
    for col, count in unique_counts.items():
        if count <= 1:
            constant_covariates.append(col)
    if constant_covariates:
        logger.info(f"Dropping constant covariate columns: {constant_covariates}")
        cleaned_data = data.drop(constant_covariates)
        config.covariate_columns = [
            col for col in config.covariate_columns if col not in constant_covariates
        ]
        config.update_row_and_column_counts(cleaned_data)
        return cleaned_data
    else:
        return data


def _create_dummy_covariates(data: pl.LazyFrame, config: StudyConfig) -> pl.LazyFrame:
    """
    Create dummy variables for categorical covariate columns.

    Parameters
    ----------
    data : pl.LazyFrame
        The input data as a Polars LazyFrame.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.LazyFrame
        The data with dummy variables created for categorical covariate columns.
    """
    # if there are no cat. covariates, return data as is
    if not config.categorical_covariate_columns:
        return data
    logger.info(
        f"Creating dummy variables for categorical covariates: "
        f"{config.categorical_covariate_columns}"
    )
    unique_values = (
        data.select(
            # This collects unique values for each categorical covariate column
            pl.col(config.categorical_covariate_columns).implode().list.unique()
        )
        .unique()
        .collect()
        # This is now a dict[str, list[Any]] of unique values in the list
        .to_dicts()[0]
    )
    new_cols = []
    for col, values in unique_values.items():
        if len(values) > 2:
            # Skip the first value to avoid multicollinearity
            new_cols.extend([f"{col}_{val}" for val in values[1:]])
            data = data.with_columns(
                [
                    pl.when(pl.col(col) == val)
                    .then(1)
                    .otherwise(0)
                    .alias(f"{col}_{val}")
                    for val in values[1:]
                ]
            ).drop(col)
    config.covariate_columns = [
        col
        for col in config.covariate_columns
        if col not in config.categorical_covariate_columns
    ]
    config.covariate_columns.extend(new_cols)
    config.update_row_and_column_counts(data)
    return data


def _write_temp_ipc_file(data: pl.LazyFrame, config: StudyConfig) -> None:
    """
    Write the preprocessed data to a temporary IPC file.

    Parameters
    ----------
    data : pl.LazyFrame
        The preprocessed data as a Polars LazyFrame.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    None
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ipc") as temp_file:
        temp_ipc_name = temp_file.name
        config.ipc_file = temp_ipc_name
        data.sink_ipc(temp_ipc_name)
    logger.info(f"Preprocessed data written to temporary IPC file: {temp_ipc_name}")
    atexit.register(cleanup_ipc, temp_ipc_name)


def cleanup_ipc(temp_ipc_file: str) -> None:
    """Remove the temporary IPC file if it exists."""
    if os.path.exists(temp_ipc_file):
        try:
            os.unlink(temp_ipc_file)
            logger.debug(f"Cleaned up IPC file: {temp_ipc_file}")
        except OSError as e:
            logger.warning(f"Failed to clean up IPC file {temp_ipc_file}: {e}")
