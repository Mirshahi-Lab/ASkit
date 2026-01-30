from pathlib import Path

import polars as pl
from loguru import logger

from .config import StudyConfig
from .constants import phecode_defs


def postprocess_results(results: pl.DataFrame, config: StudyConfig) -> pl.DataFrame:
    """
    Postprocess the regression study results according to the configuration.

    Parameters
    ----------
    results : pl.DataFrame
        The raw regression results as a Polars DataFrame.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.DataFrame
        The postprocessed regression results.

    """
    df = _calculate_corrected_pvalues(results, config)
    df = _add_phecode_definitions(df, config)
    _write_to_output(df, config)
    return df


def _calculate_corrected_pvalues(df: pl.DataFrame, config: StudyConfig) -> pl.DataFrame:
    """
    Apply multiple testing correction to p-values in the results DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The results DataFrame containing p-values.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.DataFrame
        The DataFrame with corrected p-values added.
    """
    if config.correction == "bonferroni":
        logger.info("Applying Bonferroni correction to p-values.")
        num_tests = df.filter(pl.col("pval").is_not_null()).height
        threshold = config.alpha / num_tests
        return df.with_columns(
            pl.col("pval").lt(threshold).alias("significant_bonferroni")
        )
    # TODO implement other corrections like FDR_BH
    else:
        return df  # No correction applied


def _add_phecode_definitions(df: pl.DataFrame, config: StudyConfig) -> pl.DataFrame:
    """
    Add PheCode definitions to the results DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The results DataFrame containing PheCodes.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    pl.DataFrame
        The DataFrame with PheCode definitions added.
    """
    if not config.is_phewas and not config.is_flipwas:
        return df  # No PheCode definitions needed
    logger.info(f"Adding PheCode {config.phecode_def} definitions to results.")
    phecode_def_path = phecode_defs[config.phecode_def]
    # These are always parquet files
    phecode_defs_df = pl.read_parquet(phecode_def_path)
    if config.is_phewas:
        return df.join(
            phecode_defs_df, left_on="dependent", right_on="phecode", how="left"
        )
    else:  # flipwas
        return df.join(
            phecode_defs_df, left_on="predictor", right_on="phecode", how="left"
        )


def _write_to_output(df: pl.DataFrame, config: StudyConfig) -> None:
    """
    Write the results DataFrame to the specified output path in CSV format.

    Parameters
    ----------
    df : pl.DataFrame
        The results DataFrame to write.
    config : MASConfig
        The MAS configuration object.

    Returns
    -------
    None
    """
    output_path: Path = config.output_file
    if output_path.suffix == ".csv":
        df.write_csv(output_path)
    elif output_path.suffix == ".tsv" or output_path.suffix == ".txt":
        df.write_csv(output_path, separator="\t")
    elif output_path.suffix == ".parquet":
        df.write_parquet(output_path)
    elif output_path.suffix == ".ipc":
        df.write_ipc(output_path)
    else:
        raise ValueError(f"Unsupported output file format: {output_path.suffix}")
    logger.success(f"Results written to {output_path}.")
