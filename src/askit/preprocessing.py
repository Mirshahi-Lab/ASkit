from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def create_phenotypes(
    code_file: str,
    min_code_count: int = 0,
    map_phecodes: bool = True,
    phecode_rollup: bool = True,
    phecode_exclude: bool = True,
    sex: str = None,
    n_chunks: int = 1,
    outfile: str = None,
) -> pd.DataFrame:
    """
    Create an NxP matrix, where N = number of patients and P = number of phenotypes.
    Case, control, and exclusion are assigned as 1, 0, and -9, respectively.

    Parameters
    ----------
    code_file
        Path to a csv, parquet, or feather/IPC file with columns ``id, code, index``,
        where:
            - ``id``: Patient identifier
            - ``code``: ICD9 and/or ICD10 codes
            - ``index``: Unique index such as date of diagnosis
        The actual column names don't matter if the above column order is followed.
    min_code_count
        Number of occurrences of a code required to be considered a case for a given
        code, else exclude. E.g. for min_code_count = 3, a patient with one or two
        occurrences of a code will be excluded from analysis of that code instead of
        being a case or control.
    map_phecodes
        Map ICD9/10 codes to PheCodes.
    phecode_rollup
        Perform roll-up of PheCodes, such that each PheCode is also mapped to its parent
        PheCodes.
    phecode_exclude
        Apply PheCode exclusion criteria to prevent case contamination in the control
        cohort.
    sex
        Path to a csv, parquet, or feather/IPC file containing columns ``PT_ID, sex``,
        where sex is provided as ``Male`` or ``Female``. If supplied, sex-based PheCode
        restrictions will be applied.
    n_chunks
        Number of batches to run the mapping in, subject-wise. Eg ``n_chunks=3`` will
        perform the mapping in 3 batches of subjects. The output of each batch will be
        appended to ``outfile``. Only .csv output is supported.
    outfile
        Output path for saving the phenotype file.

    Returns
    -------
    pd.DataFrame
    """

    # can't rename LazyFrame columns to their own names,
    # see https://github.com/pola-rs/polars/issues/3361
    # just force user to use column names 'PT_ID', 'code', 'index' in input file for now

    # id_name, code_name, index_name = codes.columns
    # codes = codes.rename({
    #     id_name: 'PT_ID',
    #     code_name: 'code',
    #     index_name: 'index'
    # })

    if n_chunks > 1 and Path(outfile).suffix != ".csv":
        raise NotImplementedError(
            "Only .csv output is supported for chunked PheCode mapping."
        )
    if outfile and Path(outfile).is_file():
        Path(outfile).unlink()

    with pl.StringCache():
        cohort_codes = _get_cohort_codes(
            code_file, map_phecodes, phecode_rollup, phecode_exclude
        )

        if n_chunks > 1:
            cohort_ids = (
                _read_file(code_file)
                .select("PT_ID")
                .unique()
                .with_column(pl.col("PT_ID").cast(pl.Categorical))
                .collect()["PT_ID"]
            )

            if n_chunks > cohort_ids.shape[0]:
                raise ValueError("n_chunks is greater than number of subjects")
            chunks = np.array_split(np.arange(cohort_ids.shape[0]), n_chunks)
        else:
            chunks = range(1)
            chunk_ids = None

        for chunk in chunks:
            codes = _read_file(code_file)
            codes = codes.with_columns(
                [
                    pl.col("PT_ID").cast(pl.Categorical),
                    pl.col("code").cast(pl.Categorical),
                ]
            )
            if n_chunks > 1:
                chunk_ids = cohort_ids.take(chunk)

                # https://github.com/pola-rs/polars/issues/3420
                # codes = codes.filter(pl.col("PT_ID").is_in(chunk_ids))
                # workaround join
                codes = codes.join(chunk_ids.to_frame().lazy(), how="inner", on="PT_ID")

            if map_phecodes:
                codes = map_icd_to_phecodes(codes)

            if phecode_rollup:
                codes = map_rollup(codes)

            codes = (
                codes.groupby(["PT_ID", "code"])
                .agg(pl.count())
                .with_column(pl.col("count").cast(pl.Int16))
            )

            if phecode_exclude:
                codes = map_exclusions(codes)

            if sex:
                codes = map_sex_restrictions(codes, sex, cohort_codes, chunk_ids)

            codes = (
                codes.groupby(["PT_ID", "code"])
                .agg(pl.max("count"))
                .select(
                    [
                        "PT_ID",
                        "code",
                        pl.when(pl.col("count") < min_code_count)
                        .then(-9)
                        .otherwise(1)
                        .cast(pl.Int16)
                        .alias("count"),
                    ]
                )
            )

            if n_chunks > 1:
                codes = _add_missing_codes(codes, cohort_codes, chunk_ids)
            codes = codes.collect()

            # polars pivot takes uses a LOT more ram
            codes = codes.to_pandas()
            codes = codes.pivot(
                index="PT_ID", columns="code", values="count"
            ).rename_axis(None, axis=1)
            codes = codes.fillna(0)
            codes = codes[sorted(codes.columns.tolist())].reset_index()
            if outfile:
                _pd_write_file(output=codes, filename=outfile, n_chunks=n_chunks)

    return codes


def map_icd_to_phecodes(codes: pl.LazyFrame) -> pl.LazyFrame:
    """
    Map ICD9 and/or ICD10 codes to PheCodes. The v1.2b1 mapping from the R package is
    used, from ``PheWAS::phecode_map``.

    Parameters
    ----------
    codes
        pl.LazyFrame of ``PT_ID, code, index``
    Returns
    -------
        pl.LazyFrame
            of ``PT_ID, code, index``, where codes are now PheCodes
    """
    phemap = pl.scan_parquet("resources/phecode_map.parquet")  # PheWAS::phecode_map

    phemap = phemap.with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            pl.col("phecode").cast(pl.Categorical),
        ]
    )

    codes = (
        codes.join(phemap, on=["code"], how="inner")
        .select(["PT_ID", pl.col("phecode").alias("code"), "index"])
        .unique()
    )
    return codes


def map_rollup(codes: pl.LazyFrame) -> pl.LazyFrame:
    """
    Perform roll-up of PheCodes, mapping each PheCode to its parent codes.
    Uses the ``PheWAS::phecode_rollup_map`` from the R package.

    Parameters
    ----------
    codes
        pl.LazyFrame of ``PT_ID, code, index``
    Returns
    -------
        pl.LazyFrame
            of ``PT_ID, code, index``
    """
    rollup_map = pl.scan_parquet(
        "resources/phecode_rollup_map.parquet"  # PheWAS::phecode_rollup_map
    )

    rollup_map = rollup_map.with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            pl.col("phecode_unrolled").cast(pl.Categorical),
        ]
    )

    codes = (
        codes.join(rollup_map, on="code", how="inner")
        .select(["PT_ID", pl.col("phecode_unrolled").alias("code"), "index"])
        .unique()
    )
    return codes


def map_exclusions(codes: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply PheCode exclusion criteria. Uses ``PheWAS::phecode_exclude`` from the R pkg.

    Parameters
    ----------
    codes
        pl.LazyFrame of ``PT_ID, code, index``
    Returns
    -------
        pl.LazyFrame
            of ``PT_ID, code, index``
    """
    exclusions = pl.scan_parquet(
        "resources/phecode_exclude.parquet"  # PheWAS::phecode_exclude)
    )

    exclusions = exclusions.with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            pl.col("exclusion_criteria").cast(pl.Categorical),
        ]
    )
    exclusions = (
        exclusions.join(
            codes.select(["PT_ID", pl.col("code").alias("exclusion_criteria")]),
            how="inner",
            on="exclusion_criteria",
        )
        .select(["PT_ID", "code", pl.lit(-9).alias("count").cast(pl.Int16)])
        .unique()
    )
    codes = pl.concat([codes, exclusions])
    return codes


def map_sex_restrictions(
    codes: pl.LazyFrame, sex: str, cohort_codes: pl.DataFrame, chunk_ids: pl.Series
) -> pl.LazyFrame:
    """
    Apply PheCode sex restrictions. Uses ``PheWAS::gender_restriction`` from the R pkg.
    Parameters
    ----------
    codes
        pl.LazyFrame of ``PT_ID, code, index``
    sex
        Path to a csv, parquet, or feather/IPC file containing columns ``PT_ID, sex``,
        where sex is provided as ``Male`` or ``Female``. If supplied, sex-based PheCode
        restrictions will be applied.
    Returns
    -------
        pl.LazyFrame
            of ``PT_ID, code, index``
    """
    sex = _read_file(sex, columns=["PT_ID", "sex"])
    sex = sex.select(
        [
            pl.col("PT_ID").cast(pl.Categorical).alias("PT_ID"),
            pl.col("sex").cast(pl.Categorical),
        ]
    )
    if chunk_ids:
        sex = sex.join(chunk_ids.to_frame().lazy(), how="inner", on="PT_ID")

    sex_restrictions = pl.scan_parquet(
        "resources/sex_restriction.parquet"  # PheWAS::gender_restriction
    ).with_columns(
        [
            pl.col("code").cast(pl.Categorical),
            pl.col("exclude_sex").cast(pl.Categorical),
        ]
    )

    # only keep codes that are observed in cohort
    sex_restrictions = sex_restrictions.join(
        cohort_codes.lazy(), how="inner", on="code"
    )

    # change existing codes that are inconsistent with id's sex
    codes = (
        codes.join(sex, how="inner", on="PT_ID")
        .join(sex_restrictions, how="left", on="code")
        .select(
            [
                "PT_ID",
                "code",
                pl.when(pl.col("sex") == pl.col("exclude_sex"))
                .then(-9)
                .otherwise(pl.col("count"))
                .cast(pl.Int16)
                .alias("count"),
            ]
        )
    )

    # add exclusions so they're not considered as controls
    sex_restrictions = sex_restrictions.join(
        sex, how="inner", left_on="exclude_sex", right_on="sex"
    ).select(["PT_ID", "code", pl.lit(-9).cast(pl.Int16).alias("count")])

    return pl.concat([codes, sex_restrictions])


def _get_cohort_codes(
    code_file: str,
    map_phecodes: bool = True,
    phecode_rollup: bool = True,
    phecode_exclude: bool = True,
):
    """
    Get all unique PheCodes seen in cohort, starting from ICD codes.

    Parameters
    ----------
    code_file
        ICD code file with columns ``id, ICD, index``.
    map_phecodes
    phecode_rollup
    phecode_exclude

    Returns
    -------
    pl.DataFrame
    """
    codes = (
        _read_file(code_file)
        .select("code")
        .with_column(pl.col("code").cast(pl.Categorical))
    )

    if map_phecodes:
        phemap = pl.scan_parquet("resources/phecode_map.parquet")  # PheWAS::phecode_map
        phemap = phemap.with_columns(
            [
                pl.col("code").cast(pl.Categorical),
                pl.col("phecode").cast(pl.Categorical),
            ]
        )

        codes = (
            codes.join(phemap, on=["code"], how="inner")
            .select(pl.col("phecode").alias("code"))
            .unique()
        )

        if phecode_rollup:
            rollup_map = pl.scan_parquet(
                "resources/phecode_rollup_map.parquet"  # PheWAS::phecode_rollup_map
            )

            rollup_map = rollup_map.with_columns(
                [
                    pl.col("code").cast(pl.Categorical),
                    pl.col("phecode_unrolled").cast(pl.Categorical),
                ]
            )

            codes = codes.join(rollup_map, how="inner", on="code").select(
                pl.col("phecode_unrolled").alias("code")
            )

        if phecode_exclude:
            exclusions = pl.scan_parquet(
                "resources/phecode_exclude.parquet"  # PheWAS::phecode_exclude)
            )

            exclusions = exclusions.with_columns(
                [
                    pl.col("code").cast(pl.Categorical),
                    pl.col("exclusion_criteria").cast(pl.Categorical),
                ]
            )
            exclusions = (
                exclusions.join(
                    codes.select(pl.col("code").alias("exclusion_criteria")),
                    how="inner",
                    on="exclusion_criteria",
                )
                .select("code")
                .unique()
            )
            codes = pl.concat([codes, exclusions])

        return codes.unique().collect()


def _add_missing_codes(
    codes: pl.LazyFrame, cohort_codes: pl.DataFrame, chunk_ids: pl.Series
) -> pl.LazyFrame:
    """
    Add PheCodes seen in the cohort that are not in the current chunk of subjects. This
    allows for appending the mapping output in batches.

    Parameters
    ----------
    codes
        pl.LazyFrame of current chunk ``'PT_ID','code','count'``
    cohort_codes
        pl.DataFrame of all codes seen in cohort, including mapped exclusions.
    chunk_ids
        All subject ids in the current chunk
    Returns
    -------
    pl.LazyFrame
    """
    # https://github.com/pola-rs/polars/issues/3420
    chunk_codes = codes.select("code").unique().collect()["code"]
    missing_codes = cohort_codes.filter(~pl.col("code").is_in(chunk_codes))
    missing_codes = missing_codes.with_columns(
        [
            chunk_ids[:1].alias("PT_ID").cast(pl.Categorical),
            pl.lit(None).cast(pl.Int16).alias("count"),
        ]
    ).select(["PT_ID", "code", "count"])
    return pl.concat([codes, missing_codes.lazy()])


def _read_file(filename: str, columns=None) -> pl.LazyFrame:
    path = Path(filename)
    if path.is_file():
        if path.suffix == ".csv":
            df = pl.scan_csv(path, parse_dates=True)
        elif path.suffix == ".parquet":
            df = pl.scan_parquet(path)
        elif path.suffix in (".ipc", ".feather"):
            df = pl.scan_ipc(path)
        else:
            raise ValueError(
                "Invalid input file extension - .csv, .parquet, and .feather/.IPC are "
                "accepted"
            )
        if columns:
            df = df.select(columns)
        return df
    else:
        raise FileNotFoundError(f"Input file {filename} not found!")


def _pd_write_file(output: pd.DataFrame, filename: str, n_chunks: int = 1) -> None:
    path = Path(filename)
    if path.suffix == ".csv":
        if n_chunks > 1:
            output.to_csv(filename, index=False, mode="a", header=not path.is_file())
        else:
            output.to_csv(filename, index=False)
    elif path.suffix == ".parquet":
        if n_chunks > 1:
            raise NotImplementedError(
                "Only .csv output is supported for low_memory mode"
            )
        output.to_parquet(filename, index=False)
    else:
        raise ValueError(
            f"Invalid output file extension {path.suffix} - .csv and .parquet are "
            f"accepted"
        )
