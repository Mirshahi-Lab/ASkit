import sys
from argparse import Namespace
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import polars as pl
from loguru import logger


def _log_format(record):
    """Use a compact format for INFO/SUCCESS, detailed format for everything else."""
    if record["level"].no == 20:  # INFO
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | {message}\n"
        )
    if record["level"].no == 25:  # SUCCESS
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | <green>{message}</green>\n"
        )
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | <cyan>{module}</cyan>:"
        "<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}\n"
    )


@dataclass
class MASConfig:
    """
    Config class to hold parameters for a Multiple Association Study (MAS)
    """

    dry_run: bool
    # Input Options
    input_file: Path
    output_file: Path
    predictors: str
    dependents: str
    covariates: str
    categorical_covariates: str
    null_values: str | None
    make_dirs: bool

    # Multiprocessing Options
    num_workers: int
    threads_per_worker: int

    # Model Options
    model: Literal["firth", "firth-hybrid", "logistic", "linear"]
    max_iter: int
    max_step: float
    max_halfstep: int
    gtol: float
    xtol: float
    fit_intercept: bool
    penalty_weight: float

    # Preprocessing Options
    min_case_count: int
    missing_covariates_operation: Literal["fail", "drop"]

    # PheCode Options
    is_phewas: bool
    is_flipwas: bool
    sex_col: str
    female_code: str
    male_only: bool
    female_only: bool

    # Logger Options
    verbose: bool
    quiet: bool

    # Derived attributes post-init
    reader: Callable[[Path], pl.LazyFrame] | None = field(default=None, init=False)
    column_names: list[str] = field(default_factory=list, init=False)
    total_column_count: int = field(default_factory=int, init=False)
    included_column_count: int = field(default_factory=int, init=False)
    included_row_count: int = field(default_factory=int, init=False)
    ipc_file: str | None = field(default=None, init=False)
    # Column lists
    predictor_columns: list[str] = field(default_factory=list, init=False)
    dependent_columns: list[str] = field(default_factory=list, init=False)
    covariate_columns: list[str] = field(default_factory=list, init=False)
    categorical_covariate_columns: list[str] = field(default_factory=list, init=False)
    included_columns: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        "Validate and process the config options after initialization."
        self._validate_io()
        self._parse_column_lists()
        self._assert_unique_column_sets()

    @classmethod
    def from_args(cls, args: Namespace) -> "MASConfig":
        "Create a MASConfig from CLI arguments"
        return cls(
            dry_run=args.dry_run,
            input_file=args.input_file,
            output_file=args.output_file,
            predictors=args.predictors,
            dependents=args.dependents,
            covariates=args.covariates,
            categorical_covariates=args.categorical_covariates,
            null_values=args.null_values,
            make_dirs=args.make_dirs,
            num_workers=args.num_workers,
            threads_per_worker=args.threads_per_worker,
            model=args.model,
            max_iter=args.max_iter,
            max_step=args.max_step,
            max_halfstep=args.max_halfstep,
            gtol=args.gtol,
            xtol=args.xtol,
            fit_intercept=not args.no_intercept,
            penalty_weight=args.penalty_weight,
            min_case_count=args.min_case_count,
            missing_covariates_operation=args.missing_covariates_operation,
            is_phewas=args.phewas,
            is_flipwas=args.flipwas,
            sex_col=args.sex_col,
            female_code=args.female_code,
            male_only=args.male_only,
            female_only=args.female_only,
            verbose=args.verbose,
            quiet=args.quiet,
        )

    def setup_logger(self):
        logger.remove()
        if self.quiet:
            logger.add(
                sys.stdout,
                level="SUCCESS",
                format=self._log_format,
                filter=lambda record: record["level"].no <= 25,
                enqueue=True,
            )
            logger.add(sys.stderr, level="ERROR", format=_log_format, enqueue=True)
        elif self.verbose:
            logger.add(
                sys.stdout,
                level="DEBUG",
                format=self._log_format,
                filter=lambda record: record["level"].no <= 25,
                enqueue=True,
            )
            logger.add(sys.stderr, level="WARNING", format=_log_format, enqueue=True)
        else:
            logger.add(
                sys.stdout,
                level="INFO",
                format=_log_format,
                filter=lambda record: record["level"].no <= 25,
                enqueue=True,
            )
            logger.add(
                sys.stderr,
                level="WARNING",
                format=_log_format,
                enqueue=True,
            )

    def _validate_io(self):
        "Validate input and output paths"
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {self.input_file}")
        if not self.output_file.parent.exists() and not self.make_dirs:
            raise ValueError(
                f"Output directory does not exist: {self.output_file.parent}"
            )

        # Parse the input columns
        null_values = None if self.null_values is None else self.null_values.split(",")
        if self.input_file.suffix == ".parquet":
            self.reader = pl.scan_parquet
        elif self.input_file.suffix == ".ipc":
            self.reader = pl.scan_ipc
        elif self.input_file.suffix == ".csv":
            self.reader = partial(pl.scan_csv, null_values=null_values)
        elif self.input_file.suffix == ".tsv":
            self.reader = partial(pl.scan_csv, separator="\t", null_values=null_values)
        elif self.input_file.suffix == ".txt":
            self.reader = partial(pl.scan_csv, separator="\t", null_values=null_values)
        else:
            raise ValueError(f"Unsupported input file format: {self.input_file.suffix}")

        self.column_names = self.reader(self.input_file).collect_schema().names()
        self.total_column_count = len(self.column_names)

    def _parse_column_lists(self) -> None:
        "Parse the column list arguments into lists of column names"
        self.predictor_columns = self._parse_column_list(self.predictors)
        self.dependent_columns = self._parse_column_list(self.dependents)
        self.covariate_columns = self._parse_column_list(self.covariates)
        self.categorical_covariate_columns = self._parse_column_list(
            self.categorical_covariates
        )

    def _parse_column_list(self, column_str: str | None) -> list[str]:
        "Parse a single column list argument into a list of column names"
        if column_str is None:
            return []
        col_splits = column_str.split(",")
        column_list = []
        for col in col_splits:
            # Indexed columns start with the 'i:' identifier
            if col[:2] == "i:":
                column_list.extend(self._extract_indexed_columns(col))
            else:
                if col not in self.column_names:
                    raise ValueError(f"Column {col} does not exist in the input file.")
                column_list.append(col)
        return column_list

    def _extract_indexed_columns(self, index_str: str) -> list[str]:
        "Extract the column indicies from an index column string"
        indicies = index_str.split(":")[-1]
        # Only one column index passed
        if indicies.isnumeric():
            index = int(indicies)
            if index >= self.total_column_count:
                raise ValueError(
                    f"Index {index} is out of range for \n"
                    f"input file with {self.total_column_count} columns"
                )
            return [self.column_names[index]]
        # Multiple column indices passed
        elif "-" in indicies:
            parts = indicies.split("-")
            if len(parts) != 2:
                raise ValueError("Invalid index range format. Too many '-' characters.")
            start, end = parts
            if start.isnumeric():
                start_idx = int(start)
            else:
                raise ValueError("Start index must be a numeric value")
            # End is either specified or should be all remaining columns
            end_idx = int(end) if end.isnumeric() else self.total_column_count
            if start_idx >= self.total_column_count:
                raise ValueError(
                    f"Start index {start_idx} is out of range \n"
                    f"for input file with {self.total_column_count} columns"
                )
            if end_idx > self.total_column_count:
                raise ValueError(
                    f"End index {end_idx} out of range \n"
                    f"for {self.total_column_count} columns. \n"
                    f"If you want to use all remaining columns, use {start}-."
                )
            return self.column_names[start_idx:end_idx]
        else:
            raise ValueError(
                "Invalid index format. Please use i:<index>, "
                "i:<start>-<end>, or i:<start>-."
            )

    def _assert_unique_column_sets(self):
        "Ensure that the predictor, dependent, and covariate columns are unique"
        predictor_set = set(self.predictor_columns)
        dependent_set = set(self.dependent_columns)
        covariate_set = set(self.covariate_columns)
        cat_covariate_set = set(self.categorical_covariate_columns)

        if predictor_set & dependent_set:
            raise ValueError("Predictor and dependent columns must be unique")
        if predictor_set & covariate_set:
            raise ValueError("Predictor and covariate columns must be unique")
        if dependent_set & covariate_set:
            raise ValueError("Dependent and covariate columns must be unique")
        if not cat_covariate_set:
            pass
        elif not cat_covariate_set.issubset(covariate_set):
            raise ValueError(
                "Categorical covariate columns must be a subset of covariate columns"
            )
        included_columns = list(predictor_set | dependent_set | covariate_set)
        # We do this step so that they are ordered
        # in the same order as they appear in the file
        self.included_columns = [
            col for col in self.column_names if col in included_columns
        ]

    def summary(self) -> None:
        """Log a summary of the current configuration settings."""
        sep = "─" * 50
        lines = [
            sep,
            "MAS Configuration Summary",
            sep,
            "Input/Output:",
            f"  Input file:    {self.input_file}",
            f"  Output file:   {self.output_file}",
            f"  Total columns: {self.total_column_count}",
            "",
            "Columns:",
            f"  Predictors:              "
            f"{self._format_column_list(self.predictor_columns)}",
            f"  Dependents:              "
            f"{self._format_column_list(self.dependent_columns)}",
            f"  Covariates:              "
            f"{self._format_column_list(self.covariate_columns)}",
            f"  Categorical covariates:  "
            f"{self._format_column_list(self.categorical_covariate_columns)}",
            "",
            "Model:",
            f"  Model:          {self.model}",
            f"  Fit intercept:  {self.fit_intercept}",
            f"  Max iterations: {self.max_iter}",
            f"  Max step:       {self.max_step}",
            f"  Max halfstep:   {self.max_halfstep}",
            f"  gtol:           {self.gtol}",
            f"  xtol:           {self.xtol}",
            f"  Penalty weight: {self.penalty_weight}",
            "",
            "Multiprocessing:",
            f"  Workers:            {self.num_workers}",
            f"  Threads per worker: {self.threads_per_worker}",
            "",
            "Preprocessing:",
            f"  Min case count:     {self.min_case_count}",
            f"  Missing covariates: {self.missing_covariates_operation}",
        ]
        if self.is_phewas or self.is_flipwas:
            lines += [
                "",
                "PheCode:",
                f"  PheWAS:      {self.is_phewas}",
                f"  FlipWAS:     {self.is_flipwas}",
                f"  Sex column:  {self.sex_col}",
                f"  Female code: {self.female_code}",
                f"  Male only:   {self.male_only}",
                f"  Female only: {self.female_only}",
            ]
        lines.append(sep)
        logger.info("\n".join(lines))

    @staticmethod
    def _format_column_list(columns: list[str], max_display: int = 5) -> str:
        """Format column list for display, truncating if too long."""
        n = len(columns)
        if n == 0:
            return "(none)"
        if n <= max_display:
            return f"{n} column{'s' if n != 1 else ''}: {', '.join(columns)}"
        # Show first 2 and last 2 with count
        preview = f"{columns[0]}, {columns[1]}, ... {columns[-2]}, {columns[-1]}"
        return f"{n} columns: {preview}"
