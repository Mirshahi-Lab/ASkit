from argparse import _SubParsersAction
from pathlib import Path

from askit.mas.pipeline import run_mas


def add_mas_command(subparsers: _SubParsersAction) -> None:
    """Add the 'mas' subcommand to the ASkit CLI."""
    subparser = subparsers.add_parser(
        "mas",
        help="Multiple Association Study (MAS)",
        description="Run a Multiple Association Study (MAS)",
    )
    # Set the default function to run when the 'mas' subcommand is invoked
    subparser.set_defaults(func=run_mas)
    subparser.add_argument(
        "--dry-run",
        action="store_true",
        help="Default is False. "
        "If set, see the configuration but do not actually run the MAS.",
    )
    input_group = subparser.add_argument_group(
        "Input Options", "Options for specifying input data for the MAS."
    )
    input_group.add_argument(
        "--input-file",
        "-i",
        type=Path,
        help="Input file path. Can be a .parquet, .ipc, .csv, .tsv,"
        " or .txt file. File suffix must match the format!",
        required=True,
    )
    input_group.add_argument(
        "--output-file",
        "-o",
        type=Path,
        help="Output file path for the MAS analysis results. Can be a .parquet, "
        ".ipc, .csv, .tsv, or .txt file. File suffix will match the format.",
        required=True,
    )
    input_group.add_argument(
        "--predictors",
        "-p",
        type=str,
        help="Predictor columns (comma separated list of names or i:INDEX for indices)",
        required=True,
    )
    input_group.add_argument(
        "--dependents",
        "-d",
        type=str,
        help="Dependent columns (comma separated list of names or i:INDEX for indices)",
        required=True,
    )
    input_group.add_argument(
        "--covariates",
        "-c",
        type=str,
        help="Covariate columns (comma separated list of names or i:INDEX for indices)",
    )
    input_group.add_argument(
        "-cc",
        "--categorical-covariates",
        type=str,
        help="Categorical covariate columns "
        "(comma separated list, names or 'i:INDEX for indices)",
    )
    input_group.add_argument(
        "-nv",
        "--null-values",
        type=str,
        help="Default is None (polars default). "
        "Specify the values to be considered as null/missing"
        " in the input data (comma separated list).",
        default=None,
    )
    input_group.add_argument(
        "--make-dirs",
        action="store_true",
        help="Default is False. Create the output directory if it does not exist.",
    )
    # TODO add options for save-predictors or save-dependents for filesplitting

    # Multiprocessing Options
    multiprocessing_group = subparser.add_argument_group(
        "Multiprocessing Options",
        "Options for specifying multiprocessing parameters for the MAS",
    )
    multiprocessing_group.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=1,
        help="Default is 1. Number of parallel workers to run for the MAS.",
    )
    multiprocessing_group.add_argument(
        "--threads-per-worker",
        "-t",
        type=int,
        default=1,
        help="Default is 1. Number of threads available to each worker for the MAS.",
    )

    # Model parameters
    model_group = subparser.add_argument_group(
        "Model Parameters", "Options for specifying model parameters for the MAS."
    )
    model_group.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["firth", "firth-hybrid", "logistic", "linear"],
        help="Model to use for MAS. Firth-hybrid runs Firth's logistic regression"
        " if logistic regression p-value is below alpha threshold.",
        required=True,
    )
    model_group.add_argument(
        "--max-iter",
        type=int,
        default=25,
        help="Default is 25. Maximum number of iterations for the model fitting.",
    )
    model_group.add_argument(
        "--max-step",
        type=float,
        default=5.0,
        help="Default is 5.0. "
        "Maximum step size per coefficient for Newton-Raphson solver.",
    )
    model_group.add_argument(
        "--max-halfstep",
        type=int,
        default=25,
        help="Default is 25. "
        "Maximum number of step-halvings for the Newton-Raphson solver.",
    )
    model_group.add_argument(
        "--gtol",
        type=float,
        default=1e-4,
        help="Default is 1e-4. Gradient convergence criteria. "
        "Converged when max|gradient| < gtol.",
    )
    model_group.add_argument(
        "--xtol",
        type=float,
        default=1e-4,
        help="Default is 1e-4. Parameter convergence criteria. "
        "Converged when max|delta| < xtol.",
    )
    model_group.add_argument(
        "--no-intercept",
        action="store_true",
        help="Default is False. Do not fit an intercept in the model.",
    )
    model_group.add_argument(
        "--penalty-weight",
        type=float,
        default=0.5,
        help="Default is 0.5. Penalty weight for the Firth's logistic regression.",
    )
    model_group.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Default is 0.05. Significance level for the MAS analysis.",
    )

    # Preprocessing options
    preprocessing_group = subparser.add_argument_group(
        "Preprocessing Parameters",
        "Options for specifying preprocessing parameters for the MAS.",
    )
    preprocessing_group.add_argument(
        "--min-case-count",
        "-mcc",
        type=int,
        default=20,
        help="Default is 20. "
        "Minimum count of observations required for an association to be considered. "
        "Also works as the minimum observation count for linear regression.",
    )
    preprocessing_group.add_argument(
        "--missing-covariates-operation",
        "-mco",
        type=str,
        choices=["fail", "drop", "mean", "max", "min", "zero", "one"],
        default="fail",
        help="Default is 'fail'. Specify how to handle missing values in covariates. "
        "'fail' will raise an error if missing values are present, "
        "'drop' will remove rows with missing values."
        "'mean', 'max', 'min', 'zero', and 'one' will impute.",
    )
    # TODO Add preprocessing transformations

    # PheCode Options
    phecode_group = subparser.add_argument_group(
        "PheCode Options", "PheCode-related options for the MAS."
    )
    phewas_group = phecode_group.add_mutually_exclusive_group()
    phewas_group.add_argument(
        "--phewas",
        action="store_true",
        help="This is a PheWAS analysis where PheCodes are the dependent variables.",
    )
    phewas_group.add_argument(
        "--flipwas",
        action="store_true",
        help="This is a PheWAS analysis where PheCodes are the predictor variables.",
    )
    phecode_group.add_argument(
        "--sex-col",
        type=str,
        default="sex",
        help="Default is 'sex'. Column name for the sex variable in the dataset.",
    )
    phecode_group.add_argument(
        "--female-code",
        type=str,
        default="1",
        help="Default is '1'. Code representing female in the sex column.",
    )
    sex_specific_group = phecode_group.add_mutually_exclusive_group()
    sex_specific_group.add_argument(
        "--male-only",
        action="store_true",
        help="Run the analysis only on male samples.",
    )
    sex_specific_group.add_argument(
        "--female-only",
        action="store_true",
        help="Run the analysis only on female samples.",
    )

    # Logger options
    logger_group = subparser.add_argument_group(
        "Logger Options", "Options for configuring the logger."
    )
    verbosity = logger_group.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for the logger."
    )
    verbosity.add_argument(
        "--quiet", action="store_true", help="Enable quiet output for the logger."
    )
