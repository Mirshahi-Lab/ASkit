import warnings

import numpy as np
import polars as pl
import statsmodels.api as sm
from firthmodels import FirthLogisticRegression
from sklearn.exceptions import ConvergenceWarning

from askit.run_study.config import StudyConfig


def firth_regression(X: pl.DataFrame, y: np.ndarray, config: StudyConfig) -> dict:
    """Run Firth regression on the given data.

    Uses the same default settings as the R logistf package:
    - max_iter: 25 (maxit)
    - max_halfstep: 0 (maxhs)
    - max_step: 5.0 (maxstep)
    - gtol: 1e-5 (gconv)
    - xtol: 1e-5 (xconv)

    Parameters
    ----------
    X : polars.DataFrame
        The data to use for the regression.
    y : np.ndarray
        The dependent variable.
    config : StudyConfig
        The Study configuration object.

    Returns
    -------
    dict
        The results of the regression.
    """
    with warnings.catch_warnings(record=True) as w:
        converged = True
        fl = FirthLogisticRegression(
            max_iter=config.max_iter,
            max_halfstep=config.max_halfstep,
            max_step=config.max_step,
            gtol=config.gtol,
            xtol=config.xtol,
            penalty_weight=config.penalty_weight,
        )
        fl.fit(X, y).lrt(0, warm_start=True)
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                converged = False
        return {
            "pval": fl.lrt_pvalues_[0],
            "beta": fl.coef_[0],
            "se": fl.bse_[0],
            "OR": np.e ** fl.coef_[0],
            "converged": converged,
            "beta_ci_low": fl.conf_int()[0][0],
            "beta_ci_high": fl.conf_int()[0][1],
        }


def logistic_regression(X: pl.DataFrame, y: np.ndarray, config: StudyConfig) -> dict:
    """Run standard logistic regression on the given data using statsmodels"""
    with warnings.catch_warnings(record=True) as w:
        converged = True
        fl = FirthLogisticRegression(
            max_iter=config.max_iter,
            max_halfstep=config.max_halfstep,
            max_step=config.max_step,
            gtol=config.gtol,
            xtol=config.xtol,
            penalty_weight=0,  # this is the same as logistic regression - with speedups
        )
        fl.fit(X, y)
        for warning in w:
            if issubclass(warning.category, ConvergenceWarning):
                converged = False
        return {
            "pval": fl.pvalues_[0],
            "beta": fl.coef_[0],
            "se": fl.bse_[0],
            "OR": np.e ** fl.coef_[0],
            "converged": converged,
            "beta_ci_low": fl.conf_int()[0][0],
            "beta_ci_high": fl.conf_int()[0][1],
        }


# TODO - switch this to polars-ols?
def linear_regression(X: pl.DataFrame, y: np.ndarray, config: StudyConfig) -> dict:
    X_sm = sm.add_constant(X.to_numpy(), prepend=False)
    model = sm.OLS(y, X_sm)
    result = model.fit()
    return {
        "pval": result.pvalues[0],
        "beta": result.params[0],
        "se": result.bse[0],
        "converged": True,
        "beta_ci_low": result.conf_int()[0][0],
        "beta_ci_high": result.conf_int()[0][1],
    }
