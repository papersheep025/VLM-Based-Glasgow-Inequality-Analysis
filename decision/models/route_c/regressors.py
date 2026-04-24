"""
Regressor factory for Route C.

All regressors are wrapped in a StandardScaler → estimator Pipeline so the
training code can treat them uniformly.  Each target dimension gets its own
cloned pipeline (via sklearn.base.clone in the trainer).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _clip30(X: np.ndarray) -> np.ndarray:
    return np.clip(X, -30, 30)


def build_regressor(name: str, seed: int = 42) -> Pipeline:
    if name == "ridge_cv":
        # cv=5 avoids GCV numerical issues for high-dim features.
        # clip_scaler prevents overflow in RidgeCV's internal alpha sweep when
        # low-variance columns (e.g. ego-gap) are amplified by StandardScaler.
        reg = RidgeCV(alphas=np.logspace(-2, 6, 25), cv=5)
        clip = FunctionTransformer(_clip30)
        return Pipeline([("scaler", StandardScaler()), ("clip", clip), ("reg", reg)])
    if name == "lasso_cv":
        reg = LassoCV(
            alphas=None,
            cv=5,
            max_iter=20000,
            random_state=seed,
            n_jobs=None,
        )
    elif name == "elasticnet_cv":
        reg = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=None,
            cv=5,
            max_iter=20000,
            random_state=seed,
            n_jobs=None,
        )
    else:
        raise ValueError(
            f"unknown regressor {name!r}; expected one of "
            "'ridge_cv', 'lasso_cv', 'elasticnet_cv'"
        )
    return Pipeline([("scaler", StandardScaler()), ("reg", reg)])
