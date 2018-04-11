import numpy as np
from numpy import ndarray
from typing import List, Callable
from functools import partial, wraps
from itertools import chain

from toolz import functoolz as fz

from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import ShuffleSplit

from ..common.lasso import lasso_hetero, soft_threshold
from ..common.object import Numeric, Learner, RNG


class RandomSearch(Learner, BaseEstimator):

    def __init__(
            self, estimator: BaseEstimator, cost_fn: Callable[[ndarray, ndarray], float],
            setter: Callable[[BaseEstimator, ndarray], None], lam_min: float=1e-8, lam_max: float=1,
            num_grid: int=100, num_cv: int=5, random_state=None):
        self.estimator = estimator
        self.cost_fn = cost_fn
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.num_grid = num_grid
        self.num_cv = num_cv
        if random_state is None:
            random_state = RNG(None)
        elif isinstance(random_state, int):
            random_state = RNG(random_state)
        self.random_state = random_state

    def draw_lam(self, m: int, rng: RNG) -> ndarray:
        low = np.log(self.lam_min)
        high = np.log(self.lam_max)
        log = rng.uniform(low, high, size=m)
        return np.exp(log)

    def uniform_lams(self, m: int, num: int):
        low = self.lam_min
        high = self.lam_max
        for l in np.linspace(np.log(low), np.log(high)):
            yield np.exp(l) * np.ones(m)

    def get_cost(
            self, lam: ndarray, X: ndarray, y: ndarray, splits: List[List[int]]
    ) -> float:
        estimator = clone(self.estimator)
        estimator.set_lam(lam)
        costs = []
        for train_index, test_index in splits:
            estimator.fit(X[train_index], y[train_index])
            pred = estimator.predict(X[test_index])
            costs.append(
                self.cost_fn(y[test_index], pred)
            )
        return np.mean(costs)

    def fit(self, X: ndarray, y: ndarray) -> 'RidgeGridSearch':

        n, m = X.shape

        splits = list(ShuffleSplit(
            n_splits=self.num_cv, test_size=1 / self.num_cv,
            random_state=self.random_state
        ).split(X))

        grid_lam: [[float]] = chain(
                self.uniform_lams(m, self.num_grid // 2),
                (self.draw_lam(m, self.random_state) for _ in range(self.num_grid // 2))
                )
        cost_, self.lam_ = min(
            (self.get_cost(lam, X, y, splits), lam)
            for lam in grid_lam)

        estimator = clone(self.estimator)
        estimator.set_lam(self.lam_)
        estimator.fit(X, y)
        self.chosen_ = estimator
        self.beta_ = self.chosen_.beta_
        self.sigma2_ = self.chosen_.sigma2_

        return self

    def predict(self, X):
        return self.chosen_.predict(X)

    @property
    def result(self) -> ndarray:
        return self.chosen_.beta_


class RidgeRegression(BaseEstimator, Learner):

    def __init__(self, lam: float=1e-2, fit_intercept=True):
        self.lam = lam
        self.fit_intercept = fit_intercept

        self.intercept_X_ = 0
        self.intercept_y_ = 0
        self.beta_ = None
        self.sigma2_ = None

    def fit(self, X: ndarray, y: ndarray) -> 'RidgeRegression':

        if self.fit_intercept:
            self.intercept_X_ = np.mean(X, axis=0)
            X = X - self.intercept_X_
            self.intercept_y_ = np.mean(y, axis=0)
            y = y - self.intercept_y_

        n = len(y)
        C = X.T @ X / n
        b = X.T @ y / n

        self.beta_ = self.fit_beta(C, b, self.lam)
        self.sigma2_ = self.fit_sigma2(X, y, self.beta_, self.lam)
        return self

    @property
    def result(self) -> ndarray:
        return self.beta_

    def set_lam(self, lam: ndarray):
        self.lam = lam

    def predict(self, X):
        if self.fit_intercept:
            return (X - self.intercept_X_) @ self.beta_ + self.intercept_y_
        else:
            return X @ self.beta_

    @staticmethod
    def fit_beta(C: ndarray, b: ndarray, lam: ndarray) -> ndarray:
        beta: ndarray = np.linalg.solve(C + np.diag(lam), b)
        return beta

    @staticmethod
    def fit_sigma2(X: ndarray, y: ndarray, beta: ndarray, lam: ndarray) -> float:
        return np.mean((y - X @ beta) ** 2)


class RidgeRandomSearch(RandomSearch):
    def __init__(
            self, fit_intercept: bool = True, lam_min: float=1e-8, lam_max: float=1,
            num_grid: int=100, num_cv: int=5, random_state=None):
        estimator = RidgeRegression(lam=1, fit_intercept=fit_intercept)
        RandomSearch.__init__(self, estimator, self.cost_fn, self.setter, lam_min, lam_max, num_grid, num_cv, random_state)

    @staticmethod
    def cost_fn(y, pred):
        return np.sqrt(np.mean((y - pred) ** 2))

    @staticmethod
    def setter(estimator, lam):
        estimator.set_lam(lam)


class RidgeCVProb(RidgeCV):

    @wraps(RidgeCV.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        RidgeCV.fit(self, X, y)
        pred = self.predict(X)
        self.sigma2_ = np.mean((y - pred) ** 2)


class LassoCVProb(LassoCV):

    @wraps(LassoCV.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        LassoCV.fit(self, X, y)
        pred = self.predict(X)
        self.sigma2_ = np.mean((y - pred) ** 2)
