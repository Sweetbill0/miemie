"""Simple machine learning utilities built without third-party dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


class StandardScaler:
    """Standard score feature scaler implemented using plain Python."""

    def __init__(self) -> None:
        self._means: List[float] | None = None
        self._stds: List[float] | None = None

    @property
    def means(self) -> List[float]:
        if self._means is None:
            raise ValueError("Scaler has not been fitted.")
        return self._means

    @property
    def stds(self) -> List[float]:
        if self._stds is None:
            raise ValueError("Scaler has not been fitted.")
        return self._stds

    def fit(self, rows: Iterable[Iterable[float]]) -> None:
        cached_rows = [list(row) for row in rows]
        self.fit_list(cached_rows)

    def fit_list(self, rows: List[List[float]]) -> None:
        if not rows:
            raise ValueError("Cannot fit scaler without any samples.")
        feature_count = len(rows[0])
        means = [0.0] * feature_count
        for row in rows:
            if len(row) != feature_count:
                raise ValueError("All rows must share the same number of features.")
            for idx, value in enumerate(row):
                means[idx] += value
        row_count = float(len(rows))
        means = [value / row_count for value in means]
        stds = [0.0] * feature_count
        for row in rows:
            for idx, value in enumerate(row):
                diff = value - means[idx]
                stds[idx] += diff * diff
        stds = [(value / row_count) ** 0.5 for value in stds]
        self._means = means
        self._stds = stds

    def transform_row(self, row: Iterable[float]) -> List[float]:
        means = self.means
        stds = self.stds
        row_list = list(row)
        if len(row_list) != len(means):
            raise ValueError("Unexpected number of features in row.")
        transformed: List[float] = []
        for idx, value in enumerate(row_list):
            std = stds[idx]
            if std < 1e-12:
                transformed.append(0.0)
            else:
                transformed.append((value - means[idx]) / std)
        return transformed

    def transform(self, rows: Iterable[Iterable[float]]) -> List[List[float]]:
        return [self.transform_row(row) for row in rows]


@dataclass
class RidgeRegression:
    """Closed-form ridge regression using Gaussian elimination."""

    regularization: float = 1e-3
    coefficients: List[float] | None = None

    def fit(self, features: List[List[float]], targets: List[float]) -> None:
        if not features:
            raise ValueError("Cannot train model without feature rows.")
        if len(features) != len(targets):
            raise ValueError("Feature and target counts differ.")
        feature_count = len(features[0])
        xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
        xty = [0.0 for _ in range(feature_count)]
        for row, target in zip(features, targets):
            if len(row) != feature_count:
                raise ValueError("All feature rows must share the same length.")
            for i in range(feature_count):
                value_i = row[i]
                xty[i] += value_i * target
                for j in range(i, feature_count):
                    value_j = row[j]
                    xtx[i][j] += value_i * value_j
        for i in range(feature_count):
            xtx[i][i] += self.regularization
        # Fill the lower triangle to maintain symmetry.
        for i in range(feature_count):
            for j in range(i):
                xtx[i][j] = xtx[j][i]
        solution = _solve_linear_system(xtx, xty)
        self.coefficients = solution

    def predict_row(self, row: Iterable[float]) -> float:
        if self.coefficients is None:
            raise ValueError("Model has not been trained.")
        total = 0.0
        for coef, value in zip(self.coefficients, row):
            total += coef * value
        return total

    def predict(self, rows: Iterable[Iterable[float]]) -> List[float]:
        return [self.predict_row(row) for row in rows]


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    size = len(matrix)
    if size != len(vector):
        raise ValueError("Matrix and vector dimensions do not match.")
    # Create augmented matrix for Gauss-Jordan elimination.
    aug = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]
    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(aug[r][col]))
        pivot = aug[pivot_row][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned.")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        inv_pivot = 1.0 / pivot
        aug[col] = [value * inv_pivot for value in aug[col]]
        for row_idx in range(size):
            if row_idx == col:
                continue
            factor = aug[row_idx][col]
            if abs(factor) < 1e-12:
                aug[row_idx][col] = 0.0
                continue
            aug[row_idx] = [value - factor * pivot_value for value, pivot_value in zip(aug[row_idx], aug[col])]
        # Normalise pivot column exactly.
        for row_idx in range(size):
            if row_idx == col:
                aug[row_idx][col] = 1.0
            else:
                aug[row_idx][col] = 0.0
    return [row[-1] for row in aug]
