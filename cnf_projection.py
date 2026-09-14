"""Euclidean projection onto the discrete convex-normal endpoint cone.

Dykstra's algorithm alternates exact projections onto (a) the product of the
two monotone cones and (b) the elementwise non-crossing halfspaces.  Its limit
is the unique Euclidean projection onto their closed convex intersection.
"""

from __future__ import annotations

import numpy as np


def pava(values: np.ndarray, increasing: bool = True) -> np.ndarray:
    """Exact unweighted Euclidean projection onto a monotone cone."""
    source = np.asarray(values, dtype=float)
    work = source.copy() if increasing else -source.copy()
    levels: list[float] = []
    weights: list[int] = []
    for value in work:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            levels[-2] = (weights[-2] * levels[-2] + weights[-1] * levels[-1]) / weight
            weights[-2] = weight
            levels.pop()
            weights.pop()
    result = np.empty_like(work)
    cursor = 0
    for level, weight in zip(levels, weights):
        result[cursor:cursor + weight] = level
        cursor += weight
    return result if increasing else -result


def _project_monotone(vector: np.ndarray) -> np.ndarray:
    size = len(vector) // 2
    return np.r_[pava(vector[:size], True), pava(vector[size:], False)]


def _project_non_crossing(vector: np.ndarray) -> np.ndarray:
    size = len(vector) // 2
    left, right = vector[:size].copy(), vector[size:].copy()
    mask = left > right
    midpoint = (left[mask] + right[mask]) / 2.0
    left[mask] = midpoint
    right[mask] = midpoint
    return np.r_[left, right]


def project_cnf(left: np.ndarray, right: np.ndarray, *, tolerance: float = 1e-11,
                max_iterations: int = 10000, return_iterations: bool = False):
    """Return the metric projection onto monotone, non-crossing endpoints."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("left and right must be one-dimensional arrays of equal shape")
    current = np.r_[left, right]
    correction_a = np.zeros_like(current)
    correction_b = np.zeros_like(current)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        previous = current
        intermediate = _project_monotone(previous + correction_a)
        correction_a = previous + correction_a - intermediate
        current = _project_non_crossing(intermediate + correction_b)
        correction_b = intermediate + correction_b - current
        if np.max(np.abs(current - previous)) <= tolerance * (1.0 + np.max(np.abs(previous))):
            break
    else:
        raise RuntimeError("CNF metric projection did not converge")
    size = len(left)
    result = (current[:size], current[size:])
    return (*result, iterations) if return_iterations else result
