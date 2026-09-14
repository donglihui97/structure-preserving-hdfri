"""Numerical validation of the Dykstra CNF metric projection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from cnf_projection import project_cnf
from preexperiments import violations


def constraint_matrix(levels: int) -> np.ndarray:
    rows = []
    for j in range(levels - 1):
        row = np.zeros(2 * levels); row[j + 1] = 1; row[j] = -1; rows.append(row)
        row = np.zeros(2 * levels); row[levels + j] = 1; row[levels + j + 1] = -1; rows.append(row)
    for j in range(levels):
        row = np.zeros(2 * levels); row[levels + j] = 1; row[j] = -1; rows.append(row)
    return np.asarray(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    rng = np.random.default_rng(20260912)
    levels = 11
    matrix = constraint_matrix(levels)
    agreement, feasibility, iterations = [], [], []
    for _ in range(100):
        source = rng.normal(size=2 * levels)
        left, right, used = project_cnf(source[:levels], source[levels:], return_iterations=True)
        dykstra = np.r_[left, right]
        reference = minimize(
            lambda x: 0.5 * np.sum((x - source) ** 2), source,
            jac=lambda x: x - source,
            constraints=[LinearConstraint(matrix, 0.0, np.inf)], method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not reference.success:
            raise RuntimeError(reference.message)
        agreement.append(float(np.linalg.norm(dykstra - reference.x)))
        feasibility.append(not any(violations(left, right).values()))
        iterations.append(used)

    nonexpansive = []
    for _ in range(1000):
        a = rng.normal(size=2 * levels); b = rng.normal(size=2 * levels)
        pa = np.r_[*project_cnf(a[:levels], a[levels:])]
        pb = np.r_[*project_cnf(b[:levels], b[levels:])]
        nonexpansive.append(float(np.linalg.norm(pa - pb) - np.linalg.norm(a - b)))
    payload = {
        "qp_cases": 100,
        "nonexpansive_pairs": 1000,
        "max_difference_vs_slsqp": max(agreement),
        "all_feasible": all(feasibility),
        "max_nonexpansive_excess": max(nonexpansive),
        "iterations_mean": float(np.mean(iterations)),
        "iterations_max": max(iterations),
    }
    (root / "cnf_projection_validation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
