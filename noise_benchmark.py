"""Matched knowledge-noise benchmark for grid interpolation and CP-HDFRI.

The same perturbation is applied to sparse antecedent centres and consequent
alpha-cut endpoints before either method sees the rule base.  This is the main
fairness control for any robustness claim.  Hypervector-element dropout is not
used here because it has no direct analogue for the grid baseline.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from synthetic_benchmark import (ALPHA, K_NEIGHBOURS, TEMPERATURE, cnf_violation,
                                 endpoint_decoder, project_cnf, truth, triangle,
                                 vsa_encoder, cosine_matrix, weighted_neighbours)


def noisy_rules(rng: np.random.Generator, rules: np.ndarray, left: np.ndarray,
                right: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noisy_centres = np.clip(rules + rng.normal(0.0, sigma, rules.shape), 0.0, 1.0)
    noisy_left = left + rng.normal(0.0, sigma, left.shape)
    noisy_right = right + rng.normal(0.0, sigma, right.shape)
    return noisy_centres, noisy_left, noisy_right


def run_one(seed: int, sigma: float, dimension: int, queries_n: int = 600, sparse_n: int = 30) -> list[dict]:
    rng = np.random.default_rng(seed)
    lattice = np.array([(x, y) for x in np.linspace(0.0, 1.0, 9) for y in np.linspace(0.0, 1.0, 9)])
    rules = lattice[rng.choice(len(lattice), size=sparse_n, replace=False)]
    queries = rng.uniform(0.0, 1.0, size=(queries_n, 2))
    true_center, true_width = truth(queries)
    true_left = np.stack([triangle(c, w)[0] for c, w in zip(true_center, true_width)])
    true_right = np.stack([triangle(c, w)[1] for c, w in zip(true_center, true_width)])
    rule_center, rule_width = truth(rules)
    rule_left = np.stack([triangle(c, w)[0] for c, w in zip(rule_center, rule_width)])
    rule_right = np.stack([triangle(c, w)[1] for c, w in zip(rule_center, rule_width)])
    noisy_centres, noisy_left, noisy_right = noisy_rules(rng, rules, rule_left, rule_right, sigma)

    grid_errors, grid_v = [], []
    grid_projected_errors, grid_projected_v, grid_projected_displacement = [], [], []
    for q, ideal_l, ideal_r in zip(queries, true_left, true_right):
        idx, w = weighted_neighbours(np.linalg.norm(noisy_centres - q, axis=1), K_NEIGHBOURS, "euclidean")
        out_l, out_r = w @ noisy_left[idx], w @ noisy_right[idx]
        grid_errors.append(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2)))
        grid_v.append(cnf_violation(out_l, out_r))
        projected_l, projected_r = project_cnf(out_l, out_r)
        grid_projected_errors.append(np.sqrt(np.mean(np.r_[projected_l - ideal_l, projected_r - ideal_r] ** 2)))
        grid_projected_v.append(cnf_violation(projected_l, projected_r))
        grid_projected_displacement.append(np.sqrt(np.mean(np.r_[projected_l - out_l, projected_r - out_r] ** 2)))
    rows = [{"seed": seed, "sigma": sigma, "dimension": 0, "method": "grid_euclidean",
             "rmse_mean": float(np.mean(grid_errors)), "cnf_violation_rate": float(np.mean(grid_v)),
             "projection_displacement": 0.0},
            {"seed": seed, "sigma": sigma, "dimension": 0, "method": "grid_euclidean_projected",
             "rmse_mean": float(np.mean(grid_projected_errors)),
             "cnf_violation_rate": float(np.mean(grid_projected_v)),
             "projection_displacement": float(np.mean(grid_projected_displacement))}]

    local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    encode = vsa_encoder(local_rng, dimension)
    similarities = cosine_matrix(encode(queries), encode(noisy_centres))
    projection, decoder = endpoint_decoder(local_rng, dimension, len(ALPHA))
    errors, v_rates, displacements = [], [], []
    for i, (ideal_l, ideal_r) in enumerate(zip(true_left, true_right)):
        idx, w = weighted_neighbours(similarities[i], K_NEIGHBOURS, "vsa")
        h_l = w @ (noisy_left[idx] @ projection.T)
        h_r = w @ (noisy_right[idx] @ projection.T)
        provisional_l, provisional_r = decoder @ h_l, decoder @ h_r
        out_l, out_r = project_cnf(provisional_l, provisional_r)
        errors.append(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2)))
        v_rates.append(cnf_violation(out_l, out_r))
        displacements.append(np.sqrt(np.mean(np.r_[out_l - provisional_l, out_r - provisional_r] ** 2)))
    rows.append({"seed": seed, "sigma": sigma, "dimension": dimension, "method": "cp_hdfri",
                 "rmse_mean": float(np.mean(errors)), "cnf_violation_rate": float(np.mean(v_rates)),
                 "projection_displacement": float(np.mean(displacements))})
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    seeds = (19, 23, 29, 31, 37)
    sigmas = (0.0, 0.01, 0.03, 0.06)
    dimensions = (16, 32, 64)
    rows = [row for seed in seeds for sigma in sigmas for dimension in dimensions
            for row in run_one(seed, sigma, dimension)]
    (out_dir / "noise_benchmark_results.json").write_text(json.dumps({"seeds": list(seeds), "sigmas": list(sigmas),
        "dimensions": list(dimensions), "rows": rows}, indent=2), encoding="utf-8")
    with (out_dir / "noise_benchmark_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sigma"], row["dimension"], row["method"])].append(row)
    summary = []
    for (sigma, dimension, method), group in sorted(groups.items()):
        summary.append({"sigma": sigma, "dimension": dimension, "method": method,
                        "rmse_mean": float(np.mean([r["rmse_mean"] for r in group])),
                        "rmse_std": float(np.std([r["rmse_mean"] for r in group], ddof=1)),
                        "cnf_violation_rate_mean": float(np.mean([r["cnf_violation_rate"] for r in group])),
                        "projection_displacement_mean": float(np.mean([r["projection_displacement"] for r in group]))})
    with (out_dir / "noise_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    print(json.dumps({"files": [str(out_dir / "noise_benchmark_results.json"), str(out_dir / "noise_benchmark_by_seed.csv"),
        str(out_dir / "noise_benchmark_summary.csv")]}, indent=2))


if __name__ == "__main__":
    main()
