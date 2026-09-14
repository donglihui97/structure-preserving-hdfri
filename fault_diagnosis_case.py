"""Knowledge-system case study: sparse thermal/vibration/pressure diagnosis.

The rule base is generated from an explicit expert-style severity function so
that every rule has a documented provenance and a dense reference exists for
scoring unmatched observations.  It is a controlled knowledge-engineering case,
not a claim about a particular machine or plant.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from synthetic_benchmark import ALPHA, cnf_violation, endpoint_decoder, project_cnf, vsa_encoder, cosine_matrix, weighted_neighbours


def severity(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expert-style monotone severity and uncertainty width.

    Inputs are normalised temperature, vibration and pressure in [0, 1].
    Temperature and vibration increase risk; pressure contributes a smaller
    monotone term.  The width widens mildly at high severity.
    """
    raw = 0.9 * points[:, 0] + 0.75 * points[:, 1] + 0.35 * points[:, 2] - 0.9
    center = 1.0 / (1.0 + np.exp(-4.0 * raw))
    width = 0.035 + 0.025 * center
    return center, width


def consequent(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center, width = severity(points)
    left = center[:, None] - width[:, None] + ALPHA[None, :] * width[:, None]
    right = center[:, None] + width[:, None] - ALPHA[None, :] * width[:, None]
    return left, right


def run(seed: int = 211, queries_n: int = 500, sparse_n: int = 12,
        dimensions: tuple[int, ...] = (16, 32, 64)) -> dict:
    rng = np.random.default_rng(seed)
    # Linguistic levels: low/medium/high for temperature, vibration, pressure.
    grid = np.array([(a, b, c) for a in (0.15, 0.5, 0.85)
                     for b in (0.15, 0.5, 0.85) for c in (0.15, 0.5, 0.85)])
    sparse_idx = rng.choice(len(grid), size=sparse_n, replace=False)
    rules = grid[np.sort(sparse_idx)]
    queries = rng.uniform(0.0, 1.0, size=(queries_n, 3))
    true_left, true_right = consequent(queries)
    rule_left, rule_right = consequent(rules)

    grid_errors, grid_times = [], []
    for q, ideal_l, ideal_r in zip(queries, true_left, true_right):
        idx, w = weighted_neighbours(np.linalg.norm(rules - q, axis=1), 4, "euclidean")
        out_l, out_r = w @ rule_left[idx], w @ rule_right[idx]
        grid_errors.append(float(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2))))
        grid_times.append(float(np.linalg.norm(rules[idx] - q, axis=1).mean()))
    rows = [{"dimension": 0, "method": "grid_euclidean", "rmse_mean": float(np.mean(grid_errors)),
             "rmse_p95": float(np.quantile(grid_errors, 0.95)), "cnf_violation_rate": 0.0,
             "projection_displacement": 0.0, "mean_rule_distance": float(np.mean(grid_times))}]

    for dimension in dimensions:
        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        encode = vsa_encoder(local_rng, dimension, variables=3)
        rule_h, query_h = encode(rules), encode(queries)
        similarities = cosine_matrix(query_h, rule_h)
        projection, decoder = endpoint_decoder(local_rng, dimension, len(ALPHA))
        errors, displacements, violations = [], [], []
        for i, (ideal_l, ideal_r) in enumerate(zip(true_left, true_right)):
            idx, w = weighted_neighbours(similarities[i], 4, "vsa")
            h_l = w @ (rule_left[idx] @ projection.T)
            h_r = w @ (rule_right[idx] @ projection.T)
            provisional_l, provisional_r = decoder @ h_l, decoder @ h_r
            out_l, out_r = project_cnf(provisional_l, provisional_r)
            errors.append(float(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2))))
            displacements.append(float(np.sqrt(np.mean(np.r_[out_l - provisional_l, out_r - provisional_r] ** 2))))
            violations.append(cnf_violation(out_l, out_r))
        rows.append({"dimension": dimension, "method": "cp_hdfri", "rmse_mean": float(np.mean(errors)),
                     "rmse_p95": float(np.quantile(errors, 0.95)),
                     "cnf_violation_rate": float(np.mean(violations)),
                     "projection_displacement": float(np.mean(displacements)),
                     "mean_rule_distance": None})
    return {"seed": seed, "queries": queries_n, "sparse_rules": sparse_n,
            "rule_levels": [0.15, 0.5, 0.85], "variables": ["temperature", "vibration", "pressure"],
            "rule_provenance": "3^3 linguistic grid generated from severity()",
            "results": rows}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    seeds = (211, 223, 227, 229, 233, 239, 241, 251, 257, 263)
    runs = [run(seed=s) for s in seeds]
    rows = [{"seed": s, **row} for s, payload in zip(seeds, runs) for row in payload["results"]]
    (out_dir / "fault_diagnosis_case_results.json").write_text(json.dumps({"seeds": list(seeds), "runs": runs}, indent=2), encoding="utf-8")
    with (out_dir / "fault_diagnosis_case_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"files": [str(out_dir / "fault_diagnosis_case_results.json"),
        str(out_dir / "fault_diagnosis_case_summary.csv")]}, indent=2))


if __name__ == "__main__":
    main()
