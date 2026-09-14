"""Multidimensional benchmark with a standard triangular T-FRI comparator.

The implementation follows the scale-and-move construction described in the
T-FRI literature: representative-value distances select two rules, attribute
weights form an intermediate rule, and triangular scale/move factors transform
the intermediate consequent.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA as HDT_ALPHA, interval_encoding, transform_operator
from preexperiments import project_cnf, violations
from synthetic_benchmark import cosine_matrix, truth, vsa_encoder, weighted_neighbours


ALPHA = HDT_ALPHA


SEEDS = (19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103)
QUERIES = 400
SPARSE_RULES = 30
K = 2
DIMENSIONS = (1024, 4096, 8192)
LENGTH_SCALE = 0.20


def triangle(center: float, width: float) -> np.ndarray:
    return np.array([center - width, center, center + width], dtype=float)


def triangle_endpoints(center: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return center[:, None] - width[:, None] + ALPHA[None, :] * width[:, None], center[:, None] + width[:, None] - ALPHA[None, :] * width[:, None]


def representative(tri: np.ndarray) -> float:
    return float(np.mean(tri))


def scale_triangle(tri: np.ndarray, scale: float) -> np.ndarray:
    a1, a2, a3 = tri
    return np.array([
        ((1 + 2 * scale) * a1 + (1 - scale) * a2 + (1 - scale) * a3) / 3,
        ((1 - scale) * a1 + (1 + 2 * scale) * a2 + (1 - scale) * a3) / 3,
        ((1 - scale) * a1 + (1 - scale) * a2 + (1 + 2 * scale) * a3) / 3,
    ])


def move_triangle(tri: np.ndarray, move: float) -> np.ndarray:
    h1, h2, h3 = tri
    if move >= 0:
        gamma = (h2 - h1) / 3.0
    else:
        gamma = (h3 - h2) / 3.0
    return np.array([h1 + move * gamma, h2 - 2 * move * gamma, h3 + move * gamma])


def tfri_interpolate(rule_ant: np.ndarray, rule_cons: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Return a triangular consequent for a 2-D triangular T-FRI query."""
    distances = np.sqrt(np.mean((rule_ant.mean(axis=2) - observation.mean(axis=1)) ** 2, axis=1))
    idx = np.argsort(distances)[:K]
    selected = rule_ant[idx]
    # Eq. (6): inverse distance weights, normalized independently per attribute.
    attr_weights = 1.0 / (1.0 + np.abs(selected.mean(axis=2) - observation.mean(axis=1)[None, :]))
    attr_weights /= np.sum(attr_weights, axis=0, keepdims=True)
    intermediate_ant = np.einsum("ij,ijk->jk", attr_weights, selected)
    consequent_weights = np.mean(attr_weights, axis=1)
    intermediate_cons = np.einsum("i,ij->j", consequent_weights, rule_cons[idx])

    # Translate the weighted intermediate antecedents so their representative
    # values coincide with the observation. The consequent receives the mean
    # signed translation, as in the multidimensional T-FRI construction.
    shifts = observation.mean(axis=1) - intermediate_ant.mean(axis=1)
    intermediate_ant = intermediate_ant + shifts[:, None]
    intermediate_cons = intermediate_cons + float(np.mean(shifts))

    scales = []
    moves = []
    for j in range(rule_ant.shape[1]):
        target = observation[j]
        base = intermediate_ant[j]
        support = base[2] - base[0]
        scale = (target[2] - target[0]) / max(support, 1e-12)
        scaled = scale_triangle(base, scale)
        if target[0] >= scaled[0]:
            move = 3.0 * (target[0] - scaled[0]) / max(scaled[1] - scaled[0], 1e-12)
        else:
            move = 3.0 * (target[0] - scaled[0]) / max(scaled[2] - scaled[1], 1e-12)
        scales.append(scale)
        moves.append(move)
    output = move_triangle(scale_triangle(intermediate_cons, float(np.mean(scales))), float(np.mean(moves)))
    return output


def endpoint_error(tri: np.ndarray, center: float, width: float) -> float:
    left, right = triangle_endpoints(np.array([center]), np.array([width]))
    pred_left = tri[0] + (tri[1] - tri[0]) * ALPHA
    pred_right = tri[2] - (tri[2] - tri[1]) * ALPHA
    return float(np.sqrt(np.mean(np.r_[pred_left - left[0], pred_right - right[0]] ** 2)))


def cnf_valid(tri: np.ndarray) -> bool:
    left, right = tri[0] + (tri[1] - tri[0]) * ALPHA, tri[2] - (tri[2] - tri[1]) * ALPHA
    return not any(violations(left, right).values())


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    lattice = np.array([(x, y) for x in np.linspace(0.0, 1.0, 9) for y in np.linspace(0.0, 1.0, 9)])
    rules = lattice[rng.choice(len(lattice), size=SPARSE_RULES, replace=False)]
    queries = rng.uniform(0.0, 1.0, size=(QUERIES, 2))
    rule_center, rule_width = truth(rules)
    query_center, query_width = truth(queries)
    rule_cons = np.stack([triangle(c, w) for c, w in zip(rule_center, rule_width)])
    query_cons = np.stack([triangle(c, w) for c, w in zip(query_center, query_width)])
    # Symmetric antecedent triangles with mild width variation make scale/move
    # non-degenerate while preserving the same sparse rule geometry.
    rule_ant = np.stack([[triangle(float(rules[i, j]), 0.065 + 0.015 * float(rules[i, j])) for j in range(2)] for i in range(len(rules))])
    query_ant = np.stack([[triangle(float(queries[i, j]), 0.055 + 0.01 * float(queries[i, j])) for j in range(2)] for i in range(len(queries))])

    grid_errors, tfri_errors = [], []
    grid_valid, tfri_valid = [], []
    grid_triangles = []
    for i, query in enumerate(queries):
        d = np.linalg.norm(rules - query, axis=1)
        idx, weights = weighted_neighbours(d, K, "euclidean")
        grid_tri = weights @ rule_cons[idx]
        grid_triangles.append(grid_tri)
        grid_errors.append(endpoint_error(grid_tri, query_center[i], query_width[i]))
        grid_valid.append(cnf_valid(grid_tri))
        tfri_tri = tfri_interpolate(rule_ant, rule_cons, query_ant[i])
        tfri_errors.append(endpoint_error(tfri_tri, query_center[i], query_width[i]))
        tfri_valid.append(cnf_valid(tfri_tri))
    rows = [
        {"seed": seed, "dimension": 0, "method": "grid_euclidean_2nn", "rmse_mean": float(np.mean(grid_errors)), "rmse_p95": float(np.quantile(grid_errors, .95)), "cnf_violation_rate": float(1 - np.mean(grid_valid))},
        {"seed": seed, "dimension": 0, "method": "tfri_triangular_2nn", "rmse_mean": float(np.mean(tfri_errors)), "rmse_p95": float(np.quantile(tfri_errors, .95)), "cnf_violation_rate": float(1 - np.mean(tfri_valid))},
    ]
    for dimension in DIMENSIONS:
        local_rng = np.random.default_rng(seed * 10007 + dimension)
        encoding, _ = interval_encoding(local_rng, dimension, LENGTH_SCALE)
        operator = transform_operator(encoding)
        encode = vsa_encoder(np.random.default_rng(seed * 20011 + dimension), dimension)
        similarities = cosine_matrix(encode(queries), encode(rules))
        cp_errors, cp_valid = [], []
        for i in range(QUERIES):
            idx, weights = weighted_neighbours(similarities[i], K, "vsa")
            l = weights @ np.stack([rule_cons[k][0] + (rule_cons[k][1] - rule_cons[k][0]) * ALPHA for k in idx])
            r = weights @ np.stack([rule_cons[k][2] - (rule_cons[k][2] - rule_cons[k][1]) * ALPHA for k in idx])
            raw_l, raw_r = operator @ l, operator @ r
            out_l, out_r = project_cnf(raw_l, raw_r)
            true_l, true_r = triangle_endpoints(np.array([query_center[i]]), np.array([query_width[i]]))
            cp_errors.append(float(np.sqrt(np.mean(np.r_[out_l - true_l[0], out_r - true_r[0]] ** 2))))
            cp_valid.append(not any(violations(out_l, out_r).values()))
        rows.append({"seed": seed, "dimension": dimension, "method": "cp_hdfri_vsa_2nn", "rmse_mean": float(np.mean(cp_errors)), "rmse_p95": float(np.quantile(cp_errors, .95)), "cnf_violation_rate": float(1 - np.mean(cp_valid))})
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dimension"])].append(row)
    return [{"method": method, "dimension": dimension, "seeds": len(subset), "rmse_mean": float(np.mean([r["rmse_mean"] for r in subset])), "rmse_seed_std": float(np.std([r["rmse_mean"] for r in subset], ddof=1)), "rmse_p95_mean": float(np.mean([r["rmse_p95"] for r in subset])), "cnf_violation_rate": float(np.mean([r["cnf_violation_rate"] for r in subset]))} for (method, dimension), subset in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0]))]


def paired_statistics(rows: list[dict]) -> list[dict]:
    """Return paired seed-level uncertainty for CP-HDFRI against both controls."""
    lookup = defaultdict(dict)
    for row in rows:
        lookup[(row["method"], row["dimension"])][row["seed"]] = row["rmse_mean"]
    comparisons = []
    for dimension in DIMENSIONS:
        for baseline in ("grid_euclidean_2nn", "tfri_triangular_2nn"):
            cp = lookup[("cp_hdfri_vsa_2nn", dimension)]
            control = lookup[(baseline, 0)]
            seeds = sorted(set(cp) & set(control))
            differences = np.array([cp[seed] - control[seed] for seed in seeds])
            rng = np.random.default_rng(20260911 + dimension + (0 if baseline.startswith("grid") else 1))
            bootstrap_indices = rng.integers(0, len(seeds), size=(50000, len(seeds)))
            bootstrap_means = differences[bootstrap_indices].mean(axis=1)
            positive = int(np.sum(differences > 0))
            negative = int(np.sum(differences < 0))
            nonzero = positive + negative
            tail = min(positive, negative)
            sign_p = min(1.0, 2.0 * sum(comb(nonzero, i) for i in range(tail + 1)) / (2 ** nonzero)) if nonzero else 1.0
            comparisons.append({
                "cp_dimension": dimension,
                "baseline": baseline,
                "paired_seeds": len(seeds),
                "mean_rmse_difference_cp_minus_baseline": float(np.mean(differences)),
                "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                "cp_better_fraction": float(np.mean(differences < 0)),
                "two_sided_sign_test_p": float(sign_p),
                "paired_standardized_difference": float(np.mean(differences) / np.std(differences, ddof=1)),
            })
    return comparisons


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = [row for seed in SEEDS for row in run_seed(seed)]
    summary = aggregate(rows)
    paired = paired_statistics(rows)
    payload = {"seeds": list(SEEDS), "queries_per_seed": QUERIES, "sparse_rules": SPARSE_RULES, "nearest_rules": K, "dimensions": list(DIMENSIONS), "rows": rows, "summary": summary, "paired_statistics": paired}
    (root / "tfri_multidimensional_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, data in (("tfri_multidimensional_by_seed.csv", rows), ("tfri_multidimensional_summary.csv", summary)):
        with (root / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    with (root / "tfri_multidimensional_paired_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=paired[0].keys()); w.writeheader(); w.writerows(paired)
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for method, marker in (("cp_hdfri_vsa_2nn", "s"),):
        series = [r for r in summary if r["method"] == method and r["dimension"] > 0]
        ax.plot([r["dimension"] for r in series], [r["rmse_mean"] for r in series], marker + "-", label="CP-HDFRI (VSA 2-NN)")
    grid = next(r for r in summary if r["method"] == "grid_euclidean_2nn")
    tfri = next(r for r in summary if r["method"] == "tfri_triangular_2nn")
    ax.axhline(grid["rmse_mean"], color="#64748b", linestyle="--", label="Direct grid 2-NN")
    ax.axhline(tfri["rmse_mean"], color="#b45309", linestyle=":", label="T-FRI 2-NN")
    ax.set_xscale("log", base=2); ax.set_xlabel("HDT dimension D"); ax.set_ylabel("Endpoint RMSE"); ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(root / "tfri_multidimensional_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "paired_statistics": paired}, indent=2))


if __name__ == "__main__":
    main()
