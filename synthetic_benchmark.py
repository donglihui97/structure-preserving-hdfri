"""Synthetic sparse-rule benchmark for CP-HDFRI.

The benchmark keeps the ground-truth consequent generator fixed and compares:
  * direct alpha-cut interpolation with Euclidean retrieval (grid control),
  * VSA retrieval followed by direct alpha-cut interpolation, and
  * VSA retrieval followed by hyperdimensional endpoint bundling + CNF projection.

All methods use the same sparse rule base, neighbour count, query set, and
consequent weights.  The experiment is intentionally small and deterministic;
it is a pilot for the full KH/MACI/T-FRI benchmark suite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


ALPHA = np.linspace(0.0, 1.0, 11)
K_NEIGHBOURS = 4
TEMPERATURE = 10.0


def pava(y: np.ndarray, increasing: bool) -> np.ndarray:
    x = y.copy() if increasing else -y.copy()
    vals: list[float] = []
    weights: list[int] = []
    for value in x:
        vals.append(float(value)); weights.append(1)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            total = weights[-2] + weights[-1]
            vals[-2] = (weights[-2] * vals[-2] + weights[-1] * vals[-1]) / total
            weights[-2] = total
            vals.pop(); weights.pop()
    out = np.empty_like(x); cursor = 0
    for value, weight in zip(vals, weights):
        out[cursor:cursor + weight] = value; cursor += weight
    return out if increasing else -out


def project_cnf(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # First enforce the two monotonicity constraints.  A uniform shift then
    # removes any global crossing without changing either monotonic order.
    left_p = pava(left, True)
    right_p = pava(right, False)
    gap = max(0.0, float(np.max(left_p) - np.min(right_p)))
    if gap:
        left_p = left_p - gap / 2.0
        right_p = right_p + gap / 2.0
    return left_p, right_p


def triangle(center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    left = center - width + ALPHA * width
    right = center + width - ALPHA * width
    return left, right


def truth(x: np.ndarray) -> tuple[np.ndarray, float]:
    # Smooth nonlinear map with a positive output width.
    center = 0.5 + 0.24 * (x[:, 0] - 0.5) + 0.16 * np.sin(2.0 * np.pi * x[:, 1])
    width = 0.055 + 0.02 * (0.5 + x[:, 0])
    return center, width


def vsa_encoder(rng: np.random.Generator, dimension: int, variables: int = 2):
    # Random Fourier value vectors bound to independent role vectors.
    frequencies = rng.normal(0.0, 5.0, size=(variables, dimension))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(variables, dimension))
    roles = rng.choice([-1.0, 1.0], size=(variables, dimension))

    def encode(points: np.ndarray) -> np.ndarray:
        pieces = []
        for i in range(variables):
            value = np.cos(points[:, i:i + 1] * frequencies[i:i + 1] + phases[i:i + 1])
            pieces.append(value * roles[i:i + 1])
        return np.sum(np.stack(pieces, axis=0), axis=0)

    return encode


def cosine_matrix(query_h: np.ndarray, rule_h: np.ndarray) -> np.ndarray:
    q = query_h / np.maximum(np.linalg.norm(query_h, axis=1, keepdims=True), 1e-12)
    r = rule_h / np.maximum(np.linalg.norm(rule_h, axis=1, keepdims=True), 1e-12)
    return q @ r.T


def endpoint_decoder(rng: np.random.Generator, dimension: int, m: int, ridge: float = 1e-3):
    projection = rng.normal(size=(dimension, m)) / np.sqrt(dimension)
    decoder = projection.T @ np.linalg.inv(projection @ projection.T + ridge * np.eye(dimension))
    return projection, decoder


def weighted_neighbours(distances: np.ndarray, k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "euclidean":
        idx = np.argpartition(distances, k - 1)[:k]
        scores = np.exp(-TEMPERATURE * distances[idx])
    else:
        idx = np.argpartition(-distances, k - 1)[:k]
        scores = np.exp(TEMPERATURE * distances[idx])
    return idx, scores / np.sum(scores)


def cnf_violation(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.any(np.diff(left) < -1e-8) or np.any(np.diff(right) > 1e-8) or np.any(left > right + 1e-8))


def run(seed: int = 19, queries_n: int = 600, sparse_n: int = 30,
        dimensions: tuple[int, ...] = (8, 16, 32, 64)) -> dict:
    rng = np.random.default_rng(seed)
    lattice = np.array([(x, y) for x in np.linspace(0.0, 1.0, 9) for y in np.linspace(0.0, 1.0, 9)])
    sparse_idx = rng.choice(len(lattice), size=sparse_n, replace=False)
    rules = lattice[sparse_idx]
    queries = rng.uniform(0.0, 1.0, size=(queries_n, 2))
    true_center, true_width = truth(queries)
    true_left = np.stack([triangle(c, w)[0] for c, w in zip(true_center, true_width)])
    true_right = np.stack([triangle(c, w)[1] for c, w in zip(true_center, true_width)])
    rule_center, rule_width = truth(rules)
    rule_left = np.stack([triangle(c, w)[0] for c, w in zip(rule_center, rule_width)])
    rule_right = np.stack([triangle(c, w)[1] for c, w in zip(rule_center, rule_width)])

    direct_rows = []
    euclidean_rows = []
    for q, ideal_l, ideal_r in zip(queries, true_left, true_right):
        idx, w = weighted_neighbours(np.linalg.norm(rules - q, axis=1), K_NEIGHBOURS, "euclidean")
        out_l = w @ rule_left[idx]; out_r = w @ rule_right[idx]
        direct_rows.append((out_l, out_r, ideal_l, ideal_r))

    results = []
    for method, rows in [("grid_euclidean", direct_rows)]:
        errors = [np.sqrt(np.mean(np.r_[l - il, r - ir] ** 2)) for l, r, il, ir in rows]
        results.append({"dimension": 0, "method": method, "rmse_mean": float(np.mean(errors)),
                        "rmse_p95": float(np.quantile(errors, 0.95)),
                        "cnf_violation_rate": float(np.mean([cnf_violation(l, r) for l, r, _, _ in rows])),
                        "projection_displacement": 0.0})

    for dimension in dimensions:
        # Independent encoders make every dimension a separately reproducible
        # model; the same encoder is used for rules and queries.
        local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        encode = vsa_encoder(local_rng, dimension)
        rule_h, query_h = encode(rules), encode(queries)
        similarities = cosine_matrix(query_h, rule_h)
        projection, decoder = endpoint_decoder(local_rng, dimension, len(ALPHA))
        vsa_rows, cp_rows = [], []
        for i, (ideal_l, ideal_r) in enumerate(zip(true_left, true_right)):
            idx, w = weighted_neighbours(similarities[i], K_NEIGHBOURS, "vsa")
            # VSA-only ablation: same retrieved neighbours, direct endpoint average.
            direct_l, direct_r = w @ rule_left[idx], w @ rule_right[idx]
            vsa_rows.append((direct_l, direct_r, ideal_l, ideal_r))
            h_l = w @ (rule_left[idx] @ projection.T)
            h_r = w @ (rule_right[idx] @ projection.T)
            provisional_l, provisional_r = decoder @ h_l, decoder @ h_r
            out_l, out_r = project_cnf(provisional_l, provisional_r)
            cp_rows.append((out_l, out_r, ideal_l, ideal_r, provisional_l, provisional_r))

        vsa_errors = [np.sqrt(np.mean(np.r_[l - il, r - ir] ** 2)) for l, r, il, ir in vsa_rows]
        cp_errors = [np.sqrt(np.mean(np.r_[l - il, r - ir] ** 2)) for l, r, il, ir, _, _ in cp_rows]
        displacements = [np.sqrt(np.mean(np.r_[l - pl, r - pr] ** 2)) for l, r, _, _, pl, pr in cp_rows]
        results.extend([
            {"dimension": dimension, "method": "vsa_retrieval_grid_consequent", "rmse_mean": float(np.mean(vsa_errors)),
             "rmse_p95": float(np.quantile(vsa_errors, 0.95)),
             "cnf_violation_rate": float(np.mean([cnf_violation(l, r) for l, r, _, _ in vsa_rows])),
             "projection_displacement": 0.0},
            {"dimension": dimension, "method": "cp_hdfri", "rmse_mean": float(np.mean(cp_errors)),
             "rmse_p95": float(np.quantile(cp_errors, 0.95)),
             "cnf_violation_rate": float(np.mean([cnf_violation(l, r) for l, r, _, _, _, _ in cp_rows])),
             "projection_displacement": float(np.mean(displacements))},
        ])
    return {"seed": seed, "queries": queries_n, "sparse_rules": sparse_n, "alpha_points": len(ALPHA),
            "k_neighbours": K_NEIGHBOURS, "temperature": TEMPERATURE, "results": results}


def aggregate(runs: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run_payload in runs:
        for row in run_payload["results"]:
            buckets[(row["method"], row["dimension"])].append(row)
    out = []
    for (method, dimension), rows in sorted(buckets.items(), key=lambda item: (item[0][1], item[0][0])):
        out.append({
            "method": method,
            "dimension": dimension,
            "seeds": len(rows),
            "rmse_mean": float(np.mean([r["rmse_mean"] for r in rows])),
            "rmse_std": float(np.std([r["rmse_mean"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "rmse_p95_mean": float(np.mean([r["rmse_p95"] for r in rows])),
            "cnf_violation_rate_mean": float(np.mean([r["cnf_violation_rate"] for r in rows])),
            "projection_displacement_mean": float(np.mean([r["projection_displacement"] for r in rows])),
        })
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    seeds = (19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
    runs = [run(seed=seed) for seed in seeds]
    payload = {"seeds": list(seeds), "runs": runs, "aggregate": aggregate(runs)}
    (out_dir / "synthetic_benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    per_seed_rows = [{"seed": seed, **row} for seed, run_payload in zip(seeds, runs) for row in run_payload["results"]]
    with (out_dir / "synthetic_benchmark_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_rows[0].keys())
        writer.writeheader(); writer.writerows(per_seed_rows)
    with (out_dir / "synthetic_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload["aggregate"][0].keys())
        writer.writeheader(); writer.writerows(payload["aggregate"])

    dims = sorted({r["dimension"] for r in payload["aggregate"] if r["dimension"]})
    fig, ax = plt.subplots(figsize=(4.2, 3.25))
    display = {
        "vsa_retrieval_grid_consequent": "VSA + grid consequent",
        "cp_hdfri": "CP-HDFRI",
    }
    plotted = {}
    for method, marker, color in [("vsa_retrieval_grid_consequent", "o", "#64748b"),
                                  ("cp_hdfri", "s", "#2563a6")]:
        rows = [r for r in payload["aggregate"] if r["method"] == method]
        plotted[method] = ax.errorbar([r["dimension"] for r in rows], [r["rmse_mean"] for r in rows],
                                      yerr=[r["rmse_std"] for r in rows], marker=marker, linestyle="-",
                                      color=color, capsize=3, label=display[method])
    grid_row = next(r for r in payload["aggregate"] if r["method"] == "grid_euclidean")
    direct_line = ax.axhline(grid_row["rmse_mean"], color="#111827", linestyle="--", label="Direct Euclidean")
    ax.set_xlabel(r"Hypervector dimension $D$"); ax.set_ylabel("Endpoint RMSE")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(dims, [str(d) for d in dims]); ax.grid(alpha=0.25, which="both")
    ax.legend([plotted["vsa_retrieval_grid_consequent"], plotted["cp_hdfri"], direct_line],
              ["VSA + grid consequent", "CP-HDFRI", "Direct Euclidean"],
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
              fontsize=7, handlelength=1.8, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(out_dir / "synthetic_benchmark_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"aggregate": payload["aggregate"], "files": [str(out_dir / "synthetic_benchmark_results.json"),
        str(out_dir / "synthetic_benchmark_by_seed.csv"), str(out_dir / "synthetic_benchmark_summary.csv"),
        str(out_dir / "synthetic_benchmark_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
