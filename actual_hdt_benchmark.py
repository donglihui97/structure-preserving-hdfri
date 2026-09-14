"""Paper-faithful HDT endpoint reconstruction and bundling benchmark.

Implements the interval encoding in Dewulf, Stock and De Baets
(arXiv:2310.16065, Example 1): independent Rademacher anchor vectors with
uniform random switching locations, the triangular covariance kernel, and ten
successive normalization iterations.  The forward transform is a trapezoidal
quadrature of f(alpha) Delta-phi(alpha); the inverse is the dimension-scaled
inner product with Delta-phi(alpha).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preexperiments import project_cnf, violations


ALPHA = np.linspace(0.0, 1.0, 101)
DIMENSIONS = (256, 512, 1024, 2048, 4096, 8192)
LENGTH_SCALES = (0.05, 0.10, 0.20, 0.25, 0.50)
SEEDS = (503, 509, 521, 523, 541)
TRIALS = 200


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    step = float(x[1] - x[0])
    weights = np.full(len(x), step)
    weights[[0, -1]] *= 0.5
    return weights


def normalization(alpha: np.ndarray, length_scale: float, iterations: int = 10) -> np.ndarray:
    weights = trapezoid_weights(alpha)
    kernel = np.maximum(0.0, 1.0 - np.abs(alpha[:, None] - alpha[None, :]) / length_scale)
    norm = np.sqrt(np.maximum(kernel @ weights, 1e-12))
    for _ in range(iterations):
        recovered_one = (kernel / np.maximum(norm[:, None] * norm[None, :], 1e-12)) @ weights
        norm = norm * np.sqrt(np.maximum(recovered_one, 1e-12))
    return norm


def interval_encoding(rng: np.random.Generator, dimension: int, length_scale: float) -> tuple[np.ndarray, np.ndarray]:
    intervals = int(round(1.0 / length_scale))
    # A shared random phase per component produces the stationary triangular
    # covariance: two points share a length-scale bin with probability
    # max(0, 1 - |x-x'|/length_scale).
    offsets = rng.random(dimension) * length_scale
    bin_indices = np.floor((ALPHA[:, None] - offsets[None, :]) / length_scale).astype(int)
    bin_values = rng.choice((-1.0, 1.0), size=(intervals + 2, dimension))
    phi = bin_values[bin_indices + 1, np.arange(dimension)[None, :]]
    norm = normalization(ALPHA, length_scale)
    return phi / norm[:, None], norm


def transform_operator(encoding: np.ndarray) -> np.ndarray:
    weights = trapezoid_weights(ALPHA)
    dimension = encoding.shape[1]
    return (encoding @ encoding.T / dimension) * weights[None, :]


def triangle(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = rng.uniform(-2.0, 2.0)
    b = a + rng.uniform(0.4, 2.5)
    c = b + rng.uniform(0.4, 2.5)
    return a + ALPHA * (b - a), c - ALPHA * (c - b)


def run_seed(seed: int) -> list[dict]:
    rows = []
    for length_scale in LENGTH_SCALES:
        for dimension in DIMENSIONS:
            rng = np.random.default_rng(seed + dimension + int(length_scale * 1000))
            encoding, _ = interval_encoding(rng, dimension, length_scale)
            operator = transform_operator(encoding)
            weights = trapezoid_weights(ALPHA)
            empirical_mass = (encoding @ encoding.T / dimension) @ weights
            rmse, projected_rmse, raw_invalid, projected_invalid, displacement = [], [], [], [], []
            for _ in range(TRIALS):
                l1, r1 = triangle(rng); l2, r2 = triangle(rng)
                blend = rng.dirichlet((1.0, 1.0))
                ideal_l = blend[0] * l1 + blend[1] * l2
                ideal_r = blend[0] * r1 + blend[1] * r2
                out_l = operator @ ideal_l
                out_r = operator @ ideal_r
                proj_l, proj_r = project_cnf(out_l, out_r)
                rmse.append(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2)))
                projected_rmse.append(np.sqrt(np.mean(np.r_[proj_l - ideal_l, proj_r - ideal_r] ** 2)))
                raw_invalid.append(any(violations(out_l, out_r).values()))
                projected_invalid.append(any(violations(proj_l, proj_r).values()))
                displacement.append(np.sqrt(np.mean(np.r_[proj_l - out_l, proj_r - out_r] ** 2)))
            # Requested deterministic endpoint check A=(2,5,9).
            fixed_l = 2.0 + 3.0 * ALPHA
            fixed_r = 9.0 - 4.0 * ALPHA
            fixed_out_l = operator @ fixed_l
            fixed_out_r = operator @ fixed_r
            rows.append({
                "seed": seed, "dimension": dimension, "length_scale": length_scale,
                "rmse_mean": float(np.mean(rmse)), "rmse_seed_trials_std": float(np.std(rmse, ddof=1)),
                "projected_rmse_mean": float(np.mean(projected_rmse)),
                "raw_cnf_violation_rate": float(np.mean(raw_invalid)),
                "projected_cnf_violation_rate": float(np.mean(projected_invalid)),
                "repair_displacement_mean": float(np.mean(displacement)),
                "normalization_max_abs_error": float(np.max(np.abs(empirical_mass - 1.0))),
                "fixed_triangle_rmse": float(np.sqrt(np.mean(np.r_[fixed_out_l - fixed_l, fixed_out_r - fixed_r] ** 2))),
            })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for length_scale in LENGTH_SCALES:
        for dimension in DIMENSIONS:
            subset = [r for r in rows if r["length_scale"] == length_scale and r["dimension"] == dimension]
            out.append({
                "dimension": dimension, "length_scale": length_scale,
                "rmse_mean": float(np.mean([r["rmse_mean"] for r in subset])),
                "rmse_seed_std": float(np.std([r["rmse_mean"] for r in subset], ddof=1)),
                "projected_rmse_mean": float(np.mean([r["projected_rmse_mean"] for r in subset])),
                "raw_cnf_violation_rate": float(np.mean([r["raw_cnf_violation_rate"] for r in subset])),
                "projected_cnf_violation_rate": float(np.mean([r["projected_cnf_violation_rate"] for r in subset])),
                "repair_displacement_mean": float(np.mean([r["repair_displacement_mean"] for r in subset])),
                "normalization_max_abs_error": float(np.mean([r["normalization_max_abs_error"] for r in subset])),
                "fixed_triangle_rmse": float(np.mean([r["fixed_triangle_rmse"] for r in subset])),
            })
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = [row for seed in SEEDS for row in run_seed(seed)]
    summary = aggregate(rows)
    payload = {
        "source": "Dewulf, Stock and De Baets, arXiv:2310.16065, Definitions 1/2/4 and Example 1",
        "seeds": list(SEEDS), "trials_per_setting_per_seed": TRIALS,
        "alpha_points": len(ALPHA), "normalization_iterations": 10,
        "rows": rows, "summary": summary,
    }
    (out_dir / "actual_hdt_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, data in (("actual_hdt_by_seed.csv", rows), ("actual_hdt_summary.csv", summary)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for length_scale in LENGTH_SCALES:
        subset = [r for r in summary if r["length_scale"] == length_scale]
        axes[0].errorbar([r["dimension"] for r in subset], [r["rmse_mean"] for r in subset],
                         yerr=[r["rmse_seed_std"] for r in subset], marker="o", capsize=3, label=f"lambda={length_scale}")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log"); axes[0].set_xlabel("HDT dimension D"); axes[0].set_ylabel("Endpoint RMSE")
    saturated_dims = sorted({r["dimension"] for r in summary if r["dimension"] <= 4096})
    exceptional = next(r for r in summary if r["length_scale"] == 0.50 and r["dimension"] == 8192)
    axes[1].plot(saturated_dims, np.ones(len(saturated_dims)), "o-", color="#4b5563",
                 label=r"All $\lambda$, $D\leq4096$")
    axes[1].plot([8192], [exceptional["raw_cnf_violation_rate"]], marker="D", markersize=7,
                 linestyle="none", color="#c2410c", label=r"$\lambda=0.50$, $D=8192$")
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("HDT dimension D"); axes[1].set_ylabel("Raw CNF violation rate")
    axes[1].set_xticks(saturated_dims + [8192], [str(d) for d in saturated_dims + [8192]])
    axes[1].set_ylim(0.988, 1.002); axes[1].set_yticks([0.99, 0.995, 1.00])
    for ax in axes: ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    for i, ax in enumerate(axes): ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_dir / "actual_hdt_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "files": [str(out_dir / "actual_hdt_results.json"), str(out_dir / "actual_hdt_summary.csv"), str(out_dir / "actual_hdt_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
