"""HDC/HDT value benchmark under coordinate corruption.

This experiment isolates the carrier contribution.  The direct grid and HDT
store the same smooth CNF endpoint functions, receive paired coordinate
erasures, and use the same final CNF projection.  The redesigned HDT decoder
solves the observed-coordinate inverse problem rather than rescaling a fixed
inner product; this is the operation that makes distributed storage useful
when individual coordinates are damaged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA, interval_encoding, trapezoid_weights
from preexperiments import project_cnf, violations


DIMENSIONS = (256, 1024, 4096)
ERASURE_RATES = (0.0, 0.30, 0.50, 0.70)
SEEDS = (601, 607, 613, 617, 619, 631, 641, 643, 647, 653)
TRIALS = 40
GRID_BUDGET = 64
LENGTH_SCALE = 0.20
RIDGE = 1e-8


def endpoint_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate smooth convex-normal endpoints with nonlinear shoulders."""
    a = rng.uniform(-2.0, 2.0)
    b = a + rng.uniform(0.4, 2.5)
    c = b + rng.uniform(0.4, 2.5)
    ql, qr = rng.uniform(0.55, 2.4, size=2)
    left = a + (b - a) * ALPHA**ql
    right = c - (c - b) * ALPHA**qr
    return left, right


def grid_decode(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    grid_alpha = np.linspace(0.0, 1.0, GRID_BUDGET)
    return np.interp(ALPHA, grid_alpha[keep], values[keep])


def project_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return project_cnf(left, right)


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    weights = trapezoid_weights(ALPHA)
    rows: list[dict] = []
    for dimension in DIMENSIONS:
        encoding, _ = interval_encoding(rng, dimension, LENGTH_SCALE)
        forward = encoding.T * weights[None, :]
        for rate in ERASURE_RATES:
            errors = {name: [] for name in ("grid", "hdt_fixed", "hdt_mask_aware")}
            valid = {name: [] for name in errors}
            displacement = {name: [] for name in errors}
            for trial in range(TRIALS):
                local = np.random.default_rng(seed * 100000 + dimension * 10 + trial)
                ideal_l, ideal_r = endpoint_pair(local)
                # Paired corruption: each carrier sees the same fraction, but
                # its native coordinate systems remain independent.
                h_keep = local.random(dimension) >= rate
                if not np.any(h_keep):
                    h_keep[local.integers(dimension)] = True
                g_keep = local.random(GRID_BUDGET) >= rate
                if not np.any(g_keep):
                    g_keep[local.integers(GRID_BUDGET)] = True

                grid_alpha = np.linspace(0.0, 1.0, GRID_BUDGET)
                grid_l = np.interp(grid_alpha, ALPHA, ideal_l)
                grid_r = np.interp(grid_alpha, ALPHA, ideal_r)
                raw = {
                    "grid": (grid_decode(grid_l, g_keep), grid_decode(grid_r, g_keep)),
                }
                h_l = forward @ ideal_l
                h_r = forward @ ideal_r
                # Historical HDT inverse: fixed full-coordinate inner product
                # with a surviving-coordinate rescale.
                retained = max(float(np.mean(h_keep)), 1e-12)
                raw["hdt_fixed"] = (
                    encoding[:, h_keep] @ (h_l[h_keep] / retained) / dimension,
                    encoding[:, h_keep] @ (h_r[h_keep] / retained) / dimension,
                )
                # Redesigned inverse: solve the restricted forward operator.
                observed = forward[h_keep]
                lhs = observed.T @ observed + RIDGE * np.eye(len(ALPHA))
                rhs = observed.T @ np.column_stack((h_l[h_keep], h_r[h_keep]))
                decoded = np.linalg.solve(lhs, rhs)
                raw["hdt_mask_aware"] = (decoded[:, 0], decoded[:, 1])

                for name, (left, right) in raw.items():
                    pl, pr = project_pair(left, right)
                    errors[name].append(float(np.sqrt(np.mean(np.r_[pl - ideal_l, pr - ideal_r] ** 2))))
                    valid[name].append(float(any(violations(pl, pr).values())))
                    displacement[name].append(float(np.sqrt(np.mean(np.r_[pl - left, pr - right] ** 2))))
            for name in errors:
                rows.append({
                    "seed": seed,
                    "dimension": dimension,
                    "erasure_rate": rate,
                    "method": name,
                    "rmse_mean": float(np.mean(errors[name])),
                    "rmse_p95": float(np.quantile(errors[name], 0.95)),
                    "cnf_violation_rate": float(np.mean(valid[name])),
                    "projection_displacement_mean": float(np.mean(displacement[name])),
                })
    return rows


def main() -> None:
    out = Path(__file__).resolve().parent
    rows = [row for seed in SEEDS for row in run_seed(seed)]
    summary = []
    for dimension in DIMENSIONS:
        for rate in ERASURE_RATES:
            for method in ("grid", "hdt_fixed", "hdt_mask_aware"):
                subset = [r for r in rows if r["dimension"] == dimension and r["erasure_rate"] == rate and r["method"] == method]
                summary.append({
                    "dimension": dimension,
                    "erasure_rate": rate,
                    "method": method,
                    "rmse_mean": float(np.mean([r["rmse_mean"] for r in subset])),
                    "rmse_seed_std": float(np.std([r["rmse_mean"] for r in subset], ddof=1)),
                    "rmse_p95_mean": float(np.mean([r["rmse_p95"] for r in subset])),
                    "cnf_violation_rate": float(np.mean([r["cnf_violation_rate"] for r in subset])),
                    "projection_displacement_mean": float(np.mean([r["projection_displacement_mean"] for r in subset])),
                })
    payload = {
        "dimensions": list(DIMENSIONS),
        "erasure_rates": list(ERASURE_RATES),
        "seeds": list(SEEDS),
        "trials_per_setting_per_seed": TRIALS,
        "grid_budget": GRID_BUDGET,
        "length_scale": LENGTH_SCALE,
        "ridge": RIDGE,
        "damage_model": "independent random coordinate erasure with observed mask; same CNF projection for all methods",
        "rows": rows,
        "summary": summary,
    }
    (out / "hdc_value_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for filename, data in (("hdc_value_by_seed.csv", rows), ("hdc_value_summary.csv", summary)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(12, 3.8), sharey=True)
    styles = {"grid": "o-", "hdt_fixed": "s--", "hdt_mask_aware": "^-"}
    labels = {"grid": "Grid linear", "hdt_fixed": "Fixed HDT inverse",
              "hdt_mask_aware": "Mask-aware HDT inverse"}
    for ax, dimension in zip(axes, DIMENSIONS):
        for method, style in styles.items():
            subset = [r for r in summary if r["dimension"] == dimension and r["method"] == method]
            ax.errorbar([100 * r["erasure_rate"] for r in subset], [r["rmse_mean"] for r in subset],
                        yerr=[r["rmse_seed_std"] for r in subset], fmt=style, capsize=2, label=labels[method])
        ax.set_title(f"D={dimension}")
        ax.set_xlabel("Erased coordinates (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Projected endpoint RMSE")
    axes[-1].legend(frameon=False, fontsize=8)
    for i, ax in enumerate(axes): ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "hdc_value_diagnostics.png", dpi=300)
    plt.close(fig)
    print(json.dumps({"summary": summary, "files": ["hdc_value_results.json", "hdc_value_by_seed.csv", "hdc_value_summary.csv", "hdc_value_diagnostics.png"]}, indent=2))


if __name__ == "__main__":
    main()
