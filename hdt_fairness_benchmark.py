"""Fairness controls for the coordinate-erasure HDT experiment.

The benchmark keeps the endpoint functions, erasure masks, carrier budget,
and CNF projection fixed while comparing several reconstruction assumptions:
piecewise-linear grid interpolation, PCHIP, a cubic polynomial fit, simple
repetition-coded grid storage, mask-aware HDT least squares, and an HDT
decoder that is not given the erasure mask.  The goal is to separate the
benefit of a distributed carrier from the benefit of a low-dimensional prior.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from actual_hdt_benchmark import ALPHA, interval_encoding, trapezoid_weights
from hdc_value_benchmark import (
    DIMENSIONS,
    ERASURE_RATES,
    GRID_BUDGET,
    LENGTH_SCALE,
    RIDGE,
    SEEDS,
    TRIALS,
    endpoint_pair,
)
from preexperiments import project_cnf, violations


METHODS = (
    "grid_linear",
    "grid_pchip",
    "grid_poly3",
    "grid_repetition",
    "hdt_mask_aware",
    "hdt_unknown_mask",
)
MIN_EIG_THRESHOLD = 1e-10
MAX_CONDITION_THRESHOLD = 1e12


def project_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return project_cnf(left, right)


def grid_decode(values: np.ndarray, keep: np.ndarray, method: str) -> np.ndarray:
    grid_alpha = np.linspace(0.0, 1.0, GRID_BUDGET)
    x = grid_alpha[keep]
    y = values[keep]
    if len(x) == 1:
        return np.full_like(ALPHA, y[0], dtype=float)
    if method == "grid_linear":
        return np.interp(ALPHA, x, y)
    if method == "grid_pchip":
        return PchipInterpolator(x, y, extrapolate=True)(ALPHA)
    if method == "grid_poly3":
        degree = min(3, len(x) - 1)
        design = np.vander(x, degree + 1, increasing=True)
        target = np.vander(ALPHA, degree + 1, increasing=True)
        coeff = np.linalg.lstsq(design, y, rcond=None)[0]
        return target @ coeff
    raise ValueError(method)


def repetition_decode(values: np.ndarray, keep: np.ndarray, dimension: int) -> np.ndarray:
    """Decode a 64-point grid repeated over a D-slot carrier."""
    slots_per_point = int(np.ceil(dimension / GRID_BUDGET))
    groups = []
    for j in range(GRID_BUDGET):
        lo = j * slots_per_point
        hi = min((j + 1) * slots_per_point, dimension)
        observed = values[lo:hi][keep[lo:hi]]
        groups.append(float(np.mean(observed)) if len(observed) else np.nan)
    groups = np.asarray(groups)
    good = np.isfinite(groups)
    if good.sum() == 0:
        return np.zeros_like(ALPHA)
    if good.sum() == 1:
        return np.full_like(ALPHA, groups[good][0], dtype=float)
    grid_alpha = np.linspace(0.0, 1.0, GRID_BUDGET)
    return np.interp(ALPHA, grid_alpha[good], groups[good])


def summarize(rows: list[dict]) -> list[dict]:
    def optional_mean(subset: list[dict], key: str) -> float | None:
        values = [row[key] for row in subset if row[key] is not None]
        return float(np.mean(values)) if values else None

    summary = []
    for dimension in DIMENSIONS:
        for rate in ERASURE_RATES:
            for method in METHODS:
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
                    "reliable_mask_rmse_mean": optional_mean(subset, "reliable_mask_rmse_mean") if method == "hdt_mask_aware" else None,
                    "reliable_mask_coverage": optional_mean(subset, "reliable_mask_coverage") if method == "hdt_mask_aware" else None,
                    "rejected_mask_rate": optional_mean(subset, "rejected_mask_rate") if method == "hdt_mask_aware" else None,
                })
    return summary


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    weights = trapezoid_weights(ALPHA)
    rows: list[dict] = []
    for dimension in DIMENSIONS:
        encoding, _ = interval_encoding(rng, dimension, LENGTH_SCALE)
        forward = encoding.T * weights[None, :]
        full_lhs = forward.T @ forward + RIDGE * np.eye(len(ALPHA))
        full_inverse = np.linalg.solve(full_lhs, forward.T)
        for rate in ERASURE_RATES:
            errors = {name: [] for name in METHODS}
            reliable_errors = []
            reliable_count = 0
            rejected_count = 0
            valid = {name: [] for name in METHODS}
            displacement = {name: [] for name in METHODS}
            for trial in range(TRIALS):
                local = np.random.default_rng(seed * 100000 + dimension * 10 + trial)
                ideal_l, ideal_r = endpoint_pair(local)
                h_keep = local.random(dimension) >= rate
                if not np.any(h_keep):
                    h_keep[local.integers(dimension)] = True
                grid_alpha = np.linspace(0.0, 1.0, GRID_BUDGET)
                grid_l = np.interp(grid_alpha, ALPHA, ideal_l)
                grid_r = np.interp(grid_alpha, ALPHA, ideal_r)
                g_keep = local.random(GRID_BUDGET) >= rate
                if not np.any(g_keep):
                    g_keep[local.integers(GRID_BUDGET)] = True
                h_l = forward @ ideal_l
                h_r = forward @ ideal_r
                observed = forward[h_keep]
                singular = np.linalg.svd(observed, compute_uv=False)
                eig_min = float(singular[-1] ** 2)
                eig_max = float(singular[0] ** 2)
                reliable_mask = bool(eig_min >= MIN_EIG_THRESHOLD and eig_max / max(eig_min, 1e-300) <= MAX_CONDITION_THRESHOLD)
                reliable_count += int(reliable_mask)
                rejected_count += int(not reliable_mask)
                lhs = observed.T @ observed + RIDGE * np.eye(len(ALPHA))
                rhs = observed.T @ np.column_stack((h_l[h_keep], h_r[h_keep]))
                decoded = np.linalg.solve(lhs, rhs)
                damaged_l = np.where(h_keep, h_l, 0.0)
                damaged_r = np.where(h_keep, h_r, 0.0)
                raw = {
                    "grid_linear": (grid_decode(grid_l, g_keep, "grid_linear"), grid_decode(grid_r, g_keep, "grid_linear")),
                    "grid_pchip": (grid_decode(grid_l, g_keep, "grid_pchip"), grid_decode(grid_r, g_keep, "grid_pchip")),
                    "grid_poly3": (grid_decode(grid_l, g_keep, "grid_poly3"), grid_decode(grid_r, g_keep, "grid_poly3")),
                    "grid_repetition": (
                        repetition_decode(np.repeat(grid_l, int(np.ceil(dimension / GRID_BUDGET)))[:dimension], h_keep, dimension),
                        repetition_decode(np.repeat(grid_r, int(np.ceil(dimension / GRID_BUDGET)))[:dimension], h_keep, dimension),
                    ),
                    "hdt_mask_aware": (decoded[:, 0], decoded[:, 1]),
                    "hdt_unknown_mask": (full_inverse @ damaged_l, full_inverse @ damaged_r),
                }
                for name, (left, right) in raw.items():
                    pl, pr = project_pair(left, right)
                    errors[name].append(float(np.sqrt(np.mean(np.r_[pl - ideal_l, pr - ideal_r] ** 2))))
                    if name == "hdt_mask_aware" and reliable_mask:
                        reliable_errors.append(errors[name][-1])
                    valid[name].append(float(any(violations(pl, pr).values())))
                    displacement[name].append(float(np.sqrt(np.mean(np.r_[pl - left, pr - right] ** 2))))
            for name in METHODS:
                rows.append({
                    "seed": seed,
                    "dimension": dimension,
                    "erasure_rate": rate,
                    "method": name,
                    "rmse_mean": float(np.mean(errors[name])),
                    "rmse_p95": float(np.quantile(errors[name], 0.95)),
                    "cnf_violation_rate": float(np.mean(valid[name])),
                    "projection_displacement_mean": float(np.mean(displacement[name])),
                    "reliable_mask_rmse_mean": float(np.mean(reliable_errors)) if name == "hdt_mask_aware" and reliable_errors else None,
                    "reliable_mask_coverage": float(reliable_count / TRIALS) if name == "hdt_mask_aware" else None,
                    "rejected_mask_rate": float(rejected_count / TRIALS) if name == "hdt_mask_aware" else None,
                })
    return rows


def main() -> None:
    out = Path(__file__).resolve().parent
    rows = [row for seed in SEEDS for row in run_seed(seed)]
    summary = summarize(rows)
    payload = {
        "dimensions": list(DIMENSIONS),
        "erasure_rates": list(ERASURE_RATES),
        "seeds": list(SEEDS),
        "trials_per_setting_per_seed": TRIALS,
        "grid_budget": GRID_BUDGET,
        "length_scale": LENGTH_SCALE,
        "ridge": RIDGE,
        "damage_model": "independent random coordinate erasure; mask-aware methods receive the observed mask; unknown-mask HDT receives zeros only",
        "method_definitions": {
            "grid_linear": "64-point grid with piecewise-linear interpolation over observed points",
            "grid_pchip": "64-point grid with shape-preserving PCHIP interpolation over observed points",
            "grid_poly3": "64-point grid with cubic least-squares fit over observed points",
            "grid_repetition": "64 grid values repeated across the D carrier slots and decoded by observed replica means",
            "hdt_mask_aware": "restricted ridge least-squares inverse using observed carrier rows",
            "hdt_unknown_mask": "full inverse after erased carrier coordinates are replaced by zero; no mask supplied",
        },
        "rows": rows,
        "summary": summary,
    }
    (out / "hdt_fairness_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for filename, data in (("hdt_fairness_by_seed.csv", rows), ("hdt_fairness_summary.csv", summary)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(14, 4), sharey=True)
    styles = {"grid_linear": "o-", "grid_pchip": "s-", "grid_poly3": "d-", "grid_repetition": "x--", "hdt_mask_aware": "^-", "hdt_unknown_mask": "v--"}
    labels = {"grid_linear": "Grid linear", "grid_pchip": "Grid PCHIP",
              "grid_poly3": "Grid cubic", "grid_repetition": "Grid repetition",
              "hdt_mask_aware": "HDT mask-aware", "hdt_unknown_mask": "HDT unknown-mask"}
    for ax, dimension in zip(axes, DIMENSIONS):
        for method, style in styles.items():
            subset = [r for r in summary if r["dimension"] == dimension and r["method"] == method]
            ax.errorbar([100 * r["erasure_rate"] for r in subset], [r["rmse_mean"] for r in subset], yerr=[r["rmse_seed_std"] for r in subset], fmt=style, capsize=2, label=labels[method])
        ax.set_title(f"D={dimension}")
        ax.set_xlabel("Erased coordinates (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Projected endpoint RMSE")
    axes[-1].legend(frameon=False, fontsize=7)
    for i, ax in enumerate(axes): ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "hdt_fairness_diagnostics.png", dpi=300)
    plt.close(fig)
    print(json.dumps({"files": ["hdt_fairness_results.json", "hdt_fairness_by_seed.csv", "hdt_fairness_summary.csv", "hdt_fairness_diagnostics.png"]}, indent=2))


if __name__ == "__main__":
    main()
