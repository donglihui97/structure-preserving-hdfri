"""Matched-budget coordinate-erasure benchmark for endpoint representations.

Each endpoint uses 64 stored float64 values.  The grid control stores 64
uniform alpha samples and fills erased samples by linear interpolation.  The
distributed carrier fits 12 Chebyshev coefficients and spreads them across 64
orthogonal random coordinates.  We compare a fixed inverse with an
erasure-aware least-squares inverse and apply the same CNF repair to both HDT
variants.  Results isolate representation recovery from shape repair.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preexperiments import project_cnf, violations


EVAL_ALPHA = np.linspace(0.0, 1.0, 101)
GRID_ALPHA = np.linspace(0.0, 1.0, 64)
BUDGET = 64
BASIS_SIZE = 12
ERASURE_RATES = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30)
SEEDS = (401, 409, 419, 421, 431)
TRIALS = 500


def endpoint_pair(rng: np.random.Generator, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth CNF endpoint pair with nonlinear shoulders."""
    a = rng.uniform(-2.0, 2.0)
    b = a + rng.uniform(0.4, 2.5)
    c = b + rng.uniform(0.4, 2.5)
    q_left = rng.uniform(0.55, 2.4)
    q_right = rng.uniform(0.55, 2.4)
    left = a + (b - a) * alpha**q_left
    right = c - (c - b) * alpha**q_right
    return left, right


def erase_mask(rng: np.random.Generator, size: int, rate: float) -> np.ndarray:
    keep = np.ones(size, dtype=bool)
    count = int(round(rate * size))
    if count:
        keep[rng.choice(size, size=count, replace=False)] = False
    return keep


def grid_recover(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.interp(EVAL_ALPHA, GRID_ALPHA[keep], values[keep])


def is_cnf(left: np.ndarray, right: np.ndarray) -> bool:
    return not any(violations(left, right).values())


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = 2.0 * EVAL_ALPHA - 1.0
    basis = np.polynomial.chebyshev.chebvander(x, BASIS_SIZE - 1)
    carrier, _ = np.linalg.qr(rng.normal(size=(BUDGET, BASIS_SIZE)))
    rows = []

    for rate in ERASURE_RATES:
        metrics = {name: [] for name in ("grid", "hdt_fixed", "hdt_erasure_aware")}
        valid = {name: [] for name in metrics}
        repair = {name: [] for name in metrics}
        for _ in range(TRIALS):
            l1, r1 = endpoint_pair(rng, EVAL_ALPHA)
            l2, r2 = endpoint_pair(rng, EVAL_ALPHA)
            weights = rng.dirichlet((1.0, 1.0))
            ideal_l = weights[0] * l1 + weights[1] * l2
            ideal_r = weights[0] * r1 + weights[1] * r2

            # Equal-budget grid carrier.
            grid_l = np.interp(GRID_ALPHA, EVAL_ALPHA, ideal_l)
            grid_r = np.interp(GRID_ALPHA, EVAL_ALPHA, ideal_r)
            mask_l = erase_mask(rng, BUDGET, rate)
            mask_r = erase_mask(rng, BUDGET, rate)
            out_l = grid_recover(grid_l, mask_l)
            out_r = grid_recover(grid_r, mask_r)
            metrics["grid"].append(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2)))
            valid["grid"].append(is_cnf(out_l, out_r))
            repair["grid"].append(0.0)

            coeff_l = np.linalg.lstsq(basis, ideal_l, rcond=None)[0]
            coeff_r = np.linalg.lstsq(basis, ideal_r, rcond=None)[0]
            h_l = carrier @ coeff_l
            h_r = carrier @ coeff_r
            keep_l = erase_mask(rng, BUDGET, rate)
            keep_r = erase_mask(rng, BUDGET, rate)

            keep_fraction_l = max(float(np.mean(keep_l)), 1e-12)
            keep_fraction_r = max(float(np.mean(keep_r)), 1e-12)
            fixed_l = basis @ (carrier.T @ (h_l * keep_l) / keep_fraction_l)
            fixed_r = basis @ (carrier.T @ (h_r * keep_r) / keep_fraction_r)
            aware_l = basis @ np.linalg.lstsq(carrier[keep_l], h_l[keep_l], rcond=None)[0]
            aware_r = basis @ np.linalg.lstsq(carrier[keep_r], h_r[keep_r], rcond=None)[0]

            for name, raw_l, raw_r in (("hdt_fixed", fixed_l, fixed_r), ("hdt_erasure_aware", aware_l, aware_r)):
                proj_l, proj_r = project_cnf(raw_l, raw_r)
                metrics[name].append(np.sqrt(np.mean(np.r_[proj_l - ideal_l, proj_r - ideal_r] ** 2)))
                valid[name].append(is_cnf(proj_l, proj_r))
                repair[name].append(np.sqrt(np.mean(np.r_[proj_l - raw_l, proj_r - raw_r] ** 2)))

        for name in metrics:
            rows.append({
                "seed": seed,
                "erasure_rate": rate,
                "method": name,
                "rmse_mean": float(np.mean(metrics[name])),
                "rmse_p95": float(np.quantile(metrics[name], 0.95)),
                "cnf_valid_rate": float(np.mean(valid[name])),
                "repair_displacement_mean": float(np.mean(repair[name])),
            })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    summary = []
    for rate in ERASURE_RATES:
        for method in ("grid", "hdt_fixed", "hdt_erasure_aware"):
            subset = [r for r in rows if r["erasure_rate"] == rate and r["method"] == method]
            summary.append({
                "erasure_rate": rate,
                "method": method,
                "rmse_mean": float(np.mean([r["rmse_mean"] for r in subset])),
                "rmse_seed_std": float(np.std([r["rmse_mean"] for r in subset], ddof=1)),
                "cnf_valid_rate": float(np.mean([r["cnf_valid_rate"] for r in subset])),
                "repair_displacement_mean": float(np.mean([r["repair_displacement_mean"] for r in subset])),
            })
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = [row for seed in SEEDS for row in run_seed(seed)]
    summary = aggregate(rows)
    payload = {
        "seeds": list(SEEDS), "trials_per_seed": TRIALS, "stored_values_per_endpoint": BUDGET,
        "evaluation_alpha_points": len(EVAL_ALPHA), "basis_size": BASIS_SIZE,
        "damage_model": "same fraction of stored coordinates erased; erased positions known to each decoder",
        "rows": rows, "summary": summary,
    }
    (out_dir / "representation_erasure_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, data in (("representation_erasure_by_seed.csv", rows), ("representation_erasure_summary.csv", summary)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {"grid": "o-", "hdt_fixed": "s--", "hdt_erasure_aware": "^-"}
    for method, style in styles.items():
        subset = [r for r in summary if r["method"] == method]
        ax.errorbar([100 * r["erasure_rate"] for r in subset], [r["rmse_mean"] for r in subset],
                    yerr=[r["rmse_seed_std"] for r in subset], fmt=style, capsize=3, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("Erased stored coordinates (%)"); ax.set_ylabel("Endpoint RMSE (log scale)")
    ax.set_title("Matched 64-value endpoint representation budget"); ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "representation_erasure_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "files": [str(out_dir / "representation_erasure_results.json"), str(out_dir / "representation_erasure_summary.csv"), str(out_dir / "representation_erasure_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
