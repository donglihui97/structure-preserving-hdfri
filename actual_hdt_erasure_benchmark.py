"""Coordinate-erasure benchmark for the paper-faithful interval HDT carrier."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA, interval_encoding, trapezoid_weights
from preexperiments import project_cnf, violations


DIMENSIONS = (1024, 4096, 8192)
LENGTH_SCALE = 0.20
ERASURE_RATES = (0.0, 0.05, 0.10, 0.20, 0.30)
SEEDS = (503, 509, 521, 523, 541)
TRIALS = 100


def triangle(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = rng.uniform(-2.0, 2.0)
    b = a + rng.uniform(0.4, 2.5)
    c = b + rng.uniform(0.4, 2.5)
    return a + ALPHA * (b - a), c - ALPHA * (c - b)


def grid_decode(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    known = np.flatnonzero(mask)
    if len(known) == 0:
        return np.zeros_like(values)
    return np.interp(np.arange(len(values)), known, values[known])


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    weights = trapezoid_weights(ALPHA)
    rows = []
    for dimension in DIMENSIONS:
        encoding, _ = interval_encoding(rng, dimension, LENGTH_SCALE)
        for rate in ERASURE_RATES:
            metrics = {m: {"err": [], "disp": [], "invalid": []} for m in (
                "grid", "hdt_fixed", "hdt_erasure_aware", "hdt_mass_normalized",
            )}
            for trial in range(TRIALS):
                local = np.random.default_rng(seed + 100000 + dimension * 10 + int(rate * 100) * 1000 + trial)
                left, right = triangle(local)
                mask = local.random(dimension) >= rate
                if not np.any(mask): mask[local.integers(0, dimension)] = True
                gmask = local.random(len(ALPHA)) >= rate
                if not np.any(gmask): gmask[local.integers(0, len(ALPHA))] = True
                outs = {"grid": {}, "hdt_fixed": {}, "hdt_erasure_aware": {},
                        "hdt_mass_normalized": {}}
                constant_code = encoding.T @ weights
                retained = int(np.sum(mask))
                row_mass = encoding @ (constant_code * mask) / retained
                for name, values in (("left", left), ("right", right)):
                    z = encoding.T @ (weights * values); damaged = z * mask
                    outs["hdt_fixed"][name] = encoding @ damaged / dimension
                    outs["hdt_erasure_aware"][name] = encoding @ damaged / retained
                    outs["hdt_mass_normalized"][name] = (
                        encoding @ damaged / retained / np.maximum(row_mass, 1e-12)
                    )
                    outs["grid"][name] = grid_decode(values, gmask)
                for method, values in outs.items():
                    pl, pr = project_cnf(values["left"], values["right"])
                    metrics[method]["err"].append(float(np.sqrt(np.mean(np.r_[pl - left, pr - right] ** 2))))
                    metrics[method]["disp"].append(float(np.sqrt(np.mean(np.r_[pl - values["left"], pr - values["right"]] ** 2))))
                    metrics[method]["invalid"].append(any(violations(pl, pr).values()))
            for method, metric in metrics.items():
                rows.append({"seed": seed, "dimension": dimension, "erasure_rate": rate, "method": method,
                             "rmse_mean": float(np.mean(metric["err"])), "repair_displacement_mean": float(np.mean(metric["disp"])),
                             "cnf_violation_rate": float(np.mean(metric["invalid"]))})
    return rows


def main() -> None:
    out = Path(__file__).resolve().parent
    rows = [r for seed in SEEDS for r in run_seed(seed)]
    summary = []
    for d in DIMENSIONS:
        for rate in ERASURE_RATES:
            for method in ("grid", "hdt_fixed", "hdt_erasure_aware", "hdt_mass_normalized"):
                s = [r for r in rows if r["dimension"] == d and r["erasure_rate"] == rate and r["method"] == method]
                summary.append({"dimension": d, "erasure_rate": rate, "method": method,
                                "rmse_mean": float(np.mean([r["rmse_mean"] for r in s])),
                                "rmse_seed_std": float(np.std([r["rmse_mean"] for r in s], ddof=1)),
                                "repair_displacement_mean": float(np.mean([r["repair_displacement_mean"] for r in s])),
                                "cnf_violation_rate": float(np.mean([r["cnf_violation_rate"] for r in s]))})
    payload = {"dimensions": list(DIMENSIONS), "erasure_rates": list(ERASURE_RATES), "length_scale": LENGTH_SCALE,
               "seeds": list(SEEDS), "trials_per_setting_per_seed": TRIALS, "rows": rows, "summary": summary}
    (out / "actual_hdt_erasure_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out / "actual_hdt_erasure_summary.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(12, 3.8), sharey=True)
    labels = {"grid": "Direct grid", "hdt_fixed": "Fixed HDT inverse",
              "hdt_erasure_aware": "Erasure-aware HDT inverse",
              "hdt_mass_normalized": "Mass-normalized HDT inverse"}
    for ax, d in zip(axes, DIMENSIONS):
        for method, style in (("grid", "o-"), ("hdt_fixed", "s--"),
                              ("hdt_erasure_aware", "^-"), ("hdt_mass_normalized", "d-")):
            s = [r for r in summary if r["dimension"] == d and r["method"] == method]
            ax.errorbar([r["erasure_rate"] * 100 for r in s], [r["rmse_mean"] for r in s], yerr=[r["rmse_seed_std"] for r in s], fmt=style, capsize=2, label=labels[method])
        ax.set_title(f"D={d}"); ax.set_xlabel("Erased coordinates (%)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Projected endpoint RMSE"); axes[-1].legend(frameon=False, fontsize=8)
    for i, ax in enumerate(axes): ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "actual_hdt_erasure_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "files": ["actual_hdt_erasure_results.json", "actual_hdt_erasure_summary.csv", "actual_hdt_erasure_diagnostics.png"]}, indent=2))


if __name__ == "__main__":
    main()
