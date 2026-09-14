"""Conditioning audit for the mask-aware HDT inverse.

For the same encoders, dimensions, erasure rates, seeds, and endpoint trials
used by the carrier benchmark, record the spectrum of H=F^T M F.  The audit
does not change the decoder; it identifies masks for which the conditional
error bound is informative and marks poorly identified systems as unreliable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA, interval_encoding
from hdc_value_benchmark import DIMENSIONS, ERASURE_RATES, LENGTH_SCALE, SEEDS, TRIALS


RIDGE = 1e-8
MIN_EIG_THRESHOLD = 1e-10
MAX_CONDITION_THRESHOLD = 1e12


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for dimension in DIMENSIONS:
        encoding, _ = interval_encoding(rng, dimension, LENGTH_SCALE)
        weights = np.ones(len(ALPHA), dtype=float)
        weights[[0, -1]] = 0.5
        forward = encoding.T * (weights / (len(ALPHA) - 1))[None, :]
        for rate in ERASURE_RATES:
            minimums = []
            maximums = []
            condition_numbers = []
            reliable = []
            for trial in range(TRIALS):
                local = np.random.default_rng(seed * 100000 + dimension * 10 + trial)
                keep = local.random(dimension) >= rate
                if not np.any(keep):
                    keep[local.integers(dimension)] = True
                observed = forward[keep]
                singular = np.linalg.svd(observed, compute_uv=False)
                eig_min = float(singular[-1] ** 2)
                eig_max = float(singular[0] ** 2)
                log10_cond = float(np.log10(max(eig_max, np.finfo(float).tiny)) - np.log10(max(eig_min, 1e-300)))
                minimums.append(eig_min)
                maximums.append(eig_max)
                condition_numbers.append(log10_cond)
                reliable.append(float(eig_min >= MIN_EIG_THRESHOLD and log10_cond <= np.log10(MAX_CONDITION_THRESHOLD)))
            rows.append({
                "seed": seed,
                "dimension": dimension,
                "erasure_rate": rate,
                "lambda_min_mean": float(np.mean(minimums)),
                "lambda_min_p05": float(np.quantile(minimums, 0.05)),
                "lambda_max_mean": float(np.mean(maximums)),
                "log10_condition_number_mean": float(np.mean(condition_numbers)),
                "log10_condition_number_p95": float(np.quantile(condition_numbers, 0.95)),
                "reliability_rate": float(np.mean(reliable)),
            })
    return rows


def main() -> None:
    out = Path(__file__).resolve().parent
    rows = [r for seed in SEEDS for r in run_seed(seed)]
    summary = []
    for dimension in DIMENSIONS:
        for rate in ERASURE_RATES:
            subset = [r for r in rows if r["dimension"] == dimension and r["erasure_rate"] == rate]
            summary.append({
                "dimension": dimension,
                "erasure_rate": rate,
                "lambda_min_mean": float(np.mean([r["lambda_min_mean"] for r in subset])),
                "lambda_min_p05_mean": float(np.mean([r["lambda_min_p05"] for r in subset])),
                "log10_condition_number_mean": float(np.mean([r["log10_condition_number_mean"] for r in subset])),
                "log10_condition_number_p95_mean": float(np.mean([r["log10_condition_number_p95"] for r in subset])),
                "reliability_rate": float(np.mean([r["reliability_rate"] for r in subset])),
            })
    payload = {
        "dimensions": list(DIMENSIONS),
        "erasure_rates": list(ERASURE_RATES),
        "seeds": list(SEEDS),
        "trials_per_setting_per_seed": TRIALS,
        "ridge": RIDGE,
        "min_eig_threshold": MIN_EIG_THRESHOLD,
        "max_condition_threshold": MAX_CONDITION_THRESHOLD,
        "rows": rows,
        "summary": summary,
    }
    (out / "hdt_condition_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for filename, data in (("hdt_condition_by_seed.csv", rows), ("hdt_condition_summary.csv", summary)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys())
            writer.writeheader(); writer.writerows(data)
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(12, 3.8), sharey=False)
    legend_handles = None
    for ax, dimension in zip(axes, DIMENSIONS):
        subset = [r for r in summary if r["dimension"] == dimension]
        x = [100 * r["erasure_rate"] for r in subset]
        eig_line = ax.semilogy(x, [max(r["lambda_min_mean"], 1e-16) for r in subset], "o-", label="Smallest eigenvalue")[0]
        ax2 = ax.twinx()
        condition_line = ax2.plot(x, [r["log10_condition_number_p95_mean"] for r in subset], "s--", color="tab:red", label="P95 log10 condition number")[0]
        ax2.set_ylabel("P95 log10 condition number")
        legend_handles = (eig_line, condition_line)
        ax.set_title(f"D={dimension}"); ax.set_xlabel("Erased coordinates (%)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("Smallest eigenvalue (log scale)")
    axes[-1].legend(legend_handles, [h.get_label() for h in legend_handles], frameon=False, fontsize=7)
    for i, ax in enumerate(axes): ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "hdt_condition_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"files": ["hdt_condition_results.json", "hdt_condition_by_seed.csv", "hdt_condition_summary.csv", "hdt_condition_diagnostics.png"]}, indent=2))


if __name__ == "__main__":
    main()
