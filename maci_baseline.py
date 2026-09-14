"""One-dimensional characteristic-vector MACI-compatible baseline.

This implements the 1-D piecewise-linear characteristic-vector form of MACI:
the four points (left foot, left core, right core, right foot) of two bracketing
consequents are interpolated with the same reference-point ratio used by the
antecedents.  It is evaluated beside alpha-cut KH interpolation on the same
sparse rule/query draws.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALPHA = np.linspace(0.0, 1.0, 101)


def triangle(center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    return center - width + ALPHA * width, center + width - ALPHA * width


def truth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 + 0.25 * np.sin(2.0 * np.pi * x)
    width = 0.045 + 0.012 * x
    return center, width


def cnf_violation(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.any(np.diff(left) < -1e-10) or np.any(np.diff(right) > 1e-10) or np.any(left > right + 1e-10))


def bracket(points: np.ndarray, q: float) -> tuple[int, int, float]:
    hi = int(np.searchsorted(points, q, side="left"))
    if hi <= 0:
        lo, hi = 0, 1
    elif hi >= len(points):
        lo, hi = len(points) - 2, len(points) - 1
    else:
        lo = hi - 1
    span = max(points[hi] - points[lo], 1e-12)
    return lo, hi, float(np.clip((q - points[lo]) / span, 0.0, 1.0))


def kh_interpolate(a_points: np.ndarray, consequents: list[tuple[np.ndarray, np.ndarray]], q: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi, _ = bracket(a_points, q)
    a_lo_l, a_lo_r = consequents[lo][0], consequents[lo][1]
    a_hi_l, a_hi_r = consequents[hi][0], consequents[hi][1]
    q_l, q_r = triangle(q, truth(np.array([q]))[1][0])
    d_lo = np.abs(np.stack([a_lo_l - q_l, a_lo_r - q_r], axis=0))
    d_hi = np.abs(np.stack([a_hi_l - q_l, a_hi_r - q_r], axis=0))
    w_lo = d_hi / np.maximum(d_lo + d_hi, 1e-12)
    w_hi = 1.0 - w_lo
    return w_lo[0] * consequents[lo][0] + w_hi[0] * consequents[hi][0], w_lo[1] * consequents[lo][1] + w_hi[1] * consequents[hi][1]


def maci_interpolate(a_points: np.ndarray, consequents: list[tuple[np.ndarray, np.ndarray]], q: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi, lam = bracket(a_points, q)
    # Characteristic vector interpolation preserves the piecewise-linear CNF
    # shape when both bracketing consequents are CNF.
    left = (1.0 - lam) * consequents[lo][0] + lam * consequents[hi][0]
    right = (1.0 - lam) * consequents[lo][1] + lam * consequents[hi][1]
    return left, right


def run(seed: int, queries_n: int = 800, retained_n: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    dense = np.linspace(0.0, 1.0, 21)
    retained = np.sort(rng.choice(dense, size=retained_n, replace=False))
    rc, rw = truth(retained)
    consequents = [triangle(float(c), float(w)) for c, w in zip(rc, rw)]
    queries = rng.uniform(0.0, 1.0, size=queries_n)
    tc, tw = truth(queries)
    rows = []
    for q, c, w in zip(queries, tc, tw):
        ideal_l, ideal_r = triangle(float(c), float(w))
        kh_l, kh_r = kh_interpolate(retained, consequents, float(q))
        maci_l, maci_r = maci_interpolate(retained, consequents, float(q))
        for method, left, right in (("kh_alpha_cut", kh_l, kh_r), ("maci_characteristic", maci_l, maci_r)):
            err = float(np.sqrt(np.mean(np.r_[left - ideal_l, right - ideal_r] ** 2)))
            rows.append({"method": method, "rmse": err, "cnf_violation": cnf_violation(left, right)})
    aggregate = {}
    for method in ("kh_alpha_cut", "maci_characteristic"):
        vals = [r["rmse"] for r in rows if r["method"] == method]
        aggregate[method] = {"rmse_mean": float(np.mean(vals)), "rmse_p95": float(np.quantile(vals, 0.95)),
                             "cnf_violation_rate": float(np.mean([r["cnf_violation"] for r in rows if r["method"] == method]))}
    return {"seed": seed, "retained_rules": retained.tolist(), "aggregate": aggregate}


def main() -> None:
    seeds = (19, 23, 29, 31, 37)
    records = [run(seed) for seed in seeds]
    methods = ("kh_alpha_cut", "maci_characteristic")
    summary = []
    for method in methods:
        means = [r["aggregate"][method]["rmse_mean"] for r in records]
        p95s = [r["aggregate"][method]["rmse_p95"] for r in records]
        rates = [r["aggregate"][method]["cnf_violation_rate"] for r in records]
        summary.append({"method": method, "rmse_mean": float(np.mean(means)), "rmse_std": float(np.std(means, ddof=1)),
                        "rmse_p95_mean": float(np.mean(p95s)), "cnf_violation_rate_mean": float(np.mean(rates))})
    out = Path(__file__).resolve().parent
    (out / "maci_baseline_results.json").write_text(json.dumps({"seeds": list(seeds), "records": records, "summary": summary}, indent=2), encoding="utf-8")
    with (out / "maci_baseline_summary.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    x = np.arange(len(methods)); means = [next(r["rmse_mean"] for r in summary if r["method"] == m) for m in methods]
    errs = [next(r["rmse_std"] for r in summary if r["method"] == m) for m in methods]
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); ax.bar(x, means, yerr=errs, capsize=4, color=("#4C78A8", "#F58518"))
    ax.set_xticks(x, ["KH alpha-cut", "MACI characteristic"]); ax.set_ylabel("Endpoint RMSE"); ax.set_title("KH and MACI-compatible 1-D baseline")
    fig.tight_layout(); fig.savefig(out / "maci_baseline_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "files": ["maci_baseline_results.json", "maci_baseline_summary.csv", "maci_baseline_diagnostics.png"]}, indent=2))


if __name__ == "__main__":
    main()
