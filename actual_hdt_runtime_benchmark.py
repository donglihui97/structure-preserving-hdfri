"""Runtime and memory scaling for the paper-faithful HDT consequent path."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA, interval_encoding, trapezoid_weights
from preexperiments import project_cnf


DIMENSIONS = (1024, 4096, 8192)
RULE_COUNTS = (30, 100, 300)
QUERIES = 100
REPEATS = 5
LENGTH_SCALE = 0.20


def rule_endpoints(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    center = rng.uniform(0.1, 0.9, count)
    width = rng.uniform(0.03, 0.12, count)
    left = center[:, None] - width[:, None] + ALPHA[None, :] * width[:, None]
    right = center[:, None] + width[:, None] - ALPHA[None, :] * width[:, None]
    return left, right


def query_weights(rng: np.random.Generator, count: int) -> np.ndarray:
    matrix = np.zeros((QUERIES, count))
    for row in range(QUERIES):
        idx = rng.choice(count, size=4, replace=False)
        matrix[row, idx] = rng.dirichlet(np.ones(4))
    return matrix


def timed(fn) -> tuple[float, float]:
    samples, checksum = [], 0.0
    for _ in range(REPEATS):
        start = time.perf_counter()
        checksum = float(fn())
        samples.append((time.perf_counter() - start) / QUERIES * 1e6)
    return float(np.median(samples)), checksum


def run() -> list[dict]:
    rows = []
    weights_alpha = trapezoid_weights(ALPHA)
    for rule_count in RULE_COUNTS:
        for dimension in DIMENSIONS:
            rng = np.random.default_rng(9000 + rule_count + dimension)
            left, right = rule_endpoints(rng, rule_count)
            query_matrix = query_weights(rng, rule_count)
            encoding, _ = interval_encoding(rng, dimension, LENGTH_SCALE)
            h_left = (encoding.T @ (left * weights_alpha[None, :]).T).T
            h_right = (encoding.T @ (right * weights_alpha[None, :]).T).T

            def grid_query():
                out_l = query_matrix @ left; out_r = query_matrix @ right
                return np.sum(out_l) + np.sum(out_r)

            def hdt_query():
                bundled_l = query_matrix @ h_left; bundled_r = query_matrix @ h_right
                raw_l = bundled_l @ encoding.T / dimension
                raw_r = bundled_r @ encoding.T / dimension
                total = 0.0
                for l_values, r_values in zip(raw_l, raw_r):
                    out_l, out_r = project_cnf(l_values, r_values)
                    total += float(np.sum(out_l) + np.sum(out_r))
                return total

            grid_time, grid_sum = timed(grid_query)
            hdt_time, hdt_sum = timed(hdt_query)
            rows.extend([
                {"rules": rule_count, "dimension": dimension, "method": "grid",
                 "microseconds_per_query_median": grid_time,
                 "per_rule_consequent_kib": 2 * len(ALPHA) * 8 / 1024,
                 "shared_encoder_kib": 0.0, "checksum": grid_sum},
                {"rules": rule_count, "dimension": dimension, "method": "actual_hdt_cp",
                 "microseconds_per_query_median": hdt_time,
                 "per_rule_consequent_kib": 2 * dimension * 8 / 1024,
                 "shared_encoder_kib": encoding.nbytes / 1024, "checksum": hdt_sum},
            ])
    return rows


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    plot_source = Path(sys.argv[2]) if len(sys.argv) == 3 and sys.argv[1] == "--plot-from" else None
    if plot_source is not None:
        with plot_source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in ("rules", "dimension"):
                row[key] = int(row[key])
            for key in ("microseconds_per_query_median", "per_rule_consequent_kib",
                        "shared_encoder_kib", "checksum"):
                row[key] = float(row[key])
    else:
        rows = run()
        payload = {"queries_per_batch": QUERIES, "timing_repeats": REPEATS,
                   "alpha_points": len(ALPHA), "length_scale": LENGTH_SCALE, "rows": rows}
        (out_dir / "actual_hdt_runtime_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (out_dir / "actual_hdt_runtime_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 6.2), sharex=True)
    labels = {"grid": "Direct grid", "actual_hdt_cp": "HDT + CP"}
    for method, marker in (("grid", "o"), ("actual_hdt_cp", "s")):
        subset = [r for r in rows if r["method"] == method and r["rules"] == 100]
        axes[0].plot([r["dimension"] for r in subset], [r["microseconds_per_query_median"] for r in subset], marker + "-", label=labels[method])
        axes[1].plot([r["dimension"] for r in subset], [r["per_rule_consequent_kib"] for r in subset], marker + "-", label=labels[method])
    axes[0].set_ylabel("Batch microseconds/query (N=100)")
    axes[1].set_ylabel("Consequent KiB/rule")
    axes[1].set_xlabel("HDT dimension D")
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(alpha=0.25); ax.legend(frameon=False)
    for i, ax in enumerate(axes):
        ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
    fig.tight_layout(h_pad=1.2); fig.savefig(out_dir / "actual_hdt_runtime_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"files": [str(out_dir / "actual_hdt_runtime_results.json"), str(out_dir / "actual_hdt_runtime_summary.csv"), str(out_dir / "actual_hdt_runtime_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
