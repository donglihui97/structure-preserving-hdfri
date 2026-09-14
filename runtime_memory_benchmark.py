"""Runtime and representation-memory benchmark for the two retrieval carriers."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALPHA_POINTS = 11
RULE_COUNTS = (30, 100, 300, 1000)
DIMS = (16, 32, 64)
QUERIES = 500
REPEATS = 5


def run_once(rng: np.random.Generator, n_rules: int, dimension: int, method: str) -> float:
    antecedents = rng.normal(size=(n_rules, 2)).astype(np.float64)
    query = rng.normal(size=(QUERIES, 2)).astype(np.float64)
    consequents = rng.normal(size=(n_rules, 2, ALPHA_POINTS)).astype(np.float64)
    if method == "grid":
        start = time.perf_counter()
        for q in query:
            d = np.linalg.norm(antecedents - q[None, :], axis=1)
            idx = np.argpartition(d, 4)[:4]
            w = 1.0 / np.maximum(d[idx], 1e-9)
            w /= np.sum(w)
            _ = np.tensordot(w, consequents[idx], axes=(0, 0))
    else:
        rule_h = rng.normal(size=(n_rules, dimension)).astype(np.float64)
        query_h = rng.normal(size=(QUERIES, dimension)).astype(np.float64)
        start = time.perf_counter()
        rule_norm = np.linalg.norm(rule_h, axis=1)
        for qh in query_h:
            sims = (rule_h @ qh) / np.maximum(rule_norm * np.linalg.norm(qh), 1e-12)
            idx = np.argpartition(-sims, 4)[:4]
            w = np.maximum(sims[idx], 0.0)
            if np.sum(w) == 0:
                w = np.full(4, 0.25)
            else:
                w /= np.sum(w)
            _ = np.tensordot(w, consequents[idx], axes=(0, 0))
    return (time.perf_counter() - start) / QUERIES * 1e6


def memory_bytes(n_rules: int, dimension: int, method: str) -> int:
    # Two endpoint functions per rule for the grid; two D-vectors for CP-HDFRI.
    return n_rules * (2 * ALPHA_POINTS if method == "grid" else 2 * dimension) * 8


def run() -> dict:
    records = []
    for n_rules in RULE_COUNTS:
        for dimension in DIMS:
            for method in ("grid", "vsa"):
                samples = [run_once(np.random.default_rng(7000 + rep * 101 + n_rules + dimension), n_rules, dimension, method) for rep in range(REPEATS)]
                records.append({
                    "rules": n_rules,
                    "dimension": dimension,
                    "method": method,
                    "microseconds_per_query_mean": float(np.mean(samples)),
                    "microseconds_per_query_std": float(np.std(samples, ddof=1)),
                    "memory_bytes_per_rule": memory_bytes(n_rules, dimension, method) // n_rules,
                    "memory_kib_total": memory_bytes(n_rules, dimension, method) / 1024,
                })
    return {"alpha_points": ALPHA_POINTS, "queries": QUERIES, "repeats": REPEATS, "records": records}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    payload = run()
    (out_dir / "runtime_memory_benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "runtime_memory_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload["records"][0].keys())
        writer.writeheader(); writer.writerows(payload["records"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, style in (("grid", "o-"), ("vsa", "s-")):
        subset = [r for r in payload["records"] if r["method"] == method and r["dimension"] == 32]
        axes[0].plot([r["rules"] for r in subset], [r["microseconds_per_query_mean"] for r in subset], style, label=method)
    axes[0].set_xlabel("Stored rules"); axes[0].set_ylabel("Microseconds/query (D=32)"); axes[0].legend(frameon=False)
    for method, style in (("grid", "o-"), ("vsa", "s-")):
        subset = [r for r in payload["records"] if r["rules"] == 300]
        by_dim = [r for r in subset if r["method"] == method]
        axes[1].plot([r["dimension"] for r in by_dim], [r["memory_kib_total"] for r in by_dim], style, label=method)
    axes[1].set_xlabel("Hypervector dimension"); axes[1].set_ylabel("Total representation KiB (N=300)"); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "runtime_memory_benchmark_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"files": [str(out_dir / "runtime_memory_benchmark_results.json"), str(out_dir / "runtime_memory_benchmark_summary.csv"), str(out_dir / "runtime_memory_benchmark_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
