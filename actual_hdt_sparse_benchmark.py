"""Sparse-rule benchmark using the paper-faithful HDT consequent carrier."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA, interval_encoding, transform_operator
from preexperiments import project_cnf, violations
from synthetic_benchmark import (
    K_NEIGHBOURS,
    TEMPERATURE,
    cosine_matrix,
    truth,
    vsa_encoder,
    weighted_neighbours,
)


DIMENSIONS = (1024, 2048, 4096, 8192)
LENGTH_SCALE = 0.20
SEEDS = (19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
QUERIES = 600
SPARSE_RULES = 30


def triangle(center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    return center - width + ALPHA * width, center + width - ALPHA * width


def cnf_invalid(left: np.ndarray, right: np.ndarray) -> bool:
    return any(violations(left, right).values())


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    lattice = np.array([(x, y) for x in np.linspace(0.0, 1.0, 9) for y in np.linspace(0.0, 1.0, 9)])
    rules = lattice[rng.choice(len(lattice), size=SPARSE_RULES, replace=False)]
    queries = rng.uniform(0.0, 1.0, size=(QUERIES, 2))
    true_center, true_width = truth(queries)
    true_left = np.stack([triangle(c, w)[0] for c, w in zip(true_center, true_width)])
    true_right = np.stack([triangle(c, w)[1] for c, w in zip(true_center, true_width)])
    rule_center, rule_width = truth(rules)
    rule_left = np.stack([triangle(c, w)[0] for c, w in zip(rule_center, rule_width)])
    rule_right = np.stack([triangle(c, w)[1] for c, w in zip(rule_center, rule_width)])

    direct = []
    for q, ideal_l, ideal_r in zip(queries, true_left, true_right):
        idx, weights = weighted_neighbours(np.linalg.norm(rules - q, axis=1), K_NEIGHBOURS, "euclidean")
        direct.append((weights @ rule_left[idx], weights @ rule_right[idx], ideal_l, ideal_r))

    direct_errors = [np.sqrt(np.mean(np.r_[l - il, r - ir] ** 2)) for l, r, il, ir in direct]
    rows = [{
        "seed": seed, "dimension": 0, "method": "grid_euclidean",
        "rmse_mean": float(np.mean(direct_errors)), "rmse_p95": float(np.quantile(direct_errors, 0.95)),
        "carrier_rmse_mean": 0.0, "raw_cnf_violation_rate": 0.0,
        "projected_cnf_violation_rate": 0.0, "repair_displacement_mean": 0.0,
    }]

    for dimension in DIMENSIONS:
        local_rng = np.random.default_rng(seed * 10007 + dimension)
        encoding, _ = interval_encoding(local_rng, dimension, LENGTH_SCALE)
        operator = transform_operator(encoding)
        retrieval_rng = np.random.default_rng(seed * 20011 + dimension)
        encode = vsa_encoder(retrieval_rng, dimension)
        similarities = cosine_matrix(encode(queries), encode(rules))

        errors, carrier_errors, raw_invalid, repaired_invalid, displacements = [], [], [], [], []
        vsa_grid_errors, end_to_end_errors, end_to_end_carrier = [], [], []
        end_to_end_raw_invalid, end_to_end_repaired_invalid, end_to_end_displacements = [], [], []
        for query_index, (direct_l, direct_r, ideal_l, ideal_r) in enumerate(direct):
            raw_l, raw_r = operator @ direct_l, operator @ direct_r
            out_l, out_r = project_cnf(raw_l, raw_r)
            errors.append(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2)))
            carrier_errors.append(np.sqrt(np.mean(np.r_[out_l - direct_l, out_r - direct_r] ** 2)))
            raw_invalid.append(cnf_invalid(raw_l, raw_r)); repaired_invalid.append(cnf_invalid(out_l, out_r))
            displacements.append(np.sqrt(np.mean(np.r_[out_l - raw_l, out_r - raw_r] ** 2)))
            idx, vsa_weights = weighted_neighbours(similarities[query_index], K_NEIGHBOURS, "vsa")
            vsa_l, vsa_r = vsa_weights @ rule_left[idx], vsa_weights @ rule_right[idx]
            vsa_grid_errors.append(np.sqrt(np.mean(np.r_[vsa_l - ideal_l, vsa_r - ideal_r] ** 2)))
            vsa_raw_l, vsa_raw_r = operator @ vsa_l, operator @ vsa_r
            vsa_out_l, vsa_out_r = project_cnf(vsa_raw_l, vsa_raw_r)
            end_to_end_errors.append(np.sqrt(np.mean(np.r_[vsa_out_l - ideal_l, vsa_out_r - ideal_r] ** 2)))
            end_to_end_carrier.append(np.sqrt(np.mean(np.r_[vsa_out_l - vsa_l, vsa_out_r - vsa_r] ** 2)))
            end_to_end_raw_invalid.append(cnf_invalid(vsa_raw_l, vsa_raw_r))
            end_to_end_repaired_invalid.append(cnf_invalid(vsa_out_l, vsa_out_r))
            end_to_end_displacements.append(np.sqrt(np.mean(np.r_[vsa_out_l - vsa_raw_l, vsa_out_r - vsa_raw_r] ** 2)))
        rows.append({
            "seed": seed, "dimension": dimension, "method": "actual_hdt_euclidean",
            "rmse_mean": float(np.mean(errors)), "rmse_p95": float(np.quantile(errors, 0.95)),
            "carrier_rmse_mean": float(np.mean(carrier_errors)),
            "raw_cnf_violation_rate": float(np.mean(raw_invalid)),
            "projected_cnf_violation_rate": float(np.mean(repaired_invalid)),
            "repair_displacement_mean": float(np.mean(displacements)),
        })
        rows.extend([
            {
                "seed": seed, "dimension": dimension, "method": "vsa_grid_consequent",
                "rmse_mean": float(np.mean(vsa_grid_errors)), "rmse_p95": float(np.quantile(vsa_grid_errors, 0.95)),
                "carrier_rmse_mean": 0.0, "raw_cnf_violation_rate": 0.0,
                "projected_cnf_violation_rate": 0.0, "repair_displacement_mean": 0.0,
            },
            {
                "seed": seed, "dimension": dimension, "method": "cp_hdfri_actual",
                "rmse_mean": float(np.mean(end_to_end_errors)), "rmse_p95": float(np.quantile(end_to_end_errors, 0.95)),
                "carrier_rmse_mean": float(np.mean(end_to_end_carrier)),
                "raw_cnf_violation_rate": float(np.mean(end_to_end_raw_invalid)),
                "projected_cnf_violation_rate": float(np.mean(end_to_end_repaired_invalid)),
                "repair_displacement_mean": float(np.mean(end_to_end_displacements)),
            },
        ])
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["method"], row["dimension"])].append(row)
    summary = []
    for (method, dimension), subset in sorted(buckets.items(), key=lambda item: item[0][1]):
        summary.append({
            "method": method, "dimension": dimension, "seeds": len(subset),
            "rmse_mean": float(np.mean([r["rmse_mean"] for r in subset])),
            "rmse_seed_std": float(np.std([r["rmse_mean"] for r in subset], ddof=1)),
            "rmse_p95_mean": float(np.mean([r["rmse_p95"] for r in subset])),
            "carrier_rmse_mean": float(np.mean([r["carrier_rmse_mean"] for r in subset])),
            "raw_cnf_violation_rate": float(np.mean([r["raw_cnf_violation_rate"] for r in subset])),
            "projected_cnf_violation_rate": float(np.mean([r["projected_cnf_violation_rate"] for r in subset])),
            "repair_displacement_mean": float(np.mean([r["repair_displacement_mean"] for r in subset])),
        })
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = [row for seed in SEEDS for row in run_seed(seed)]
    summary = aggregate(rows)
    payload = {"seeds": list(SEEDS), "queries_per_seed": QUERIES, "sparse_rules": SPARSE_RULES,
               "length_scale": LENGTH_SCALE, "alpha_points": len(ALPHA), "rows": rows, "summary": summary}
    (out_dir / "actual_hdt_sparse_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, data in (("actual_hdt_sparse_by_seed.csv", rows), ("actual_hdt_sparse_summary.csv", summary)):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    hdt = [r for r in summary if r["method"] == "actual_hdt_euclidean"]
    end_to_end = [r for r in summary if r["method"] == "cp_hdfri_actual"]
    vsa_grid = [r for r in summary if r["method"] == "vsa_grid_consequent"]
    grid = next(r for r in summary if r["method"] == "grid_euclidean")
    ax.errorbar([r["dimension"] for r in hdt], [r["rmse_mean"] for r in hdt],
                yerr=[r["rmse_seed_std"] for r in hdt], fmt="o-", capsize=3, label="paper-faithful HDT + CP")
    ax.plot([r["dimension"] for r in vsa_grid], [r["rmse_mean"] for r in vsa_grid], "^-", label="VSA + grid consequent")
    ax.plot([r["dimension"] for r in end_to_end], [r["rmse_mean"] for r in end_to_end], "s-", label="end-to-end CP-HDFRI")
    ax.axhline(grid["rmse_mean"], color="#64748b", linestyle="--", label="direct alpha-cut grid")
    ax.set_xscale("log", base=2); ax.set_xlabel("HDT dimension D"); ax.set_ylabel("Endpoint RMSE")
    ax.grid(alpha=0.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(out_dir / "actual_hdt_sparse_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "files": [str(out_dir / "actual_hdt_sparse_results.json"), str(out_dir / "actual_hdt_sparse_summary.csv"), str(out_dir / "actual_hdt_sparse_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
