"""Rule-density sensitivity for the paper-faithful fault-diagnosis case.

The experiment uses nested subsets of the same 27-rule linguistic grid so that
each seed compares methods and rule counts on identical queries.  CP-HDFRI is
evaluated at the manuscript's principal dimension, D=4096.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA, interval_encoding, transform_operator
from fault_diagnosis_case import severity
from preexperiments import project_cnf, violations
from synthetic_benchmark import cosine_matrix, vsa_encoder, weighted_neighbours


SEEDS = (211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
         269, 271, 277, 281, 283, 293, 307, 311, 313, 317)
RULE_COUNTS = (6, 9, 12, 18, 24)
QUERIES = 500
DIMENSION = 4096
K = 4
LENGTH_SCALE = 0.20


def consequent(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center, width = severity(points)
    left = center[:, None] - width[:, None] + ALPHA[None, :] * width[:, None]
    right = center[:, None] + width[:, None] - ALPHA[None, :] * width[:, None]
    return left, right


def invalid(left: np.ndarray, right: np.ndarray) -> bool:
    return any(violations(left, right).values())


def row(seed: int, rule_count: int, method: str, errors: list[float],
        raw_invalid: list[bool] | None = None,
        projected_invalid: list[bool] | None = None,
        carrier_errors: list[float] | None = None,
        displacements: list[float] | None = None) -> dict:
    return {
        "seed": seed,
        "rule_count": rule_count,
        "rule_density": rule_count / 27.0,
        "dimension": 0 if method == "grid_euclidean" else DIMENSION,
        "method": method,
        "rmse_mean": float(np.mean(errors)),
        "rmse_p95": float(np.quantile(errors, 0.95)),
        "carrier_rmse_mean": float(np.mean(carrier_errors)) if carrier_errors else 0.0,
        "raw_cnf_violation_rate": float(np.mean(raw_invalid)) if raw_invalid else 0.0,
        "projected_cnf_violation_rate": float(np.mean(projected_invalid)) if projected_invalid else 0.0,
        "repair_displacement_mean": float(np.mean(displacements)) if displacements else 0.0,
    }


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    grid = np.array([(a, b, c) for a in (0.15, 0.5, 0.85)
                     for b in (0.15, 0.5, 0.85)
                     for c in (0.15, 0.5, 0.85)])
    order = rng.permutation(len(grid))
    queries = rng.uniform(0.0, 1.0, size=(QUERIES, 3))
    true_left, true_right = consequent(queries)
    grid_left, grid_right = consequent(grid)

    encoder_rng = np.random.default_rng(seed * 30011 + DIMENSION)
    hdt_encoding, _ = interval_encoding(encoder_rng, DIMENSION, LENGTH_SCALE)
    operator = transform_operator(hdt_encoding)
    vsa_encode = vsa_encoder(
        np.random.default_rng(seed * 40009 + DIMENSION), DIMENSION, variables=3
    )
    grid_h = vsa_encode(grid)
    query_h = vsa_encode(queries)

    rows: list[dict] = []
    for rule_count in RULE_COUNTS:
        keep = np.sort(order[:rule_count])
        rules = grid[keep]
        rule_left, rule_right = grid_left[keep], grid_right[keep]
        similarities = cosine_matrix(query_h, grid_h[keep])

        grid_errors: list[float] = []
        vsa_errors: list[float] = []
        cp_errors: list[float] = []
        carrier_errors: list[float] = []
        raw_invalid: list[bool] = []
        projected_invalid: list[bool] = []
        displacements: list[float] = []
        for i, query in enumerate(queries):
            idx, weights = weighted_neighbours(
                np.linalg.norm(rules - query, axis=1), K, "euclidean"
            )
            direct_l, direct_r = weights @ rule_left[idx], weights @ rule_right[idx]
            grid_errors.append(float(np.sqrt(np.mean(
                np.r_[direct_l - true_left[i], direct_r - true_right[i]] ** 2
            ))))

            idx, weights = weighted_neighbours(similarities[i], K, "vsa")
            vsa_l, vsa_r = weights @ rule_left[idx], weights @ rule_right[idx]
            vsa_errors.append(float(np.sqrt(np.mean(
                np.r_[vsa_l - true_left[i], vsa_r - true_right[i]] ** 2
            ))))
            provisional_l, provisional_r = operator @ vsa_l, operator @ vsa_r
            out_l, out_r = project_cnf(provisional_l, provisional_r)
            cp_errors.append(float(np.sqrt(np.mean(
                np.r_[out_l - true_left[i], out_r - true_right[i]] ** 2
            ))))
            carrier_errors.append(float(np.sqrt(np.mean(
                np.r_[out_l - vsa_l, out_r - vsa_r] ** 2
            ))))
            raw_invalid.append(invalid(provisional_l, provisional_r))
            projected_invalid.append(invalid(out_l, out_r))
            displacements.append(float(np.sqrt(np.mean(
                np.r_[out_l - provisional_l, out_r - provisional_r] ** 2
            ))))

        rows.extend([
            row(seed, rule_count, "grid_euclidean", grid_errors),
            row(seed, rule_count, "vsa_grid_consequent", vsa_errors),
            row(seed, rule_count, "cp_hdfri_actual", cp_errors,
                raw_invalid, projected_invalid, carrier_errors, displacements),
        ])
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in rows:
        groups[(item["method"], item["rule_count"])].append(item)
    summary = []
    for (method, rule_count), subset in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        summary.append({
            "method": method,
            "rule_count": rule_count,
            "rule_density": rule_count / 27.0,
            "dimension": 0 if method == "grid_euclidean" else DIMENSION,
            "seeds": len(subset),
            "rmse_mean": float(np.mean([r["rmse_mean"] for r in subset])),
            "rmse_seed_std": float(np.std([r["rmse_mean"] for r in subset], ddof=1)),
            "rmse_p95_mean": float(np.mean([r["rmse_p95"] for r in subset])),
            "carrier_rmse_mean": float(np.mean([r["carrier_rmse_mean"] for r in subset])),
            "raw_cnf_violation_rate": float(np.mean([r["raw_cnf_violation_rate"] for r in subset])),
            "projected_cnf_violation_rate": float(np.mean([r["projected_cnf_violation_rate"] for r in subset])),
            "repair_displacement_mean": float(np.mean([r["repair_displacement_mean"] for r in subset])),
        })
    return summary


def paired_statistics(rows: list[dict]) -> list[dict]:
    lookup: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for item in rows:
        lookup[(item["method"], item["rule_count"])][item["seed"]] = item["rmse_mean"]
    comparisons = []
    for rule_count in RULE_COUNTS:
        for baseline in ("grid_euclidean", "vsa_grid_consequent"):
            cp = lookup[("cp_hdfri_actual", rule_count)]
            control = lookup[(baseline, rule_count)]
            seeds = sorted(set(cp) & set(control))
            differences = np.array([cp[seed] - control[seed] for seed in seeds])
            rng = np.random.default_rng(20260911 + 101 * rule_count + len(baseline))
            sample_idx = rng.integers(0, len(seeds), size=(50000, len(seeds)))
            bootstrap_means = differences[sample_idx].mean(axis=1)
            positive = int(np.sum(differences > 0))
            negative = int(np.sum(differences < 0))
            nonzero = positive + negative
            tail = min(positive, negative)
            sign_p = min(
                1.0,
                2.0 * sum(comb(nonzero, i) for i in range(tail + 1)) / (2 ** nonzero),
            ) if nonzero else 1.0
            comparisons.append({
                "rule_count": rule_count,
                "baseline": baseline,
                "paired_seeds": len(seeds),
                "mean_rmse_difference_cp_minus_baseline": float(np.mean(differences)),
                "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                "cp_better_fraction": float(np.mean(differences < 0)),
                "two_sided_sign_test_p": float(sign_p),
                "paired_standardized_difference": float(
                    np.mean(differences) / np.std(differences, ddof=1)
                ),
            })
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = [item for seed in SEEDS for item in run_seed(seed)]
    summary = aggregate(rows)
    paired = paired_statistics(rows)
    payload = {
        "seeds": list(SEEDS),
        "queries_per_seed": QUERIES,
        "dense_rules": 27,
        "rule_counts": list(RULE_COUNTS),
        "dimension": DIMENSION,
        "nearest_rules": K,
        "length_scale": LENGTH_SCALE,
        "alpha_points": len(ALPHA),
        "nested_rule_subsets": True,
        "rows": rows,
        "summary": summary,
        "paired_statistics": paired,
    }
    (root / "fault_sparsity_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(root / "fault_sparsity_by_seed.csv", rows)
    write_csv(root / "fault_sparsity_summary.csv", summary)
    write_csv(root / "fault_sparsity_paired_stats.csv", paired)

    fig, (ax_rmse, ax_cnf) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    styles = {
        "grid_euclidean": ("o", "#64748b", "Direct Euclidean"),
        "vsa_grid_consequent": ("^", "#b45309", "VSA + grid consequent"),
        "cp_hdfri_actual": ("s", "#2563eb", "CP-HDFRI"),
    }
    for method, (marker, color, label) in styles.items():
        subset = [r for r in summary if r["method"] == method]
        ax_rmse.errorbar(
            [r["rule_count"] for r in subset],
            [r["rmse_mean"] for r in subset],
            yerr=[r["rmse_seed_std"] for r in subset],
            marker=marker, color=color, capsize=3, label=label,
        )
    cp = [r for r in summary if r["method"] == "cp_hdfri_actual"]
    ax_cnf.plot([r["rule_count"] for r in cp],
                [r["raw_cnf_violation_rate"] for r in cp],
                "o-", color="#b91c1c", label="Raw decode")
    ax_cnf.plot([r["rule_count"] for r in cp],
                [r["projected_cnf_violation_rate"] for r in cp],
                "s-", color="#15803d", label="After projection")
    ax_rmse.set_xlabel("Retained rules (of 27)")
    ax_rmse.set_ylabel("Endpoint RMSE")
    ax_cnf.set_xlabel("Retained rules (of 27)")
    ax_cnf.set_ylabel("CNF violation rate")
    for ax in (ax_rmse, ax_cnf):
        ax.set_xticks(RULE_COUNTS)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    for i, ax in enumerate((ax_rmse, ax_cnf)): ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(root / "fault_sparsity_diagnostics.png", dpi=300)
    plt.close(fig)
    print(json.dumps({"summary": summary, "paired_statistics": paired}, indent=2))


if __name__ == "__main__":
    main()
