"""Neighbour-count sensitivity for the paper-faithful sparse-rule benchmark."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from actual_hdt_benchmark import ALPHA as HDT_ALPHA, interval_encoding, transform_operator
from preexperiments import project_cnf, violations
from synthetic_benchmark import cosine_matrix, truth, vsa_encoder


ALPHA = HDT_ALPHA


SEEDS = (19, 23, 29, 31, 37)
QUERIES = 600
SPARSE_RULES = 30
NEIGHBOURS = (1, 2, 4, 8)
DIMENSION = 4096
LENGTH_SCALE = 0.20


def endpoints(center: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    return center - width + ALPHA * width, center + width - ALPHA * width


def choose(scores: np.ndarray, k: int, similarity: bool) -> tuple[np.ndarray, np.ndarray]:
    if similarity:
        idx = np.argpartition(-scores, k - 1)[:k]
        logits = 10.0 * scores[idx]
    else:
        idx = np.argpartition(scores, k - 1)[:k]
        logits = -10.0 * scores[idx]
    logits -= np.max(logits)
    weights = np.exp(logits)
    return idx, weights / np.sum(weights)


def invalid(left: np.ndarray, right: np.ndarray) -> bool:
    return any(violations(left, right).values())


def run_seed(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    lattice = np.array([(x, y) for x in np.linspace(0.0, 1.0, 9) for y in np.linspace(0.0, 1.0, 9)])
    rules = lattice[rng.choice(len(lattice), size=SPARSE_RULES, replace=False)]
    queries = rng.uniform(0.0, 1.0, size=(QUERIES, 2))
    rule_center, rule_width = truth(rules)
    query_center, query_width = truth(queries)
    rule_left = np.stack([endpoints(c, w)[0] for c, w in zip(rule_center, rule_width)])
    rule_right = np.stack([endpoints(c, w)[1] for c, w in zip(rule_center, rule_width)])
    true_left = np.stack([endpoints(c, w)[0] for c, w in zip(query_center, query_width)])
    true_right = np.stack([endpoints(c, w)[1] for c, w in zip(query_center, query_width)])
    encoder = vsa_encoder(np.random.default_rng(seed * 20011 + DIMENSION), DIMENSION)
    similarities = cosine_matrix(encoder(queries), encoder(rules))
    hdt_rng = np.random.default_rng(seed * 10007 + DIMENSION)
    encoding, _ = interval_encoding(hdt_rng, DIMENSION, LENGTH_SCALE)
    operator = transform_operator(encoding)
    rows: list[dict] = []
    for k in NEIGHBOURS:
        euclidean_errors, vsa_errors, cp_errors = [], [], []
        cp_invalid = []
        distances = np.linalg.norm(rules[None, :, :] - queries[:, None, :], axis=2)
        for i in range(QUERIES):
            idx, weights = choose(distances[i], k, False)
            grid_l, grid_r = weights @ rule_left[idx], weights @ rule_right[idx]
            euclidean_errors.append(float(np.sqrt(np.mean(np.r_[grid_l - true_left[i], grid_r - true_right[i]] ** 2))))
            idx, weights = choose(similarities[i], k, True)
            vsa_l, vsa_r = weights @ rule_left[idx], weights @ rule_right[idx]
            vsa_errors.append(float(np.sqrt(np.mean(np.r_[vsa_l - true_left[i], vsa_r - true_right[i]] ** 2))))
            raw_l, raw_r = operator @ vsa_l, operator @ vsa_r
            out_l, out_r = project_cnf(raw_l, raw_r)
            cp_errors.append(float(np.sqrt(np.mean(np.r_[out_l - true_left[i], out_r - true_right[i]] ** 2))))
            cp_invalid.append(invalid(out_l, out_r))
        rows.extend([
            {"seed": seed, "neighbours": k, "method": "grid_euclidean", "rmse_mean": float(np.mean(euclidean_errors)), "rmse_p95": float(np.quantile(euclidean_errors, .95)), "cnf_violation_rate": 0.0},
            {"seed": seed, "neighbours": k, "method": "vsa_grid_consequent", "rmse_mean": float(np.mean(vsa_errors)), "rmse_p95": float(np.quantile(vsa_errors, .95)), "cnf_violation_rate": 0.0},
            {"seed": seed, "neighbours": k, "method": "cp_hdfri", "rmse_mean": float(np.mean(cp_errors)), "rmse_p95": float(np.quantile(cp_errors, .95)), "cnf_violation_rate": float(np.mean(cp_invalid))},
        ])
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["neighbours"])].append(row)
    return [{"method": method, "neighbours": neighbours, "seeds": len(subset), "rmse_mean": float(np.mean([r["rmse_mean"] for r in subset])), "rmse_seed_std": float(np.std([r["rmse_mean"] for r in subset], ddof=1)), "rmse_p95_mean": float(np.mean([r["rmse_p95"] for r in subset])), "cnf_violation_rate": float(np.mean([r["cnf_violation_rate"] for r in subset]))} for (method, neighbours), subset in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0]))]


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = [row for seed in SEEDS for row in run_seed(seed)]
    summary = aggregate(rows)
    payload = {"seeds": list(SEEDS), "queries_per_seed": QUERIES, "sparse_rules": SPARSE_RULES, "dimension": DIMENSION, "length_scale": LENGTH_SCALE, "rows": rows, "summary": summary}
    (root / "neighbour_sensitivity_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, data in (("neighbour_sensitivity_by_seed.csv", rows), ("neighbour_sensitivity_summary.csv", summary)):
        with (root / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys()); writer.writeheader(); writer.writerows(data)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for method, marker in (("grid_euclidean", "o"), ("vsa_grid_consequent", "^"), ("cp_hdfri", "s")):
        series = [r for r in summary if r["method"] == method]
        ax.errorbar([r["neighbours"] for r in series], [r["rmse_mean"] for r in series], yerr=[r["rmse_seed_std"] for r in series], marker=marker, capsize=3, label=method)
    ax.set_xlabel("number of retrieved rules k"); ax.set_ylabel("endpoint RMSE"); ax.set_xticks(NEIGHBOURS); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(root / "neighbour_sensitivity_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
