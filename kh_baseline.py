"""Transparent 1-D KH-style alpha-cut interpolation baseline.

For two bracketing antecedent rules, KH interpolation uses relative distances
to compute non-negative weights and linearly combines consequent alpha-cut
endpoints.  This script uses triangular fuzzy sets and reports the baseline
error on a nonlinear continuous generator.  It is a pilot implementation, not
a claim of complete reproduction of every KH variant.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALPHA = np.linspace(0.0, 1.0, 11)


def consequent(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = 0.2 + 0.6 * x + 0.06 * np.sin(2.0 * np.pi * x)
    width = 0.04 + 0.01 * x
    left = center[:, None] - width[:, None] + ALPHA[None, :] * width[:, None]
    right = center[:, None] + width[:, None] - ALPHA[None, :] * width[:, None]
    return center, left, right


def kh_pair(query: float, rules: np.ndarray, rule_left: np.ndarray, rule_right: np.ndarray):
    order = np.argsort(rules)
    sorted_rules = rules[order]
    pos = int(np.searchsorted(sorted_rules, query))
    if pos <= 0:
        idx = order[:2]
    elif pos >= len(sorted_rules):
        idx = order[-2:]
    else:
        idx = np.array([order[pos - 1], order[pos]])
    d = np.abs(rules[idx] - query)
    if float(np.sum(d)) < 1e-12:
        weights = np.array([0.5, 0.5])
    else:
        weights = np.array([d[1], d[0]]) / np.sum(d)
    return weights @ rule_left[idx], weights @ rule_right[idx], idx, weights


def run(seed: int = 101, repeats: int = 5, queries_n: int = 800) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    for rep in range(repeats):
        lattice = np.linspace(0.0, 1.0, 21)
        keep = rng.choice(len(lattice), size=12, replace=False)
        rules = lattice[np.sort(keep)]
        _, rule_left, rule_right = consequent(rules)
        queries = rng.uniform(0.0, 1.0, queries_n)
        _, true_left, true_right = consequent(queries)
        errors = []
        for q, ideal_l, ideal_r in zip(queries, true_left, true_right):
            out_l, out_r, _, _ = kh_pair(q, rules, rule_left, rule_right)
            errors.append(float(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2))))
        records.append({"rep": rep, "rmse_mean": float(np.mean(errors)),
                        "rmse_p95": float(np.quantile(errors, 0.95)), "rules": len(rules)})
    return {"seed": seed, "repeats": repeats, "queries_per_repeat": queries_n,
            "alpha_points": len(ALPHA), "records": records,
            "summary": {"rmse_mean": float(np.mean([r["rmse_mean"] for r in records])),
                        "rmse_std": float(np.std([r["rmse_mean"] for r in records], ddof=1)),
                        "rmse_p95_mean": float(np.mean([r["rmse_p95"] for r in records]))}}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    payload = run()
    (out_dir / "kh_baseline_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "kh_baseline_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload["records"][0].keys())
        writer.writeheader(); writer.writerows(payload["records"])
    print(json.dumps({"summary": payload["summary"], "files": [str(out_dir / "kh_baseline_results.json"),
        str(out_dir / "kh_baseline_summary.csv")]}, indent=2))


if __name__ == "__main__":
    main()
