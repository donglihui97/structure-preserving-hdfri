"""Rule-count sensitivity benchmark on the leakage-safe UCI WDBC protocol.

The split, scaling, consequent construction, retrieval, and CP decoder match
``uci_wdbc_benchmark.py``.  Only the number of training-derived prototypes is
varied, so the experiment measures knowledge-base coverage directly.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from actual_hdt_benchmark import ALPHA, interval_encoding, transform_operator
from preexperiments import project_cnf, violations
from uci_wdbc_benchmark import DIMENSION, NEIGHBOURS, SEEDS, TEST_SIZE, VSA_FREQUENCY_SCALE, WEIGHT_TEMPERATURE, endpoint_rules, load_wdbc, vsa_encoder


RULE_COUNTS = (8, 16, 32, 64)


def weights_for(values: np.ndarray, largest: bool, neighbours: int = NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    if largest:
        idx = np.argpartition(-values, neighbours - 1)[:neighbours]
        selected = values[idx]
    else:
        idx = np.argpartition(values, neighbours - 1)[:neighbours]
        selected = -values[idx]
    logits = WEIGHT_TEMPERATURE * (selected - np.max(selected))
    score = np.exp(logits)
    return idx, score / score.sum()


def invalid(left: np.ndarray, right: np.ndarray) -> bool:
    return any(violations(left, right).values())


def metric_row(seed: int, rules: int, method: str, truth: np.ndarray,
               probability: np.ndarray, elapsed: float,
               raw_invalid: list[bool] | None = None,
               projected_invalid: list[bool] | None = None,
               repair_displacement: list[float] | None = None,
               carrier_rmse: list[float] | None = None) -> dict:
    probability = np.clip(probability, 0.0, 1.0)
    return {
        "seed": seed,
        "rules": rules,
        "method": method,
        "test_cases": len(truth),
        "brier_score": float(brier_score_loss(truth, probability)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "accuracy": float(accuracy_score(truth, probability >= 0.5)),
        "milliseconds_per_query": 1000.0 * elapsed / len(truth),
        "raw_cnf_violation_rate": float(np.mean(raw_invalid)) if raw_invalid else 0.0,
        "projected_cnf_violation_rate": float(np.mean(projected_invalid)) if projected_invalid else 0.0,
        "repair_displacement_mean": float(np.mean(repair_displacement)) if repair_displacement else 0.0,
        "carrier_rmse_mean": float(np.mean(carrier_rmse)) if carrier_rmse else 0.0,
    }


def run_seed(seed: int, features: np.ndarray, labels: np.ndarray) -> list[dict]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(splitter.split(features, labels))
    scaler = MinMaxScaler(clip=True)
    x_train = scaler.fit_transform(features[train_idx])
    x_test = scaler.transform(features[test_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]
    rows: list[dict] = []

    for rules in RULE_COUNTS:
        kmeans = KMeans(n_clusters=rules, n_init=20, random_state=seed)
        assignments = kmeans.fit_predict(x_train)
        centers = kmeans.cluster_centers_
        counts = np.bincount(assignments, minlength=rules).astype(float)
        positives = np.bincount(assignments, weights=y_train, minlength=rules)
        posterior = (positives + 1.0) / (counts + 2.0)
        rule_left, rule_right = endpoint_rules(posterior, counts)

        start = time.perf_counter()
        grid_probability = []
        for query in x_test:
            idx, blend = weights_for(np.linalg.norm(centers - query, axis=1), largest=False)
            out_l, out_r = blend @ rule_left[idx], blend @ rule_right[idx]
            grid_probability.append(float(np.mean((out_l + out_r) / 2.0)))
        rows.append(metric_row(seed, rules, "prototype_grid", y_test,
                               np.asarray(grid_probability), time.perf_counter() - start))

        encode = vsa_encoder(np.random.default_rng(seed * 40009 + DIMENSION), x_train.shape[1])
        center_h = encode(centers)
        start = time.perf_counter()
        test_h = encode(x_test)
        similarities = test_h @ center_h.T
        vsa_probability = []
        vsa_left, vsa_right = [], []
        for similarity in similarities:
            idx, blend = weights_for(similarity, largest=True)
            out_l, out_r = blend @ rule_left[idx], blend @ rule_right[idx]
            vsa_left.append(out_l)
            vsa_right.append(out_r)
            vsa_probability.append(float(np.mean((out_l + out_r) / 2.0)))
        vsa_elapsed = time.perf_counter() - start
        rows.append(metric_row(seed, rules, "prototype_vsa_grid", y_test,
                               np.asarray(vsa_probability), vsa_elapsed))

        encoding, _ = interval_encoding(
            np.random.default_rng(seed * 30011 + DIMENSION), DIMENSION, 0.20
        )
        operator = transform_operator(encoding)
        cp_probability, raw_invalid, projected_invalid = [], [], []
        displacements, carrier_errors = [], []
        start = time.perf_counter()
        for direct_l, direct_r in zip(vsa_left, vsa_right):
            provisional_l, provisional_r = operator @ direct_l, operator @ direct_r
            out_l, out_r = project_cnf(provisional_l, provisional_r)
            cp_probability.append(float(np.mean((out_l + out_r) / 2.0)))
            raw_invalid.append(invalid(provisional_l, provisional_r))
            projected_invalid.append(invalid(out_l, out_r))
            displacements.append(float(np.sqrt(np.mean(np.r_[out_l - provisional_l, out_r - provisional_r] ** 2))))
            carrier_errors.append(float(np.sqrt(np.mean(np.r_[out_l - direct_l, out_r - direct_r] ** 2))))
        cp_elapsed = time.perf_counter() - start + vsa_elapsed
        rows.append(metric_row(seed, rules, "cp_hdfri_actual", y_test,
                               np.asarray(cp_probability), cp_elapsed,
                               raw_invalid, projected_invalid,
                               displacements, carrier_errors))
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["rules"], row["method"])].append(row)
    result = []
    for (rules, method), subset in sorted(groups.items()):
        result.append({
            "rules": rules,
            "method": method,
            "seeds": len(subset),
            "brier_score_mean": float(np.mean([r["brier_score"] for r in subset])),
            "brier_score_seed_std": float(np.std([r["brier_score"] for r in subset], ddof=1)),
            "roc_auc_mean": float(np.mean([r["roc_auc"] for r in subset])),
            "roc_auc_seed_std": float(np.std([r["roc_auc"] for r in subset], ddof=1)),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in subset])),
            "accuracy_seed_std": float(np.std([r["accuracy"] for r in subset], ddof=1)),
            "raw_cnf_violation_rate_mean": float(np.mean([r["raw_cnf_violation_rate"] for r in subset])),
            "projected_cnf_violation_rate_mean": float(np.mean([r["projected_cnf_violation_rate"] for r in subset])),
            "repair_displacement_mean": float(np.mean([r["repair_displacement_mean"] for r in subset])),
            "carrier_rmse_mean": float(np.mean([r["carrier_rmse_mean"] for r in subset])),
        })
    return result


def paired_statistics(rows: list[dict]) -> list[dict]:
    result = []
    for rules in RULE_COUNTS:
        grid = {(r["seed"], r["method"]): r for r in rows if r["rules"] == rules}
        for metric in ("brier_score", "roc_auc", "accuracy"):
            diffs = np.array([
                grid[(seed, "cp_hdfri_actual")][metric] - grid[(seed, "prototype_grid")][metric]
                for seed in SEEDS
            ])
            result.append({
                "rules": rules,
                "metric": metric,
                "baseline": "prototype_grid",
                "paired_seeds": len(SEEDS),
                "cp_minus_grid_mean": float(diffs.mean()),
                "cp_better_fraction": float(np.mean(diffs < 0 if metric != "accuracy" else diffs > 0)),
            })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    features, labels = load_wdbc(root / "data" / "wdbc.data")
    rows = [row for seed in SEEDS for row in run_seed(seed, features, labels)]
    summary = aggregate(rows)
    paired = paired_statistics(rows)
    payload = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "instances": int(features.shape[0]),
        "features": int(features.shape[1]),
        "seeds": list(SEEDS),
        "test_size": TEST_SIZE,
        "rule_counts": list(RULE_COUNTS),
        "nearest_rules": NEIGHBOURS,
        "dimension": DIMENSION,
        "vsa_frequency_scale": VSA_FREQUENCY_SCALE,
        "weight_temperature": WEIGHT_TEMPERATURE,
        "rows": rows,
        "summary": summary,
        "paired_statistics": paired,
    }
    (root / "uci_rule_count_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(root / "uci_rule_count_by_seed.csv", rows)
    write_csv(root / "uci_rule_count_summary.csv", summary)
    write_csv(root / "uci_rule_count_paired_stats.csv", paired)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for method, color, label in (
        ("prototype_grid", "#64748b", "grid rules"),
        ("prototype_vsa_grid", "#c2410c", "VSA+grid"),
        ("cp_hdfri_actual", "#2563eb", "CP-HDFRI"),
    ):
        subset = [r for r in summary if r["method"] == method]
        axes[0].errorbar([r["rules"] for r in subset], [r["roc_auc_mean"] for r in subset],
                         yerr=[r["roc_auc_seed_std"] for r in subset], marker="o",
                         color=color, label=label)
        axes[1].plot([r["rules"] for r in subset], [r["projected_cnf_violation_rate_mean"] for r in subset],
                     marker="o", color=color, label=label)
    axes[0].set_xlabel("Training-derived rules"); axes[0].set_ylabel("ROC AUC")
    axes[1].set_xlabel("Training-derived rules"); axes[1].set_ylabel("Projected CNF violation rate")
    axes[0].set_xticks(RULE_COUNTS); axes[1].set_xticks(RULE_COUNTS)
    axes[0].set_ylim(0.8, 1.0); axes[1].set_ylim(-0.02, 0.05)
    axes[0].legend(frameon=True, fontsize=8); axes[1].legend(frameon=True, fontsize=8)
    fig.suptitle("UCI WDBC rule-count sensitivity")
    fig.tight_layout()
    fig.savefig(root / "uci_rule_count_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
