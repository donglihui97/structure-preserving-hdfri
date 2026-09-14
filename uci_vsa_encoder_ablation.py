"""VSA antecedent-encoder ablation for the UCI WDBC benchmark.

This isolates retrieval from the HDT consequent carrier.  K-means prototypes,
scaling, and splits match ``uci_wdbc_benchmark.py``; only the VSA encoder and
its frequency scale change.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from uci_wdbc_benchmark import (
    DIMENSION, NEIGHBOURS, RULES, SEEDS, TEST_SIZE, WEIGHT_TEMPERATURE,
    endpoint_rules, load_wdbc, weights,
)


MODES = ("sum", "block")
FREQUENCY_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)


def encoder(rng: np.random.Generator, variables: int, mode: str, scale: float):
    if mode == "sum":
        frequencies = rng.normal(0.0, scale, size=(variables, DIMENSION))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(variables, DIMENSION))
        roles = rng.choice((-1.0, 1.0), size=(variables, DIMENSION))

        def encode(points: np.ndarray) -> np.ndarray:
            result = np.zeros((len(points), DIMENSION), dtype=np.float64)
            for j in range(variables):
                result += np.cos(points[:, j:j + 1] * frequencies[j] + phases[j]) * roles[j]
            return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)

        return encode

    block = DIMENSION // variables
    frequencies = rng.normal(0.0, scale, size=(variables, block))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(variables, block))
    roles = rng.choice((-1.0, 1.0), size=(variables, block))

    def encode(points: np.ndarray) -> np.ndarray:
        result = np.zeros((len(points), block * variables), dtype=np.float64)
        for j in range(variables):
            result[:, j * block:(j + 1) * block] = (
                np.cos(points[:, j:j + 1] * frequencies[j] + phases[j]) * roles[j]
            )
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)

    return encode


def run_seed(seed: int, features: np.ndarray, labels: np.ndarray) -> list[dict]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(splitter.split(features, labels))
    scaler = MinMaxScaler(clip=True)
    x_train = scaler.fit_transform(features[train_idx])
    x_test = scaler.transform(features[test_idx])
    y_train, y_test = labels[train_idx], labels[test_idx]
    kmeans = KMeans(n_clusters=RULES, n_init=20, random_state=seed)
    assignments = kmeans.fit_predict(x_train)
    counts = np.bincount(assignments, minlength=RULES).astype(float)
    positives = np.bincount(assignments, weights=y_train, minlength=RULES)
    posterior = (positives + 1.0) / (counts + 2.0)
    rule_left, rule_right = endpoint_rules(posterior, counts)
    rows = []
    for mode in MODES:
        for scale in FREQUENCY_SCALES:
            # The sum/2.0 setting is bit-for-bit identical to the main WDBC
            # benchmark. Other settings receive deterministic offsets.
            offset = int(round((scale - 2.0) * 100)) * 101 + (0 if mode == "sum" else 1)
            rng = np.random.default_rng(seed * 40009 + DIMENSION + offset)
            encode = encoder(rng, x_train.shape[1], mode, scale)
            center_h = encode(kmeans.cluster_centers_)
            query_h = encode(x_test)
            similarity = query_h @ center_h.T
            probability = []
            for row in similarity:
                idx, blend = weights(row, largest=True)
                probability.append(float(np.mean((blend @ rule_left[idx] + blend @ rule_right[idx]) / 2.0)))
            probability = np.clip(np.asarray(probability), 0.0, 1.0)
            rows.append({
                "seed": seed,
                "mode": mode,
                "frequency_scale": scale,
                "test_cases": len(y_test),
                "brier_score": float(brier_score_loss(y_test, probability)),
                "roc_auc": float(roc_auc_score(y_test, probability)),
                "accuracy": float(accuracy_score(y_test, probability >= 0.5)),
            })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["frequency_scale"])].append(row)
    result = []
    for (mode, scale), subset in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        result.append({
            "mode": mode,
            "frequency_scale": scale,
            "seeds": len(subset),
            "brier_score_mean": float(np.mean([r["brier_score"] for r in subset])),
            "brier_score_seed_std": float(np.std([r["brier_score"] for r in subset], ddof=1)),
            "roc_auc_mean": float(np.mean([r["roc_auc"] for r in subset])),
            "roc_auc_seed_std": float(np.std([r["roc_auc"] for r in subset], ddof=1)),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in subset])),
            "accuracy_seed_std": float(np.std([r["accuracy"] for r in subset], ddof=1)),
        })
    return result


def paired(rows: list[dict]) -> list[dict]:
    current = {(r["seed"], r["frequency_scale"]): r for r in rows if r["mode"] == "sum"}
    result = []
    reference_scale = 2.0
    for row in aggregate(rows):
        if row["mode"] == "sum" and row["frequency_scale"] == reference_scale:
            continue
        for metric in ("brier_score", "roc_auc"):
            diffs = np.array([
                next(r[metric] for r in rows
                     if r["seed"] == seed and r["mode"] == row["mode"]
                     and r["frequency_scale"] == row["frequency_scale"])
                - current[(seed, reference_scale)][metric]
                for seed in SEEDS
            ])
            result.append({
                "mode": row["mode"],
                "frequency_scale": row["frequency_scale"],
                "metric": metric,
                "paired_seeds": len(SEEDS),
                "mean_difference_vs_sum_scale_2": float(diffs.mean()),
                "cp_better_fraction": float(np.mean(diffs < 0 if metric == "brier_score" else diffs > 0)),
            })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    features, labels = load_wdbc(root / "data" / "wdbc.data")
    rows = [row for seed in SEEDS for row in run_seed(seed, features, labels)]
    summary = aggregate(rows)
    comparisons = paired(rows)
    payload = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "seeds": list(SEEDS),
        "test_size": TEST_SIZE,
        "rules": RULES,
        "nearest_rules": NEIGHBOURS,
        "dimension": DIMENSION,
        "weight_temperature": WEIGHT_TEMPERATURE,
        "modes": list(MODES),
        "frequency_scales": list(FREQUENCY_SCALES),
        "rows": rows,
        "summary": summary,
        "paired_comparisons": comparisons,
    }
    (root / "uci_vsa_encoder_ablation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(root / "uci_vsa_encoder_ablation_by_seed.csv", rows)
    write_csv(root / "uci_vsa_encoder_ablation_summary.csv", summary)
    write_csv(root / "uci_vsa_encoder_ablation_paired.csv", comparisons)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for mode, color in (("sum", "#64748b"), ("block", "#2563eb")):
        subset = [r for r in summary if r["mode"] == mode]
        axes[0].errorbar([r["frequency_scale"] for r in subset], [r["roc_auc_mean"] for r in subset],
                         yerr=[r["roc_auc_seed_std"] for r in subset], marker="o", capsize=3,
                         color=color, label=mode)
        axes[1].errorbar([r["frequency_scale"] for r in subset], [r["brier_score_mean"] for r in subset],
                         yerr=[r["brier_score_seed_std"] for r in subset], marker="o", capsize=3,
                         color=color, label=mode)
    axes[0].set_xscale("log", base=2); axes[0].set_ylim(0.94, 1.00); axes[0].set_ylabel("ROC AUC")
    axes[1].set_xscale("log", base=2); axes[1].set_ylabel("Brier score")
    for ax in axes:
        ax.set_xlabel("VSA frequency scale"); ax.set_xticks(FREQUENCY_SCALES)
        ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(root / "uci_vsa_encoder_ablation_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "paired_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
