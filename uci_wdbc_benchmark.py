"""Repeated holdout benchmark on UCI WDBC with data-derived fuzzy rules.

The experiment is deliberately leakage-safe: every split fits the scaler,
prototype locations, prototype consequents, and supervised baselines on the
training partition only.  The positive class is malignant.  This is a
classification benchmark for the inference mechanism, not a clinical claim.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from actual_hdt_benchmark import ALPHA, interval_encoding, transform_operator
from preexperiments import project_cnf, violations


SEEDS = (211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
         269, 271, 277, 281, 283, 293, 307, 311, 313, 317)
TEST_SIZE = 0.30
RULES = 16
NEIGHBOURS = 4
DIMENSION = 4096
LENGTH_SCALE = 0.20
VSA_FREQUENCY_SCALE = 2.0
WEIGHT_TEMPERATURE = 4.0
BOOTSTRAP_RESAMPLES = 50_000


def load_wdbc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(path, delimiter=",", dtype=str)
    if raw.shape != (569, 32):
        raise ValueError(f"unexpected WDBC shape: {raw.shape}")
    labels = (raw[:, 1] == "M").astype(int)
    features = raw[:, 2:].astype(float)
    if not np.isfinite(features).all():
        raise ValueError("WDBC contains a non-finite feature")
    return features, labels


def endpoint_rules(probability: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # A symmetric fuzzy probability consequent.  The half-width combines a
    # small irreducible width with the prototype's binomial uncertainty.
    uncertainty = 1.96 * np.sqrt(probability * (1.0 - probability) / np.maximum(counts + 2.0, 1.0))
    width = np.clip(0.03 + uncertainty, 0.03, 0.35)
    taper = 1.0 - ALPHA[None, :]
    left = np.clip(probability[:, None] - width[:, None] * taper, 0.0, 1.0)
    right = np.clip(probability[:, None] + width[:, None] * taper, 0.0, 1.0)
    return left, right


def weights(values: np.ndarray, largest: bool) -> tuple[np.ndarray, np.ndarray]:
    if largest:
        idx = np.argpartition(-values, NEIGHBOURS - 1)[:NEIGHBOURS]
        selected = values[idx]
    else:
        idx = np.argpartition(values, NEIGHBOURS - 1)[:NEIGHBOURS]
        selected = -values[idx]
    logits = WEIGHT_TEMPERATURE * (selected - np.max(selected))
    score = np.exp(logits)
    return idx, score / score.sum()


def vsa_encoder(rng: np.random.Generator, variables: int):
    frequencies = rng.normal(0.0, VSA_FREQUENCY_SCALE, size=(variables, DIMENSION))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(variables, DIMENSION))
    roles = rng.choice((-1.0, 1.0), size=(variables, DIMENSION))

    def encode(points: np.ndarray) -> np.ndarray:
        encoded = np.zeros((len(points), DIMENSION), dtype=np.float64)
        for j in range(variables):
            encoded += np.cos(points[:, j:j + 1] * frequencies[j] + phases[j]) * roles[j]
        encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)
        return encoded

    return encode


def invalid(left: np.ndarray, right: np.ndarray) -> bool:
    return any(violations(left, right).values())


def metrics(seed: int, method: str, truth: np.ndarray, probability: np.ndarray,
            elapsed: float, raw_invalid: list[bool] | None = None,
            projected_invalid: list[bool] | None = None,
            repair_displacement: list[float] | None = None,
            carrier_rmse: list[float] | None = None) -> dict:
    probability = np.clip(probability, 0.0, 1.0)
    return {
        "seed": seed,
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


def run_seed(seed: int, features: np.ndarray, labels: np.ndarray) -> tuple[list[dict], dict]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(splitter.split(features, labels))
    x_train_raw, x_test_raw = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    scaler = MinMaxScaler(clip=True)
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    kmeans = KMeans(n_clusters=RULES, n_init=20, random_state=seed)
    assignments = kmeans.fit_predict(x_train)
    centers = kmeans.cluster_centers_
    counts = np.bincount(assignments, minlength=RULES).astype(float)
    positives = np.bincount(assignments, weights=y_train, minlength=RULES)
    posterior = (positives + 1.0) / (counts + 2.0)
    rule_left, rule_right = endpoint_rules(posterior, counts)

    start = time.perf_counter()
    grid_probability = []
    for query in x_test:
        idx, blend = weights(np.linalg.norm(centers - query, axis=1), largest=False)
        out_l, out_r = blend @ rule_left[idx], blend @ rule_right[idx]
        grid_probability.append(float(np.mean((out_l + out_r) / 2.0)))
    grid_elapsed = time.perf_counter() - start

    encode = vsa_encoder(np.random.default_rng(seed * 40009 + DIMENSION), x_train.shape[1])
    center_h = encode(centers)
    start = time.perf_counter()
    test_h = encode(x_test)
    similarities = test_h @ center_h.T
    vsa_probability = []
    vsa_left, vsa_right = [], []
    for similarity in similarities:
        idx, blend = weights(similarity, largest=True)
        out_l, out_r = blend @ rule_left[idx], blend @ rule_right[idx]
        vsa_left.append(out_l)
        vsa_right.append(out_r)
        vsa_probability.append(float(np.mean((out_l + out_r) / 2.0)))
    vsa_elapsed = time.perf_counter() - start

    encoding, _ = interval_encoding(
        np.random.default_rng(seed * 30011 + DIMENSION), DIMENSION, LENGTH_SCALE
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

    logistic = LogisticRegression(max_iter=3000, random_state=seed)
    logistic.fit(x_train, y_train)
    start = time.perf_counter()
    logistic_probability = logistic.predict_proba(x_test)[:, 1]
    logistic_elapsed = time.perf_counter() - start

    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(x_train, y_train)
    start = time.perf_counter()
    knn_probability = knn.predict_proba(x_test)[:, 1]
    knn_elapsed = time.perf_counter() - start

    rows = [
        metrics(seed, "prototype_grid", y_test, np.asarray(grid_probability), grid_elapsed),
        metrics(seed, "prototype_vsa_grid", y_test, np.asarray(vsa_probability), vsa_elapsed),
        metrics(seed, "cp_hdfri_actual", y_test, np.asarray(cp_probability), cp_elapsed,
                raw_invalid, projected_invalid, displacements, carrier_errors),
        metrics(seed, "logistic_regression", y_test, logistic_probability, logistic_elapsed),
        metrics(seed, "knn_7", y_test, knn_probability, knn_elapsed),
    ]
    split = {
        "seed": seed,
        "training_cases": len(train_idx),
        "test_cases": len(test_idx),
        "training_malignant": int(y_train.sum()),
        "test_malignant": int(y_test.sum()),
        "prototype_min_members": int(counts.min()),
        "prototype_max_members": int(counts.max()),
        "train_indices_sha256": hashlib.sha256(train_idx.astype("<i8").tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(test_idx.astype("<i8").tobytes()).hexdigest(),
    }
    return rows, split


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    result = []
    for method in ("prototype_grid", "prototype_vsa_grid", "cp_hdfri_actual",
                   "logistic_regression", "knn_7"):
        subset = groups[method]
        result.append({
            "method": method,
            "seeds": len(subset),
            **{f"{metric}_mean": float(np.mean([row[metric] for row in subset]))
               for metric in ("brier_score", "roc_auc", "accuracy", "milliseconds_per_query",
                              "raw_cnf_violation_rate", "projected_cnf_violation_rate",
                              "repair_displacement_mean", "carrier_rmse_mean")},
            **{f"{metric}_seed_std": float(np.std([row[metric] for row in subset], ddof=1))
               for metric in ("brier_score", "roc_auc", "accuracy")},
        })
    return result


def paired_statistics(rows: list[dict]) -> list[dict]:
    lookup: dict[tuple[str, int], dict] = {(row["method"], row["seed"]): row for row in rows}
    result = []
    for metric in ("brier_score", "roc_auc"):
        for baseline in ("prototype_grid", "prototype_vsa_grid", "logistic_regression", "knn_7"):
            differences = np.array([
                lookup[("cp_hdfri_actual", seed)][metric] - lookup[(baseline, seed)][metric]
                for seed in SEEDS
            ])
            rng = np.random.default_rng(20260911 + len(metric) * 101 + len(baseline))
            sample_idx = rng.integers(0, len(SEEDS), size=(BOOTSTRAP_RESAMPLES, len(SEEDS)))
            bootstrap_means = differences[sample_idx].mean(axis=1)
            positive = int(np.sum(differences > 0)); negative = int(np.sum(differences < 0))
            nonzero = positive + negative; tail = min(positive, negative)
            sign_p = min(1.0, 2.0 * sum(comb(nonzero, i) for i in range(tail + 1)) / (2 ** nonzero)) if nonzero else 1.0
            result.append({
                "metric": metric,
                "baseline": baseline,
                "paired_seeds": len(SEEDS),
                "mean_difference_cp_minus_baseline": float(differences.mean()),
                "bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                "cp_better_fraction": float(np.mean(differences < 0 if metric == "brier_score" else differences > 0)),
                "two_sided_sign_test_p": float(sign_p),
            })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    features, labels = load_wdbc(root / "data" / "wdbc.data")
    rows, splits = [], []
    for seed in SEEDS:
        seed_rows, split = run_seed(seed, features, labels)
        rows.extend(seed_rows); splits.append(split)
    summary = aggregate(rows)
    paired = paired_statistics(rows)
    payload = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "dataset_doi": "10.24432/C5DW2B",
        "positive_class": "malignant",
        "instances": len(labels),
        "features": features.shape[1],
        "malignant_instances": int(labels.sum()),
        "seeds": list(SEEDS),
        "test_size": TEST_SIZE,
        "rules": RULES,
        "nearest_rules": NEIGHBOURS,
        "dimension": DIMENSION,
        "length_scale": LENGTH_SCALE,
        "vsa_frequency_scale": VSA_FREQUENCY_SCALE,
        "weight_temperature": WEIGHT_TEMPERATURE,
        "alpha_points": len(ALPHA),
        "leakage_control": "split first; scaler, KMeans rules, consequents, and supervised baselines fit on training data only",
        "rows": rows,
        "summary": summary,
        "paired_statistics": paired,
        "splits": splits,
    }
    (root / "uci_wdbc_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(root / "uci_wdbc_by_seed.csv", rows)
    write_csv(root / "uci_wdbc_summary.csv", summary)
    write_csv(root / "uci_wdbc_paired_stats.csv", paired)

    fig, (ax_perf, ax_cnf) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    methods = [row["method"] for row in summary]
    labels_short = ["Grid rules", "VSA+grid", "CP-HDFRI", "Logistic", "7-NN"]
    means = [row["roc_auc_mean"] for row in summary]
    errors = [row["roc_auc_seed_std"] for row in summary]
    colors = ["#64748b", "#b45309", "#2563eb", "#7c3aed", "#15803d"]
    ax_perf.bar(labels_short, means, yerr=errors, capsize=3, color=colors)
    ax_perf.set_ylim(0.80, 1.00); ax_perf.set_ylabel("ROC AUC")
    ax_perf.tick_params(axis="x", rotation=25); ax_perf.grid(axis="y", alpha=0.25)
    cp = next(row for row in summary if row["method"] == "cp_hdfri_actual")
    ax_cnf.bar(["Raw decode", "After projection"],
               [cp["raw_cnf_violation_rate_mean"], cp["projected_cnf_violation_rate_mean"]],
               color=["#b91c1c", "#15803d"])
    ax_cnf.set_ylim(0.0, 1.05); ax_cnf.set_ylabel("CNF violation rate")
    ax_cnf.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(root / "uci_wdbc_diagnostics.png", dpi=300); plt.close(fig)
    print(json.dumps({"summary": summary, "paired_statistics": paired}, indent=2))


if __name__ == "__main__":
    main()
