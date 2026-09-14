"""Nested, leakage-safe WDBC benchmark for Evidence-HDFRI.

The redesigned model learns a low-dimensional evidence coordinate from three
diverse supervised learners, represents a sparse fuzzy confidence rule base in
that coordinate, retrieves a small candidate set with random-feature VSA, and
reconstructs alpha-cut consequents with a constant-preserving paper-faithful
HDT operator before exact Euclidean CNF projection.

The direct evidence stack is reported as an explicit ablation.  Consequently,
predictive gains caused by the supervised evidence layer are never attributed
to the VSA/HDT carrier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import warnings
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from actual_hdt_benchmark import ALPHA, interval_encoding, trapezoid_weights
from cnf_projection import project_cnf
from preexperiments import violations
from uci_wdbc_benchmark import SEEDS, TEST_SIZE, load_wdbc

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RULE_OPTIONS = (16, 32, 64)
VSA_SCALE_OPTIONS = (2.0, 4.0, 8.0)
VSA_DIMENSION = 4096
CANDIDATE_POOL = 12
RETRIEVAL_K = 8
HDT_DIMENSION = 4096
HDT_LENGTH_SCALE = 0.20
BOOTSTRAP_RESAMPLES = 50_000


def tuned_base_probabilities(seed: int, x_train: np.ndarray, y_train: np.ndarray,
                             x_test: np.ndarray):
    """Return train OOF and test probabilities for three nested-tuned bases."""
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 7001)
    svc = Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(probability=False, random_state=seed)),
    ])
    svc_grid = {
        "model__C": [1, 2, 5, 10, 20],
        "model__gamma": ["scale", 0.005, 0.01, 0.02, 0.05],
    }
    qda = Pipeline([
        ("scale", StandardScaler()),
        ("model", QuadraticDiscriminantAnalysis()),
    ])
    qda_grid = {"model__reg_param": [0.01, 0.05, 0.1, 0.2, 0.5]}
    trees = ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    tree_grid = {"max_features": ["sqrt", 0.5, 1.0], "min_samples_leaf": [1, 2]}

    outputs, parameters = [], {}
    for name, estimator, grid in (
        ("rbf_svc", svc, svc_grid),
        ("qda", qda, qda_grid),
        ("extra_trees", trees, tree_grid),
    ):
        search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner, n_jobs=1)
        search.fit(x_train, y_train)
        best = search.best_estimator_
        parameters[name] = search.best_params_
        if name == "rbf_svc":
            # Explicit sigmoid calibration replaces the deprecated SVC
            # probability switch.  It is itself cross-fitted for the meta layer.
            probabilistic = CalibratedClassifierCV(
                estimator=clone(best), method="sigmoid", cv=5, ensemble=False,
            )
        else:
            probabilistic = clone(best)
        oof = cross_val_predict(
            probabilistic, x_train, y_train, cv=inner, method="predict_proba",
            n_jobs=1,
        )[:, 1]
        probabilistic.fit(x_train, y_train)
        test = probabilistic.predict_proba(x_test)[:, 1]
        outputs.append((oof, test))
    return (np.column_stack([item[0] for item in outputs]),
            np.column_stack([item[1] for item in outputs]), parameters)


def fit_evidence_stack(seed: int, oof: np.ndarray, y_train: np.ndarray,
                       test: np.ndarray):
    """Select meta-regularization on training-only OOF evidence."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 9001)
    search = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=seed),
        {"C": [0.1, 1.0, 10.0, 100.0]}, scoring="neg_brier_score", cv=cv,
        n_jobs=1,
    )
    search.fit(oof, y_train)
    model = search.best_estimator_
    train_probability = model.predict_proba(oof)[:, 1]
    test_probability = model.predict_proba(test)[:, 1]
    return train_probability, test_probability, {
        "C": search.best_params_["C"],
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def encoder(seed: int, scale: float):
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(0.0, scale, size=VSA_DIMENSION)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=VSA_DIMENSION)

    def encode(values: np.ndarray) -> np.ndarray:
        result = np.cos(np.asarray(values)[:, None] * frequencies[None, :] + phases[None, :])
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return encode


def confidence_grid(probability: np.ndarray, rules: int):
    z = logit(np.clip(probability, 1e-6, 1.0 - 1e-6))
    span = max(float(z.max() - z.min()), 1e-6)
    lo = float(z.min() - 0.10 * span)
    hi = float(z.max() + 0.10 * span)
    centers = np.linspace(0.0, 1.0, rules)
    consequent = expit(lo + centers * (hi - lo))
    return z, lo, hi, centers, consequent


def exact_grid(values: np.ndarray, lo: float, hi: float, centers: np.ndarray,
               consequent: np.ndarray) -> np.ndarray:
    normalized = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return np.interp(normalized, centers, consequent)


def vsa_grid(values: np.ndarray, lo: float, hi: float, centers: np.ndarray,
             consequent: np.ndarray, seed: int, scale: float, with_diagnostics: bool = False):
    normalized = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    encode = encoder(seed, scale)
    center_h = encode(centers)
    query_h = encode(normalized)
    similarity = query_h @ center_h.T
    pool_size = min(CANDIDATE_POOL, len(centers))
    candidates = np.argpartition(-similarity, pool_size - 1, axis=1)[:, :pool_size]
    predictions, pairs = [], []
    for value, candidate in zip(normalized, candidates):
        candidate_centers = centers[candidate]
        lower = candidate[candidate_centers <= value]
        upper = candidate[candidate_centers >= value]
        left = lower[np.argmax(centers[lower])] if len(lower) else candidate[np.argmin(candidate_centers)]
        right = upper[np.argmin(centers[upper])] if len(upper) else candidate[np.argmax(candidate_centers)]
        if left == right or centers[right] <= centers[left]:
            blend = 0.0
        else:
            blend = float((value - centers[left]) / (centers[right] - centers[left]))
            blend = float(np.clip(blend, 0.0, 1.0))
        predictions.append((1.0 - blend) * consequent[left] + blend * consequent[right])
        pairs.append((int(left), int(right), blend))
    if not with_diagnostics:
        return np.asarray(predictions), pairs, None

    k = min(RETRIEVAL_K, len(centers))
    exact_order = np.argsort(np.abs(normalized[:, None] - centers[None, :]), axis=1)
    vsa_order = np.argsort(-similarity, axis=1)
    overlaps, spearman, distortion = [], [], []
    exact_distance = np.abs(normalized[:, None] - centers[None, :])
    hv_distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))
    for i in range(len(normalized)):
        overlaps.append(len(set(exact_order[i, :k]) & set(vsa_order[i, :k])) / k)
        exact_rank = rankdata(exact_distance[i])
        hv_rank = rankdata(hv_distance[i])
        spearman.append(float(np.corrcoef(exact_rank, hv_rank)[0, 1]))
        scale_fit = float(np.dot(exact_distance[i], hv_distance[i]) /
                          max(np.dot(hv_distance[i], hv_distance[i]), 1e-12))
        distortion.append(float(np.sqrt(np.mean(
            (scale_fit * hv_distance[i] - exact_distance[i]) ** 2,
        ))))
    diagnostics = {
        "recall_at_k": float(np.mean(overlaps)),
        "top_k_overlap": float(np.mean(overlaps)),
        "rank_correlation": float(np.mean(spearman)),
        "distance_distortion_rmse": float(np.mean(distortion)),
    }
    return np.asarray(predictions), pairs, diagnostics


def select_rule_hyperparameters(seed: int, evidence_probability: np.ndarray,
                                y_train: np.ndarray):
    z = logit(np.clip(evidence_probability, 1e-6, 1.0 - 1e-6))
    candidates = []
    for rules in RULE_OPTIONS:
        _, lo, hi, centers, consequent = confidence_grid(evidence_probability, rules)
        for scale in VSA_SCALE_OPTIONS:
            prediction, _, diagnostics = vsa_grid(
                z, lo, hi, centers, consequent,
                seed * 100003 + rules * 101 + int(scale * 10), scale, True,
            )
            candidates.append({
                "rules": rules, "vsa_scale": scale,
                "training_oof_brier": float(brier_score_loss(y_train, prediction)),
                **diagnostics,
            })
    # Primary objective: OOF Brier.  Retrieval recall breaks numerical ties.
    selected = min(candidates, key=lambda item: (item["training_oof_brier"],
                                                  -item["recall_at_k"], item["rules"]))
    return selected, candidates


def fuzzy_endpoints(consequent: np.ndarray):
    width = np.minimum(0.10, 0.80 * np.minimum(consequent, 1.0 - consequent))
    taper = 1.0 - ALPHA[None, :]
    return consequent[:, None] - width[:, None] * taper, consequent[:, None] + width[:, None] * taper


def hdt_decode(seed: int, rule_left: np.ndarray, rule_right: np.ndarray,
               pairs: list[tuple[int, int, float]]):
    encoding, _ = interval_encoding(
        np.random.default_rng(seed * 30011 + HDT_DIMENSION),
        HDT_DIMENSION, HDT_LENGTH_SCALE,
    )
    # The redesigned inverse solves the observed-coordinate HDT system.  With
    # the full mask used in the clean predictive benchmark this is a fixed
    # ridge inverse; under coordinate damage the same solve is restricted to
    # surviving rows (see hdc_value_benchmark.py).
    forward = encoding.T * trapezoid_weights(ALPHA)[None, :]
    normal = forward.T @ forward + 1e-8 * np.eye(len(ALPHA))
    inverse = np.linalg.solve(normal, forward.T)
    probability, raw_invalid, projected_invalid = [], [], []
    displacement, carrier_rmse = [], []
    for left, right, blend in pairs:
        direct_l = (1.0 - blend) * rule_left[left] + blend * rule_left[right]
        direct_r = (1.0 - blend) * rule_right[left] + blend * rule_right[right]
        provisional_l = inverse @ (forward @ direct_l)
        provisional_r = inverse @ (forward @ direct_r)
        out_l, out_r = project_cnf(provisional_l, provisional_r)
        probability.append(float(np.mean((out_l + out_r) / 2.0)))
        raw_invalid.append(any(violations(provisional_l, provisional_r).values()))
        projected_invalid.append(any(violations(out_l, out_r).values()))
        displacement.append(float(np.sqrt(np.mean(np.r_[out_l - provisional_l,
                                                        out_r - provisional_r] ** 2))))
        carrier_rmse.append(float(np.sqrt(np.mean(np.r_[out_l - direct_l,
                                                        out_r - direct_r] ** 2))))
    return np.asarray(probability), raw_invalid, projected_invalid, displacement, carrier_rmse


def metric_row(seed: int, method: str, truth: np.ndarray, probability: np.ndarray,
               elapsed: float, **extra):
    probability = np.clip(probability, 0.0, 1.0)
    return {
        "seed": seed, "method": method, "test_cases": len(truth),
        "brier_score": float(brier_score_loss(truth, probability)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "accuracy": float(accuracy_score(truth, probability >= 0.5)),
        "milliseconds_per_query": 1000.0 * elapsed / len(truth),
        "raw_cnf_violation_rate": float(np.mean(extra.get("raw_invalid", [False]))),
        "projected_cnf_violation_rate": float(np.mean(extra.get("projected_invalid", [False]))),
        "projection_displacement_mean": float(np.mean(extra.get("displacement", [0.0]))),
        "carrier_rmse_mean": float(np.mean(extra.get("carrier_rmse", [0.0]))),
        "retrieval_recall_at_8": float(extra.get("retrieval", {}).get("recall_at_k", 0.0)),
        "retrieval_rank_correlation": float(extra.get("retrieval", {}).get("rank_correlation", 0.0)),
        "retrieval_distance_distortion": float(extra.get("retrieval", {}).get("distance_distortion_rmse", 0.0)),
    }


def run_seed(seed: int, features: np.ndarray, labels: np.ndarray):
    train, test = next(StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=seed,
    ).split(features, labels))
    x_train, x_test = features[train], features[test]
    y_train, y_test = labels[train], labels[test]

    oof, test_base, base_parameters = tuned_base_probabilities(
        seed, x_train, y_train, x_test,
    )
    evidence_train, evidence_test, meta_parameters = fit_evidence_stack(
        seed, oof, y_train, test_base,
    )
    selected, candidate_results = select_rule_hyperparameters(
        seed, evidence_train, y_train,
    )
    rules, scale = int(selected["rules"]), float(selected["vsa_scale"])
    _, lo, hi, centers, consequent = confidence_grid(evidence_train, rules)
    z_test = logit(np.clip(evidence_test, 1e-6, 1.0 - 1e-6))

    start = time.perf_counter()
    direct_probability = exact_grid(z_test, lo, hi, centers, consequent)
    direct_elapsed = time.perf_counter() - start

    retrieval_seed = seed * 100003 + rules * 101 + int(scale * 10)
    start = time.perf_counter()
    vsa_probability, pairs, retrieval = vsa_grid(
        z_test, lo, hi, centers, consequent, retrieval_seed, scale, True,
    )
    vsa_elapsed = time.perf_counter() - start

    rule_left, rule_right = fuzzy_endpoints(consequent)
    start = time.perf_counter()
    hdt_probability, raw_invalid, projected_invalid, displacement, carrier_rmse = hdt_decode(
        seed, rule_left, rule_right, pairs,
    )
    hdt_elapsed = time.perf_counter() - start + vsa_elapsed

    scaler = MinMaxScaler(clip=True)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    logistic = LogisticRegression(max_iter=3000, random_state=seed).fit(x_train_scaled, y_train)
    start = time.perf_counter(); logistic_probability = logistic.predict_proba(x_test_scaled)[:, 1]
    logistic_elapsed = time.perf_counter() - start
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance").fit(x_train_scaled, y_train)
    start = time.perf_counter(); knn_probability = knn.predict_proba(x_test_scaled)[:, 1]
    knn_elapsed = time.perf_counter() - start

    rows = [
        metric_row(seed, "rbf_svc_calibrated", y_test, test_base[:, 0], 0.0),
        metric_row(seed, "qda_tuned", y_test, test_base[:, 1], 0.0),
        metric_row(seed, "extra_trees_tuned", y_test, test_base[:, 2], 0.0),
        metric_row(seed, "direct_evidence_stack", y_test, evidence_test, 0.0),
        metric_row(seed, "evidence_rule_grid", y_test, direct_probability, direct_elapsed),
        metric_row(seed, "evidence_vsa_grid", y_test, vsa_probability, vsa_elapsed,
                   retrieval=retrieval),
        metric_row(seed, "evidence_hdfri", y_test, hdt_probability, hdt_elapsed,
                   raw_invalid=raw_invalid, projected_invalid=projected_invalid,
                   displacement=displacement, carrier_rmse=carrier_rmse,
                   retrieval=retrieval),
        metric_row(seed, "logistic_regression", y_test, logistic_probability, logistic_elapsed),
        metric_row(seed, "knn_7", y_test, knn_probability, knn_elapsed),
    ]
    split = {
        "seed": seed, "training_cases": len(train), "test_cases": len(test),
        "training_malignant": int(y_train.sum()), "test_malignant": int(y_test.sum()),
        "train_indices_sha256": hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
        "base_parameters": base_parameters, "meta_parameters": meta_parameters,
        "selected_rule_parameters": selected, "rule_candidates": candidate_results,
    }
    return rows, split


def aggregate(rows: list[dict]):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    result = []
    for method in sorted(groups):
        subset = groups[method]
        result.append({
            "method": method, "seeds": len(subset),
            **{f"{metric}_mean": float(np.mean([row[metric] for row in subset]))
               for metric in ("brier_score", "roc_auc", "accuracy", "milliseconds_per_query",
                              "raw_cnf_violation_rate", "projected_cnf_violation_rate",
                              "projection_displacement_mean", "carrier_rmse_mean",
                              "retrieval_recall_at_8", "retrieval_rank_correlation",
                              "retrieval_distance_distortion")},
            **{f"{metric}_seed_std": float(np.std([row[metric] for row in subset], ddof=1))
               for metric in ("brier_score", "roc_auc", "accuracy")},
        })
    return result


def paired_statistics(rows: list[dict]):
    lookup = {(row["method"], row["seed"]): row for row in rows}
    baselines = ("direct_evidence_stack", "rbf_svc_calibrated", "qda_tuned",
                 "extra_trees_tuned", "logistic_regression", "knn_7")
    result = []
    for metric in ("brier_score", "roc_auc"):
        for baseline in baselines:
            difference = np.array([
                lookup[("evidence_hdfri", seed)][metric] - lookup[(baseline, seed)][metric]
                for seed in SEEDS
            ])
            rng = np.random.default_rng(20260911 + len(metric) * 101 + len(baseline))
            samples = difference[rng.integers(0, len(SEEDS),
                                               size=(BOOTSTRAP_RESAMPLES, len(SEEDS)))].mean(axis=1)
            positive, negative = int(np.sum(difference > 0)), int(np.sum(difference < 0))
            nonzero, tail = positive + negative, min(positive, negative)
            sign_p = min(1.0, 2.0 * sum(comb(nonzero, i) for i in range(tail + 1)) /
                         (2 ** nonzero)) if nonzero else 1.0
            result.append({
                "metric": metric, "baseline": baseline, "paired_seeds": len(SEEDS),
                "mean_difference_hdfri_minus_baseline": float(difference.mean()),
                "bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
                "hdfri_better_fraction": float(np.mean(
                    difference < 0 if metric == "brier_score" else difference > 0,
                )),
                "two_sided_sign_test_p": float(sign_p),
            })
    return result


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    features, labels = load_wdbc(root / "data" / "wdbc.data")
    rows, splits = [], []
    for index, seed in enumerate(SEEDS, 1):
        seed_rows, split = run_seed(seed, features, labels)
        rows.extend(seed_rows); splits.append(split)
        print(f"completed seed {index}/{len(SEEDS)}: {seed}", flush=True)
    summary = aggregate(rows)
    paired = paired_statistics(rows)
    payload = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "positive_class": "malignant", "instances": len(labels),
        "features": features.shape[1], "outer_seeds": list(SEEDS),
        "outer_test_size": TEST_SIZE, "inner_folds": 5,
        "vsa_dimension": VSA_DIMENSION, "candidate_pool": CANDIDATE_POOL,
        "retrieval_k": RETRIEVAL_K, "hdt_dimension": HDT_DIMENSION,
        "hdt_length_scale": HDT_LENGTH_SCALE,
        "splits": splits, "rows": rows, "summary": summary,
        "paired_statistics": paired,
    }
    (root / "evidence_hdfri_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    write_csv(root / "evidence_hdfri_by_seed.csv", rows)
    write_csv(root / "evidence_hdfri_summary.csv", summary)
    write_csv(root / "evidence_hdfri_paired_stats.csv", paired)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
