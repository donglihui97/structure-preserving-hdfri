"""Repeated nested validation on three additional real binary tasks.

The datasets ship with scikit-learn, so the benchmark is network independent.
Wine uses class 1 versus the remaining classes; digits uses 3 versus 5 after
filtering to those two classes; iris uses versicolor versus virginica after
filtering out setosa.  Every outer split repeats all tuning and rule fitting on
its training partition.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evidence_hdfri_benchmark import (
    SEEDS, confidence_grid, fuzzy_endpoints, hdt_decode, select_rule_hyperparameters,
    vsa_grid,
)

warnings.filterwarnings("ignore", category=Warning, module="sklearn")


def datasets():
    wine = load_wine()
    digits = load_digits()
    iris = load_iris()
    digit_mask = np.isin(digits.target, (3, 5))
    iris_mask = np.isin(iris.target, (1, 2))
    return {
        "wine_class1_vs_rest": (wine.data.astype(float), (wine.target == 1).astype(int)),
        "digits_3_vs_5": (digits.data[digit_mask].astype(float),
                          (digits.target[digit_mask] == 5).astype(int)),
        "iris_versicolor_vs_virginica": (iris.data[iris_mask].astype(float),
                                         (iris.target[iris_mask] == 2).astype(int)),
    }


def base_probabilities(seed: int, x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 17001)
    candidates = (
        ("rbf_svc", Pipeline([
            ("scale", StandardScaler()), ("model", SVC(probability=False, random_state=seed)),
        ]), {"model__C": [1, 5, 10], "model__gamma": ["scale", 0.01, 0.05]}),
        ("qda", Pipeline([
            ("scale", StandardScaler()), ("model", QuadraticDiscriminantAnalysis()),
        ]), {"model__reg_param": [0.05, 0.2, 0.5]}),
        ("extra_trees", ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1,
        ), {"max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 2]}),
    )
    outputs, params = [], {}
    for name, model, grid in candidates:
        search = GridSearchCV(model, grid, scoring="roc_auc", cv=inner, n_jobs=1).fit(x_train, y_train)
        best = search.best_estimator_; params[name] = search.best_params_
        probabilistic = (CalibratedClassifierCV(
            estimator=clone(best), method="sigmoid", cv=3, ensemble=False,
        ) if name == "rbf_svc" else clone(best))
        oof = cross_val_predict(
            probabilistic, x_train, y_train, cv=inner, method="predict_proba", n_jobs=1,
        )[:, 1]
        probabilistic.fit(x_train, y_train)
        outputs.append((oof, probabilistic.predict_proba(x_test)[:, 1]))
    return (np.column_stack([item[0] for item in outputs]),
            np.column_stack([item[1] for item in outputs]), params)


def metric(dataset: str, seed: int, method: str, truth: np.ndarray,
           probability: np.ndarray, retrieval: dict | None = None):
    probability = np.clip(probability, 0.0, 1.0)
    return {
        "dataset": dataset, "seed": seed, "method": method,
        "brier_score": float(brier_score_loss(truth, probability)),
        "roc_auc": float(roc_auc_score(truth, probability)),
        "accuracy": float(accuracy_score(truth, probability >= 0.5)),
        "retrieval_recall_at_8": float((retrieval or {}).get("recall_at_k", 0.0)),
        "retrieval_rank_correlation": float((retrieval or {}).get("rank_correlation", 0.0)),
    }


def run_seed(dataset: str, seed: int, x: np.ndarray, y: np.ndarray):
    train, test = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=seed,
    ).split(x, y))
    x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
    oof, test_base, base_params = base_probabilities(seed, x_train, y_train, x_test)
    meta_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 19001)
    meta_search = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=seed),
        {"C": [1.0, 10.0, 100.0]}, scoring="neg_brier_score", cv=meta_cv, n_jobs=1,
    ).fit(oof, y_train)
    meta = meta_search.best_estimator_
    train_evidence = meta.predict_proba(oof)[:, 1]
    test_evidence = meta.predict_proba(test_base)[:, 1]
    selected, _ = select_rule_hyperparameters(seed, train_evidence, y_train)
    rules, scale = int(selected["rules"]), float(selected["vsa_scale"])
    z_train, lo, hi, centers, consequent = confidence_grid(train_evidence, rules)
    z_test = np.log(np.clip(test_evidence, 1e-6, 1 - 1e-6) /
                    np.clip(1 - test_evidence, 1e-6, 1 - 1e-6))
    retrieval_seed = seed * 100003 + rules * 101 + int(scale * 10)
    vsa_probability, pairs, retrieval = vsa_grid(
        z_test, lo, hi, centers, consequent, retrieval_seed, scale, True,
    )
    left, right = fuzzy_endpoints(consequent)
    hdfri_probability, raw_invalid, projected_invalid, displacement, carrier = hdt_decode(
        seed, left, right, pairs,
    )

    scale_model = StandardScaler().fit(x_train)
    xtr, xte = scale_model.transform(x_train), scale_model.transform(x_test)
    logistic = LogisticRegression(max_iter=3000, random_state=seed).fit(xtr, y_train)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance").fit(xtr, y_train)
    rows = [
        metric(dataset, seed, "direct_evidence_stack", y_test, test_evidence),
        metric(dataset, seed, "evidence_vsa_grid", y_test, vsa_probability, retrieval),
        metric(dataset, seed, "evidence_hdfri", y_test, hdfri_probability, retrieval),
        metric(dataset, seed, "rbf_svc_calibrated", y_test, test_base[:, 0]),
        metric(dataset, seed, "qda_tuned", y_test, test_base[:, 1]),
        metric(dataset, seed, "extra_trees_tuned", y_test, test_base[:, 2]),
        metric(dataset, seed, "logistic_regression", y_test, logistic.predict_proba(xte)[:, 1]),
        metric(dataset, seed, "knn_7", y_test, knn.predict_proba(xte)[:, 1]),
    ]
    detail = {
        "dataset": dataset, "seed": seed, "train_cases": len(train), "test_cases": len(test),
        "positive_train": int(y_train.sum()), "positive_test": int(y_test.sum()),
        "base_parameters": base_params,
        "meta_C": meta_search.best_params_["C"],
        "rule_parameters": selected,
        "raw_cnf_violation_rate": float(np.mean(raw_invalid)),
        "projected_cnf_violation_rate": float(np.mean(projected_invalid)),
        "projection_displacement_mean": float(np.mean(displacement)),
        "carrier_rmse_mean": float(np.mean(carrier)),
    }
    return rows, detail


def aggregate(rows: list[dict]):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)
    result = []
    for (dataset, method), subset in sorted(groups.items()):
        result.append({
            "dataset": dataset, "method": method, "seeds": len(subset),
            **{f"{name}_mean": float(np.mean([row[name] for row in subset]))
               for name in ("brier_score", "roc_auc", "accuracy",
                            "retrieval_recall_at_8", "retrieval_rank_correlation")},
            **{f"{name}_seed_std": float(np.std([row[name] for row in subset], ddof=1))
               for name in ("brier_score", "roc_auc", "accuracy")},
        })
    return result


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    rows, details = [], []
    all_data = datasets()
    total = len(all_data) * len(SEEDS)
    completed = 0
    for name, (x, y) in all_data.items():
        for seed in SEEDS:
            seed_rows, detail = run_seed(name, seed, x, y)
            rows.extend(seed_rows); details.append(detail)
            completed += 1
            print(f"completed {completed}/{total}: {name}, seed={seed}", flush=True)
    summary = aggregate(rows)
    payload = {
        "outer_seeds": list(SEEDS), "outer_test_size": 0.30, "inner_folds": 3,
        "task_definitions": {
            "wine_class1_vs_rest": "Wine class 1 versus classes 0 and 2",
            "digits_3_vs_5": "Digits 3 versus 5 after filtering",
            "iris_versicolor_vs_virginica": "Iris versicolor versus virginica after filtering",
        },
        "details": details, "rows": rows, "summary": summary,
    }
    (root / "multidataset_evidence_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    write_csv(root / "multidataset_evidence_by_seed.csv", rows)
    write_csv(root / "multidataset_evidence_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
