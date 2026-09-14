"""Direct retrieval-quality benchmark for Gaussian random-feature VSA.

The experiment varies hypervector dimension D, rule-base size N, and
antecedent dimension p around a fixed anchor.  It reports Recall@8/top-k
overlap, full-ranking Spearman correlation, distance distortion, query time,
and representation memory.  No labels or downstream task metrics are used.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import rankdata


SEEDS = (401, 409, 419, 421, 431, 433, 439, 443, 449, 457)
QUERIES = 32
K = 8
ANCHOR_D = 2048
ANCHOR_N = 256
ANCHOR_P = 8


def settings():
    result = set()
    for dimension in (128, 256, 512, 1024, 2048, 4096):
        result.add((dimension, ANCHOR_N, ANCHOR_P, "D"))
    for rules in (64, 128, 256, 512, 1024):
        result.add((ANCHOR_D, rules, ANCHOR_P, "N"))
    for variables in (2, 4, 8, 16, 32):
        result.add((ANCHOR_D, ANCHOR_N, variables, "p"))
    return sorted(result, key=lambda row: (row[3], row[0], row[1], row[2]))


def run_setting(seed: int, dimension: int, rules: int, variables: int, sweep: str):
    rng = np.random.default_rng(seed * 1000003 + dimension * 101 + rules * 17 + variables)
    centers = rng.random((rules, variables))
    queries = rng.random((QUERIES, variables))
    pair_sample = centers[rng.choice(rules, size=min(rules, 256), replace=False)]
    bandwidth = float(np.median(pdist(pair_sample)))
    bandwidth = max(bandwidth, 1e-6)
    frequencies = rng.normal(0.0, 1.0 / bandwidth, size=(variables, dimension))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=dimension)

    def encode(points: np.ndarray):
        encoded = np.sqrt(2.0 / dimension) * np.cos(points @ frequencies + phases)
        return encoded / np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-12)

    center_h = encode(centers)
    start = time.perf_counter()
    query_h = encode(queries)
    similarity = query_h @ center_h.T
    elapsed = time.perf_counter() - start
    exact_distance = cdist(queries, centers)
    exact_order = np.argsort(exact_distance, axis=1)
    vsa_order = np.argsort(-similarity, axis=1)

    overlap, correlation, distortion = [], [], []
    # Invert the target RBF kernel to a distance estimate.  Clipping is only
    # needed for finite-D values outside the analytic kernel range.
    estimated_distance = bandwidth * np.sqrt(
        np.maximum(0.0, -2.0 * np.log(np.clip(similarity, 1e-8, 1.0))),
    )
    for i in range(QUERIES):
        overlap.append(len(set(exact_order[i, :K]) & set(vsa_order[i, :K])) / K)
        correlation.append(float(np.corrcoef(
            rankdata(exact_distance[i]), rankdata(estimated_distance[i]),
        )[0, 1]))
        distortion.append(float(np.sqrt(np.mean(
            (estimated_distance[i] - exact_distance[i]) ** 2,
        )) / max(np.mean(exact_distance[i]), 1e-12)))
    return {
        "seed": seed, "sweep": sweep, "dimension_D": dimension,
        "rules_N": rules, "antecedent_dimension_p": variables,
        "queries": QUERIES, "k": K, "bandwidth": bandwidth,
        "recall_at_8": float(np.mean(overlap)),
        "top_k_overlap": float(np.mean(overlap)),
        "rank_correlation": float(np.mean(correlation)),
        "relative_distance_distortion": float(np.mean(distortion)),
        "milliseconds_per_query": 1000.0 * elapsed / QUERIES,
        "representation_megabytes": float(center_h.nbytes / (1024 ** 2)),
    }


def aggregate(rows: list[dict]):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["sweep"], row["dimension_D"], row["rules_N"],
                row["antecedent_dimension_p"])].append(row)
    result = []
    for key in sorted(groups):
        subset = groups[key]
        result.append({
            "sweep": key[0], "dimension_D": key[1], "rules_N": key[2],
            "antecedent_dimension_p": key[3], "seeds": len(subset),
            **{f"{metric}_mean": float(np.mean([row[metric] for row in subset]))
               for metric in ("recall_at_8", "top_k_overlap", "rank_correlation",
                              "relative_distance_distortion", "milliseconds_per_query",
                              "representation_megabytes")},
            **{f"{metric}_seed_std": float(np.std([row[metric] for row in subset], ddof=1))
               for metric in ("recall_at_8", "rank_correlation",
                              "relative_distance_distortion")},
        })
    return result


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = []
    all_settings = settings()
    for index, (dimension, rules, variables, sweep) in enumerate(all_settings, 1):
        rows.extend(run_setting(seed, dimension, rules, variables, sweep) for seed in SEEDS)
        print(f"completed setting {index}/{len(all_settings)}: {sweep} D={dimension} N={rules} p={variables}",
              flush=True)
    summary = aggregate(rows)
    payload = {
        "seeds": list(SEEDS), "queries_per_setting_per_seed": QUERIES,
        "k": K, "anchor": {"D": ANCHOR_D, "N": ANCHOR_N, "p": ANCHOR_P},
        "rows": rows, "summary": summary,
    }
    (root / "retrieval_scaling_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    write_csv(root / "retrieval_scaling_by_seed.csv", rows)
    write_csv(root / "retrieval_scaling_summary.csv", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
