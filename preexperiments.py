"""Pre-experiments for HDT/VSA alpha-cut endpoint interpolation.

This is deliberately a transparent proxy for an HDT implementation: a dense
random projection maps the sampled endpoint functions to a real-valued
hypervector and a ridge pseudoinverse decodes it.  The script tests the two
claims that must hold before building a full FRI system: (i) endpoint shape is
recoverable at low dimension and (ii) convex-normal fuzzy sets survive
hypervector bundling.  Results are written to ``preexperiment_results.json``
and ``preexperiment_summary.csv``; a diagnostic plot is also produced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cnf_projection import pava, project_cnf


# The first sanity check follows the proposed protocol exactly: 11 alpha
# levels.  A dense-grid variant can be enabled by changing this constant.
ALPHA = np.linspace(0.0, 1.0, 11)


def triangle(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    a = rng.uniform(-5.0, 5.0)
    b = a + rng.uniform(0.5, 5.0)
    c = b + rng.uniform(0.5, 5.0)
    left = a + ALPHA * (b - a)
    right = c - ALPHA * (c - b)
    return left, right


def reference_triangle() -> tuple[np.ndarray, np.ndarray]:
    """The deterministic A=(2,5,9) sanity case from the study protocol."""
    return 2.0 + 3.0 * ALPHA, 9.0 - 4.0 * ALPHA


def violations(left: np.ndarray, right: np.ndarray, tol: float = 1e-8) -> dict[str, bool]:
    return {
        "left_monotone": bool(np.any(np.diff(left) < -tol)),
        "right_monotone": bool(np.any(np.diff(right) > tol)),
        "left_le_right": bool(np.any(left > right + tol)),
    }


def make_decoder(rng: np.random.Generator, dimension: int, m: int, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    # Rows are normalized so changing dimension does not change signal scale.
    projection = rng.normal(size=(dimension, m)) / np.sqrt(dimension)
    gram = projection @ projection.T + ridge * np.eye(dimension)
    decoder = projection.T @ np.linalg.inv(gram)
    return projection, decoder


def run(seed: int = 7, repeats: int = 1000, dimensions: tuple[int, ...] = (4, 8, 16, 32, 64)) -> dict:
    master = np.random.default_rng(seed)
    results: list[dict] = []
    curves: dict[str, list[dict]] = {str(d): [] for d in dimensions}
    for dimension in dimensions:
        for rep in range(repeats):
            rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
            projection, decoder = make_decoder(rng, dimension, len(ALPHA), ridge=1e-3)
            l1, r1 = triangle(rng)
            l2, r2 = triangle(rng)
            weights = rng.dirichlet([1.0, 1.0])
            ideal_l = weights[0] * l1 + weights[1] * l2
            ideal_r = weights[0] * r1 + weights[1] * r2

            # Encode/bundle/decode.  This is the proposed operation in its
            # simplest form, with no post-hoc shape repair.
            h_l = weights[0] * (projection @ l1) + weights[1] * (projection @ l2)
            h_r = weights[0] * (projection @ r1) + weights[1] * (projection @ r2)
            out_l = decoder @ h_l
            out_r = decoder @ h_r
            raw_v = violations(out_l, out_r)
            proj_l, proj_r = project_cnf(out_l, out_r)
            proj_v = violations(proj_l, proj_r)

            row = {
                "dimension": dimension,
                "endpoint_rmse": float(np.sqrt(np.mean(np.r_[out_l - ideal_l, out_r - ideal_r] ** 2))),
                "endpoint_mae": float(np.mean(np.abs(np.r_[out_l - ideal_l, out_r - ideal_r]))),
                "raw_cnf_violation": bool(any(raw_v.values())),
                "raw_left_monotone_violation": raw_v["left_monotone"],
                "raw_right_monotone_violation": raw_v["right_monotone"],
                "raw_left_gt_right_violation": raw_v["left_le_right"],
                "projected_endpoint_rmse": float(np.sqrt(np.mean(np.r_[proj_l - ideal_l, proj_r - ideal_r] ** 2))),
                "projected_cnf_violation": bool(any(proj_v.values())),
            }
            curves[str(dimension)].append(row)
            results.append(row)

    summary = []
    for dim in dimensions:
        rows = curves[str(dim)]
        summary.append({
            "dimension": dim,
            "rmse_mean": float(np.mean([r["endpoint_rmse"] for r in rows])),
            "rmse_p95": float(np.quantile([r["endpoint_rmse"] for r in rows], 0.95)),
            "raw_cnf_violation_rate": float(np.mean([r["raw_cnf_violation"] for r in rows])),
            "raw_left_monotone_rate": float(np.mean([r["raw_left_monotone_violation"] for r in rows])),
            "raw_right_monotone_rate": float(np.mean([r["raw_right_monotone_violation"] for r in rows])),
            "raw_left_gt_right_rate": float(np.mean([r["raw_left_gt_right_violation"] for r in rows])),
            "projected_rmse_mean": float(np.mean([r["projected_endpoint_rmse"] for r in rows])),
            "projected_cnf_violation_rate": float(np.mean([r["projected_cnf_violation"] for r in rows])),
        })

    reference_l, reference_r = reference_triangle()
    reference = []
    for dimension in dimensions:
        rng = np.random.default_rng(1000003 + dimension)
        projection, decoder = make_decoder(rng, dimension, len(ALPHA), ridge=1e-3)
        out_l = decoder @ (projection @ reference_l)
        out_r = decoder @ (projection @ reference_r)
        raw_v = violations(out_l, out_r)
        proj_l, proj_r = project_cnf(out_l, out_r)
        reference.append({
            "dimension": dimension,
            "endpoint_rmse": float(np.sqrt(np.mean(np.r_[out_l - reference_l, out_r - reference_r] ** 2))),
            "projected_endpoint_rmse": float(np.sqrt(np.mean(np.r_[proj_l - reference_l, proj_r - reference_r] ** 2))),
            "raw_cnf_violation": bool(any(raw_v.values())),
            "projected_cnf_violation": bool(any(violations(proj_l, proj_r).values())),
        })

    return {
        "seed": seed,
        "repeats": repeats,
        "alpha_points": len(ALPHA),
        "reference_triangle": {"parameters": [2.0, 5.0, 9.0], "rows": reference},
        "grid_control": {
            "endpoint_rmse": 0.0,
            "cnf_violation_rate": 0.0,
            "description": "direct weighted alpha-cut endpoint averaging",
        },
        "summary": summary,
        "raw": results,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    payload = run()
    (out_dir / "preexperiment_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "preexperiment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload["summary"][0].keys())
        writer.writeheader(); writer.writerows(payload["summary"])

    dims = [row["dimension"] for row in payload["summary"]]
    rmse = [row["rmse_mean"] for row in payload["summary"]]
    violation = [row["raw_cnf_violation_rate"] for row in payload["summary"]]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax1.plot(dims, rmse, "o-", color="#155e75", label="endpoint RMSE")
    ax1.set_xlabel("hypervector dimension D")
    ax1.set_ylabel("RMSE (input units)", color="#155e75")
    ax1.set_xticks(dims)
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(dims, violation, "s--", color="#b91c1c", label="raw CNF violation rate")
    ax2.set_ylabel("raw CNF violation rate", color="#b91c1c")
    ax2.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "preexperiment_diagnostics.png", dpi=300)
    plt.close(fig)

    print(json.dumps({"summary": payload["summary"], "files": [
        str(out_dir / "preexperiment_results.json"),
        str(out_dir / "preexperiment_summary.csv"),
        str(out_dir / "preexperiment_diagnostics.png"),
    ]}, indent=2))


if __name__ == "__main__":
    main()
