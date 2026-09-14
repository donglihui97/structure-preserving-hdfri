"""Reproduce the published KH CNF benchmark of Alzubi and Kovacs.

The benchmark cases and trapezoidal parameters are transcribed from Tables I-VII
of arXiv:1911.05041.  For each alpha level, the original KH construction
interpolates lower and upper endpoints using the corresponding antecedent
endpoint distances.  The CP repair is the same monotone-and-order projection
used by CP-HDFRI; it is evaluated here as a structural postcondition test, not
as a claim that this is a full reimplementation of every KH toolbox detail.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ALPHA = np.linspace(0.0, 1.0, 101)


CASES = {
    1: ((1, 2, 2, 3), (7, 8, 8, 9), (2, 2, 2, 2), (8, 8, 8, 8), (4, 5, 5, 6)),
    2: ((1, 2.5, 2.5, 4), (6, 7.5, 7.5, 9), (1, 2.5, 2.5, 4), (6, 7.5, 7.5, 9), (4.5, 5, 5, 5.5)),
    3: ((1, 2, 3, 4), (6, 7, 8, 9), (1, 2, 3, 4), (6, 7, 8, 9), (4, 4.8, 5.2, 6)),
    4: ((1.5, 2, 2, 2.5), (6.5, 7, 7, 7.5), (1, 2, 3, 4), (6, 7, 8, 9), (4.5, 4.5, 4.5, 4.5)),
    5: ((2, 2, 2, 2), (8, 8, 8, 8), (1, 2, 3, 4), (6, 7, 8, 9), (4.5, 5, 5, 5.5)),
    6: ((1, 2, 3, 4), (6, 7, 8, 9), (1.5, 2.5, 2.5, 3.8), (6.5, 7.5, 7.5, 9), (4.2, 5.2, 5.2, 6.7)),
    7: ((1, 2.5, 2.5, 4), (5.5, 7.5, 7.5, 9), (1, 2, 3, 4.5), (6.5, 7, 8, 9.5), (4.5, 4.9, 5.1, 5.5)),
    8: ((1.5, 2.5, 2.5, 4.3), (6.5, 7.5, 7.5, 8.8), (1, 2, 3, 3.5), (6, 7, 8, 8.9), (4.5, 4.9, 5.1, 5.5)),
    9: ((2, 2, 2.5, 3), (6, 7.5, 8, 8), (2, 2, 2, 2), (8, 8, 8, 8), (5, 5, 5, 5)),
}


def alpha_cut(trap: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    a1, a2, a3, a4 = trap
    left = a1 + (a2 - a1) * ALPHA
    right = a4 - (a4 - a3) * ALPHA
    return left, right


def kh_endpoint_interpolation(a1, a2, b1, b2, obs):
    """KH endpoint interpolation with alpha-dependent lower/upper distances."""
    a1_l, a1_r = alpha_cut(a1)
    a2_l, a2_r = alpha_cut(a2)
    b1_l, b1_r = alpha_cut(b1)
    b2_l, b2_r = alpha_cut(b2)
    o_l, o_r = alpha_cut(obs)

    def blend(x1, x2, xo, y1, y2):
        d1 = np.abs(xo - x1)
        d2 = np.abs(x2 - xo)
        total = d1 + d2
        w1 = np.divide(d2, total, out=np.full_like(total, 0.5), where=total > 1e-12)
        w2 = 1.0 - w1
        return w1 * y1 + w2 * y2

    # Lower and upper endpoints use their matching antecedent distances.
    left = blend(a1_l, a2_l, o_l, b1_l, b2_l)
    right = blend(a1_r, a2_r, o_r, b1_r, b2_r)
    return left, right


def project_cnf(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone endpoint repair followed by a common translation if needed."""
    def pava(values: np.ndarray, increasing: bool) -> np.ndarray:
        x = values.copy()
        if not increasing:
            x = -x
        blocks: list[list[float]] = []
        for value in x:
            blocks.append([float(value), 1.0])
            while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
                total = blocks[-2][1] + blocks[-1][1]
                mean = (blocks[-2][0] * blocks[-2][1] + blocks[-1][0] * blocks[-1][1]) / total
                blocks[-2:] = [[mean, total]]
        out = np.concatenate([np.full(int(weight), mean) for mean, weight in blocks])
        return out if increasing else -out

    l = pava(left, True)
    r = pava(right, False)
    gap = float(np.max(l - r))
    if gap > 0:
        l = l - gap / 2.0
        r = r + gap / 2.0
    return l, r


def diagnostics(left: np.ndarray, right: np.ndarray) -> dict:
    return {
        "cnf_valid": bool(np.all(np.diff(left) >= -1e-10) and np.all(np.diff(right) <= 1e-10) and np.all(left <= right + 1e-10)),
        "monotonicity_violation": float(max(0.0, -np.min(np.diff(left)), -np.min(-np.diff(right)))),
        "max_crossing": float(max(0.0, np.max(left - right))),
        "min_gap": float(np.min(right - left)),
    }


def run() -> dict:
    records = []
    for case_id, (a1, a2, b1, b2, obs) in CASES.items():
        raw_l, raw_r = kh_endpoint_interpolation(a1, a2, b1, b2, obs)
        repaired_l, repaired_r = project_cnf(raw_l, raw_r)
        raw = diagnostics(raw_l, raw_r)
        repaired = diagnostics(repaired_l, repaired_r)
        records.append({
            "case": case_id,
            "published_group": "normal" if case_id <= 5 else "abnormal",
            "raw_cnf_valid": raw["cnf_valid"],
            "raw_monotonicity_violation": raw["monotonicity_violation"],
            "raw_max_crossing": raw["max_crossing"],
            "raw_min_gap": raw["min_gap"],
            "repaired_cnf_valid": repaired["cnf_valid"],
            "repair_rms_displacement": float(np.sqrt(np.mean(np.r_[repaired_l - raw_l, repaired_r - raw_r] ** 2))),
        })
    return {"source": "Alzubi and Kovacs, arXiv:1911.05041, Tables I-VII", "alpha_points": len(ALPHA), "records": records}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    payload = run()
    (out_dir / "published_cnf_benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "published_cnf_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload["records"][0].keys())
        writer.writeheader(); writer.writerows(payload["records"])

    cases = [r["case"] for r in payload["records"]]
    raw_cross = [r["raw_max_crossing"] for r in payload["records"]]
    repaired_disp = [r["repair_rms_displacement"] for r in payload["records"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(np.array(cases) - 0.18, raw_cross, width=0.36, label="KH max crossing")
    ax.bar(np.array(cases) + 0.18, repaired_disp, width=0.36, label="CP repair RMS displacement")
    ax.set_xlabel("Published benchmark case")
    ax.set_ylabel("Endpoint scale")
    ax.set_xticks(cases)
    ax.legend(frameon=False)
    ax.set_title("Published CNF benchmark: KH output and CP repair")
    fig.tight_layout()
    fig.savefig(out_dir / "published_cnf_benchmark_diagnostics.png", dpi=300)
    plt.close(fig)
    print(json.dumps({"files": [str(out_dir / "published_cnf_benchmark_results.json"), str(out_dir / "published_cnf_benchmark_summary.csv"), str(out_dir / "published_cnf_benchmark_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
