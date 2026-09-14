"""Aggregate reliable-mask RMSE and deployment coverage from fairness rows."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
payload = json.loads((ROOT / "hdt_fairness_results.json").read_text(encoding="utf-8"))
for row in payload["summary"]:
    subset = [r for r in payload["rows"] if r["dimension"] == row["dimension"] and r["erasure_rate"] == row["erasure_rate"] and r["method"] == row["method"]]
    if row["method"] == "hdt_mask_aware":
        values = [r["reliable_mask_rmse_mean"] for r in subset if r["reliable_mask_rmse_mean"] is not None]
        row["reliable_mask_rmse_mean"] = float(np.mean(values)) if values else None
        row["reliable_mask_coverage"] = float(np.mean([r["reliable_mask_coverage"] for r in subset]))
        row["rejected_mask_rate"] = float(np.mean([r["rejected_mask_rate"] for r in subset]))
    else:
        row["reliable_mask_rmse_mean"] = None
        row["reliable_mask_coverage"] = None
        row["rejected_mask_rate"] = None
(ROOT / "hdt_fairness_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
with (ROOT / "hdt_fairness_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=payload["summary"][0].keys()); writer.writeheader(); writer.writerows(payload["summary"])
print(json.dumps([r for r in payload["summary"] if r["dimension"] == 1024 and r["erasure_rate"] == 0.7 and r["method"] == "hdt_mask_aware"], indent=2))
