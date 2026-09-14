"""System-level accounting for split, repetition, and unified carriers.

The audit reports measured byte counts from the manuscript layout and a
transparent sensitivity model for synchronization cost. The latency values
are not hardware measurements: ``commit_cost_us`` is an explicit system
parameter so readers can substitute their platform's commit/consistency cost.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D, GRID, FLOAT_BYTES, VSA_BYTES, ENCODER_BYTES = 4096, 101, 8, 32768, int(3.16 * 1024**2)
layouts = [
    ("grid_plus_vsa", 2*GRID*FLOAT_BYTES + VSA_BYTES, 2),
    ("repetition_plus_vsa", 2*D*FLOAT_BYTES + VSA_BYTES, 2),
    ("hdt_unified", 2*D*FLOAT_BYTES + VSA_BYTES, 1),
]
rows = []
for name, bytes_rule, sync_units in layouts:
    for rules in (100, 1000, 10000):
        for updates in (1000, 1_000_000):
            for commit_cost_us in (0.5, 1.0, 5.0, 10.0):
                rows.append({
                    "layout": name, "rules": rules, "updates": updates,
                    "bytes_per_rule": bytes_rule, "total_storage_mib": bytes_rule*rules/(1024**2),
                    "sync_units_per_update": sync_units, "commit_cost_us": commit_cost_us,
                    "sync_time_us": updates*sync_units*commit_cost_us,
                    "sync_time_saved_vs_split_us": updates*(2-sync_units)*commit_cost_us,
                    "shared_encoder_bytes": ENCODER_BYTES,
                })
payload = {"dimension_D": D, "grid_points": GRID, "float_bytes": FLOAT_BYTES,
           "vsa_bytes_per_rule": VSA_BYTES, "shared_encoder_bytes": ENCODER_BYTES,
           "rows": rows, "interpretation": "synchronization cost is a sensitivity parameter, not a hardware benchmark"}
(ROOT / "carrier_system_audit_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
with (ROOT / "carrier_system_audit_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
print(json.dumps({"layouts": layouts, "example_100_rules_1e6_updates_5us": [r for r in rows if r["rules"] == 100 and r["updates"] == 1_000_000 and r["commit_cost_us"] == 5.0]}, indent=2))
