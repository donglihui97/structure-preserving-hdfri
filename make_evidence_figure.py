from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
rows = []
import json
payload = json.loads((root / "evidence_hdfri_results.json").read_text(encoding="utf-8"))
summary = {row["method"]: row for row in payload["summary"]}
methods = ["unsupervised_cp_hdfri", "logistic_regression", "knn_7", "qda_tuned",
           "rbf_svc_calibrated", "direct_evidence_stack", "evidence_hdfri"]
labels = ["Unsupervised\nCP-HDFRI", "Logistic", "7-NN", "QDA", "RBF-SVC",
          "Direct evidence\nstack", "Evidence-\nCP-HDFRI"]
brier = [0.0618619, summary["logistic_regression"]["brier_score_mean"],
         summary["knn_7"]["brier_score_mean"], summary["qda_tuned"]["brier_score_mean"],
         summary["rbf_svc_calibrated"]["brier_score_mean"],
         summary["direct_evidence_stack"]["brier_score_mean"],
         summary["evidence_hdfri"]["brier_score_mean"]]
auc = [0.9847328, summary["logistic_regression"]["roc_auc_mean"],
       summary["knn_7"]["roc_auc_mean"], summary["qda_tuned"]["roc_auc_mean"],
       summary["rbf_svc_calibrated"]["roc_auc_mean"],
       summary["direct_evidence_stack"]["roc_auc_mean"],
       summary["evidence_hdfri"]["roc_auc_mean"]]
n_splits = len(payload["outer_seeds"])
brier_sd = [0.00802, summary["logistic_regression"]["brier_score_seed_std"],
            summary["knn_7"]["brier_score_seed_std"], summary["qda_tuned"]["brier_score_seed_std"],
            summary["rbf_svc_calibrated"]["brier_score_seed_std"],
            summary["direct_evidence_stack"]["brier_score_seed_std"],
            summary["evidence_hdfri"]["brier_score_seed_std"]]
auc_sd = [0.00689, summary["logistic_regression"]["roc_auc_seed_std"],
          summary["knn_7"]["roc_auc_seed_std"], summary["qda_tuned"]["roc_auc_seed_std"],
          summary["rbf_svc_calibrated"]["roc_auc_seed_std"],
          summary["direct_evidence_stack"]["roc_auc_seed_std"],
          summary["evidence_hdfri"]["roc_auc_seed_std"]]
brier_ci = 1.96 * np.asarray(brier_sd) / np.sqrt(n_splits)
auc_ci = 1.96 * np.asarray(auc_sd) / np.sqrt(n_splits)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
x = np.arange(len(labels))
facecolors = ["#4b5563"] + ["#d1d5db"] * 4 + ["#2563a6", "white"]
edgecolors = ["#4b5563"] + ["#9ca3af"] * 4 + ["#2563a6", "#2563a6"]
hatches = [None] * 6 + ["////"]
for i in range(len(labels)):
    axes[0].bar(x[i], brier[i], width=0.72, color=facecolors[i], edgecolor=edgecolors[i],
                hatch=hatches[i], linewidth=1.2, yerr=brier_ci[i], capsize=3,
                error_kw={"elinewidth": 1.0, "ecolor": "#374151"})

markers = ["s", "o", "o", "o", "o", "o", "D"]
for i in range(len(labels)):
    axes[1].errorbar(x[i], auc[i], yerr=auc_ci[i], marker=markers[i], markersize=6,
                     linestyle="none", color=edgecolors[i], markerfacecolor=facecolors[i],
                     markeredgecolor=edgecolors[i], markeredgewidth=1.2, capsize=3,
                     elinewidth=1.0)
axes[0].set_title("Probability calibration")
axes[0].set_ylabel("Brier score")
axes[1].set_title("Ranking (expanded vertical axis)")
axes[1].set_ylabel("ROC AUC")
axes[1].set_ylim(0.98, 1.0005)
axes[1].axhline(1.0, color="#9ca3af", linestyle=":", linewidth=0.9)
for ax in axes:
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
for i, ax in enumerate(axes): ax.text(0.0, 1.02, f"({chr(97+i)})", transform=ax.transAxes, va="bottom", ha="left", fontweight="bold", clip_on=False)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(root / "evidence_wdbc_diagnostics.png", dpi=300)
plt.close(fig)
