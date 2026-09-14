"""Paired bootstrap/sign tests with Holm correction for multi-dataset transfer."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'multidataset_evidence_by_seed.csv').open(encoding='utf-8')))
datasets=sorted({r['dataset'] for r in rows}); metrics=('brier_score','roc_auc','accuracy')
out=[]; rng=np.random.default_rng(917)
for ds in datasets:
    by={}
    for r in rows:
        if r['dataset']==ds: by.setdefault(r['method'],{})[int(r['seed'])]=r
    common=sorted(set(by['direct_evidence_stack']) & set(by['evidence_vsa_grid']))
    for metric in metrics:
        diff=np.array([float(by['evidence_vsa_grid'][s][metric])-float(by['direct_evidence_stack'][s][metric]) for s in common])
        boot=np.mean(diff[rng.integers(0,len(diff),size=(50000,len(diff)))],axis=1)
        nonzero=diff[diff!=0]
        p=float(binomtest(int(np.sum(nonzero<0)),len(nonzero),0.5,alternative='two-sided').pvalue) if len(nonzero) else 1.0
        out.append({'dataset':ds,'metric':metric,'baseline':'direct_evidence_stack','method':'evidence_vsa_grid','paired_seeds':len(common),'difference_mean':float(diff.mean()),'bootstrap_ci95_low':float(np.quantile(boot,.025)),'bootstrap_ci95_high':float(np.quantile(boot,.975)),'sign_test_p':p})
order=sorted(range(len(out)),key=lambda i:out[i]['sign_test_p'])
m=len(out)
for rank,i in enumerate(order): out[i]['holm_p']=min(1.0,(m-rank)*out[i]['sign_test_p'])
payload={'datasets':datasets,'metrics':list(metrics),'correction':'Holm step-down over dataset-metric paired sign tests','rows':out}
(ROOT/'multidataset_paired_stats.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
with (ROOT/'multidataset_paired_stats.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
print(json.dumps(out,indent=2))
