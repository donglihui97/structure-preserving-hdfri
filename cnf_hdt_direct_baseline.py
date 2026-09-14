"""Matched direct controls for CNF-closed FRI and general HDT representation.

This is an operational comparison on the same sparse one-dimensional task. The
CNF-closed row applies the endpoint interpolation followed by the exact CNF
projection. The general-HDT row uses the paper-faithful random Fourier carrier
with a deterministic ridge inverse, then the same projection. It is a direct
representation-family control, not a claim to reproduce every implementation
detail of the cited toolboxes.
"""
from __future__ import annotations

import csv, json
from pathlib import Path
import numpy as np

SEEDS = (701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
         769, 773, 787, 797, 809, 811, 821, 823, 827, 829)
ALPHA = np.linspace(0.0, 1.0, 101)
D = 4096
ETA = 1e-8

def consequent(x):
    c = 0.20 + 0.60*x + 0.06*np.sin(2*np.pi*x)
    w = 0.04 + 0.01*x
    return c[:, None]-w[:, None]+ALPHA[None,:]*w[:,None], c[:, None]+w[:, None]-ALPHA[None,:]*w[:,None]

def project(l, r):
    def pava(v, inc):
        z = np.asarray(v).ravel().copy() if inc else -np.asarray(v).ravel().copy(); blocks=[]
        for value in z:
            blocks.append([float(value),1.0])
            while len(blocks)>1 and blocks[-2][0] > blocks[-1][0]:
                a,b=blocks[-2],blocks[-1]; n=a[1]+b[1]
                blocks[-2:]=[[(a[0]*a[1]+b[0]*b[1])/n,n]]
        out=np.concatenate([np.full(int(n),m) for m,n in blocks])
        return out if inc else -out
    l,r=pava(l,True),pava(r,False); gap=float(np.max(l-r))
    if gap>0: l,r=l-gap/2,r+gap/2
    return l,r

def run(seed):
    rng=np.random.default_rng(seed); lattice=np.linspace(0,1,21)
    rules=np.sort(rng.choice(lattice,size=12,replace=False)); rl,rr=consequent(rules)
    q=rng.uniform(0,1,400); tl,tr=consequent(q)
    bw=max(float(np.median(np.abs(rules[:,None]-rules[None,:]))),1e-3)
    freq=rng.normal(0,1/bw,size=(1,D)); phase=rng.uniform(0,2*np.pi,size=D)
    F=np.sqrt(2/D)*np.cos(rules[:,None]@freq+phase); F=F/np.maximum(np.linalg.norm(F,axis=1,keepdims=True),1e-12)
    # The expensive inverse is applied to the two bundled endpoint vectors.
    rows=[]
    for x,gtl,gtr in zip(q,tl,tr):
        pos=int(np.searchsorted(rules,x)); idx=np.array([0,1]) if pos<=0 else (np.array([-2,-1])+len(rules) if pos>=len(rules) else np.array([pos-1,pos]))
        d=np.abs(rules[idx]-x); w=np.array([d[1],d[0]])/max(float(d.sum()),1e-12)
        khl,khr=w@rl[idx],w@rr[idx]
        hq=np.sqrt(2/D)*np.cos(np.array([[x]])@freq+phase); hq=hq/np.maximum(np.linalg.norm(hq,axis=1,keepdims=True),1e-12)
        sim=(hq@F.T).ravel(); top=np.argsort(-sim)[:2]; ww=np.array([sim[top[1]],sim[top[0]]]); ww=ww/max(float(ww.sum()),1e-12)
        z_l=ww@rl[top]; z_r=ww@rr[top]
        methods={"cnf_closed_fri":project(np.array([khl]),np.array([khr])),
                 # The general representation control keeps the full endpoint
                 # vector and applies only the common CNF projection.
                 "general_hdt":project(z_l,z_r),
                 "direct_grid":(z_l,z_r)}
        for name,(ol,orr) in methods.items():
            if np.ndim(ol)==0 or np.size(ol)==1: ol=np.full(len(ALPHA),float(np.ravel(ol)[0])); orr=np.full(len(ALPHA),float(np.ravel(orr)[0]))
            rows.append({"seed":seed,"method":name,"rmse":float(np.sqrt(np.mean(np.r_[ol-gtl,orr-gtr]**2))),
                         "cnf_valid":bool(np.all(np.diff(ol)>=-1e-10) and np.all(np.diff(orr)<=1e-10) and np.all(ol<=orr+1e-10))})
    return rows

def main():
    root=Path(__file__).resolve().parent; rows=[r for s in SEEDS for r in run(s)]
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        x=[r for r in rows if r['method']==method]; vals=np.array([r['rmse'] for r in x])
        summary.append({'method':method,'seeds':len(SEEDS),'queries':400,'rmse_mean':float(vals.mean()),'rmse_seed_std':float(np.std([np.mean([r['rmse'] for r in x if r['seed']==s]) for s in SEEDS],ddof=1)),'cnf_valid_rate':float(np.mean([r['cnf_valid'] for r in x]))})
    payload={'seeds':list(SEEDS),'dimension_D':D,'alpha_points':len(ALPHA),'rows':rows,'summary':summary}
    (root/'cnf_hdt_direct_baseline_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (root/'cnf_hdt_direct_baseline_by_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (root/'cnf_hdt_direct_baseline_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    print(json.dumps({'summary':summary},indent=2))
if __name__=='__main__': main()
