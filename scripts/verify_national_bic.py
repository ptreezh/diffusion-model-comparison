#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 paper_final.md Table 2 的 GE national 三系列 ΔBIC（增量口径+BIC，与 ge_state_bic.py 完全一致）"""
import csv, math, json
import numpy as np
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import laplace_bf as lb
_DATA = os.path.join(_HERE, "..", "data", "usda_ge_2000_24.csv")

ATTRS = {"corn":"All GE varieties (percent of all corn planted)",
         "soybeans":"All GE varieties (percent of all soybeans planted)",
         "cotton":"All GE varieties (percent of all upland cotton planted)"}

def load_national(path, attr):
    out=[]
    for r in csv.DictReader(open(path,encoding="utf-8",errors="replace")):
        if r["State"].strip()=="United States" and r["Attribute"].strip()==attr:
            v=r["Value"].strip()
            if v in ("",".","D","S"): continue
            out.append((int(r["Year"]),float(v)))
    out.sort(); return np.array([v for _,v in out],float)

results={}
for crop,attr in ATTRS.items():
    share=load_national(_DATA,attr)
    d=np.diff(share); T=np.arange(len(d),dtype=float); n=len(d)
    thb=lb.fit(d,T,"bass"); thl=lb.fit(d,T,"logistic"); thg=lb.fit(d,T,"gompertz")
    llb=lb.lml(d,lb.bass_sales(T,thb[0],thb[1],thb[2]))
    lll=lb.lml(d,lb.logistic_sales(T,thl[0],thl[1],thl[2]))
    llg=lb.lml(d,lb.gompertz_sales(T,thg[0],thg[1],thg[2],thg[3]))
    dBL=(-2*lll+3*math.log(n))-(-2*llb+3*math.log(n))
    dBG=(-2*llg+4*math.log(n))-(-2*llb+3*math.log(n))
    bnd_b=any(thb[i]<=lb.spec_dims("bass")[i][0]*1.001 or thb[i]>=lb.spec_dims("bass")[i][1]*0.999 for i in range(3))
    results[crop]={"n":n,"max":float(share.max()),"ll_B":llb,"ll_L":lll,"ll_G":llg,
        "DeltaBIC_BL":dBL,"DeltaBIC_BG":dBG,"bnd_B":bool(bnd_b),"theta_B":list(thb),"theta_L":list(thl)}
    print(f"national {crop:8s} n={n:2d} max={share.max():4.0f}%  ΔBIC_BL={dBL:+6.2f} ΔBIC_BG={dBG:+6.2f}"+(f"  [B边界]" if bnd_b else ""),flush=True)

print("="*60)
for crop in ATTRS:
    r=results[crop]
    print(f"{crop}: ΔBIC_BL={r['DeltaBIC_BL']:.4f} (论文 Table 2: corn 0.00 / cotton +0.24 / soybeans +6.50)")
json.dump(results,open(os.path.join(_HERE,"..","results","ge_national_bic_results.json"),"w"),indent=2,default=str)
print("saved ge_national_bic_results.json")
