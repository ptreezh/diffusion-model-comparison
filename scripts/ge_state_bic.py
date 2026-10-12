#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""上篇重构：GE 州面板 37 序列正确口径（增量）+ BIC（确定性、域无关）"""
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
STATES = {"corn":["Illinois","Indiana","Iowa","Kansas","Michigan","Minnesota","Missouri","Nebraska","North Dakota","Ohio","South Dakota","Texas","Wisconsin"],
          "soybeans":["Arkansas","Illinois","Indiana","Iowa","Kansas","Michigan","Minnesota","Mississippi","Missouri","Nebraska","North Dakota","Ohio","South Dakota","Wisconsin"],
          "cotton":["Alabama","Arkansas","California","Georgia","Louisiana","Mississippi","Missouri","North Carolina","Tennessee","Texas"]}

def load(path, attr, state):
    out=[]
    for r in csv.DictReader(open(path,encoding="utf-8",errors="replace")):
        if r["State"].strip()==state and r["Attribute"].strip()==attr:
            v=r["Value"].strip()
            if v in ("",".","D","S"): continue
            out.append((int(r["Year"]),float(v)))
    out.sort(); return np.array([v for _,v in out],float)

results={}
rows=[]
for crop,states in STATES.items():
    for st in states:
        share=load(_DATA,ATTRS[crop],st)
        if len(share)<8: continue
        d=np.diff(share); T=np.arange(len(d),dtype=float); n=len(d)
        thb=lb.fit(d,T,"bass"); thl=lb.fit(d,T,"logistic"); thg=lb.fit(d,T,"gompertz")
        llb=lb.lml(d,lb.bass_sales(T,thb[0],thb[1],thb[2]))
        lll=lb.lml(d,lb.logistic_sales(T,thl[0],thl[1],thl[2]))
        llg=lb.lml(d,lb.gompertz_sales(T,thg[0],thg[1],thg[2],thg[3]))
        dBL=(-2*lll+3*math.log(n))-(-2*llb+3*math.log(n))
        dBG=(-2*llg+4*math.log(n))-(-2*llb+3*math.log(n))
        # MAP 边界命中检测
        bnd_b=any(thb[i]<=lb.spec_dims("bass")[i][0]*1.001 or thb[i]>=lb.spec_dims("bass")[i][1]*0.999 for i in range(3))
        bnd_l=any(thl[i]<=lb.spec_dims("logistic")[i][0]*1.001 or thl[i]>=lb.spec_dims("logistic")[i][1]*0.999 for i in range(3))
        results[f"{st}__{crop}"]={"n":n,"max":float(share.max()),"ll_B":llb,"ll_L":lll,"ll_G":llg,
            "DeltaBIC_BL":dBL,"DeltaBIC_BG":dBG,"bnd_B":bool(bnd_b),"bnd_L":bool(bnd_l)}
        rows.append((st,crop,n,share.max(),llb,lll,llg,dBL,dBG))
        print(f"{st:16s} {crop:8s} n={n:2d} max={share.max():4.0f}%  ΔBIC_BL={dBL:+6.2f} ΔBIC_BG={dBG:+6.2f}"+(f"  [B边界]" if bnd_b else "")+(f"  [L边界]" if bnd_l else ""),flush=True)

# 统计
import numpy as np
dBLs=[r[7] for r in rows]; dBGs=[r[8] for r in rows]; llb=[r[4] for r in rows]; lll=[r[5] for r in rows]
print("="*100)
print(f"n_series={len(rows)}  ΔBIC_BL: mean={np.mean(dBLs):+.3f} median={np.median(dBLs):+.3f} "
      f"min={min(dBLs):+.2f} max={max(dBLs):+.2f}")
print(f"ΔBIC_BG: mean={np.mean(dBGs):+.3f} median={np.median(dBGs):+.3f} min={min(dBGs):+.2f} max={max(dBGs):+.2f}")
eq=np.mean([abs(a-b)<0.01 for a,b in zip(llb,lll)]); print(f"ll_B=ll_L 精确相等比例: {eq*100:.0f}%")
print(f"ΔBIC_BL<=0 比例: {np.mean([x<=0.01 for x in dBLs])*100:.0f}%；ΔBIC_BL∈[0,1] 比例: {np.mean([0<x<=1 for x in dBLs])*100:.0f}%")
print(f"ΔBIC_BL>1（Bass明显好）: {np.mean([x>1 for x in dBLs])*100:.0f}%；ΔBIC_BG>0（Bass好于Gompertz）: {np.mean([x>0 for x in dBGs])*100:.0f}%")
json.dump(results,open(os.path.join(_HERE,"..","results","ge_state_bic_results.json"),"w"),indent=2,default=str)
print("saved ge_state_bic_results.json")
