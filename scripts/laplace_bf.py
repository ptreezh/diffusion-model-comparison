#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Laplace 近似（Kass & Raftery 1995）——无网格依赖的贝叶斯因子
   对数均匀先验：(log 维) pi(x)=1/(x ln(hi/lo))；(lin 维) pi(x)=1/(hi-lo)
   ln Z ≈ ll(MAP)+ln prior(MAP)+ (d/2)ln(2π) - 0.5 ln|det(-H)|，H=∇²(ll+ln prior)@MAP
"""
import csv, math, json
import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln

P_LO,P_HI=1e-5,0.5; Q_LO,Q_HI=0.01,5.0; R_LO,R_HI=0.005,3.0
B_LO,B_HI=0.01,10.0; C_LO,C_HI=0.005,3.0; T0_LO,T0_HI=-5.0,30.0; M_LO,M_HI=1.0,100.0

def bass_sales(t,m,p,q):
    s=p+q
    return m*s**2/p*np.exp(-s*t)/(1.0+(q/p)*np.exp(-s*t))**2
def logistic_sales(t,m,r,t0):
    z=np.exp(-r*(t-t0)); return m*r*z/(1.0+z)**2
def gompertz_sales(t,m,b,c,t0):
    z=np.exp(-c*(t-t0)); return m*b*c*z*np.exp(-b*z)

def lml(data,pred):
    n=len(data); sse=float(np.sum((data-pred)**2))
    if sse<=0: return -np.inf
    return -0.5*n*math.log(2*math.pi)+gammaln(n/2.0)-(n/2.0)*math.log(sse/2.0)-math.log(2.0)

def log_prior(th):
    # th: (m, ...)  先验密度对数
    lp=-math.log(M_LO);  # 常数，域内
    # log 维：pi=1/(x ln(hi/lo))
    lp-=math.log(th[0])  # m 的 log-uniform
    # 各模型追加维
    return lp

def spec_dims(kind):
    if kind=="bass": return [(M_LO,M_HI,"log"),(P_LO,P_HI,"log"),(Q_LO,Q_HI,"log")]
    if kind=="logistic": return [(M_LO,M_HI,"log"),(R_LO,R_HI,"log"),(T0_LO,T0_HI,"lin")]
    return [(M_LO,M_HI,"log"),(B_LO,B_HI,"log"),(C_LO,C_HI,"log"),(T0_LO,T0_HI,"lin")]

def lnprior_full(th,spec):
    lp=0.0
    for x,(lo,hi,kind) in zip(th,spec):
        if kind=="log":
            lp-=math.log(x*math.log(hi/lo))
        else:
            lp-=math.log(hi-lo)
    return lp

def fit(data,T,kind):
    d=len(spec_dims(kind))
    def pred(th):
        if kind=="bass": return bass_sales(T,th[0],th[1],th[2])
        if kind=="logistic": return logistic_sales(T,th[0],th[1],th[2])
        return gompertz_sales(T,th[0],th[1],th[2],th[3])
    lo=[s[0] for s in spec_dims(kind)]; hi=[s[1] for s in spec_dims(kind)]
    # 起始：log 维取几何中值，lin 取中点
    x0=[math.sqrt(a*b) if s[2]=="log" else (a+b)/2 for a,b,s in zip(lo,hi,spec_dims(kind))]
    def resid(th):
        return (pred(th)-data)/math.sqrt(max(np.var(data),1e-9))
    r=least_squares(resid,x0,bounds=(lo,hi),max_nfev=20000,xtol=1e-12,ftol=1e-12)
    if not r.success and r.cost>1e-6:
        r=least_squares(resid,x0,bounds=(lo,hi),method="dogbox",max_nfev=50000)
    return r.x

def hessian_ll(data,T,kind,th):
    d=len(th); eps=1e-4
    def f(x):
        if kind=="bass": return lml(data,bass_sales(T,x[0],x[1],x[2]))
        if kind=="logistic": return lml(data,logistic_sales(T,x[0],x[1],x[2]))
        return lml(data,gompertz_sales(T,x[0],x[1],x[2],x[3]))
    H=np.zeros((d,d))
    for i in range(d):
        for j in range(i,d):
            h=eps*(1+abs(th[i])); h2=eps*(1+abs(th[j]))
            xpp=th.copy(); xpp[i]+=h; xpp[j]+=h2
            xpm=th.copy(); xpm[i]+=h; xpm[j]-=h2
            xmp=th.copy(); xmp[i]-=h; xmp[j]+=h2
            xmm=th.copy(); xmm[i]-=h; xmm[j]-=h2
            v=(f(xpp)-f(xpm)-f(xmp)+f(xmm))/(4*h*h2)
            H[i,j]=H[j,i]=v
    return -H  # -Hessian(ll) 正定近似

def lnZ_laplace(data,T,kind):
    th=fit(data,T,kind)
    spec=spec_dims(kind); d=len(th)
    if kind=="bass": ll=lml(data,bass_sales(T,th[0],th[1],th[2]))
    elif kind=="logistic": ll=lml(data,logistic_sales(T,th[0],th[1],th[2]))
    else: ll=lml(data,gompertz_sales(T,th[0],th[1],th[2],th[3]))
    lp=lnprior_full(th,spec)
    H=hessian_ll(data,T,kind,th)
    # 加入先验曲率：d²(-ln prior)/dx²
    for i,(x,(lo,hi,k)) in enumerate(zip(th,spec)):
        if k=="log": H[i,i]+=1.0/(x*x)  # d²(-(-ln x·ln))/dx² = d²(ln x)/dx²=-1/x² → -H_prior=-(-1/x²)=1/x²
        else: H[i,i]+=0.0
    sign,logdet=np.linalg.slogdet(H)
    if sign<=0 or logdet<=-50:
        return None,th,ll,lp
    return ll+lp+(d/2)*math.log(2*math.pi)-0.5*logdet, th, ll, lp

def run_seq(name, share):
    d=np.diff(share); T=np.arange(len(d),dtype=float)
    out={"name":name,"n":len(d),"max":float(share.max())}
    for k in ("bass","logistic","gompertz"):
        z,th,ll,lp=lnZ_laplace(d,T,k)
        out[k+"_MAP"]=[float(v) for v in th]
        out[k+"_ll"]=ll; out[k+"_lnZ"]=None if z is None else z
    if None in (out["bass_lnZ"],out["logistic_lnZ"],out["gompertz_lnZ"]):
        out["BL"]=None; out["BG"]=None
    else:
        out["BL"]=out["bass_lnZ"]-out["logistic_lnZ"]
        out["BG"]=out["bass_lnZ"]-out["gompertz_lnZ"]
    print(f"{name:14s} n={len(d):2d} max={out['max']:5.1f}%  BL={out['BL'] if out['BL'] is None else round(out['BL'],3)}  "
          f"BG={out['BG'] if out['BG'] is None else round(out['BG'],3)}", flush=True)
    return out

if __name__=="__main__":
    import importlib.util
    spec=importlib.util.spec_from_file_location("cr","correct_rerun.py")
    cr=importlib.util.module_from_spec(spec); spec.loader.exec_module(cr)
    seqs=cr.load_all()
    res=[run_seq(nm,seqs[nm]) for nm in ["EV World","EV China","EV Europe","EV United States","Smartphone"]]
    json.dump(res,open("laplace_results.json","w"),indent=2,default=str)
    print("saved laplace_results.json")
