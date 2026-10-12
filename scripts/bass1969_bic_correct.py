#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bass(1969) 11 durables -- CORRECT BIC protocol (increments, MAP, no quadrature).
Reconstruct each product's sales curve deterministically from Bass 1969 Table I
(m,p,q,period), scale-normalize window max to 100, add 0.2% peak noise (seed 2026),
fit increments to Bass/Logistic/Gompertz densities, report BIC differences.
"""
import json, math
import numpy as np
from scipy.optimize import least_squares
import os
import laplace_bf as lbf
_HERE = os.path.dirname(os.path.abspath(__file__))

PRODUCTS = [
    ("Electric refrigerators", 1920, 1940, 40001, 0.002617, 0.21566),
    ("Home freezers", 1946, 1961, 21973, 0.018119, 0.17110),
    ("B&W television", 1946, 1961, 96717, 0.027877, 0.25105),
    ("Water softeners", 1949, 1961, 5793, 0.017703, 0.29695),
    ("Room air conditioners", 1946, 1961, 16895, 0.010399, 0.41861),
    ("Clothes dryers", 1948, 1961, 15092, 0.017206, 0.35688),
    ("Power lawn-mowers", 1948, 1961, 44751, 0.009184, 0.33790),
    ("Electric bed coverings", 1949, 1961, 76589, 0.005876, 0.24387),
    ("Automatic coffee makers", 1948, 1961, 58838, 0.017135, 0.30145),
    ("Steam irons", 1949, 1960, 55696, 0.028632, 0.32791),
    ("Record players", 1952, 1961, 21937, 0.024796, 0.65410),
]

def bass_curve(t, m, p, q):
    s = p + q
    return m * s**2 / p * np.exp(-s * t) / (1.0 + (q / p) * np.exp(-s * t))**2

def bic_diff(share):
    d = np.diff(share)
    n = len(d)
    T = np.arange(n, dtype=float)
    res = {}
    for k in ("bass", "logistic", "gompertz"):
        th = lbf.fit(d, T, k)
        if k == "bass":
            pred = lbf.bass_sales(T, th[0], th[1], th[2])
        elif k == "logistic":
            pred = lbf.logistic_sales(T, th[0], th[1], th[2])
        else:
            pred = lbf.gompertz_sales(T, th[0], th[1], th[2], th[3])
        res[k] = {"MAP": [float(v) for v in th], "ll": lbf.lml(d, pred)}
    dBL = 2.0 * (res["logistic"]["ll"] - res["bass"]["ll"])      # penalty cancels
    dBG = 2.0 * (res["gompertz"]["ll"] - res["bass"]["ll"]) - math.log(n)  # (4-3) ln n
    return dBL, dBG, res

rng = np.random.default_rng(2026)
out = {}
for name, y0, y1, m, p, q in PRODUCTS:
    N = y1 - y0 + 1
    t = np.arange(N, dtype=float)
    y = bass_curve(t, m, p, q)
    y = y / np.max(y) * 100.0
    y = y + rng.normal(0.0, 0.2, y.size)
    dBL, dBG, res = bic_diff(y)
    out[name] = {"N": N, "DeltaBIC_BL": dBL, "DeltaBIC_BG": dBG,
                 "ll_B": res["bass"]["ll"], "ll_L": res["logistic"]["ll"], "ll_G": res["gompertz"]["ll"]}
    print(f"{name:26s} N={N:2d}  dBL={dBL:+8.4f}  dBG={dBG:+8.4f}")

bl = [v["DeltaBIC_BL"] for v in out.values()]
bg = [v["DeltaBIC_BG"] for v in out.values()]
zero = sum(1 for x in bl if abs(x) < 1e-6)
print("=" * 70)
print(f"ΔBIC_BL range [{min(bl):.4f}, {max(bl):.4f}]  |Δ|<1e-6 exact: {zero}/11")
print(f"ΔBIC_BG range [{min(bg):.4f}, {max(bg):.4f}]  median {np.median(bg):.4f}")
with open(os.path.join(_HERE,"..","results","bass1969_bic_correct_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved bass1969_bic_correct_results.json")
