# Supplementary Materials — README

**Manuscript:** Window-Dependent Identifiability of Three Diffusion-Model Families: Corrected Bayesian Evidence from Fifty-One Agricultural and Durable-Goods Adoption Series
**Revised resubmission of withdrawn manuscript TFS-D-26-09877**

## 1. Reproducibility commitment

All quantitative claims in the manuscript are regenerated from the public USDA dataset and from deterministic reconstructions of Bass (1969), using a corrected, deterministic protocol (increment-based likelihood, correctly mapped parameter domain, BIC/MAP statistics). No random draws are used in the reported empirical results; the simulation benchmark (Section 5) uses seeded generation as described in the manuscript.

## 2. Data sources

| Dataset | Source | File |
|---|---|---|
| GE crop adoption, state-level and national, 2000–2024 | USDA Economic Research Service | `data/usda_ge_2000_24.csv` (columns: Attribute, State, Year, Value, Table) |
| Bass (1969) durable-goods parameters | Bass (1969), Table I, reproduced in the manuscript | encoded in `scripts/bass1969_bic_correct.py` |

The state panel comprises 37 series (13 corn, 14 soybeans, 10 cotton); the national series comprise corn, soybeans, and cotton.

## 3. Protocol (identical to the manuscript)

1. **Increments:** For each share series s(t), fit the model to the annual increments d(t) = s(t) − s(t−1).
2. **Model families:** Bass (m,p,q), logistic (m,a,b), Gompertz (m,a,b,c) density forms, as defined in the manuscript Section 3.1.
3. **Optimization:** multi-start least squares on the increment likelihood (`laplace_bf.fit`), MAP parameters recovered.
4. **Comparison:** ΔBIC(A,B) = BIC_A − BIC_B with BIC = −2ℓ + k·ln n (Kass & Raftery, 1995). ΔBIC(Bass,Logistic) reduces to 2(ℓ_L − ℓ_B) since both families have three parameters.
5. **Boundary detection:** a fit is flagged as boundary-hitting when any MAP parameter sits within 0.1% of its domain edge (`bnd_B` / `bnd_L` flags in the result JSONs).

## 4. Scripts and how to reproduce each table

| Table (manuscript) | Script | Output |
|---|---|---|
| Table 1 (37 state series) | `python3 scripts/ge_state_bic.py` | `results/ge_state_bic_results.json` |
| Table 2 (3 national series) | `python3 scripts/verify_national_bic.py` | `results/ge_national_bic_results.json` |
| Table 2 (11 Bass 1969 reconstructions) | `python3 scripts/bass1969_bic_correct.py` | `results/bass1969_bic_correct_results.json` |

Dependency: `scripts/laplace_bf.py` must be importable (the three scripts import it from the same directory). Requires numpy, scipy, Python ≥ 3.8.

## 5. Verification results (run 2026-10-12, identical to the manuscript)

- **37 state series:** ΔBIC(Bass,Logistic) median exactly 0.000, mean +0.164, range [−7.25, +11.20]; 18/37 ≤ 0; boundary-free subsample (n=16) all equal to 0.000 to machine precision, 15/16 exact.
- **National series:** corn ΔBIC_BL = 0.00 (exact), cotton +0.24, soybeans +6.50 (Bass MAP at domain boundary; domain-sensitivity protocol in manuscript Section 4.3 quantifies this as an artifact).
- **Bass (1969) reconstructions (11):** ΔBIC_BL ∈ [−2.86, 0.00], 9/11 exact zero; ΔBIC_BG ∈ [−26.30, −2.96].

## 6. Relationship to the withdrawn version

The earlier submission (TFS-D-26-09877) reported constant log Bayes factors near three computed by quadrature; that computation contained two implementation errors (a data-scale error — shares vs. increments — and an integration-domain mapping error), discovered in an independent mpmath (60-digit) re-implementation. Both are corrected here by construction: the protocol uses increment likelihoods and no quadrature. The artifact diagnosis in manuscript Section 6 documents, on the corrected data, why the conventional quadrature Bayes factors still produce the "near three" band.

## 7. Environment and numerical note

- Requires Python >= 3.8, numpy, scipy. Verified with numpy 1.26.x / scipy 1.11.x (2026-10-12).
- All reported statistics are reproducible to machine precision (IEEE-754 double). The three scripts are self-contained: run them from this package root as `python3 scripts/<script>.py`; data are read from `data/`, results are written to `results/`.
- For a small number of state series, the MAP log-likelihood recovered by `scipy.optimize.least_squares` differs between runs at the <= 1e-9 level (convergence tolerance); none of the reported BIC statistics (medians, means, ranges, counts at the 1e-9 threshold) changes under this perturbation.
- Independent verification (2026-10-12): scripts regenerated `ge_state_bic_results.json`, `ge_national_bic_results.json`, and `bass1969_bic_correct_results.json` from scratch; national and Bass-1969 outputs match the committed results bit-for-bit, and the state output matches all manuscript statistics.
