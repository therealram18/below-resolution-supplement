To remove everything: delete `camera_ready/_refenv` and run `brew uninstall micromamba` (Homebrew also pulled small dependencies of micromamba, e.g. rpm/openssl; `brew autoremove` is not run here).

# Cross-check against reference software (DESIGN_ROUND2.md, S2), 45 GIFT-Eval pairs, MASE_Rank

Versions: micromamba 2.9.0; R 4.5.3 (conda-forge, `_refenv/r`); sandwich 3.1.3, lmtest 0.9.40, collapse 2.1.9 (conda-forge);
clubSandwich 0.7.0 and dfadjust 1.1.0 (CRAN, source-built into `_refenv/rlib`; conda-forge has neither). Python (`_refenv/py`, uv): wildboottest 0.3.2,
statsmodels 0.15.0, pandas 3.1.1, numpy 2.5.4, scipy 1.18.1. fwildclusterboot 0.14.3 installed but is unusable here (see c).
Files: `refcheck_export.py` (data and our values; asserts equality with `retest_gift.json` to 1e-10), `refcheck.R`, `refcheck_wild.py`
(run with `_refenv/py/bin/python`), `refcheck_summarize.py`, results in `refcheck_results.json` (summary plus per-pair rows).

## a. clubSandwich: PASS
CR2 SE and Satterthwaite df, independent working model, `coef_test(lm(d ~ 1), "CR2", cluster = dataset, "Satterthwaite")`.
Largest relative difference over 45 pairs: SE 2.5e-15, df 6.2e-16 (p-value 8.6e-15). Thresholds were 1e-8 and 1e-6.

## b. dfadjust
What it does (read from the source of `dfadjustSE`): the reported "HC2 se" is the CR2 variance with the symmetric inverse square root of I - H_gg
per cluster, the same as ours (SE agrees to 2.5e-15). With `IK = TRUE` it estimates a working covariance sig*I + rho*(block of ones) from the
OLS residuals u: rho = (sum_g (sum_{i in g} u_i)^2 - sum u^2) / (sum_g n_g^2 - N), sig = max(sum u^2 / N - rho, 0). So its rho is a residual
cross-product moment, not our ANOVA intraclass correlation. It does NOT floor rho at zero by default (12 of 45 pairs have negative rho);
`rho0 = TRUE` floors rho at 0 (then sig = sum u^2/N - rho). sig was never floored here. Our correlation parameter equals rho / (sig + rho)
(range -0.084 to 0.324; ours: 0 to 0.511, floored at 0).
Note: dfadjustSE 1.1.0 crashes on an intercept-only model (`Q[ord, ]` loses its matrix shape when K = 1). `refcheck.R` evaluates an in-memory copy
with `drop = FALSE` added at that one spot; the installed package is untouched.

(i) Our `satterthwaite_df(cl, rho)` fed its correlation (rho/(sig+rho)): max relative difference to its df 1.5e-14. PASS (also with rho0 = TRUE).
(ii) Shipped df and p against ours (CR2, rho = icc_hat): largest df difference 4.80 (dfadjust larger), largest p difference 0.0037
(with rho0 = TRUE: 2.33 and 0.0032). p for dfadjust = 2 pt(-|t|, its df), t = estimate / HC2 se.

| version | neighbours p<.05 (of 9) | pairs p<.05 (of 45) | Holm over 9 | Holm over 45 |
|---|---|---|---|---|
| ours, rho = icc_hat | 1 | 12 | 0 | 0 |
| ours, independent (rho = 0) | 1 | 13 | 0 | 0 |
| dfadjust as shipped | 1 | 12 | 0 | 1 |
| dfadjust, rho0 = TRUE | 1 | 12 | 0 | 0 |

The single Holm-over-45 rejection under dfadjust as shipped is a pair that ours does not reject: conclusions differ only through the correlation estimator.

## c. wildboottest (Python): PASS
It bootstraps the CR1-type studentized statistic with the null imposed: t = beta / sqrt(ssc * R V R'), ssc = (N-1)/(N-k) * G/(G-1) (identical in the
bootstrap denominators, so it cancels in p; with k = 1 it equals ours exactly). Bootstrap t uses the "11" scores (null-restricted residuals, cluster sums).
p = mean(|t_obs| < |t_boot|), with no +1 correction. It switches to full enumeration only if 2^G < B; here 2^28 > 99,999, so it uses random draws.
Settings: Rademacher, `bootstrap_type="11"`, B = 99,999, seed 20261111.
- t statistic vs ours: largest absolute difference 8.9e-16.
- p vs stored WCB-Rad (rule 4 x sqrt(2p(1-p)/B)): 45 of 45 pass; largest difference 0.0048 (worst pair uses 64% of its tolerance).
- p vs exact studentized p of `exact_gift.json` (rule 4 x sqrt(p(1-p)/B)): 45 of 45 pass; largest difference 0.0029 (worst uses 50%).
- fwildclusterboot: installed from r-universe/CRAN (0.14.3), but `boottest()` errors on an intercept-only `lm` (both fnw11 and 11 algorithms), so no second reference.
  Adding a regressor would change the test, so it was not done.
